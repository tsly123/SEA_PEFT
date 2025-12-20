"""PEFT unit registry and wrappers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MethodType
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover
    from torch.nn import LayerNorm, BatchNorm1d, BatchNorm2d, BatchNorm3d

from .manifest import FamilySpec, LevelSpec, Manifest, SiteSpec


class Attachment(nn.Module):
    """Base class for additive PEFT attachments."""

    def __init__(self) -> None:
        super().__init__()
        self.active: bool = False
        self.register_buffer("alpha", torch.tensor(1.0))

    def configure(self, active: bool, alpha: float = 1.0) -> None:
        self.active = active
        self.alpha.fill_(alpha)
        for param in self.parameters():
            param.requires_grad = active

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return base
        return base + self.alpha * self.forward_delta(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LoRAAttachment(Attachment):
    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class AdapterAttachment(Attachment):
    def __init__(self, in_features: int, out_features: int, bottleneck: int, activation: nn.Module) -> None:
        super().__init__()
        self.down = nn.Linear(in_features, bottleneck, bias=True)
        self.act = activation
        self.up = nn.Linear(bottleneck, out_features, bias=True)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


class IA3Attachment(Attachment):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.scales = nn.Parameter(torch.ones(features))

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return base
        return base * (1.0 + self.alpha * self.scales)

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - unused
        raise RuntimeError("IA3Attachment does not support forward_delta")

    @property
    def num_parameters(self) -> int:  # type: ignore[override]
        return self.scales.numel()


class BitFitController:
    def __init__(self, linear: nn.Linear) -> None:
        self.linear = linear
        self.active = False

    def configure(self, active: bool) -> None:
        self.active = active
        bias = self._ensure_bias()
        bias.requires_grad = active

    def _ensure_bias(self) -> nn.Parameter:
        if self.linear.bias is None:
            self.linear.bias = nn.Parameter(torch.zeros(self.linear.out_features, device=self.linear.weight.device))
        return self.linear.bias

    def parameters(self) -> List[nn.Parameter]:
        if self.linear.bias is None:
            return []
        return [self.linear.bias]


class PEFTLinearWrapper(nn.Module):
    """Wrapper that hosts PEFT attachments around a linear layer."""

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        self.base = base
        self.attachments: nn.ModuleDict = nn.ModuleDict()
        self.active_keys: List[str] = []
        self.bitfit = BitFitController(self.base)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        for key in self.active_keys:
            out = self.attachments[key](x, out)
        if self.bitfit.active:
            bias = self.bitfit._ensure_bias()
            out = out + bias
        return out

    def _key(self, family: str, topology: str, level: str) -> str:
        return f"{family}:{topology}:{level}"

    def _make_attachment(self, family: str, topology: str, level: LevelSpec) -> Attachment:
        if family == "lora":
            rank = level.params.get("rank")
            if rank is None:
                raise ValueError("LoRA level missing 'rank'")
            return LoRAAttachment(self.in_features, self.out_features, rank)
        if family == "adapter":
            bottleneck = level.params.get("bottleneck")
            if bottleneck is None:
                raise ValueError("Adapter level missing 'bottleneck'")
            return AdapterAttachment(self.in_features, self.out_features, bottleneck, nn.ReLU())
        if family == "adaptformer":
            bottleneck = level.params.get("bottleneck")
            if bottleneck is None:
                raise ValueError("AdaptFormer level missing 'bottleneck'")
            return AdapterAttachment(self.in_features, self.out_features, bottleneck, nn.GELU())
        if family == "ia3":
            return IA3Attachment(self.out_features)
        raise ValueError(f"Unsupported family '{family}'")

    def configure(self, family: str, topology: str, level: LevelSpec, alpha: float = 1.0) -> None:
        if family == "bitfit":
            self.bitfit.configure(True)
            return
        key = self._key(family, topology, level.name)
        if key in self.attachments:
            attachment = self.attachments[key]
        else:
            attachment = self._make_attachment(family, topology, level)
            attachment = attachment.to(self.base.weight.device)
            self.attachments[key] = attachment
        attachment.configure(True, alpha)
        if key not in self.active_keys:
            self.active_keys.append(key)

    def disable_family(self, family: str) -> None:
        if family == "bitfit":
            self.bitfit.configure(False)
            return
        prefix = family + ":"
        for key in list(self.active_keys):
            if key.startswith(prefix):
                self.attachments[key].configure(False, 0.0)
                self.active_keys.remove(key)

    def parameters_for(self, family: str) -> Iterable[nn.Parameter]:
        if family == "bitfit":
            return self.bitfit.parameters()
        params: List[nn.Parameter] = []
        prefix = family + ":"
        for key, attachment in self.attachments.items():
            if key.startswith(prefix):
                params.extend(p for p in attachment.parameters() if p.requires_grad)
        return params


class AffineNormController:
    """Controls affine norm parameters via requires_grad toggling."""

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.alpha: float = 1.0

    def configure(self, active: bool, alpha: float = 1.0) -> None:
        self.alpha = alpha
        requires = bool(active and alpha > 0.0)
        weight = getattr(self.module, "weight", None)
        bias = getattr(self.module, "bias", None)
        if weight is not None:
            weight.requires_grad = requires
        if bias is not None:
            bias.requires_grad = requires

    def parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        weight = getattr(self.module, "weight", None)
        bias = getattr(self.module, "bias", None)
        if weight is not None:
            params.append(weight)
        if bias is not None:
            params.append(bias)
        return params


class PEFTNormWrapper(nn.Module):
    """Wrapper for LayerNorm / BatchNorm style modules supporting affine gating."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.controller = AffineNormController(base)
        self.active: bool = False

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def configure(self, family: str, topology: str, level: LevelSpec, alpha: float = 1.0) -> None:
        if family != "affine":
            raise ValueError(f"Unsupported family '{family}' for norm wrapper")
        self.controller.configure(True, alpha)
        self.active = True

    def disable_family(self, family: str) -> None:
        if family != "affine":
            return
        self.controller.configure(False, 0.0)
        self.active = False

    def parameters_for(self, family: str) -> Iterable[nn.Parameter]:
        if family != "affine":
            return []
        return [p for p in self.controller.parameters() if p.requires_grad]

    @property
    def in_features(self) -> int:
        weight = getattr(self.base, "weight", None)
        if weight is None:
            return 0
        return weight.numel()

    @property
    def out_features(self) -> int:
        return self.in_features


class SwinFFNCompositeWrapper(nn.Module):
    """Wrapper that hosts PEFT attachments around Swin block forward_part2."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "_block", block)
        self.attachments: nn.ModuleDict = nn.ModuleDict()
        self.active_keys: List[str] = []
        self.features = getattr(block, "dim", 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block = getattr(self, "_block")
        normed = block.norm2(x)
        base = block.drop_path(block.mlp(normed))
        out = base
        for key in self.active_keys:
            out = self.attachments[key](normed, out)
        return out

    @property
    def in_features(self) -> int:
        return self.features

    @property
    def out_features(self) -> int:
        return self.features

    def _key(self, family: str, topology: str, level: str) -> str:
        return f"{family}:{topology}:{level}"

    def _device(self) -> torch.device:
        block = getattr(self, "_block")
        for param in block.parameters():
            return param.device
        return torch.device("cpu")

    def _make_attachment(self, family: str, topology: str, level: LevelSpec) -> Attachment:
        if family == "lora":
            rank = level.params.get("rank")
            if rank is None:
                raise ValueError("LoRA level missing 'rank'")
            return LoRAAttachment(self.in_features, self.out_features, rank)
        if family in {"adapter", "adaptformer"}:
            bottleneck = level.params.get("bottleneck")
            if bottleneck is None:
                raise ValueError("Adapter level missing 'bottleneck'")
            activation: nn.Module
            activation = nn.ReLU() if family == "adapter" else nn.GELU()
            return AdapterAttachment(self.in_features, self.out_features, bottleneck, activation)
        raise ValueError(f"Unsupported family '{family}' for composite wrapper")

    def configure(self, family: str, topology: str, level: LevelSpec, alpha: float = 1.0) -> None:
        key = self._key(family, topology, level.name)
        if key in self.attachments:
            attachment = self.attachments[key]
        else:
            attachment = self._make_attachment(family, topology, level)
            attachment = attachment.to(self._device())
            self.attachments[key] = attachment
        attachment.configure(True, alpha)
        if key not in self.active_keys:
            self.active_keys.append(key)

    def disable_family(self, family: str) -> None:
        prefix = family + ":"
        for key in list(self.active_keys):
            if key.startswith(prefix):
                self.attachments[key].configure(False, 0.0)
                self.active_keys.remove(key)

    def parameters_for(self, family: str) -> Iterable[nn.Parameter]:
        params: List[nn.Parameter] = []
        prefix = family + ":"
        for key, attachment in self.attachments.items():
            if key.startswith(prefix):
                params.extend(p for p in attachment.parameters() if p.requires_grad)
        return params


@dataclass
class UnitState:
    active: bool = False
    guard: int = 0
    promotion_votes: int = 0
    demotion_votes: int = 0
    last_transition: Optional[Tuple[str, str, int, int]] = None


@dataclass(eq=False)
class PEFTUnit:
    name: str
    site: SiteSpec
    family: str
    topology: str
    level: LevelSpec
    module_name: str
    wrapper: Union[PEFTLinearWrapper, PEFTNormWrapper, SwinFFNCompositeWrapper]
    params: int
    level_index: int
    state: UnitState = field(default_factory=UnitState)
    site_instance: str = ""

    def activate(self, alpha: float = 1.0) -> None:
        self.wrapper.configure(self.family, self.topology, self.level, alpha)
        self.state.active = True

    def deactivate(self) -> None:
        self.wrapper.disable_family(self.family)
        self.state = UnitState()

    def param_count(self) -> int:
        return self.params if self.state.active else 0

    def parameters(self) -> List[nn.Parameter]:
        return list(self.wrapper.parameters_for(self.family))

    @property
    def block_name(self) -> str:
        parts = self.module_name.split(".")
        block_parts: List[str] = []
        for part in parts:
            block_parts.append(part)
            if part.startswith("blocks"):
                break
        if not block_parts:
            block_parts = parts[:-1]
        return ".".join(block_parts) if block_parts else self.module_name

    def __hash__(self) -> int:
        return hash(self.name)


class UnitRegistry:
    SUPPORTED_FAMILIES = {"lora", "adapter", "adaptformer", "ia3", "bitfit", "affine"}

    def __init__(self, manifest: Manifest, model: nn.Module) -> None:
        self.manifest = manifest
        self.model = model
        self.units: List[PEFTUnit] = []
        self.lookup: Dict[str, PEFTUnit] = {}
        self._build_units()

    def _build_units(self) -> None:
        name_to_module = dict(self.model.named_modules())
        self._ensure_composite_sites(name_to_module)
        name_to_module = dict(self.model.named_modules())
        wrappers: Dict[str, Union[PEFTLinearWrapper, PEFTNormWrapper, SwinFFNCompositeWrapper]] = {}
        for site, module_name in self.manifest.iter_site_matches(name_to_module.keys()):
            module = name_to_module[module_name]
            wrapper = wrappers.get(module_name)

            if isinstance(module, nn.Linear):
                if wrapper is None:
                    parent_name, child_name = module_name.rsplit(".", 1)
                    parent = name_to_module[parent_name]
                    wrapper = PEFTLinearWrapper(module)
                    setattr(parent, child_name, wrapper)
                    name_to_module[module_name] = wrapper
                    wrappers[module_name] = wrapper
                self._register_units(site, module_name, wrapper)
                continue

            if isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
                if wrapper is None:
                    parent_name, child_name = module_name.rsplit(".", 1)
                    parent = name_to_module[parent_name]
                    wrapper = PEFTNormWrapper(module)
                    setattr(parent, child_name, wrapper)
                    name_to_module[module_name] = wrapper
                    wrappers[module_name] = wrapper
                self._register_units(site, module_name, wrapper)
                continue

            if isinstance(module, SwinFFNCompositeWrapper):
                if wrapper is None:
                    wrapper = module
                    wrappers[module_name] = wrapper
                self._register_units(site, module_name, wrapper)
                continue

    def _register_units(
        self,
        site: SiteSpec,
        module_name: str,
        wrapper: Union[PEFTLinearWrapper, PEFTNormWrapper, SwinFFNCompositeWrapper],
    ) -> None:
        for family in site.families.values():
            if family.name not in self.SUPPORTED_FAMILIES:
                continue
            for topo in family.topologies.values():
                if family.name == "adaptformer" and topo.name != "pa":
                    continue
                for idx, level in enumerate(topo.levels):
                    params = self._estimate_params(family, wrapper, level)
                    unit_name = f"{module_name}:{family.name}:{topo.name}:{level.name}"
                    unit = PEFTUnit(
                        name=unit_name,
                        site=site,
                        family=family.name,
                        topology=topo.name,
                        level=level,
                        module_name=module_name,
                        wrapper=wrapper,
                        params=params,
                        level_index=idx,
                        site_instance=(
                            f"{self._derive_block_scope(module_name)}:{site.name}"
                            if site.scope == "block"
                            else site.name
                        ),
                    )
                    self.units.append(unit)
                    self.lookup[unit_name] = unit

    @staticmethod
    def _derive_block_scope(module_name: str) -> str:
        parts = module_name.split(".")
        block_parts: List[str] = []
        for part in parts:
            block_parts.append(part)
            if part.startswith("blocks"):
                break
        if not block_parts:
            block_parts = parts[:-1]
        return ".".join(block_parts) if block_parts else module_name

    @staticmethod
    def _estimate_params(
        family: FamilySpec,
        wrapper: Union[PEFTLinearWrapper, PEFTNormWrapper, SwinFFNCompositeWrapper],
        level: LevelSpec,
    ) -> int:
        if family.name == "lora":
            rank = level.params.get("rank", 0)
            return rank * (wrapper.in_features + wrapper.out_features)
        if family.name in {"adapter", "adaptformer"}:
            bottleneck = level.params.get("bottleneck", 0)
            return bottleneck * (wrapper.in_features + wrapper.out_features)
        if family.name == "ia3":
            return wrapper.out_features
        if family.name == "bitfit":
            return wrapper.out_features
        if family.name == "affine":
            params = 0
            for param in wrapper.parameters_for("affine"):
                params += param.numel()
            return params
        return 0

    def _ensure_composite_sites(self, name_to_module: Dict[str, nn.Module]) -> None:
        for module in list(name_to_module.values()):
            if not hasattr(module, "mlp") or not hasattr(module, "norm2") or not hasattr(module, "forward_part2"):
                continue
            if hasattr(module, "ffn_peft"):
                continue
            wrapper = SwinFFNCompositeWrapper(module)
            module.add_module("ffn_peft", wrapper)

            def _forward_part2_with_wrapper(self, x, _wrapper=wrapper):  # type: ignore[override]
                return _wrapper(x)

            module.forward_part2 = MethodType(_forward_part2_with_wrapper, module)


    def __iter__(self) -> Iterator[PEFTUnit]:
        return iter(self.units)

    def get(self, name: str) -> PEFTUnit:
        return self.lookup[name]

    def active_param_count(self) -> int:
        return sum(unit.param_count() for unit in self.units)

    def units_by_block(self) -> Dict[str, List[PEFTUnit]]:
        blocks: Dict[str, List[PEFTUnit]] = {}
        for unit in self.units:
            blocks.setdefault(unit.block_name, []).append(unit)
        return blocks


__all__ = ["PEFTUnit", "UnitRegistry", "UnitState"]
