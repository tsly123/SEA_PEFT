"""Manifest loader for SA-PEFT search space."""

from __future__ import annotations

import fnmatch
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import yaml


@dataclass
class LevelSpec:
    name: str
    params: Dict[str, int]


@dataclass
class TopologySpec:
    name: str
    levels: List[LevelSpec]


@dataclass
class FamilySpec:
    name: str
    topologies: Dict[str, TopologySpec]


@dataclass
class SiteSpec:
    name: str
    pattern: str
    capacity_key: str
    families: Dict[str, FamilySpec]
    scope: str = "global"


@dataclass
class Manifest:
    version: int
    sites: Dict[str, SiteSpec]

    def iter_site_matches(self, module_names: Iterable[str]):
        for site in self.sites.values():
            for module_name in module_names:
                if fnmatch.fnmatch(module_name, site.pattern):
                    yield site, module_name


def _load_levels(raw_levels: List[dict]) -> List[LevelSpec]:
    levels: List[LevelSpec] = []
    for entry in raw_levels:
        if "name" not in entry:
            raise ValueError("Level entry missing 'name'")
        params = {k: v for k, v in entry.items() if k != "name"}
        levels.append(LevelSpec(name=entry["name"], params=params))
    return levels


def _load_topologies(raw_tops: List[dict]) -> Dict[str, TopologySpec]:
    topo: Dict[str, TopologySpec] = {}
    for entry in raw_tops:
        name = entry.get("name")
        if not name:
            raise ValueError("Topology entry missing 'name'")
        topo[name] = TopologySpec(name=name, levels=_load_levels(entry.get("levels", [])))
    return topo


def _load_families(raw_families: List[dict]) -> Dict[str, FamilySpec]:
    families: Dict[str, FamilySpec] = {}
    for entry in raw_families:
        name = entry.get("name")
        if not name:
            raise ValueError("Family entry missing 'name'")
        families[name] = FamilySpec(name=name, topologies=_load_topologies(entry.get("topologies", [])))
    return families


def _load_sites(raw_sites: List[dict]) -> Dict[str, SiteSpec]:
    sites: Dict[str, SiteSpec] = {}
    for entry in raw_sites:
        name = entry.get("name")
        if not name:
            raise ValueError("Site entry missing 'name'")
        pattern = entry.get("pattern")
        if not pattern:
            raise ValueError(f"Site '{name}' missing 'pattern'")
        capacity_key = entry.get("capacity_key", "out_features")
        scope = entry.get("scope", "global")
        if scope not in {"global", "block"}:
            raise ValueError(
                f"Site '{name}' has invalid scope '{scope}'. Expected 'global' or 'block'."
            )
        sites[name] = SiteSpec(
            name=name,
            pattern=pattern,
            capacity_key=capacity_key,
            families=_load_families(entry.get("families", [])),
            scope=scope,
        )
    return sites


def load_manifest(path: Optional[str] = None) -> Manifest:
    if path is None:
        path = pathlib.Path(__file__).with_name("default.yaml")
    else:
        path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    version = raw.get("version", 1)
    sites = _load_sites(raw.get("sites", []))
    return Manifest(version=version, sites=sites)


__all__ = [
    "Manifest",
    "SiteSpec",
    "FamilySpec",
    "TopologySpec",
    "LevelSpec",
    "load_manifest",
]
