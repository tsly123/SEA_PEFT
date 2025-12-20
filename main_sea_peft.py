"""Entry point for SA-PEFT search episodes."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from fseft.datasets.configs import get_task_config
from fseft.datasets.utils import load_query
from utils.misc import set_seeds

from sa_peft.episode import EpisodeConfig, SAPEFTEpisode
from main_fseft import test


def schedule_for_shots(args, k: int) -> EpisodeConfig:
    if k == 1:
        if args.tsteps is None:
            steps = 6000
        else:
            steps = args.tsteps
    elif k == 5:
        if args.tsteps is None:
            steps = 8000
        else:
            steps = args.tsteps
    elif k == 10:
        steps = 12000
    else:
        steps = 6000
    return EpisodeConfig(episode_steps=steps, refinement_steps=200)

def run(args) -> None:
    args.method = "sa_peft"
    get_task_config(args)
    manifest_path = args.manifest_path or None
    config = schedule_for_shots(args, args.k)
    config.param_budget = args.param_budget
    config.use_promotion_fsm = args.enable_promotion_fsm
    config.top_k = args.top_k
    config.max_single_swaps = args.max_single_swaps
    config.use_adapter_lr_restarts = args.use_adapter_lr_restarts
    config.adapter_restart_steps = args.adapter_restart_steps
    out_path = Path(args.out_path).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    args.out_path = str(out_path)
    log_root = out_path / "logs"
    run_name = f"{args.organ}/shot{args.k}"
    args.path_results = os.path.join(args.out_path, "logs", run_name)
    dice_all = []
    # expose CLI overrides for downstream components
    args.validation_batches = max(1, args.validation_batches)
    args.final_config_dir = args.final_config_dir
    print(args)
    for fold in args.folds:
        args.iFold = fold
        print(f"SA-PEFT_fold_{fold}")
        episode = SAPEFTEpisode(
            args,
            manifest_path=manifest_path,
            config=config,
            run_name=f"{run_name}/fold-{fold}",
            log_root=log_root,
            disable_progress=args.disable_progress,
            force_progress=args.force_progress,
        )
        t0 = time.time()
        episode.run()
        t1 = time.time()
        minutes, seconds = divmod(t1 - t0, 60)
        hours, minutes = divmod(minutes, 60)
        print(f"Episode time: {int(hours)}:{int(minutes):02d}:{int(seconds):02d}")
        dice_test = []
        for iFold in range(args.ntest):
            args.iFold = iFold
            load_query(args)
            dice_sample = test(args, episode.model)
            dice_test.append(dice_sample)
        dice_test = dict(
            zip(
                list(dice_test[0].keys()),
                [
                    round(sum(sample[organ] for sample in dice_test) * 100 / len(dice_test), 2)
                    for organ in args.selected_organs
                ],
            )
        )
        dice_all.append(dice_test)
    dice_all = dict(
        zip(
            list(dice_all[0].keys()),
            [
                round(sum(sample[organ] for sample in dice_all) / len(dice_all), 2)
                for organ in args.selected_organs
            ],
        )
    )
    print(f"dice_all: {dice_all}")
    # path_results = os.path.join(args.out_path, f"{args.organ}_results.json")
    path_results = os.path.join(args.path_results, f"{args.organ}_results.json")
    exp_id = f"sa_peft_k_{args.k}_{args.decoder}Decoder"
    if not os.path.isfile(path_results):
        data = {exp_id: dice_all}
    else:
        with open(path_results) as infile:
            data = json.load(infile)
        data[exp_id] = dice_all
    with open(path_results, "w") as fp:
        json.dump(data, fp)
    print("Results across folds.")
    print(f"Average DICE: {dice_all}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", default="./results/")
    parser.add_argument("--model_id", default="fseft")
    parser.add_argument("--dataset", default="flare")
    parser.add_argument('--organ', default='pancreas',
                        help='Target organ - e.g.: selected - spleen - kidney_left - gallbladder - esophagus          '
                             '                     liver - pancreas - stomach - duodenum - aorta - heart_myocardium   '
                             '                     heart_atrium_left - heart_atrium_right - heart_ventricle_left      '
                             '                     heart_ventricle_right - lung_lower_lobe_left                       '
                             '                     lung_lower_lobe_right - lung_middle_lobe_right                     '
                             '                     lung_upper_lobe_left - lung_upper_lobe_right                       '
                             '                     gluteus_maximus_left - gluteus_maximus_right - gluteus_medius_left '
                             '                     gluteus_medius_right - gluteus_minimus_left - gluteus_minimus_right'
                        )
    parser.add_argument("--k", default=1, type=int)
    parser.add_argument("--seeds", default=1, type=int)
    parser.add_argument("--decoder", default="frozen")
    parser.add_argument("--bottleneck", default="frozen")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--post_process", default=False, type=lambda x: str(x).lower() == "true")
    parser.add_argument("--visualization", default=False, type=lambda x: str(x).lower() == "true")
    parser.add_argument("--objective", default="binary")
    parser.add_argument("--early_stop_criteria", default="train")
    parser.add_argument("--manifest_path", default="./sa_peft/manifest/default_composite.yaml")
    parser.add_argument("--param_budget", default=0.05, type=float)
    parser.add_argument("--enable_promotion_fsm", action="store_false")
    parser.add_argument("--top_k", type=int, default=5, help="Number of configurations to keep during final sweep")
    parser.add_argument("--max_single_swaps", type=int, default=5, help="Maximum single-swap neighbors to evaluate")
    parser.add_argument("--final_config_dir", default=None, help="Directory for exporting final configuration files")
    parser.add_argument("--validation_batches", type=int, default=2, help="Number of held-out validation batches")
    parser.add_argument("--use_adapter_lr_restarts", action="store_false", help="Enable cosine annealing warm restarts for adapter LR")
    parser.add_argument("--no_adapter_lr_restarts", dest="use_adapter_lr_restarts", action="store_false", help="Disable adapter LR restarts")
    parser.add_argument("--adapter_restart_steps", type=int, default=None, help="Restart period for adapter LR (defaults to audit_cadence)")
    parser.add_argument("--disable_progress", action="store_true", help="Disable live progress bars and console summaries")
    parser.add_argument("--force_progress", action="store_true", help="Force-enable progress bars even when stdout is not a TTY")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for SA-PEFT episodes")
    parser.add_argument("--tsteps", type=int, default=None, help="custom total steps for SA-PEFT episodes")
    parser.add_argument("--enable_refinement_delegation", action="store_false")
    args, _ = parser.parse_known_args()

    # Check training hardware gpu/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set seeds for reproducibility
    set_seeds(42, use_cuda=device == 'cuda')
    run(args)


if __name__ == "__main__":
    main()
