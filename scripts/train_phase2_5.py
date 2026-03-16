from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import time

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.config import ModelConfig, TrainingConfig
from enzyme_software.liquid_nn_v2.data.dataset_loader import create_dataloaders
from enzyme_software.liquid_nn_v2.model.model import LiquidMetabolismNetV2
from enzyme_software.liquid_nn_v2.training.trainer import Trainer


def _initialized_state_dict(model) -> dict:
    state = {}
    uninitialized_type = getattr(torch.nn.parameter, "UninitializedParameter", ())
    for key, value in model.state_dict().items():
        if isinstance(value, uninitialized_type):
            continue
        state[key] = value.detach().cpu() if hasattr(value, "detach") else value
    return state


def _resolve_device(name: str | None):
    if name:
        return torch.device(name)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _latest_candidate(paths):
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return None


def _maybe_override_bool(current: bool, enabled: bool, disabled: bool) -> bool:
    if enabled and disabled:
        raise ValueError("Conflicting enable/disable flags provided")
    if enabled:
        return True
    if disabled:
        return False
    return current


def _should_abort_screen(
    history,
    train_stats,
    val_metrics,
    *,
    loss_jump_ratio: float,
    max_hidden_norm: float,
    max_energy: float,
    max_tunnel_msg_norm: float,
    val_drop_patience: int,
):
    reasons = []
    current_loss = float(train_stats.get("total_loss", float("nan")))
    if not torch.isfinite(torch.tensor(current_loss)):
        reasons.append("non_finite_loss")
    if history:
        prev_loss = float(history[-1]["train"].get("total_loss", current_loss))
        if prev_loss > 0.0 and current_loss > prev_loss * loss_jump_ratio:
            reasons.append(f"loss_jump>{loss_jump_ratio}x")
    atom_norm = float(train_stats.get("atom_hidden_norm_mean", 0.0))
    mol_norm = float(train_stats.get("mol_hidden_norm_mean", 0.0))
    if atom_norm > max_hidden_norm or mol_norm > max_hidden_norm:
        reasons.append("hidden_norm_explosion")
    energy_max = float(train_stats.get("energy_max", 0.0))
    if energy_max > max_energy:
        reasons.append("energy_max_out_of_range")
    tunnel_msg = float(train_stats.get("tunnel_msg_norm_mean", 0.0))
    if tunnel_msg > max_tunnel_msg_norm:
        reasons.append("tunnel_msg_out_of_range")
    if val_drop_patience > 0 and len(history) >= val_drop_patience:
        metric_key = "site_top1_acc"
        current_val = float(val_metrics.get(metric_key, 0.0))
        recent = [float(entry["val"].get(metric_key, 0.0)) for entry in history[-val_drop_patience:]]
        if all(current_val < prev for prev in recent):
            reasons.append("consecutive_val_drop")
    return reasons


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description='Phase 2.5 training on the 580-drug dataset')
    parser.add_argument('--dataset', default='data/training_dataset_580.json')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--device', default=None)
    parser.add_argument('--structure-sdf', default=None)
    parser.add_argument('--model-variant', choices=['baseline', 'light_advanced', 'full_advanced', 'hybrid_selective'], default='baseline')
    parser.add_argument('--use-manual-engine-priors', action='store_true')
    parser.add_argument('--disable-3d', action='store_true')
    parser.add_argument('--warm-start', default=None)
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--early-stopping-patience', type=int, default=20)
    parser.add_argument('--use-energy-module', action='store_true')
    parser.add_argument('--disable-energy-module', action='store_true')
    parser.add_argument('--use-tunneling-module', action='store_true')
    parser.add_argument('--disable-tunneling-module', action='store_true')
    parser.add_argument('--use-graph-tunneling', action='store_true')
    parser.add_argument('--disable-graph-tunneling', action='store_true')
    parser.add_argument('--use-deliberation-loop', action='store_true')
    parser.add_argument('--disable-deliberation-loop', action='store_true')
    parser.add_argument('--num-deliberation-steps', type=int, default=None)
    parser.add_argument('--screen-abort-on-instability', action='store_true')
    parser.add_argument('--loss-jump-ratio', type=float, default=3.0)
    parser.add_argument('--max-hidden-norm', type=float, default=20.0)
    parser.add_argument('--max-energy-max', type=float, default=8.0)
    parser.add_argument('--max-tunnel-msg-norm', type=float, default=15.0)
    parser.add_argument('--val-drop-patience', type=int, default=2)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset not found: {dataset_path}')

    print('=' * 60, flush=True)
    print('PHASE 2.5: Training on 580 Drugs', flush=True)
    print('=' * 60, flush=True)

    device = _resolve_device(args.device)
    print(f'Using device: {device}', flush=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        str(dataset_path),
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        structure_sdf=args.structure_sdf,
    )

    if args.model_variant == 'light_advanced':
        model_config = ModelConfig.light_advanced(
            use_manual_engine_priors=args.use_manual_engine_priors,
            use_3d_branch=not args.disable_3d,
            return_intermediate_stats=True,
        )
    elif args.model_variant == 'full_advanced':
        model_config = ModelConfig.full_advanced(
            use_manual_engine_priors=args.use_manual_engine_priors,
            use_3d_branch=not args.disable_3d,
            return_intermediate_stats=True,
        )
    elif args.model_variant == 'hybrid_selective':
        model_config = ModelConfig.hybrid_selective(
            use_manual_engine_priors=args.use_manual_engine_priors,
            use_3d_branch=not args.disable_3d,
            return_intermediate_stats=True,
        )
    else:
        model_config = ModelConfig.baseline(
            use_manual_engine_priors=args.use_manual_engine_priors,
            use_3d_branch=not args.disable_3d,
            return_intermediate_stats=True,
        )

    model_config.use_energy_module = _maybe_override_bool(
        model_config.use_energy_module,
        args.use_energy_module,
        args.disable_energy_module,
    )
    model_config.use_tunneling_module = _maybe_override_bool(
        model_config.use_tunneling_module,
        args.use_tunneling_module,
        args.disable_tunneling_module,
    )
    model_config.use_graph_tunneling = _maybe_override_bool(
        model_config.use_graph_tunneling,
        args.use_graph_tunneling,
        args.disable_graph_tunneling,
    )
    model_config.use_deliberation_loop = _maybe_override_bool(
        model_config.use_deliberation_loop,
        args.use_deliberation_loop,
        args.disable_deliberation_loop,
    )
    if args.num_deliberation_steps is not None:
        model_config.num_deliberation_steps = max(0, int(args.num_deliberation_steps))
    if not model_config.use_deliberation_loop:
        model_config.num_deliberation_steps = 0

    print(f'model_variant: {args.model_variant}', flush=True)
    print(f'use_3d_branch: {model_config.use_3d_branch}', flush=True)
    print(f'use_manual_engine_priors: {model_config.use_manual_engine_priors}', flush=True)
    print(f'use_energy_module: {model_config.use_energy_module}', flush=True)
    print(f'use_tunneling_module: {model_config.use_tunneling_module}', flush=True)
    print(f'use_graph_tunneling: {model_config.use_graph_tunneling}', flush=True)
    print(f'use_deliberation_loop: {model_config.use_deliberation_loop}', flush=True)
    print(f'num_deliberation_steps: {model_config.num_deliberation_steps}', flush=True)

    model = LiquidMetabolismNetV2(model_config)

    warm_start = args.warm_start
    if warm_start is None:
        warm = _latest_candidate([
            'checkpoints/phase2_530drugs_latest.pt',
            'checkpoints/training_dataset_530_latest.pt',
            'checkpoints/training_dataset_530_best.pt',
        ])
    else:
        warm = Path(warm_start) if Path(warm_start).exists() else None

    if warm is not None:
        print(f'Loading warm start from {warm}', flush=True)
        checkpoint = torch.load(warm, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        print('No Phase 2 checkpoint found; starting from current initialization', flush=True)

    trainer = Trainer(
        model=model,
        config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping_patience,
        ),
        device=device,
    )

    history = []
    best_val_top1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    train_start = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_stats = trainer.train_loader_epoch(train_loader)
        val_metrics = trainer.evaluate_loader(val_loader)
        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_seconds = time.perf_counter() - train_start
        history.append({'epoch': epoch + 1, 'train': train_stats, 'val': val_metrics})
        val_top1 = float(val_metrics.get('site_top1_acc', 0.0))
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_state = _initialized_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (epoch + 1) % max(1, int(args.log_every)) == 0 or epoch == 0:
            avg_epoch_seconds = elapsed_seconds / float(epoch + 1)
            eta_seconds = avg_epoch_seconds * max(0, args.epochs - (epoch + 1))
            print(
                f"Epoch {epoch + 1:3d} | loss={train_stats.get('total_loss', float('nan')):.4f} | "
                f"site_loss={train_stats.get('site_loss', float('nan')):.4f} | "
                f"cyp_loss={train_stats.get('cyp_loss', float('nan')):.4f} | "
                f"energy_loss={train_stats.get('energy_loss', 0.0):.4f} | "
                f"delib_loss={train_stats.get('deliberation_loss', 0.0):.4f} | "
                f"site_top1={val_metrics.get('site_top1_acc', 0.0):.3f} | "
                f"site_top3={val_metrics.get('site_top3_acc', 0.0):.3f} | "
                f"cyp_acc={val_metrics.get('accuracy', 0.0):.3f} | "
                f"cyp_f1={val_metrics.get('f1_macro', 0.0):.3f} | "
                f"energy={train_stats.get('energy_mean', 0.0):.3f} | "
                f"tunnel={train_stats.get('tunnel_prob_mean', 0.0):.3f} | "
                f"tmsg={train_stats.get('tunnel_msg_norm_mean', 0.0):.3f} | "
                f"tau={train_stats.get('tau_mean', 0.0):.3f} | "
                f"critic={train_stats.get('critic_mean', 0.0):.3f} | "
                f"atom_norm={train_stats.get('atom_hidden_norm_mean', 0.0):.3f} | "
                f"mol_norm={train_stats.get('mol_hidden_norm_mean', 0.0):.3f} | "
                f"epoch_time={epoch_seconds:.1f}s | "
                f"elapsed={elapsed_seconds/60.0:.1f}m | "
                f"eta={eta_seconds/60.0:.1f}m",
                flush=True,
            )
        if args.screen_abort_on_instability:
            stop_reasons = _should_abort_screen(
                history[:-1],
                train_stats,
                val_metrics,
                loss_jump_ratio=args.loss_jump_ratio,
                max_hidden_norm=args.max_hidden_norm,
                max_energy=args.max_energy_max,
                max_tunnel_msg_norm=args.max_tunnel_msg_norm,
                val_drop_patience=args.val_drop_patience,
            )
            if stop_reasons:
                print(
                    f"Screening abort after epoch {epoch + 1}: {', '.join(stop_reasons)}",
                    flush=True,
                )
                break
        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping after epoch {epoch + 1}: no site_top1 improvement for "
                f"{args.early_stopping_patience} epochs.",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    print('\n' + '=' * 60, flush=True)
    print('TEST SET EVALUATION', flush=True)
    print('=' * 60, flush=True)
    test_metrics = trainer.evaluate_loader(test_loader)
    print(json.dumps(test_metrics, indent=2), flush=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / 'phase2_5_580drugs_latest.pt'
    archive_path = output_dir / f'phase2_5_580drugs_{timestamp}.pt'
    report_path = output_dir / f'phase2_5_report_{timestamp}.json'

    checkpoint = {
        'phase': 2.5,
        'num_drugs': 580,
        'model_state_dict': _initialized_state_dict(model),
        'config': model_config.__dict__,
        'training_config': TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ).__dict__,
        'best_val_top1': best_val_top1,
        'test_metrics': test_metrics,
        'history': history,
    }
    torch.save(checkpoint, latest_path)
    torch.save(checkpoint, archive_path)
    report_path.write_text(json.dumps({'phase': 2.5, 'num_drugs': 580, 'best_val_top1': best_val_top1, 'test_metrics': test_metrics}, indent=2))

    print(f'\nSaved checkpoint: {archive_path}', flush=True)
    print(f'Saved latest checkpoint: {latest_path}', flush=True)
    print(f'Saved report: {report_path}', flush=True)


if __name__ == '__main__':
    main()
