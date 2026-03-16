from __future__ import annotations

import argparse
from pathlib import Path

from enzyme_software.liquid_nn_v2.analysis.visualization import export_architecture_html
from enzyme_software.liquid_nn_v2.config import ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate an interactive HTML graph for LiquidMetabolismNetV2')
    parser.add_argument('--output', default='artifacts/liquid_nn_v2_architecture.html')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--liquid-layers', type=int, default=2)
    parser.add_argument('--ode-steps', type=int, default=6)
    parser.add_argument('--model-variant', choices=['baseline', 'light_advanced', 'full_advanced', 'hybrid_selective'], default='baseline')
    args = parser.parse_args()

    if args.model_variant == 'light_advanced':
        config = ModelConfig.light_advanced(hidden_dim=args.hidden_dim, num_liquid_layers=args.liquid_layers, ode_steps=args.ode_steps)
    elif args.model_variant == 'full_advanced':
        config = ModelConfig.full_advanced(hidden_dim=args.hidden_dim, num_liquid_layers=args.liquid_layers, ode_steps=args.ode_steps)
    elif args.model_variant == 'hybrid_selective':
        config = ModelConfig.hybrid_selective(hidden_dim=args.hidden_dim, num_liquid_layers=args.liquid_layers, ode_steps=args.ode_steps)
    else:
        config = ModelConfig.baseline(hidden_dim=args.hidden_dim, num_liquid_layers=args.liquid_layers, ode_steps=args.ode_steps)
    output = export_architecture_html(args.output, config=config, checkpoint_path=args.checkpoint)
    print(f'architecture_html={Path(output).resolve()}')


if __name__ == '__main__':
    main()
