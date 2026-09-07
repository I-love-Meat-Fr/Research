"""
Quick start script: tests all 3 models with minimal training.

Usage:
    python src/quick_start.py
    python src/quick_start.py --model fdt --epochs 3
"""
import sys
import subprocess
from pathlib import Path

# Run smoke test for each model
models = ['fdt', 'ilcnn', 'gla-fd']

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='all', choices=['all'] + models)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    to_test = models if args.model == 'all' else [args.model]

    for model_name in to_test:
        print(f'\n{"="*70}')
        print(f'TESTING: {model_name}')
        print(f'{"="*70}')

        cmd = [
            'python3', 'src/train.py',
            '--model', model_name,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--quick_test',
        ]

        subprocess.run(cmd, cwd=Path(__file__).parent.parent)