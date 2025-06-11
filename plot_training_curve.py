import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Plot training reward curve from CSV log")
    parser.add_argument("csv_path", type=str, help="Path to train_log_<N>.csv")
    parser.add_argument("--show", action="store_true", help="Show plot instead of saving")
    parser.add_argument("--output", type=str, default=None, help="Path to save figure (PNG)")
    args = parser.parse_args()

    csv_file = Path(args.csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Cannot find {csv_file}")

    data = pd.read_csv(csv_file)

    required = {'epoch', 'income', 'expense'}
    if not required.issubset(data.columns):
        raise ValueError("CSV must contain columns 'epoch', 'income' and 'expense'")

    reward = data['income'] - data['expense']

    plt.figure(figsize=(8, 4))
    plt.plot(data['epoch'], reward, label='Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Training Reward Curve')
    plt.grid(True)
    plt.legend()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight')
    if args.show or not args.output:
        plt.show()


if __name__ == '__main__':
    main()
