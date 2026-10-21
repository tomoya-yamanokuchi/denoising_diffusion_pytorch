import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PRODUCTS = ['Polisher', 'PowerCutter', 'SheetSander']
METRICS = {
    'cutting_error_volume': ('Cutting Error Volume [voxels]', [6, 0, 0], [0, 0, 0], [1, 0, 84], [2.45, 0, 80.14]),
    'part_remaining_rate': ('Part Remaining Rate [%]', [82.86, 100, 100], [0, 0, 0], [97.14, 100, 44.93], [7, 0, 50.45]),
    'part_occupancy_rate': ('Part Occupancy Rate [%]', [46.99, 6.58, 59.74], [3.29, 0, 0], [45.09, 5.85, 18.06], [3.75, 0, 22.89]),
}

def main():
    parser = argparse.ArgumentParser(description='Compare I2SB and Conditional Diffusion across three products.')
    parser.add_argument('--out_dir', type=Path, default=Path('analysis/model_comparison/figures'))
    parser.add_argument('--swap_labels', action='store_true', help='Use first set for Conditional Diffusion and second set for I2SB.')
    parser.add_argument('--format', choices=['png', 'pdf', 'both'], default='both')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    names = ['I2SB', 'Conditional Diffusion'] if not args.swap_labels else ['Conditional Diffusion', 'I2SB']
    x = np.arange(len(PRODUCTS))
    width = 0.36
    for key, (ylabel, first_mean, first_std, second_mean, second_std) in METRICS.items():
        fig, ax = plt.subplots(figsize=(8.3, 4.6))
        for offset, means, stds, label in [(-width / 2, first_mean, first_std, names[0]), (width / 2, second_mean, second_std, names[1])]:
            ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=label, edgecolor='black', linewidth=0.5, error_kw={'elinewidth': 1.1, 'capthick': 1.1})
        ax.set_xticks(x, PRODUCTS)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        if key != 'cutting_error_volume':
            ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        fig.tight_layout()
        formats = ['png', 'pdf'] if args.format == 'both' else [args.format]
        for ext in formats:
            path = args.out_dir / f'{key}.{ext}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f'Saved: {path}')
        plt.close(fig)

if __name__ == '__main__':
    main()
