"""Redraw the published figures from the bundled CSVs (NumPy + Matplotlib)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
KEYS = [(m, t) for m in ['1p5b', '7b'] for t in ['countdown', 'gsm8k', 'math']]
MODEL = {'1p5b': '1.5B', '7b': '7B'}
TASK = {'countdown': 'Countdown', 'gsm8k': 'GSM8K', 'math': 'MATH'}


def read(name):
    with (ROOT / 'results' / name).open() as handle:
        return list(csv.DictReader(handle))


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    grad = {r['condition']: r for r in read('gradient_metrics.csv') if r['predictor'] == 'predictor'}
    repeat = {(r['model'], r['task']): r for r in read('seed_replications.csv') if r['comparison'] == 'adaptive - mean'}
    labels = [f'{MODEL[m]} / {TASK[t]}' for m, t in KEYS]
    y = np.arange(6)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6))
    for target, offset, color, label in [('activation', -.18, '#2167b0', 'Activation gradient'), ('lora', .18, '#e08c24', 'LoRA A/B gradient')]:
        values = [float(grad[f'{m}_{t}_gentle'][target + '_cosine']) for m, t in KEYS]
        axes[0].barh(y + offset, values, height=.32, color=color, label=label)
        for yi, value in zip(y + offset, values):
            axes[0].text(value + .015, yi, f'{value:.3f}', va='center', fontsize=9)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.11)
    axes[0].set_xlabel('Per-question cosine on 128 audit questions')
    axes[0].set_title('Static prediction at gentle warmup\nPrimary seed; activation and LoRA differ', loc='left', pad=12)
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2, fontsize=9)
    delta = np.array([float(repeat[k]['delta_pp']) for k in KEYS])
    low = np.array([float(repeat[k]['ci95_low_pp']) for k in KEYS])
    high = np.array([float(repeat[k]['ci95_high_pp']) for k in KEYS])
    axes[1].axvline(0, color='#9298a0', linewidth=1)
    axes[1].errorbar(delta, y, xerr=[delta-low, high-delta], fmt='o', color='#2167b0', capsize=4)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-3, 2.4)
    axes[1].set_xlabel('Adaptive minus Mean accuracy (percentage points)')
    axes[1].set_title('Downstream results after 32 updates\nThree-seed mean and paired question 95% CI', loc='left', pad=12)
    for ax in axes:
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('Gentle warmup: learnable gradients, no clear gain over Mean', fontsize=15, y=.98)
    fig.text(.5, .025, 'Repeated seeds share warmup and splits. Intervals condition on these runs; all six cross zero.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .91))
    fig.savefig(ROOT / 'figures/overview.png', dpi=160)
    plt.close(fig)

    rows = read('step_metrics.csv')
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.1), sharex=True, sharey=True)
    for ax, (m, t) in zip(axes.flat, KEYS):
        rr = [r for r in rows if r['condition'] == f'{m}_{t}_gentle']
        assert len(rr) == 32
        x = [int(r['step']) for r in rr]
        for metric, color, style, label in [('cosine', '#2167b0', '-', 'Cosine (ideal 1)'),
                                             ('relative_l2', '#b84842', '--', 'Relative L2 (ideal 0)'),
                                             ('norm_ratio', '#23836f', ':', 'Norm ratio (ideal 1)')]:
            ax.plot(x, [float(r['validation16_after_activation_' + metric]) for r in rr], color=color,
                    linestyle=style, linewidth=2, label=label)
        ax.set_title(f"{MODEL[m]} / {TASK[t]}  |  adapter LR {float(rr[0]['lr']):g}")
        ax.set_ylim(-.04, 1.04)
        ax.set_xticks([1, 8, 16, 24, 32])
        ax.grid(alpha=.18)
    fig.suptitle('Gentle Adaptive: activation-gradient metrics at every step', fontsize=15)
    fig.text(.5, .92, 'Fixed 16 checkpoint-selection questions; after predictor selection, before the LoRA update', ha='center', fontsize=10)
    fig.supxlabel('Step t: evaluated after t-1 completed adapter updates', y=.09)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, .01))
    fig.tight_layout(rect=(0, .13, 1, .9))
    fig.savefig(ROOT / 'figures/gentle_steps.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
