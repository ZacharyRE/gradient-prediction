"""Publication figure from released paired metrics; no models or GPUs required."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / 'results/summary.json').read_text())
rows = list(csv.DictReader((ROOT / 'results/paired_updates.csv').open()))
primary = {(int(r['seed']), int(r['probe']), r['method']): float(r['gradient_relative_l2'])
           for r in rows if r['model_step'] == '32' and r['method'] in ('carry', 'reset')}
colors = dict(baseline='#737c8a', carry='#147d80', reset='#d48a38', frozen='#a4aab3', oracle='#7660a9')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'axes.spines.top':False, 'axes.spines.right':False})
fig, (left, right) = plt.subplots(1, 2, figsize=(11.8, 4.9), gridspec_kw={'width_ratios':[1, 1.35]})
fig.suptitle('Predictor history transfers, but net model improvement remains unresolved',
             x=.06, y=.99, ha='left', fontsize=14, fontweight='bold')
means = [summary['main']['32']['methods'][m]['heldout_gradient']['relative_l2'] for m in ('reset', 'carry')]
left.bar([0, 1], means, width=.55, color=[colors['reset'], colors['carry']], alpha=.18)
for i, (seed, probe) in enumerate((s,p) for s in (101,102,103) for p in range(4)):
    offset = (i-5.5)*.012
    values = [primary[(seed, probe, m)] for m in ('reset', 'carry')]
    left.plot([offset, 1+offset], values, color='#6f7681', alpha=.45, linewidth=.9,
              marker='o', markersize=3)
left.scatter([0, 1], means, c=[colors['reset'], colors['carry']], s=85, marker='D', zorder=5)
left.axhline(1, color='#959ca5', linestyle=':', linewidth=1)
left.text(.02, 1.015, 'Zero-gradient error = 1', color='#777f89', fontsize=8)
left.set_xticks([0, 1], [f'Reset\n{means[0]:.3f}', f'Carry history\n{means[1]:.3f}'])
left.set_xlim(-.45, 1.45)
left.set_ylim(0, 2.12)
left.set_ylabel('Held-out batch relative L2  (lower is better)')
left.set_title('Gradient prediction: mean error −14.6%', loc='left', fontsize=11, pad=15)
left.grid(axis='y', alpha=.15)
left.set_axisbelow(True)

methods = ('baseline', 'carry', 'reset', 'frozen', 'oracle')
labels = ('Before update', 'Carry history', 'Reset each episode', 'Frozen predictor', 'True gradient')
values = [summary['accuracy']['mean_percent'][m] for m in methods]
y = np.arange(len(methods))
right.barh(y, values, color=[colors[m] for m in methods], alpha=.9, height=.62)
for i, method in enumerate(methods):
    for seed, dy in zip((101,102,103), (-.16, 0, .16)):
        value = 100*summary['accuracy']['by_seed'][str(seed)][method]['accuracy']
        right.scatter(value, i+dy, color='white', edgecolor='#343b43', linewidth=.7, s=20, zorder=4)
    right.text(10.4, i, f'{values[i]:.2f}%', va='center', ha='right', fontsize=10, fontweight='bold')
right.axvline(values[0], color='#46515e', linestyle='--', linewidth=.9, alpha=.7)
right.set_yticks(y, labels)
right.invert_yaxis()
right.set_xlim(0, 10.6)
right.set_xlabel('Exact accuracy (%) · bars: mean, dots: three seeds')
right.set_title('One-update accuracy: carry > reset, near baseline', loc='left', fontsize=11, pad=15)
right.grid(axis='x', alpha=.15)
right.set_axisbelow(True)
fig.text(.06, .055, 'Primary model state 32 · 3 paired seeds · left: 4 held-out batches per seed · right: prespecified probe 0, 1,024 shared questions',
         fontsize=8.5, color='#525b67')
fig.text(.06, .015, 'Qwen2.5-0.5B · layer-8 rank-64 LoRA · full input prompts preserved · 2,048-token limit applies only to new output',
         fontsize=8.5, color='#525b67')
fig.subplots_adjust(top=.80, bottom=.20, left=.07, right=.98, wspace=.48)
fig.savefig(ROOT / 'figures/overview.png', dpi=180, bbox_inches='tight')
fig.savefig(ROOT / 'figures/overview.pdf', bbox_inches='tight')
plt.close(fig)
