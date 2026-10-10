"""Regenerate overview.png from the compact numerical results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
r = json.loads((ROOT / 'results.json').read_text())
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

def forest(ax, rows, title, subtitle):
    for i, (label, stat) in enumerate(rows):
        mean = stat['delta_pp']
        lo, hi = stat['familywise_ci95']
        color = '#17836b' if lo > 0 else '#636e80'
        ax.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt='o', color=color, capsize=4, lw=2)
        ax.annotate(f'{mean:+.2f}', (mean, i), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9)
    ax.set_yticks(range(len(rows)), [x[0] for x in rows])
    ax.set_ylim(len(rows)-.5, -.55)
    ax.axvline(0, color='#999999', lw=1, ls='--')
    ax.grid(axis='x', alpha=.18)
    ax.set_title(title + '\n' + subtitle, fontsize=11, pad=12)
    ax.set_xlabel('Gain over own Warmup (percentage points)')

forest(axes[0,0], [(label,r['scope_search'][task]['comparison_family11']) for task,label in [('boolq','BoolQ'),('arc_easy','ARC-Easy*'),('arc_challenge','ARC-Challenge')]], 'Single-layer Oracle: fixed Warmup', '5 seeds; adjusted 95% intervals, family 11')
forest(axes[0,1], [(label,r['fresh_warmup'][task]['comparison_family11']) for task,label in [('boolq','BoolQ'),('arc_easy','ARC-Easy*'),('arc_challenge','ARC-Challenge')]], 'New Warmup checkpoints', '3 seeds; adjusted 95% intervals, family 11')
ax=axes[1,0]
labels=['Single\nanswer','Three-layer\nanswer','Single\nteacher','Three-layer\nteacher']
order=['single_answer','multi_answer','single_teacher','multi_teacher']
x=np.arange(4)
for offset,task,label,color in [(-.18,'openbookqa','OpenBookQA (128 steps)','#377eb8'),(.18,'arc_challenge','ARC-Challenge (256 steps)','#e7963b')]:
    vals={d['arm']:d['gain_over_warmup_pp'] for d in r['target_layer_experiments'][task]['matched']}
    bars=ax.bar(x+offset,[vals[k] for k in order],width=.35,color=color,label=label)
    ax.bar_label(bars,fmt='%+.2f',padding=3,fontsize=9)
ax.set_xticks(x,labels)
ax.axhline(0,color='#999999',lw=1)
ax.set_ylim(-5,9)
ax.set_ylabel('Gain over own Warmup (pp)')
ax.set_title('Targets and layers: matched LR 1e-4\nDescriptive means; additional-layer intervals cross zero',fontsize=11)
ax.legend(fontsize=9,loc='upper right')
ax.grid(axis='y',alpha=.18)
rows=[('BoolQ: original pool',r['boolq_pool_bridge']['old_minus_warmup']),('OpenBookQA: old protocol',r['openbookqa_old_protocol']['comparisons']['step128_minus_warmup'])]
forest(axes[1,1],rows,'Gains also exist with original update pools','BoolQ: 5 seeds / family 11; OpenBookQA: 3 seeds / family 3')
fig.suptitle('True-gradient updates can improve downstream accuracy\nQwen2.5-1.5B-Instruct | *ARC-Easy includes formatting gains (see report)',fontsize=16)
fig.savefig(ROOT/'overview.png',dpi=180)
plt.close(fig)
