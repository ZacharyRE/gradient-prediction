"""Regenerate overview.png/pdf from published statistics (requires Matplotlib)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
tasks = json.loads((ROOT / 'statistics.json').read_text())['tasks']
complete = [task for task, result in tasks.items() if result['final_complete']]
names = dict(zip(complete, ['MNLI', 'PAWS', 'HellaSwag', 'RACE',
    'CommonsenseQA', 'SciQ', 'SocialIQA', 'WinoGrande', 'SST-2']))
fig, ax = plt.subplots(figsize=(10, max(3.8, .55 * len(complete) + 1.8)))
for key, color, offset, title in [
    ('oracle_vs_warmup', '#2563a6', -.12, 'Oracle'),
    ('predictor_vs_warmup', '#d77924', .12, 'Predictor'),
]:
    for index, task in enumerate(complete):
        stat = tasks[task][key]
        mean = stat['mean_pp']
        lo, hi = stat['ci95_pp']
        ax.errorbar(mean, index + offset, xerr=[[mean-lo], [hi-mean]],
                    fmt='o', capsize=3, color=color,
                    label=title if index == 0 else None)
ax.set_yticks(range(len(complete)), [names[task] for task in complete])
ax.invert_yaxis()
ax.axvline(0, color='gray', lw=1)
ax.set_xlabel('Final accuracy gain over fixed warmup (percentage points)')
ax.legend()
ax.grid(axis='x', alpha=.2)
ax.set_title('Qwen2.5-1.5B-Instruct: oracle gains and predictor retention\n'
             'Nominal 95% paired intervals; fixed warmup seed 201')
fig.tight_layout()
fig.savefig(ROOT / 'overview.png', dpi=180)
fig.savefig(ROOT / 'overview.pdf')
plt.close(fig)
