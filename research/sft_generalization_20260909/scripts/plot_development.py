"""Regenerate descriptive development plots. These are not confirmation evidence."""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

rows=json.loads((S/'results/dev_analysis.json').read_text())
base=next(r['accuracy'] for r in rows if r['name']=='base')
groups={}
for r in rows:
 m=re.fullmatch(r'(.+)_epoch(\d+)',r['name'])
 if m and re.search(r'_s43(?:_serial_replay)?$',m[1]) and m[1]!='sample_augmented_lr1e5_s43':groups.setdefault(m[1],[]).append((int(m[2]),r['accuracy']))
panels=[('^greedy','Previous greedy-correct targets'),('^sample_all_lr','Student targets, QKVO'),('^teacher_all_lr','7B targets, QKVO'),('^teacher32_all_qkvo','32B targets, QKVO'),('^sample_all_all','Student targets, all modules'),('^teacher(32)?_all_all','Teacher targets, all modules'),('^(sample|teacher)_common_lr','Matched student / 7B questions'),('^sample_(augmented|augmented_rawhard|greedy_budget)_lr','Question coverage and replay'),('^sample_(multi|repeat)_lr','Answer-text diversity')]
fig,axes=plt.subplots(3,3,figsize=(18,13.5),sharey=True)
for ax,(pattern,title) in zip(axes.flat,panels):
 for name,points in sorted(groups.items()):
  if re.search(pattern,name):
   label=name.removesuffix('_s43').replace('lr1e5','LR=1e-5').replace('lr5e5','LR=5e-5').replace('lr1e4','LR=1e-4')
   points=sorted(points);ax.plot([x[0] for x in points],[x[1] for x in points],marker='o',label=label)
 if not ax.lines:ax.text(.5,.5,'Results pending',transform=ax.transAxes,ha='center',color='gray')
 ax.axhline(base,color='black',ls='--',lw=1,label='Frozen baseline')
 last=max([int(x) for line in ax.lines for x in line.get_xdata() if x>0],default=4)
 ax.set(title=title,xlabel='Training epoch',xticks=list(range(1,max(4,last)+1)));ax.grid(alpha=.2);ax.legend(fontsize=7)
for ax in axes[:,0]:ax.set_ylabel('Reused MATH development accuracy (%)')
fig.suptitle('Exploratory seed43 trajectories; checkpoints and recipes are being searched',fontsize=11)
fig.tight_layout();(S/'figures').mkdir(exist_ok=True)
for ext in ['png','pdf']:fig.savefig(S/f'figures/development_trajectories.{ext}',dpi=180)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for f in sorted((S/'results/training').glob('*/history.json')):
 h=json.loads(f.read_text());points=[r for r in h if 'dev_ce' in r]
 if not points:continue
 name=f.parent.name
 if name.startswith(('sample_all_','teacher_all_')) and '_all16_' not in name:
  axes[0].plot([r.get('epoch',0) for r in points],[r['dev_ce'] for r in points],marker='.',label=name)
  axes[1].plot([r.get('epoch',0) for r in points],[r.get('probe_ce',{}).get('probe_'+name.split('_')[0],float('nan')) for r in points],marker='.',label=name)
for ax,title in zip(axes,['Reference-target CE on dev64','Same-source CE on matched training probe64']):
 ax.set(title=title,xlabel='Epoch',ylabel='Completion token-mean CE');ax.grid(alpha=.2)
 if ax.lines:ax.legend(fontsize=7)
fig.suptitle('Teacher-forced diagnostics; lower CE does not establish answer-accuracy gains',fontsize=11)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(S/f'figures/development_ce.{ext}',dpi=180)
plt.close(fig)
