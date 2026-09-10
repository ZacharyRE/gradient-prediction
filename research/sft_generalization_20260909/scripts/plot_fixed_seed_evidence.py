"""Plot fixed three-seed families without conflating development and confirmation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from matplotlib.lines import Line2D
from common import *

files=[S/'audits/fixed_seed_development.json',S/'audits/fixed_seed_math_reused1500.json'];objects=[json.loads(p.read_text()) for p in files]
keys=['sample_epoch2','teacher7_epoch4','sample_avg12','teacher7_avg1234'];labels=['Student targets · epoch 2','7B targets · epoch 4','Student · mean epochs 1–2','7B · mean epochs 1–4']
fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True,sharex=True)
for ax,obj,title in zip(axes,objects,['Development (n = 500)','Historically reused validation (n = 1500)']):
 for i,key in enumerate(keys):
  r=obj['families'][key];assert not r['missing'];mean=r['mean_delta_pp'];q=r['question_ci95_pp'];sq=r['seed_question_ci95_pp']
  ax.plot(sq,[i,i],color='#CED8DF',linewidth=8,solid_capstyle='round',zorder=1)
  ax.plot(q,[i,i],color='#247BA0',linewidth=2.2,zorder=2)
  ax.scatter([mean],[i],marker='D',s=38,color='#18242B',zorder=4)
  ax.scatter([x['delta_pp'] for x in r['per_seed']],[i-.17]*3,facecolors='none',edgecolors='#555F66',s=24,zorder=3)
 ax.axvline(0,color='#666666',ls='--',lw=1);ax.set_title(title,fontsize=12);ax.set_xlabel('Accuracy change versus baseline (percentage points)')
 ax.set_xlim(-5,6);ax.set_xticks([-4,-2,0,2,4,6]);ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
axes[0].set_yticks(range(4),labels);axes[0].invert_yaxis()
legend=[Line2D([0],[0],color='#247BA0',lw=2,marker='D',markerfacecolor='#18242B',label='Mean + question CI'),Line2D([0],[0],color='#CED8DF',lw=7,label='Seed + question CI'),Line2D([0],[0],color='none',marker='o',markeredgecolor='#555F66',markerfacecolor='none',label='Individual training seeds')]
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.55,.065),ncol=3,frameon=False,fontsize=10)
fig.suptitle('Fixed endpoints across seeds 43 / 44 / 45',fontsize=14,y=.98)
fig.text(.55,.015,'95% bootstrap intervals are unadjusted; four-family sensitivity is reported separately. Fresh confirmation pending.',ha='center',fontsize=9,color='#555F66')
fig.subplots_adjust(left=.23,right=.985,top=.87,bottom=.25,wspace=.12)
out=S/'figures';out.mkdir(exist_ok=True)
for ext in ['png','svg','pdf']:fig.savefig(out/f'fixed_seed_evidence.{ext}',dpi=180,bbox_inches='tight')
write(out/'fixed_seed_evidence_sources.json',{str(p):sha(p) for p in files})
table=[]
for source,obj in zip(files,objects):
 for key in keys:
  r=obj['families'][key]
  for seed in r['per_seed']:
   table.append(dict(source=str(source),family=key,seed=seed['seed'],seed_delta_pp=seed['delta_pp'],mean_delta_pp=r['mean_delta_pp'],question_ci_low=r['question_ci95_pp'][0],question_ci_high=r['question_ci95_pp'][1],seed_question_ci_low=r['seed_question_ci95_pp'][0],seed_question_ci_high=r['seed_question_ci95_pp'][1]))
with (out/'fixed_seed_evidence.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
