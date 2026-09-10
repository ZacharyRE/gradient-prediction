"""Five-seed selected-candidate followup; preserve the earlier three-seed figure."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import csv
from common import *
files=[S/'audits/teacher_average_five_seed_development.json',S/'audits/teacher_average_five_seed_math_reused1500.json']
objects=[json.loads(p.read_text()) for p in files];keys=['average1234','epoch4'];table=[]
fig,axes=plt.subplots(1,2,figsize=(12,3.9),sharey=True,sharex=True)
colors=['#007F73','#2A5E9A','#9452A0','#CE7922','#A83441']
for ax,obj,title in zip(axes,objects,['Development: 500 questions','Reused validation: 1500 questions']):
 for i,key in enumerate(keys):
  r=obj['families'][key];stats=r['vs_base'];mean=stats['mean_delta_pp']
  ax.plot(stats['seed_question_ci95_pp'],[i,i],color='#D3DCE2',lw=9,solid_capstyle='round')
  ax.plot(stats['question_ci95_pp'],[i,i],color='#244B63',lw=2.2)
  ax.scatter([mean],[i],marker='D',s=42,color='#17252E',zorder=3)
  for j,seed in enumerate(r['per_seed']):
   ax.scatter([seed['delta_pp']],[i-.18-j*.025],s=26,facecolors='white',edgecolors=colors[j],linewidth=1.2,zorder=4)
   table.append(dict(dataset=obj['n'],family=key,seed=seed['seed'],correct=seed['correct'],base_correct=obj['base_correct'],delta_pp=seed['delta_pp'],mean_delta_pp=mean,question_ci_low=stats['question_ci95_pp'][0],question_ci_high=stats['question_ci95_pp'][1],seed_question_ci_low=stats['seed_question_ci95_pp'][0],seed_question_ci_high=stats['seed_question_ci95_pp'][1]))
  ax.text(6.3,i,f"{mean:+.2f} pp",ha='right',va='center',fontsize=10,color='#17252E')
 ax.axvline(0,color='#777',ls='--',lw=1);ax.set_title(title,fontsize=12)
 ax.set_xlim(-3.6,6.5);ax.set_ylim(1.4,-.6);ax.set_xlabel('Accuracy change versus own baseline (pp)')
 ax.grid(axis='x',alpha=.17);ax.spines[['top','right']].set_visible(False)
axes[0].set_yticks([0,1],['Mean of epoch 1–4 updates','Fixed epoch 4'])
handles=[Line2D([0],[0],color='#244B63',lw=2,marker='D',label='Mean + question 95% CI'),Line2D([0],[0],color='#D3DCE2',lw=8,label='Seed + question 95% CI')]
handles += [Line2D([0],[0],color='none',marker='o',markerfacecolor='white',markeredgecolor=c,label=str(seed)) for c,seed in zip(colors,range(43,48))]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.56,.065),ncol=7,frameon=False,fontsize=9,handlelength=1.3,columnspacing=1.1)
fig.suptitle('General 7B targets: all five training seeds retained',fontsize=14,y=.99)
fig.text(.55,.015,'Selected-candidate development evidence; unadjusted intervals. Reserved OOD and full MATH evaluation pending.',ha='center',fontsize=9,color='#555')
fig.subplots_adjust(left=.23,right=.985,top=.85,bottom=.28,wspace=.13)
for ext in ['png','svg','pdf']:fig.savefig(S/f'figures/five_seed_evidence.{ext}',dpi=180,bbox_inches='tight')
write(S/'figures/five_seed_evidence_sources.json',{str(p):sha(p) for p in files})
with (S/'figures/five_seed_evidence.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
