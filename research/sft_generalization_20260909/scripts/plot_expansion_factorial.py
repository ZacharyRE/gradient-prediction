"""Plot complete five-seed expansion/averaging results with paired uncertainty."""
import csv,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
s=Path(__file__).resolve().parents[1]
p=s/'results/expanded_unaveraged_supplement_analysis.json';a=json.loads(p.read_text())
families=['teacher7_avg1234','teacher7_epoch4','teacher7_expanded_avg1234','teacher7_expanded_epoch4']
labels=['Original data, average 1–4\n(original primary)','Original data, epoch4',
        'Expanded data, average 1–4','Expanded data, epoch4\n(supplement)']
fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained',sharey=True)
records=[];colors=['#3f6b8e','#669590','#9b7148','#875b85']
for row,(mode,result) in enumerate(a['results'].items()):
    for col,metric in enumerate(['math_reused5000','mathematical_ood_macro']):
        ax=axes[row,col];ax.axvline(0,color='#888888',linewidth=.8)
        for j,f in enumerate(families):
            v=result['datasets'][metric]['families'][f]['vs_base'] if col==0 else result['ood_macro'][f]
            y=3-j;c=colors[j]
            ax.plot(v['seed_question_ci95'],[y,y],color=c,linewidth=2,alpha=.4)
            ax.plot(v['question_ci95'],[y,y],color=c,linewidth=4)
            ax.scatter(v['per_seed_delta_pp'],y+np.linspace(-.11,.11,5),s=23,color=c,zorder=3)
            ax.scatter([v['delta_pp']],[y],s=60,color=c,marker='D',edgecolor='white',zorder=4)
            records.append(dict(scoring=mode,metric=metric,family=f,mean_delta_pp=v['delta_pp'],
                question_ci_low=v['question_ci95'][0],question_ci_high=v['question_ci95'][1],
                seed_question_ci_low=v['seed_question_ci95'][0],seed_question_ci_high=v['seed_question_ci95'][1],
                **{f'seed{seed}_delta_pp':delta for seed,delta in zip(a['seeds'],v['per_seed_delta_pp'])}))
        ax.set_yticks([3,2,1,0],labels=labels,fontsize=9);ax.set_ylim(-.45,3.45)
        ax.set_xlabel('Accuracy change versus baseline (percentage points)')
        ax.set_title(('Primary scoring' if row==0 else 'Strict last-box scoring')+' | '+('MATH 5000 (reused)' if col==0 else 'Mathematical OOD macro'),fontsize=11)
        ax.grid(axis='x',alpha=.13)
fig.suptitle('Expansion and checkpoint averaging: all five training seeds',fontsize=15)
fig.text(.5,-.025,'Dots: seeds43–47. Diamond: mean. Dark bar: paired-question95%CI. Pale bar: seed/question95%CI.\nMinerva numeric repair applied; secondary comparisons exploratory. Strict OOD gains depend strongly on missing-box penalties.',ha='center',fontsize=9)
dest=s/'figures/expansion_factorial'
for ext in ['png','svg','pdf']:fig.savefig(str(dest)+'.'+ext,dpi=180,bbox_inches='tight')
with Path(str(dest)+'.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
Path(str(dest)+'.sources.json').write_text(json.dumps(dict(analysis=str(p),analysis_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
print(dest)
