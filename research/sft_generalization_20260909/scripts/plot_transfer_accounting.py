"""Held-out development gains and losses, fixed endpoints and all three seeds."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import numpy as np
from common import *

root=S/'results/evaluation/dev';bp=root/'base/math_predictions.jsonl';base=read(bp)
b=np.array([r['correct'] for r in base],int);records=[]
for family,epoch in [('sample_all_lr1e5',2),('teacher_all_lr5e5',4)]:
 for seed in [43,44,45]:
  name=f'{family}_s{seed}_epoch{epoch}';p=root/name/'math_predictions.jsonl';rr=read(p)
  assert [r['sample_hash'] for r in rr]==[r['sample_hash'] for r in base]
  a=np.array([r['correct'] for r in rr],int);d=a-b
  records.append(dict(model=name,seed=seed,n=len(b),wins=int((d>0).sum()),losses=int((d<0).sum()),net=int(d.sum()),prediction_sha256=sha(p)))
fig,ax=plt.subplots(figsize=(9,4.8));x=np.array([0,1,2,4,5,6]);wins=np.array([r['wins'] for r in records]);losses=np.array([r['losses'] for r in records]);net=wins-losses
ax.bar(x,wins,color='#247BA0',width=.62,label='Baseline wrong → SFT correct')
ax.bar(x,-losses,color='#DD8452',width=.62,label='Baseline correct → SFT wrong')
ax.scatter(x,net,color='#18242B',marker='D',s=34,zorder=4,label='Net gain')
for xx,w,l,d in zip(x,wins,losses,net):
 ax.text(xx,w+1,f'+{w}',ha='center',va='bottom',fontsize=10)
 ax.text(xx,-l-1,f'−{l}',ha='center',va='top',fontsize=10)
 ax.annotate(f'{d:+d}',(xx,d),xytext=(11,0),textcoords='offset points',va='center',fontsize=9)
ax.axhline(0,color='#7D8990',lw=.8);ax.set_ylim(-43,52)
ax.set_xticks(x,[str(r['seed']) for r in records]);ax.set_xlabel('Training seed')
ax.text(1,47,'Student-sampled targets · epoch 2',ha='center',fontsize=11)
ax.text(5,47,'General 7B targets · epoch 4',ha='center',fontsize=11)
ax.set_ylabel('Number of questions (fixed development set, n = 500)')
ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,frameon=False,fontsize=9)
fig.suptitle('Useful answer transfers are largely offset by regressions',fontsize=14,y=.98)
fig.subplots_adjust(bottom=.23,top=.89,left=.10,right=.97)
out=S/'figures';out.mkdir(exist_ok=True)
for ext in ['png','svg','pdf']:fig.savefig(out/f'transfer_accounting.{ext}',dpi=180,bbox_inches='tight')
write(out/'transfer_accounting_data.json',dict(time=time.time(),base_predictions_sha256=sha(bp),records=records,scope='Exploratory development accounting, fixed previously selected endpoints. The decomposition itself does not establish a unique causal mechanism or general efficacy.'))
print(json.dumps(records))
with (out/'transfer_accounting.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
