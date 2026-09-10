"""Standalone fixed-pair likelihood/accuracy evidence with raw CSV and hashes."""
import csv,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s=Path(__file__).resolve().parents[1];p=s/'audits/loss_accuracy_alignment.json';a=json.loads(p.read_text());rows=[]
for pair in a['pairs']:
 for kind in ['raw','generated']:
  r=pair[kind];label=('Student-matched' if r['epoch']==2 else '7B-matched')+' '+kind
  rows.append(dict(label=label,name=r['name'],epoch=r['epoch'],raw_reference_ce=r['endpoint_raw_reference_ce'],ce_delta=r['raw_reference_ce_change'],same64_correct=r['correct_same64'],same64_delta_pp=(r['correct_same64']-r['base_correct_same64'])/64*100,dev500_correct=r['correct_dev500'],dev500_delta_pp=(r['correct_dev500']-r['base_correct_dev500'])/500*100,kind=kind))
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,key,title in zip(axes,['same64_delta_pp','dev500_delta_pp'],['Free generation on the SAME 64 questions','Free generation on all 500 development questions']):
 ax.axvline(0,color='#b0b0b0',linewidth=.8);ax.axhline(0,color='#b0b0b0',linewidth=.8)
 for r in rows:
  c='#b54a43' if r['kind']=='raw' else '#297ca3';mark='o' if r['epoch']==2 else 's'
  ax.scatter(r['ce_delta'],r[key],s=62,color=c,marker=mark,zorder=3)
  ax.annotate(r['label'],(r['ce_delta'],r[key]),xytext=((-7,22) if r['epoch']==2 and r['kind']=='generated' else (7,-16 if r['epoch']==4 else 18)),ha=('right' if r['epoch']==2 and r['kind']=='generated' else 'left'),textcoords='offset points',fontsize=9)
 ax.scatter(0,0,s=65,color='#444444',marker='x',label='Baseline');ax.annotate('Baseline',(0,0),xytext=(5,-16),textcoords='offset points',fontsize=9)
 ax.set_xlabel('Change in teacher-forced raw-reference CE (64 questions)')
 ax.set_ylabel('Accuracy change versus baseline (percentage points)')
 ax.set_title(title,fontsize=11);ax.set_xlim(-.095,.225);ax.set_ylim(-38,10);ax.grid(alpha=.12)
fig.suptitle('Lower reference loss can accompany worse free-generation accuracy',fontsize=14)
fig.text(.5,-.035,'Fixed seed43 endpoints: student pair epoch2, 7B pair epoch4. Descriptive paired-source evidence; no uncertainty bands.',ha='center',fontsize=9)
out=s/'figures/loss_accuracy_alignment'
for ext in ['png','svg','pdf']:fig.savefig(str(out)+'.'+ext,dpi=180,bbox_inches='tight')
with Path(str(out)+'.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
Path(str(out)+'.sources.json').write_text(json.dumps(dict(audit=str(p),audit_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
print(out)
