"""Snapshot the actual experiment inventory, distinguishing training from derived models."""
import csv
from common import *

def main():
 queue=json.loads((S/'results/train_queue.json').read_text());rows=[]
 for job in queue:
  out=S/'results/training'/job['name'];mp=out/'manifest.json';manifest=json.loads(mp.read_text()) if mp.exists() else {}
  complete=out/'complete.json';failure=S/'logs'/('failed_'+job['name']+'.json')
  status='complete' if complete.exists() else 'failed' if failure.exists() else 'started' if out.exists() else 'pending'
  history=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
  checkpoints=[]
  for p in sorted(out.glob('epoch*/ready.json'),key=lambda p:int(p.parent.name[5:])):
   ready=json.loads(p.read_text());evalfile=S/'results/evaluation/dev'/ready['name']/'math_summary.json'
   result=json.loads(evalfile.read_text()) if evalfile.exists() else None
   checkpoints.append(dict(epoch=ready['epoch'],name=ready['name'],path=str(p.parent),dev_correct=result['correct'] if result else None,dev_n=result['samples'] if result else None))
  rows.append(dict(kind='resumed_training' if manifest.get('resumed_from') else 'training',name=job['name'],status=status,seed=job.get('seed',43),lr=job.get('lr',1e-5),rank=job.get('rank',16),alpha=job.get('alpha',16),modules=job.get('modules','qkvo'),objective=job.get('objective','token_mean'),batch=job.get('batch',16),horizon_epochs=job.get('epochs',8),planned_stop_epoch=job.get('stop',4),data_file=job['train_file'],data_sha256=manifest.get('data_sha'),n_examples=manifest.get('examples'),trainable=manifest.get('trainable'),start=manifest.get('start'),gpu=manifest.get('gpu'),script_sha256=manifest.get('script_sha'),last_saved_step=history[-1]['step'] if history else None,checkpoints=checkpoints))
 for row,job in zip(rows,queue):
  row['trainer_script']=job.get('trainer_script','train.py')
  row['kl_coefficient']=job.get('kl_coefficient',0.)
  row['role']='numerical_replay_control' if job['name'] in ['teacher_all_kl0_lr5e5_s43','sample_augmented_lr1e5_s43_serial_replay'] else 'experimental_training'
 derived=[]
 for mp in sorted((S/'results/training').glob('*/manifest.json')):
  m=json.loads(mp.read_text())
  if 'arguments' in m:continue
  cps=[]
  for p in sorted(mp.parent.glob('epoch*/ready.json')):
   ready=json.loads(p.read_text());ef=S/'results/evaluation/dev'/ready['name']/'math_summary.json';v=json.loads(ef.read_text()) if ef.exists() else {}
   cps.append(dict(name=ready['name'],path=str(p.parent),dev_correct=v.get('correct'),dev_n=v.get('samples')))
  derived.append(dict(name=mp.parent.name,kind=m.get('kind','derived_checkpoint'),manifest_sha256=sha(mp),checkpoints=cps))
 evals=[]
 for p in sorted((S/'results/evaluation').glob('*/*/math_summary.json')):
  m=json.loads(p.read_text());evals.append(dict(split=p.parents[1].name,name=m['variant'],correct=m['correct'],n=m['samples'],summary=str(p)))
 write(S/'results/experiment_inventory.json',dict(time=time.time(),deadline=DEADLINE,training=rows,derived=derived,evaluations=evals,interpretation='Queued jobs are plans, not evidence. Multiple checkpoints and derived adapters are not independent training runs. Resumed branch shares the source run history. Dev outcomes are exploratory; official-test reuse and OOD roles are defined in PLAN.md.'))
 flat=[]
 for r in rows:
  item={k:v for k,v in r.items() if k!='checkpoints'};item['dev_trajectory']=';'.join(f"e{x['epoch']}:{x['dev_correct'] if x['dev_correct'] is not None else 'pending'}" for x in r['checkpoints']);flat.append(item)
 with (S/'results/experiment_inventory.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 print(json.dumps(dict(training_complete=sum(r['status']=='complete' for r in rows),training_started=sum(r['status']=='started' for r in rows),training_pending=sum(r['status']=='pending' for r in rows),derived_models=len(derived),evaluation_summaries=len(evals))))

if __name__=='__main__':main()
