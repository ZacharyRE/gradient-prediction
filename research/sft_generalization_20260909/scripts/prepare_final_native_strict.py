"""Expose completed native outputs read-only to the unchanged strict scorer."""
import subprocess
from common import *
plan=json.loads((S/'results/confirmation_plan.json').read_text());assert sha(__file__)==plan['native_strict_prepare_script_sha256']
assert sha(S/'scripts/score_sensitivity.py')==plan['strict_scoring_script_sha256']
names=['base']+[plan['families'][plan['primary_family']][str(seed)] for seed in plan['seeds']]
source=S/'results/native/final_dev500';view=S/'results/native/final_dev500_strict_view'
for name in names:
 summary=json.loads((source/name/'summary.json').read_text());assert summary['n']==500
 dest=view/name;dest.mkdir(parents=True,exist_ok=True)
 for filename,target in [('math_predictions.jsonl','predictions.jsonl'),('math_summary.json','summary.json')]:
  p=dest/filename;actual=(source/name/target).resolve()
  if p.exists():assert p.is_symlink() and p.resolve()==actual
  else:p.symlink_to(actual)
for tag,root in [('final_native_dev500',view),('final_primary_dev500',S/'results/evaluation/dev')]:
 p=S/f'audits/scoring_{tag}.json'
 if p.exists():
  r=json.loads(p.read_text());assert set(r['models'])==set(names)
  assert r['script_sha256']==plan['strict_scoring_script_sha256']
  assert all(r['models'][n]['predictions_sha256']==sha(root/n/'math_predictions.jsonl') for n in names)
  continue
 command=[sys.executable,str(S/'scripts/score_sensitivity.py'),'--dataset','dev','--prediction-root',str(root),'--tag',tag,'--models',*names]
 with (S/f'logs/strict_{tag}.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
write(S/'audits/final_native_strict_dispatch.json',dict(time=time.time(),models=names,native_view=str(view),scope='Symlinks to immutable completed outputs only; native and vLLM originals unchanged. Same strict scorer and reference data.'))
