"""Run the registered whole-workflow timing after primary final validation."""
from common import *
import sys
from surrogate_freeze import FREEZE

def main():
    required=[EXP/'results'/f'queue_surrogate_final_{name}.json' for name in ['correctness_bundle','summarize_timing']]
    while not all(p.exists() for p in required):
        if deadline(3600):raise RuntimeError('Whole-pipeline validation reserve reached before prerequisites completed')
        time.sleep(30)
    for p in required:assert json.loads(p.read_text())['returncode']==0,str(p)
    policy=EXP/'configs/surrogate_cost_policy.json';frozen=json.loads(FREEZE.read_text());assert frozen['recipes'][str(policy)]['sha256']==sha(policy)
    plan=json.loads(policy.read_text())['measured_selection_pipeline_plan'];queue=[]
    for repeat in range(1,plan['repetitions']+1):
        offset=repeat-1;conditions=plan['conditions'][offset:]+plan['conditions'][:offset]
        for family in plan['families']:
            for condition in conditions:
                name=f'surrogate_pipeline_{family}_{condition["name"]}_r{repeat}';queue.append(dict(name=name,command=[sys.executable,str(EXP/'scripts/stage.py'),'--name',name,sys.executable,str(EXP/'scripts/benchmark_selection_pipeline.py'),'--condition',condition['name'],'--family',family,'--repeat',str(repeat)]))
    queue.append(dict(name='surrogate_pipeline_summary',command=[sys.executable,str(EXP/'scripts/summarize_selection_pipeline.py')]))
    path=EXP/'configs/queue_selection_pipeline.json';assert not path.exists();write(path,queue);write(EXP/'results/selection_pipeline_queue_receipt.json',dict(freeze_sha256=sha(FREEZE),policy_sha256=sha(policy),queue_sha256=sha(path),stages=len(queue)))
    subprocess.run([sys.executable,str(EXP/'scripts/run_queue.py'),'--queue',str(path)],env=dict(os.environ,BENCH_FINAL_VALIDATION='1'),check=True,cwd=ROOT)

if __name__=='__main__':main()
