"""Run a registered, development-only diversity ablation after final timing."""
from common import *
import sys

def main():
    summary=EXP/'results/selection_pipeline_summary.json'
    while not (summary.exists() and json.loads(summary.read_text()).get('complete')):
        if deadline(9000):
            write(EXP/'results/diversity_launch.json',dict(launched=False,reason='Insufficient fitting reserve',epoch=time.time()))
            return
        time.sleep(15)
    write(EXP/'results/diversity_launch.json',dict(launched=True,epoch=time.time(),workflow_summary_sha256=sha(summary)))
    subprocess.run([sys.executable,str(EXP/'scripts/run_queue.py'),'--queue',str(EXP/'configs/queue_diversity2048.json')],check=True,env=dict(os.environ,BENCH_RUN_ID=EXP.name),cwd=ROOT)

if __name__=='__main__':main()
