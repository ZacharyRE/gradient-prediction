"""Join completed final readouts and timing, then produce reviewable artifacts."""
from common import *
import sys

def main():
    required=[EXP/'results'/f'queue_surrogate_final_{name}.json' for name in ['correctness_bundle','summarize_timing']]
    while not all(p.exists() for p in required):
        if deadline(900):raise RuntimeError('Final artifact reserve reached before both queues completed')
        time.sleep(30)
    for p in required:assert json.loads(p.read_text())['returncode']==0,str(p)
    scripts=['export_surrogate_preparation.py','summarize_reuse_timings.py','surrogate_cost_comparison.py','render_surrogate_results.py','plot_surrogate_results.py','export_reuse_index.py']
    for script in scripts:subprocess.run([sys.executable,str(EXP/'scripts'/script)],check=True,cwd=ROOT)
    write(EXP/'results/surrogate_artifacts_ready.json',dict(complete=True,epoch=time.time(),scripts=scripts))

if __name__=='__main__':main()
