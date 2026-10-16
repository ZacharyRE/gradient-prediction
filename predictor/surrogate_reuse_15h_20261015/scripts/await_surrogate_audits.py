"""Run independent CPU fidelity/statistical checks after final numerical artifacts."""
from common import *
import sys

def main():
    ready=EXP/'results/surrogate_artifacts_ready.json'
    while not ready.exists():
        if deadline(600):raise RuntimeError('Final audit reserve reached before numerical artifacts completed')
        time.sleep(30)
    assert json.loads(ready.read_text())['complete']
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    completed=[]
    for script in ['analyze_behavior_fidelity.py','audit_surrogate_analysis_reproduction.py','render_surrogate_results.py']:
        tick=time.time();subprocess.run([sys.executable,str(EXP/'scripts'/script)],env=env,check=True,cwd=ROOT)
        completed.append(dict(script=script,seconds=time.time()-tick))
    write(EXP/'results/surrogate_cpu_audits_complete.json',dict(complete=True,epoch=time.time(),scripts=completed))

if __name__=='__main__':main()
