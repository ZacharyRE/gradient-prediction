"""Run one registered behavior diagnostic after the existing KD fits finish."""
from common import *
import sys

def main():
    plan=json.loads((EXP/'configs/token_behavior_readout_plan.json').read_text())
    fit=EXP/'results/stage_fit_distill_critic.json'
    while not fit.exists():
        if time.time()>=plan['cutoff_epoch']-300:return
        time.sleep(10)
    assert json.loads(fit.read_text())['returncode']==0
    assert time.time()<plan['cutoff_epoch']-300
    subprocess.run([sys.executable,str(EXP/'scripts/stage.py'),'--name','token_behavior_dev',sys.executable,str(EXP/'scripts/evaluate_token_behavior.py')],check=True,cwd=ROOT)

if __name__=='__main__':main()
