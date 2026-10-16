"""Complete initial-screen validation sequentially on the single reserved card."""
from common import *
import sys

def call(command):
    subprocess.run(command,check=True,cwd=ROOT)

def main():
    waiting=EXP/'results/stage_screen_paws_v1.json'
    while not waiting.exists():
        if deadline(5400):raise RuntimeError('Deadline before first screen finished')
        time.sleep(2)
    assert json.loads(waiting.read_text())['returncode']==0
    py=str(ROOT/'.venv/bin/python');stage=str(EXP/'scripts/stage.py')
    call([py,stage,'--name','fast_backend_audit',py,str(EXP/'scripts/audit_fast_backend.py')])
    entries=str(EXP/'configs/paws_screen_v1_entries.json')
    call([py,str(EXP/'scripts/make_entries.py'),'--task','paws','--jobs',str(EXP/'configs/screen_paws_v1.json'),'--output',entries])
    call([py,stage,'--name','paws_screen_v1_eval',py,str(EXP/'scripts/evaluate.py'),'--model','1p5b','--entries',entries,'--name','paws_screen_v1','--split','accuracy_dev','--max-new-tokens','256'])

if __name__=='__main__':main()
