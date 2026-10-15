"""Sequential, resumable study queue; each GPU stage enforces the reservation/deadline."""
from common import *
import argparse,sys

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--queue',required=True);ap.add_argument('--wait-stage');a=ap.parse_args()
    if a.wait_stage:
        f=EXP/'results'/f'stage_{a.wait_stage}.json'
        while not f.exists():
            if deadline(5400):raise RuntimeError('Validation reserve reached while waiting')
            time.sleep(2)
        assert json.loads(f.read_text())['returncode']==0
    queue=json.loads(Path(a.queue).read_text())
    for item in queue:
        if deadline(5400):break
        record=EXP/'results'/f'queue_{item["name"]}.json'
        signature=hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
        if record.exists():
            old=json.loads(record.read_text())
            if old.get('returncode')==0 and old['signature']==signature:continue
        start=time.time();code=subprocess.call(item['command'],cwd=ROOT)
        write(record,dict(name=item['name'],signature=signature,started=start,finished=time.time(),returncode=code))
        if code:raise SystemExit(code)

if __name__=='__main__':main()
