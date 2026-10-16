from common import *
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--predictor-seed',type=int,default=123);ap.add_argument('--output',required=True);a=ap.parse_args()
    jobs=json.loads(Path(a.jobs).read_text());entries=[]
    for j in jobs:
        name=f'{a.task}__{j["name"]}_u{a.seed}_p{a.predictor_seed}'
        path=EXP/'models'/name/'summary.json'
        assert path.exists(),name
        result=json.loads(path.read_text());assert result['complete']
        entries.extend(result['entries'])
    warm=json.loads((EXP/'configs'/f'{a.task}_warmup.json').read_text())
    entries.append(dict(task=a.task,tag=f'{a.task}__warmup',adapter=warm['adapter'],method='warmup',steps=0,lr=warm['lr'],seed=201,adapter_sha256=warm['adapter_sha256'],online_seconds=0))
    assert len({(e['task'],e['tag']) for e in entries})==len(entries),'Duplicate evaluation identifiers'
    write(a.output,entries);log(dict(entries=len(entries),output=a.output))

if __name__=='__main__':main()
