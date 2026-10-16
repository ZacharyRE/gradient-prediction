"""Assemble old and additional teacher-state labels without duplicating shards."""
from common import *
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='critic_coverage_v1');ap.add_argument('--sources',nargs='+',default=['critic_v1','critic_fast_v1']);a=ap.parse_args()
    out=EXP/'cache'/a.name;out.mkdir(exist_ok=True);states=[];sources={}
    for name in a.sources:
        root=EXP/'cache'/name;receipt=json.loads((root/'complete.json').read_text());assert receipt['complete'];sources[name]=sha(root/'complete.json')
        for state in receipt['states']:
            label=state['state']['name'];assert label not in {s['state']['name'] for s in states}
            target=out/label
            if target.exists():assert target.resolve()==(root/label).resolve()
            else:target.symlink_to(root/label,target_is_directory=True)
            states.append(state)
    assert len(states)==12
    write(out/'complete.json',dict(complete=True,states=states,source_cache_sha256=sources,purpose='Original8 teacherstates plus4 high-LR Oracle states; same512 fitting examples and129 predictor-dev examples. No heldoutstarts, no selectionoutcomes.',source_sha256=source_signature(__file__)))
    log(dict(states=len(states),shared_shards=True))

if __name__=='__main__':main()
