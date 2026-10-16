"""Check current selected inputs against group-separation and membership claims."""
from common import *
from itertools import combinations
from collections import Counter

def main():
    names=['warmup','predictor_train','predictor_dev','accuracy_dev','update','selection_dev','selection_holdout']+[f'candidate_{i}' for i in range(5)]+[f'genre_{g}' for g in ['fiction','government','slate','telephone','travel']]
    rows={n:read(data('mnli',n)) for n in names};groups={n:{r['group_id'] for r in rr} for n,rr in rows.items()};ids={n:{identity(r) for r in rr} for n,rr in rows.items()}
    overlaps=[]
    for a,b in combinations(names,2):
        ng=len(groups[a]&groups[b]);ni=len(ids[a]&ids[b]);allowed=(a=='update' or a.startswith('candidate_')) and (b=='update' or b.startswith('candidate_'))
        assert allowed or (ng==0 and ni==0),(a,b,ng,ni)
        if ng or ni:overlaps.append(dict(first=a,second=b,source_groups=ng,examples=ni,allowed_original_update_pool_overlap=True))
    manifest=json.loads((EXP/'configs/data_manifest.json').read_text());pool=rows['update']
    for i in range(5):
        name=f'candidate_{i}';r=manifest[name];assert rows[name]==[pool[j] for j in r['indices_in_update']];assert len(ids[name])==len(rows[name])==1024 and sha(data('mnli',name))==r['sha256']
    excluded=set();excluded_ids=set();source_receipts=[]
    for path in manifest['excluded_sources']:
        rr=read(path);excluded.update(r['group_id'] for r in rr);excluded_ids.update(identity(r) for r in rr);source_receipts.append(dict(path=path,sha256=sha(path)))
    for name in ['selection_dev','selection_holdout']+[n for n in names if n.startswith('genre_')]:
        assert not groups[name]&excluded and not ids[name]&excluded_ids,name
    frozen=EXP/'configs/surrogate_final_freeze.json'
    if frozen.exists():
        f=json.loads(frozen.read_text())
        for n in ['selection_dev','selection_holdout']:assert sha(data('mnli',n))==f['data_sha256']['mnli'][n]
    report=dict(passed=True,epoch=time.time(),splits={n:dict(examples=len(rr),source_groups=len(groups[n]),labels=dict(Counter(r['gold'] for r in rr)),sha256=sha(data('mnli',n))) for n,rr in rows.items()},allowed_overlap=overlaps,excluded_source_receipts=source_receipts,scope='Actual current inputs: disjoint source groups and whitespace-normalized problem identities outside the explicitly overlapping random-subset/original-pool family; exact registered random-subset membership; selection and genre groups excluded from all registered previous-study inputs. This does not establish absence from base-model pretraining.',freeze_sha256=sha(frozen) if frozen.exists() else None)
    write(EXP/'results/split_integrity_audit.json',report);log(dict(passed=True,splits=len(rows),allowed_overlapping_pairs=len(overlaps),selection_holdout=report['splits']['selection_holdout']))

if __name__=='__main__':main()
