"""Predeclare equal-size candidate subsets and fresh grouped evaluation splits."""
from common import *
from collections import defaultdict,Counter
import pyarrow.parquet as pq

def main():
    task='mnli';manifest={};pool=read(data(task,'update'))
    for i in range(5):
        rng=np.random.default_rng(601+i);ids=list(map(int,rng.choice(len(pool),1024,replace=False)))
        rows=[pool[j] for j in ids];path=data(task,f'candidate_{i}')
        if path.exists():assert read(path)==rows
        else:jsonl(path,rows)
        manifest[f'candidate_{i}']=dict(n=len(rows),sha256=sha(path),indices_in_update=ids,labels=dict(Counter(r['gold'] for r in rows)),seed=601+i)
    # Exclude source-study training, development and evaluation groups, including
    # its 0.5B source; these fresh train-population readouts are not MNLI's standard test.
    usedgroups=set();usedkeys=set();checked=[]
    for source in ['benchmark_0p5b_16h_20261010','benchmark_1p5b_11h_20261011','cv4_effectiveness_9h_20261014']:
        for p in (EXP.parent/source/'data/mnli').glob('*.jsonl'):
            rows=read(p);usedgroups.update(r['group_id'] for r in rows);usedkeys.update(identity(r) for r in rows);checked.append(str(p))
    raw=EXP.parent/'benchmark_0p5b_16h_20261010/raw/mnli/data/train-00000-of-00001.parquet'
    records=pq.read_table(raw).to_pylist();groups=defaultdict(list);seen=set()
    key=lambda s:hashlib.sha256(re.sub(r'\s+',' ',s).strip().casefold().encode()).hexdigest()
    for i,r in enumerate(records):
        g=key(r['premise'])
        if g in usedgroups or r['label'] not in [0,1,2]:continue
        problem='Premise:\n'+r['premise']+'\n\nHypothesis:\n'+r['hypothesis']+'\n\nA) entailment\nB) neutral\nC) contradiction'
        gold='ABC'[r['label']];row=dict(task=task,problem=problem,solution='\\boxed{'+gold+'}',gold=gold,answer_type='choice',source_index=i,source_split='train',group_id=g,raw_source={'choices':{'text':['entailment','neutral','contradiction'],'label':['A','B','C']}},genre=r.get('genre'))
        ident=identity(row)
        if ident in seen or ident in usedkeys:continue
        seen.add(ident);row['sample_id']=key(f'surrogate20261015/{i}/{ident}');groups[g].append(row)
    rng=np.random.default_rng(20261015);keys=list(groups);rng.shuffle(keys);cursor=0;newgroups=set()
    for split,target in [('selection_dev',1024),('selection_holdout',2048)]:
        rows=[]
        while len(rows)<target:
            g=keys[cursor];cursor+=1;assert g not in newgroups;newgroups.add(g);rows.extend(groups[g])
        path=data(task,split);assert not path.exists();jsonl(path,rows)
        manifest[split]=dict(n=len(rows),groups=len({r['group_id'] for r in rows}),sha256=sha(path),labels=dict(Counter(r['gold'] for r in rows)),population='Fresh MNLI training-population source groups; distinct from standard validation_matched',purpose='Candidate choice' if split=='selection_dev' else 'Frozen final capability and candidate-selection regret')
    manifest.update(created=time.time(),raw_sha256=sha(raw),excluded_sources=checked,exclusion_groups=len(usedgroups),candidate_design='Five uniform equal-size 1024-example subsets of fixed original update pool, allowed overlap; membership fixed before outcomes',selection='No labels from selection_dev/holdout may train a predictor or adaptation; only selection_dev may choose a data subset. Holdout is final only.')
    write(EXP/'configs/data_manifest.json',manifest);log({k:{x:v[x] for x in ['n','sha256'] if x in v} for k,v in manifest.items() if isinstance(v,dict)})

if __name__=='__main__':main()
