"""Predeclare a secondary, unseen-example domain-composition selection test."""
from common import *
from collections import Counter,defaultdict
import pyarrow.parquet as pq

def main():
    used=set()
    for p in (EXP/'data/mnli').glob('*.jsonl'):used.update(r['group_id'] for r in read(p))
    raw=EXP.parent/'benchmark_0p5b_16h_20261010/raw/mnli/data/train-00000-of-00001.parquet'
    source=pq.read_table(raw).to_pylist();key=lambda s:hashlib.sha256(re.sub(r'\s+',' ',s).strip().casefold().encode()).hexdigest()
    pools=defaultdict(list);seen=set()
    for i,r in enumerate(source):
        group=key(r['premise'])
        if group in used or r['label'] not in [0,1,2]:continue
        ident=key(r['premise']+'\n'+r['hypothesis'])
        if ident in seen:continue
        seen.add(ident);pools[(r['genre'],r['label'])].append((i,r,group))
    result={}
    for gi,genre in enumerate(['fiction','government','slate','telephone','travel']):
        picked=[];rng=np.random.default_rng(701+gi)
        for label,n in enumerate([342,341,341]):
            eligible=[r for r in pools[(genre,label)] if r[2] not in used]
            chosen=rng.choice(len(eligible),n,replace=False);picked.extend(eligible[int(j)] for j in chosen)
        rng.shuffle(picked);rows=[]
        for i,r,g in picked:
            gold='ABC'[r['label']];problem='Premise:\n'+r['premise']+'\n\nHypothesis:\n'+r['hypothesis']+'\n\nA) entailment\nB) neutral\nC) contradiction'
            rows.append(dict(task='mnli',problem=problem,solution='\\boxed{'+gold+'}',gold=gold,answer_type='choice',source_index=i,source_split='train',group_id=g,sample_id=key(f'genre20261015/{i}/{problem}'),genre=genre,raw_source={'choices':{'text':['entailment','neutral','contradiction'],'label':['A','B','C']}}))
        groups={r['group_id'] for r in rows};assert not groups&used;used.update(groups)
        name=f'genre_{genre}';path=data('mnli',name);assert not path.exists();jsonl(path,rows)
        result[name]=dict(n=len(rows),groups=len(groups),sha256=sha(path),labels=dict(Counter(r['gold'] for r in rows)),seed=701+gi)
    write(EXP/'configs/genre_candidates.json',dict(created=time.time(),candidates=result,raw_sha256=sha(raw),role='Secondary candidate-selection task, registered before any candidate selection readout; five natural genres with matched label counts and1024examples each',source_group_overlap_with_existing_splits=0,source_group_overlap_between_candidates=0,not_selected_based_on_performance=True))
    log(result)

if __name__=='__main__':main()
