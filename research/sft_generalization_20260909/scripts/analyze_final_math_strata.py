"""Complete descriptive subject/level partitions; never select favorable strata."""
import ast
import csv
from collections import Counter,defaultdict
import numpy as np
from common import *
planpath=S/'audits/final_math_strata_plan.json';plan=json.loads(planpath.read_text())
primarypath=S/'results/confirmation_plan.json';primary=json.loads(primarypath.read_text())
assert plan['primary_plan_sha256']==sha(primarypath)
data=read(S/'data/math_reused5000.jsonl');assert sha(S/'data/math_reused5000.jsonl')==plan['data_sha256']
main=json.loads((S/'results/confirmation_analysis.json').read_text())
mainstrict=json.loads((S/'results/confirmation_analysis_strict.json').read_text())
assert main['plan_sha256']==mainstrict['plan_sha256']==sha(primarypath)
families=dict(primary['families']);extra_analysis=None
suppstatus=json.loads((S/'audits/expanded_unaveraged_supplement_status.json').read_text())
audit=json.loads((S/'audits/scoring_confirmation_math_reused5000.json').read_text())
scoremodels=dict(audit['models'])
if suppstatus['all_complete']:
    extra_analysis=json.loads((S/'results/expanded_unaveraged_supplement_analysis.json').read_text())
    supplement=json.loads((S/'audits/expanded_unaveraged_supplement_plan.json').read_text())
    families['teacher7_expanded_epoch4']=supplement['models']
    scoremodels.update(json.loads((S/'audits/scoring_supplement_math_reused5000.json').read_text())['models'])
names=['base']+[n for f in families.values() for n in f.values()];root=S/'results/evaluation/math_reused5000'
raw={};strict={};identities={}
for name in names:
    p=root/name/'math_predictions.jsonl';rows=read(p);assert len(rows)==5000
    assert sha(p)==scoremodels[name]['predictions_sha256']
    raw[name]=np.array([r['correct'] for r in rows],int)
    strict[name]=np.array(scoremodels[name]['strict_vector'],int)
    identities[name]=sha(p)
source=S/'scripts/analyze_confirmation.py';assert sha(source)==primary['analysis_script_sha256']
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compare')
namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
compare=namespace['compare'];records=[]
for mode,values,original in [('raw',raw,main),('strict',strict,mainstrict)]:
    b=values['base']
    for family,mapping in families.items():
        group=np.stack([values[mapping[str(seed)]] for seed in primary['seeds']]);d=group-b
        if family in original['datasets']['math_reused5000']['families']:
            expected=original['datasets']['math_reused5000']['families'][family]['vs_base']['delta_pp']
        else:
            key='primary_numeric_repaired' if mode=='raw' else 'strict_numeric_repaired'
            expected=extra_analysis['results'][key]['datasets']['math_reused5000']['families'][family]['vs_base']['delta_pp']
        assert abs(d.mean()*100-expected)<1e-10
        for field,categories in plan['fields'].items():
            total=0;weighted=0.
            for category in categories:
                keep=np.array([r[field]==category for r in data]);n=int(keep.sum());assert n>0
                effect=compare(d[:,keep]);total+=n;weighted+=n*effect['delta_pp']
                counts=group[:,keep].sum(1).tolist()
                records.append(dict(scoring=mode,family=family,field=field,category=category,n=n,
                    base_correct=int(b[keep].sum()),per_seed_correct=counts,
                    mean_wrong_to_right=float(((group[:,keep]==1)&(b[keep]==0)).sum(1).mean()),
                    mean_right_to_wrong=float(((group[:,keep]==0)&(b[keep]==1)).sum(1).mean()),
                    **effect))
            assert total==5000 and abs(weighted/total-expected)<1e-10
training={}
for label,path in plan['training_sources'].items():
    rows=read(S/path);entry=dict(n=len(rows),data_sha256=sha(S/path),partitions={})
    for field in plan['fields']:
        counts=Counter(r[field] for r in rows);tokens=defaultdict(int)
        for r in rows:tokens[r[field]]+=r['target_tokens']
        entry['partitions'][field]={k:dict(n=v,target_tokens=tokens[k]) for k,v in sorted(counts.items())}
        assert sum(counts.values())==len(rows)
    training[label]=entry
out=S/'results/final_math_strata.json'
write(out,dict(time=time.time(),plan_sha256=sha(planpath),script_sha256=sha(__file__),
    source_analysis_sha256=sha(source),identities=identities,training=training,rows=records,
    partition_conservation_passed=True,scope=plan['scope']))
flat=[]
for r in records:
    v={k:x for k,x in r.items() if not isinstance(x,list)}
    v.update({f'seed{seed}_correct':x for seed,x in zip(primary['seeds'],r['per_seed_correct'])})
    for key in ['question_ci95','seed_question_ci95']:v[key+'_low'],v[key+'_high']=r[key]
    flat.append(v)
with (S/'results/final_math_strata.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
print(json.dumps(dict(rows=len(records),families=list(families),partition_conservation_passed=True)))
