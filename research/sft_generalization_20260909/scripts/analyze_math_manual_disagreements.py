"""Post-outcome partial manual sensitivity; immutable original statistics and scores."""
import ast
from collections import Counter
import numpy as np
from common import *

plan_path=S/'audits/math_scoring_disagreement_review_plan.json'
decision_path=S/'audits/math_scoring_disagreement_review_decisions.json'
cases_path=S/'audits/math_scoring_disagreement_review_cases.jsonl'
plan=json.loads(plan_path.read_text()); review=json.loads(decision_path.read_text())
cases=read(cases_path); frozen=json.loads((S/'results/confirmation_plan.json').read_text())
source=S/'scripts/analyze_confirmation.py'
assert sha(source)==plan['statistics_source_sha256']==frozen['analysis_script_sha256']
assert review['all_reviewed'] and review['n_reviewed']==len(cases)==47
assert review['plan_sha256']==sha(plan_path) and review['case_file_sha256']==sha(cases_path)
assert sha(S/'results/confirmation_plan.json')==plan['primary_plan_sha256']
fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compare')
namespace={'np':np}; exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
compare=namespace['compare']
strict_path=S/'audits/scoring_confirmation_math_reused5000.json'
assert sha(strict_path)==plan['strict_audit_sha256']
strict=json.loads(strict_path.read_text())['models']
names=plan['models']; assert names==['base']+[frozen['families'][frozen['primary_family']][str(s)] for s in frozen['seeds']]
raw={}; hashes={}; expected=set()
for name in names:
    path=S/'results/evaluation/math_reused5000'/name/'math_predictions.jsonl'
    rows=read(path); assert len(rows)==5000
    hashes[str(path.relative_to(S))]=sha(path)
    assert sha(path)==strict[name]['predictions_sha256']
    raw[name]=np.array([r['correct'] for r in rows],int)
    expected.update((name,int(i)) for i in np.flatnonzero(raw[name]!=strict[name]['strict_vector']))
assert expected=={(c['model'],c['index']) for c in cases}
low={n:v.copy() for n,v in raw.items()}; high={n:v.copy() for n,v in raw.items()}
for c,d in zip(cases,review['decisions']):
    assert c['case_id']==d['case_id'] and d['reviewed'] and d['complete_text_read']
    assert d['case_sha256']==hashlib.sha256(json.dumps(c,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    assert d['prediction_sha256']==c['prediction_sha256']
    name=c['model']; i=c['index']; val=d['mathematical_final_answer_correct']
    assert c['predictions_file_sha256']==hashes[f'results/evaluation/math_reused5000/{name}/math_predictions.jsonl']
    if val is None:
        low[name][i]=int(name=='base'); high[name][i]=int(name!='base')
    else:
        assert type(val) is bool; low[name][i]=high[name][i]=int(val)

results={}
for partition,indices in [('full5000',np.arange(5000)),('remaining3500',np.array(frozen['validation_excluded_sensitivity']['keep_indices']))]:
    results[partition]={}
    for label,v in [('conservative_lower',low),('conservative_upper',high)]:
        b=v['base'][indices]; g=np.stack([v[n][indices] for n in names[1:]])
        results[partition][label]={'n':len(indices),'base_correct':int(b.sum()),'per_seed_correct':g.sum(1).astype(int).tolist(),'vs_base':compare(g-b)}
    assert np.all(np.stack([low[n]-low['base'] for n in names[1:]])<=np.stack([high[n]-high['base'] for n in names[1:]]))
counts={}
for name in names:
    ds=[d for d in review['decisions'] if d['model']==name]
    counts[name]={'disagreements':len(ds),'correct':sum(d['mathematical_final_answer_correct'] is True for d in ds),'incorrect':sum(d['mathematical_final_answer_correct'] is False for d in ds),'unresolved':sum(d['mathematical_final_answer_correct'] is None for d in ds),'categories':dict(Counter(d['category'] for d in ds))}
sources={str(p.relative_to(S)):sha(p) for p in [plan_path,decision_path,cases_path,strict_path,source,S/'results/confirmation_plan.json']}
sources.update(hashes)
out={'time':time.time(),'script_sha256':sha(__file__),'source_sha256':sources,'all47_reviewed':True,'counts':counts,'unresolved':[{'model':d['model'],'index':d['index'],'case_id':d['case_id']} for d in review['decisions'] if d['mathematical_final_answer_correct'] is None],'results':results,'scope':'Post-outcome, unblinded, partial manual final-answer sensitivity on all47 raw/strict disagreements for frozen primary plus baseline. Agreed judgments elsewhere are retained, not certified mathematically correct. Unresolved cases receive adverse/favorable assignments, with one shared baseline across all seeds. Bounds and their fixed-source paired bootstrap intervals are sensitivity envelopes, not a new prespecified efficacy criterion or a guarantee of universal scoring correctness. Correct final answer does not certify reasoning validity.'}
write(S/'results/math_manual_disagreement_sensitivity.json',out)
print(json.dumps({'counts':counts,'results':results},ensure_ascii=False,indent=2))
