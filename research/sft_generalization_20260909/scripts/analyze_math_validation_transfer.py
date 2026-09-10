"""Describe net transfer across the already fixed1500/3500 split, all five seeds."""
import ast
import numpy as np
from common import *
pp=S/'audits/math_validation_transfer_plan.json';plan=json.loads(pp.read_text())
primarypath=S/'results/confirmation_plan.json';primary=json.loads(primarypath.read_text())
assert sha(primarypath)==plan['primary_plan_sha256']
assert sha(S/'audits/math_without_current_validation_plan.json')==plan['partition_plan_sha256']
assert (S/'results/confirmation_analysis.json').exists() and (S/'results/confirmation_analysis_strict.json').exists()
families=dict(primary['families']);scores=json.loads((S/'audits/scoring_confirmation_math_reused5000.json').read_text())['models']
suppstatus=json.loads((S/'audits/expanded_unaveraged_supplement_status.json').read_text())
if suppstatus['all_complete']:
    supplement=json.loads((S/'audits/expanded_unaveraged_supplement_plan.json').read_text())
    assert (S/'results/expanded_unaveraged_supplement_analysis.json').exists()
    families['teacher7_expanded_epoch4']=supplement['models']
    scores=dict(scores,**json.loads((S/'audits/scoring_supplement_math_reused5000.json').read_text())['models'])
keep=primary['validation_excluded_sensitivity']['keep_indices'];other=sorted(set(range(5000))-set(keep))
assert len(keep)==3500 and len(other)==1500
partitions={'current_selected1500':np.array(other,int),'remaining3500':np.array(keep,int)}
root=S/'results/evaluation/math_reused5000';base=read(root/'base/math_predictions.jsonl');assert len(base)==5000
source=S/'scripts/analyze_confirmation.py';assert sha(source)==primary['analysis_script_sha256']
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['compare','macro_compare']]
namespace={'np':np,'seeds':primary['seeds']};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
compare,macro_compare=namespace['compare'],namespace['macro_compare'];raw={};strict={};texts={};identities={}
for name in ['base']+[n for f in families.values() for n in f.values()]:
    path=root/name/'math_predictions.jsonl';rows=read(path);assert len(rows)==5000
    assert sha(path)==scores[name]['predictions_sha256']
    assert [r['sample_hash'] for r in rows]==[r['sample_hash'] for r in base]
    raw[name]=np.array([r['correct'] for r in rows],int);strict[name]=np.array(scores[name]['strict_vector'],int)
    texts[name]=np.array([r['prediction']==b['prediction'] for r,b in zip(rows,base)],bool);identities[name]=sha(path)
results={}
for mode,values in [('raw',raw),('strict',strict)]:
    b=values['base'];results[mode]={}
    for family,mapping in families.items():
        names=[mapping[str(seed)] for seed in primary['seeds']];group=np.stack([values[n] for n in names]);dd=group-b
        sub={}
        for label,index in partitions.items():
            wins=((group[:,index]==1)&(b[index]==0)).sum(1);losses=((group[:,index]==0)&(b[index]==1)).sum(1)
            assert np.array_equal(wins-losses,dd[:,index].sum(1))
            sub[label]=dict(n=len(index),base_correct=int(b[index].sum()),per_seed_correct=group[:,index].sum(1).tolist(),
                per_seed_rescues=wins.tolist(),per_seed_losses=losses.tolist(),
                baseline_text_identical=[int(texts[n][index].sum()) for n in names],
                mean_rescue_rate_among_baseline_wrong=float(wins.mean()/(len(index)-b[index].sum())*100),
                mean_loss_rate_among_baseline_correct=float(losses.mean()/b[index].sum()*100),vs_base=compare(dd[:,index]))
        contrast=macro_compare([dd[:,other],-dd[:,keep]])
        for key in ['delta_pp','per_seed_delta_pp','question_ci95','seed_question_ci95']:
            contrast[key]=[2*x for x in contrast[key]] if isinstance(contrast[key],list) else 2*contrast[key]
        assert abs(contrast['delta_pp']-(dd[:,other].mean()-dd[:,keep].mean())*100)<1e-10
        results[mode][family]=dict(partitions=sub,selected1500_minus_remaining3500=contrast)
write(S/'results/math_validation_transfer.json',dict(time=time.time(),plan_sha256=sha(pp),script_sha256=sha(__file__),
    statistics_source_sha256=sha(source),identities=identities,results=results,scope=plan['scope']))
print(json.dumps({mode:{f:v['selected1500_minus_remaining3500']['delta_pp'] for f,v in r.items()} for mode,r in results.items()},indent=2))
