"""Collect explicit percentage/degree answers for context-aware review, without rescoring."""
import re
from common import *
from evaluation.audit_lora_sft_review import boxed_contents
pp=S/'audits/minerva_unit_case_audit_plan.json';plan=json.loads(pp.read_text())
primary=json.loads((S/'results/confirmation_plan.json').read_text());assert sha(S/'results/confirmation_plan.json')==plan['primary_plan_sha256']
cp=S/'audits/minerva_numeric_contract.json';contract=json.loads(cp.read_text());assert sha(cp)==plan['numeric_contract_sha256']
refs={r['index'] for r in contract['numeric_references']}
names=list(json.loads((S/'results/confirmation_manifest.json').read_text()))
old=json.loads((S/'audits/scoring_minerva_numeric_corrected.json').read_text());numeric=dict(old['models'])
suppstatus=json.loads((S/'audits/expanded_unaveraged_supplement_status.json').read_text())
if suppstatus['all_complete']:
    extra=json.loads((S/'audits/expanded_unaveraged_supplement_plan.json').read_text())
    names+=list(extra['models'].values())
    numeric.update(json.loads((S/'audits/scoring_supplement_minerva_numeric.json').read_text())['models'])
rows=read(S/'data/ood_minerva.jsonl');pattern=re.compile(r'%|\\circ\b|°|\bdegrees?\b',re.I)
groups={};identities={};per_model={}
for name in names:
    path=S/'results/evaluation/ood_minerva'/name/'math_predictions.jsonl';pred=read(path);assert len(pred)==272
    assert sha(path)==numeric[name]['predictions_sha256'];identities[name]=sha(path);count=0
    for i in sorted(refs):
        boxes=boxed_contents(pred[i]['prediction'])
        if not boxes or not pattern.search(boxes[-1]):continue
        box=boxes[-1];key=(i,box);count+=1
        if key not in groups:
            groups[key]=dict(index=i,problem=rows[i]['problem'],reference_answer=rows[i]['answer'],solution=rows[i]['solution'],last_box=box,outputs=[])
        groups[key]['outputs'].append(dict(model=name,prediction=pred[i]['prediction'],
            numeric_primary_correct=bool(numeric[name]['primary_vector'][i]),numeric_strict_correct=bool(numeric[name]['strict_vector'][i]),
            legacy_correct=bool(pred[i]['correct']),predictions_sha256=sha(path)))
    per_model[name]=count
cases=[]
for j,key in enumerate(sorted(groups)):
    case=groups[key];case['case_id']=f'minerva_units_{j:03d}';cases.append(case)
jsonl(S/'audits/minerva_explicit_unit_cases.jsonl',cases)
write(S/'audits/minerva_explicit_unit_inventory.json',dict(time=time.time(),plan_sha256=sha(pp),
    script_sha256=sha(__file__),identities=identities,n_models=len(names),per_model_counts=per_model,
    unique_question_box_cases=len(cases),unique_question_indices=sorted({c['index'] for c in cases}),
    case_file_sha256=sha(S/'audits/minerva_explicit_unit_cases.jsonl'),scope=plan['scope']))
print(json.dumps(dict(unique_cases=len(cases),per_model=per_model),indent=2))
