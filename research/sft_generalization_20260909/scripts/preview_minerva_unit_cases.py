"""Advance only the frozen unit-case reading; never score or approve cases.

The authoritative preparer still runs through the existing CPU collector after
its dependencies finish. This separate preview permits reading actual answers
before the slower native evaluation terminates. All 21 Minerva files must be
complete, and final case identities must subsequently match exactly.
"""
import re
from common import *
from evaluation.audit_lora_sft_review import boxed_contents

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
pp = S/'audits/minerva_unit_case_audit_plan.json'
plan = json.loads(pp.read_text())
assert sha(S/'results/confirmation_plan.json') == plan['primary_plan_sha256']
cp = S/'audits/minerva_numeric_contract.json'
assert sha(cp) == plan['numeric_contract_sha256']
contract = json.loads(cp.read_text())
execution = json.loads((S/'audits/additional_collection_execution_amendment.json').read_text())
original = S/'scripts/prepare_minerva_unit_review.py'
assert sha(original) == execution['unchanged_dependency_sha256'][original.name]
supp_path = S/'audits/expanded_unaveraged_supplement_plan.json'
assert sha(supp_path) == execution['supplement_plan_sha256']
supp = json.loads(supp_path.read_text())
assert json.loads((S/'audits/expanded_supplement_gpu2_protocol_replay.json').read_text())['passed']
names = list(json.loads((S/'results/confirmation_manifest.json').read_text()))
names += list(supp['models'].values())
assert len(names) == len(set(names)) == 21
data_path = S/'data/ood_minerva.jsonl'
assert sha(data_path) == supp['data_sha256']['ood_minerva']
data = read(data_path)
refs = {r['index'] for r in contract['numeric_references']}
pattern = re.compile(r'%|\\circ\b|°|\bdegrees?\b', re.I)
groups = {}
identities = {}
per_model = {}
for name in names:
    root = S/'results/evaluation/ood_minerva'/name
    summary = json.loads((root/'math_summary.json').read_text())
    assert summary['samples'] == 272
    path = root/'math_predictions.jsonl'
    pred = read(path)
    assert len(pred) == 272
    identities[name] = sha(path)
    count = 0
    for i in sorted(refs):
        assert pred[i]['index'] == i
        boxes = boxed_contents(pred[i]['prediction'])
        if not boxes or not pattern.search(boxes[-1]):
            continue
        box = boxes[-1]
        key = (i, box)
        count += 1
        if key not in groups:
            groups[key] = dict(index=i, problem=data[i]['problem'],
                reference_answer=data[i]['answer'], solution=data[i]['solution'],
                last_box=box, outputs=[])
        groups[key]['outputs'].append(dict(model=name,
            prediction=pred[i]['prediction'], legacy_correct=bool(pred[i]['correct']),
            predictions_sha256=identities[name]))
    per_model[name] = count
cases = []
for j, key in enumerate(sorted(groups)):
    case = groups[key]
    case['case_id'] = f'minerva_units_{j:03d}'
    cases.append(case)
case_path = S/'audits/minerva_explicit_unit_preview_cases.jsonl'
audit_path = S/'audits/minerva_explicit_unit_preview_inventory.json'
assert not case_path.exists() and not audit_path.exists()
jsonl(case_path, cases)
write(audit_path, dict(time=time.time(), plan_sha256=sha(pp),
    script_sha256=sha(__file__), authoritative_preparer_sha256=sha(original),
    primary_plan_sha256=plan['primary_plan_sha256'], supplement_plan_sha256=sha(supp_path),
    numeric_contract_sha256=sha(cp), dataset_sha256=sha(data_path),
    identities=identities, n_models=len(names), per_model_counts=per_model,
    unique_question_box_cases=len(cases),
    unique_question_indices=sorted({c['index'] for c in cases}),
    case_file_sha256=sha(case_path), scope=plan['scope'],
    preview_only=True, reviewed=False,
    final_reconciliation_required='Compare every question/box/case_id/output model/full text/file hash and per-model count against the unchanged authoritative preparer. Numerical sidecar judgments are deliberately absent here; review them when available.'))
print(json.dumps(dict(unique_cases=len(cases), per_model=per_model)))
