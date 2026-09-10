"""Gather every supplemental warning without changing scores or claiming review."""
from common import *
pp = S/'audits/expanded_unaveraged_supplement_plan.json'; plan = json.loads(pp.read_text())
tags = [('supplement_'+d, d) for d in plan['datasets']]
tags.append(('supplement_minerva_numeric', 'ood_minerva'))
entries = []; inventory = []
for tag, dataset in tags:
    ap = S/f'audits/scoring_{tag}.json'; a = json.loads(ap.read_text())
    assert a['dataset_sha256'] == plan['data_sha256'][dataset]
    rows = read(S/f'data/{dataset}.jsonl'); cache = {}
    for j, warning in enumerate(a['parser_comparison_warnings']):
        i = warning['index']; item = dict(case_id=f'{tag}:{j}', tag=tag,
            dataset=dataset, warning=warning, problem=rows[i]['problem'], solution=rows[i]['solution'])
        if 'model' in warning:
            name = warning['model']; p = Path(a['prediction_root'])/name/'math_predictions.jsonl'
            assert a['models'][name]['predictions_sha256'] == sha(p)
            if name not in cache: cache[name] = read(p)
            r = cache[name][i]
            item.update(model=name, prediction=r['prediction'], original_correct=r['correct'],
                strict_correct=bool(a['models'][name]['strict_vector'][i]),
                finish_reason=r['finish_reason'], generated_tokens=r['generated_tokens'], predictions_sha256=sha(p))
        entries.append(item)
    inventory.append(dict(tag=tag, audit_sha256=sha(ap), n_models=len(a['models']),
        n_judgments=sum(x['n'] for x in a['models'].values()), warning_count=len(a['parser_comparison_warnings'])))
jsonl(S/'audits/supplement_scoring_warning_cases.jsonl', entries)
write(S/'audits/supplement_scoring_warning_inventory.json', dict(time=time.time(),
    plan_sha256=sha(pp), audits=inventory, n_warning_events=len(entries),
    case_file_sha256=sha(S/'audits/supplement_scoring_warning_cases.jsonl'),
    scope='Preparation only; every warning needs explicit review or exact prior-case identity.'))
print(json.dumps(dict(n_warning_events=len(entries), audits=inventory), indent=2))
