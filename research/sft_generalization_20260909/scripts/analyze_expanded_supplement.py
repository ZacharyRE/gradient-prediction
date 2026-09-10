"""Secondary full factorial comparison; primary campaign outputs remain unchanged."""
import ast
import logging
import subprocess
import numpy as np
from common import *
from minerva_numeric import gold_numeric, numeric_prediction, numeric_equal

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
pp = S/'audits/expanded_unaveraged_supplement_plan.json'; plan = json.loads(pp.read_text())
fp = S/'audits/expanded_unaveraged_supplement_frozen.json'; frozen = json.loads(fp.read_text())
assert frozen['plan_sha256'] == sha(pp) and frozen['analysis_sha256'] == sha(__file__)
primary_path = S/'results/confirmation_plan.json'; primary = json.loads(primary_path.read_text())
assert frozen['primary_plan_sha256'] == sha(primary_path)
assert json.loads((S/'audits/expanded_unaveraged_supplement_status.json').read_text())['all_complete']
assert json.loads((S/'audits/expanded_supplement_gpu1_protocol_replay.json').read_text())['passed']
assert json.loads((S/'audits/final_collection_status.json').read_text())['all_complete']
out = S/'results/expanded_unaveraged_supplement_analysis.json'; assert not out.exists()
seeds = plan['seeds']; extra = 'teacher7_expanded_epoch4'
families = dict(primary['families'], **{extra: plan['models']})
assert len(families) == 4 and seeds == [43, 44, 45, 46, 47]
new_names = list(plan['models'].values())
all_names = ['base'] + [n for family in families.values() for n in family.values()]
audits, raw, strict, identities = {}, {}, {}, {}
for dataset in plan['datasets']:
    ap = S/f'audits/scoring_supplement_{dataset}.json'
    if not ap.exists():
        with (S/f'logs/strict_supplement_{dataset}.log').open('w') as log:
            subprocess.run([sys.executable, str(S/'scripts/score_sensitivity.py'),
                '--dataset', dataset, '--tag', 'supplement_'+dataset, '--models', *new_names],
                stdout=log, stderr=subprocess.STDOUT, check=True,
                timeout=max(1, DEADLINE-time.time()))
    new = json.loads(ap.read_text())
    old = json.loads((S/f'audits/scoring_confirmation_{dataset}.json').read_text())
    assert set(new['models']) == set(new_names)
    assert new['script_sha256'] == old['script_sha256'] == plan['strict_script_sha256']
    assert new['dataset_sha256'] == old['dataset_sha256'] == plan['data_sha256'][dataset]
    combined = dict(old['models'], **new['models']); assert set(combined) == set(all_names)
    root = S/'results/evaluation'/dataset
    bb = read(root/'base/math_predictions.jsonl')
    bm = json.loads((root/'base/math_manifest.json').read_text())
    raw[dataset] = {}; strict[dataset] = {}; identities[dataset] = {}
    for name in all_names:
        path = root/name/'math_predictions.jsonl'; rr = read(path)
        assert len(rr) == len(bb) and [r['sample_hash'] for r in rr] == [r['sample_hash'] for r in bb]
        assert sha(path) == combined[name]['predictions_sha256']
        mm = json.loads((root/name/'math_manifest.json').read_text())
        for key, value in primary['evaluation_protocol'].items(): assert mm[key] == value
        assert mm['model'] == bm['model'] and mm['data_sha256'] == plan['data_sha256'][dataset]
        if name != 'base':
            expected = frozen['artifacts'][name]['files'] if name in new_names else primary['adapter_files_sha256'][name]
            assert mm['adapter']['files'] == expected
        raw[dataset][name] = np.array([r['correct'] for r in rr], int)
        strict[dataset][name] = np.array(combined[name]['strict_vector'], int)
        identities[dataset][name] = dict(predictions_sha256=sha(path), manifest_sha256=sha(root/name/'math_manifest.json'))
    audits[dataset] = dict(supplement_strict_sha256=sha(ap),
                          original_strict_sha256=sha(S/f'audits/scoring_confirmation_{dataset}.json'))

contract_path = S/'audits/minerva_numeric_contract.json'
assert sha(contract_path) == plan['numeric_contract_sha256']
assert sha(S/'scripts/minerva_numeric.py') == plan['numeric_helper_sha256']
contract = json.loads(contract_path.read_text()); assert contract['passed']
refs = {r['index']: r['numeric_value'] for r in contract['numeric_references']}
original_numeric_path = S/'audits/scoring_minerva_numeric_corrected.json'
original_numeric = json.loads(original_numeric_path.read_text())
assert original_numeric['plan_sha256'] == sha(primary_path)
numeric_models = {}; warnings = []; current = {}
class Capture(logging.Handler):
    def emit(self, record):
        if record.name.startswith('math_verify'):
            warnings.append(dict(current, logger=record.name, message=record.getMessage()))
handler = Capture(level=logging.WARNING); logging.getLogger().addHandler(handler)
try:
    rows = read(S/'data/ood_minerva.jsonl')
    for name in new_names:
        path = S/'results/evaluation/ood_minerva'/name/'math_predictions.jsonl'; rr = read(path)
        v = raw['ood_minerva'][name].copy(); st = strict['ood_minerva'][name].copy()
        sensitivity = {str(t): {'primary_vector':v.copy(), 'strict_vector':st.copy()} for t in contract['sensitivity_rtols']}
        details = []
        for i, gold in refs.items():
            assert gold_numeric(rows[i]['answer']) == gold
            current.clear(); current.update(phase='minerva_numeric_prediction', model=name, index=i)
            start = len(warnings); pred = numeric_prediction(rr[i]['prediction'])
            ok = int(numeric_equal(pred['value'], gold, contract['primary_rtol'])); v[i] = st[i] = ok
            for w in warnings[start:]: w['final_strict_correct'] = bool(ok)
            if pred['reason'] in ['numeric_evaluation_timeout', 'parse_exception']:
                warnings.append(dict(current, logger='minerva_numeric', message=pred['reason'], final_strict_correct=bool(ok)))
            for tolerance in contract['sensitivity_rtols']:
                value = int(numeric_equal(pred['value'], gold, tolerance))
                sensitivity[str(tolerance)]['primary_vector'][i] = value
                sensitivity[str(tolerance)]['strict_vector'][i] = value
            details.append(dict(index=i, gold=gold, parsed_prediction=pred, correct=bool(ok)))
        numeric_models[name] = dict(n=272, predictions_sha256=sha(path), primary_vector=v.tolist(),
            strict_vector=st.tolist(), numeric_cases=details,
            sensitivity_vectors={t:{k:x.tolist() for k,x in vv.items()} for t,vv in sensitivity.items()})
finally: logging.getLogger().removeHandler(handler)
npth = S/'audits/scoring_supplement_minerva_numeric.json'; assert not npth.exists()
write(npth, dict(time=time.time(), script_sha256=sha(__file__), plan_sha256=sha(pp),
    helper_sha256=plan['numeric_helper_sha256'], contract_sha256=sha(contract_path),
    dataset_sha256=plan['data_sha256']['ood_minerva'],
    prediction_root=str(S/'results/evaluation/ood_minerva'), models=numeric_models,
    parser_comparison_warnings=warnings, scope='Same already validated numeric helper/policy as primary; no original outputs changed.'))
legacy_raw = {n:v.copy() for n,v in raw['ood_minerva'].items()}
legacy_strict = {n:v.copy() for n,v in strict['ood_minerva'].items()}
merged_numeric = dict(original_numeric['models'], **numeric_models)
for name in all_names:
    assert merged_numeric[name]['predictions_sha256'] == identities['ood_minerva'][name]['predictions_sha256']
    raw['ood_minerva'][name] = np.array(merged_numeric[name]['primary_vector'], int)
    strict['ood_minerva'][name] = np.array(merged_numeric[name]['strict_vector'], int)

source = S/'scripts/analyze_confirmation.py'; assert sha(source) == plan['analysis_script_sha256']
contract_stats = json.loads((S/'audits/final_statistics_contract.json').read_text())
assert contract_stats['passed'] and contract_stats['analysis_script_sha256'] == sha(source)
tree = ast.parse(source.read_text()); namespace = {'np': np, 'seeds': seeds}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['compare', 'macro_compare']]
assert len(nodes) == 2
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
compare, macro_compare = namespace['compare'], namespace['macro_compare']
pairs = [('teacher7_expanded_epoch4', 'teacher7_epoch4'),
         ('teacher7_expanded_avg1234', 'teacher7_expanded_epoch4'),
         ('teacher7_avg1234', 'teacher7_epoch4'),
         ('teacher7_expanded_avg1234', 'teacher7_avg1234')]
results = {}
for mode, vv in [('primary_numeric_repaired', raw), ('strict_numeric_repaired', strict)]:
    datasets = {}; differences = {}
    for dataset, values in vv.items():
        b = values['base']; groups = {f:np.stack([values[m[str(seed)]] for seed in seeds]) for f,m in families.items()}
        differences[dataset] = {f:g-b for f,g in groups.items()}
        datasets[dataset] = dict(n=len(b), base_correct=int(b.sum()),
            families={f:dict(per_seed_correct=g.sum(1).tolist(), vs_base=compare(g-b)) for f,g in groups.items()},
            contrasts={t+'_minus_'+c:compare(groups[t]-groups[c]) for t,c in pairs})
    macro = {f:macro_compare([differences[d][f] for d in plan['macro_datasets']]) for f in families}
    macro_controls = {t+'_minus_'+c:macro_compare([differences[d][t]-differences[d][c] for d in plan['macro_datasets']]) for t,c in pairs}
    numerical_sensitivity = {}
    legacy = legacy_strict if mode.startswith('strict') else legacy_raw
    field = 'strict_vector' if mode.startswith('strict') else 'primary_vector'
    for tolerance in ['legacy_saved_scoring', *map(str, contract['sensitivity_rtols'])]:
        vals = legacy if tolerance == 'legacy_saved_scoring' else {
            n:np.array(merged_numeric[n]['sensitivity_vectors'][tolerance][field], int) for n in all_names}
        b = vals['base']; g = np.stack([vals[n] for n in new_names]); dd = g-b
        numerical_sensitivity[tolerance] = dict(minerva_base_correct=int(b.sum()),
            minerva_per_seed_correct=g.sum(1).tolist(), minerva_vs_base=compare(dd),
            ood_macro=macro_compare([dd if d == 'ood_minerva' else differences[d][extra] for d in plan['macro_datasets']]))
    mm = datasets['math_reused5000']['families'][extra]['vs_base']; oo = macro[extra]
    results[mode] = dict(datasets=datasets, ood_macro=macro, ood_macro_contrasts=macro_controls,
        supplemental_minerva_sensitivity=numerical_sensitivity,
        descriptive_criteria=dict(math_all_seeds_positive=mm['every_seed_positive'],
            math_question_ci_positive=mm['question_ci95'][0]>0, math_seed_question_ci_positive=mm['seed_question_ci95'][0]>0,
            ood_all_seeds_positive=oo['every_seed_positive'], ood_question_ci_positive=oo['question_ci95'][0]>0,
            ood_seed_question_ci_positive=oo['seed_question_ci95'][0]>0))
write(out, dict(time=time.time(), script_sha256=sha(__file__), plan_sha256=sha(pp),
    frozen_sha256=sha(fp), original_primary_plan_sha256=sha(primary_path), seeds=seeds,
    identities=identities, scoring_audits=audits, numeric_audit_sha256=sha(npth),
    results=results, scope=plan['scope']))
print(json.dumps({k:v['descriptive_criteria'] for k,v in results.items()}, indent=2))
