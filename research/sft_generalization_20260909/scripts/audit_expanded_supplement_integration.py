"""Synthetic-only end-to-end supplementary analyzer test; no model inference."""
import subprocess
import shutil
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
f = S/'audits/expanded_supplement_synthetic_fixture'; assert not f.exists()
for d in ['scripts', 'audits', 'data', 'results']: (f/d).mkdir(parents=True, exist_ok=True)
for name in ['analyze_confirmation.py', 'minerva_numeric.py']:
    (f/'scripts'/name).symlink_to(S/'scripts'/name)
source = S/'scripts/analyze_expanded_supplement.py'
plan = json.loads((S/'audits/expanded_unaveraged_supplement_plan.json').read_text())
oldfamilies = json.loads((S/'audits/final_selection_rule.json').read_text())['families']
newnames = list(plan['models'].values()); oldnames = ['base']+[n for v in oldfamilies.values() for n in v.values()]
names = oldnames+newnames
original_contract = S/'audits/minerva_numeric_contract.json'
contract = json.loads(original_contract.read_text())
synthetic_correct = read(S/'audits/minerva_synthetic_fixture/results/evaluation/ood_minerva/synthetic_correct/math_predictions.jsonl')
all_files = {n:{'adapter_model.safetensors':'synthetic-'+n} for n in names if n != 'base'}
protocol = dict(schema=1, script_sha256=plan['evaluation_script_sha256'], engine={'synthetic':True},
                batch_invariant=True, max_tokens=2048, temperature=0, chunk_size=500)
for dataset in plan['datasets']:
    if dataset == 'ood_minerva':
        rows = read(S/'data/ood_minerva.jsonl')
    else: rows = [dict(problem=f'SYNTHETIC TEST ONLY {i}', solution='\\boxed{0}') for i in range(4)]
    jsonl(f/f'data/{dataset}.jsonl', rows); plan['data_sha256'][dataset] = sha(f/f'data/{dataset}.jsonl')
    model_scores = {}
    for name in names:
        root = f/'results/evaluation'/dataset/name; root.mkdir(parents=True)
        predictions = [dict(index=i, sample_hash='synthetic-'+str(i), correct=False,
            prediction=synthetic_correct[i]['prediction'] if dataset == 'ood_minerva' and name in newnames else '\\boxed{0}',
            finish_reason='stop', generated_tokens=1) for i in range(len(rows))]
        jsonl(root/'math_predictions.jsonl', predictions)
        write(root/'math_summary.json', dict(samples=len(rows), correct=0, synthetic=True))
        write(root/'math_manifest.json', dict(protocol, model={'synthetic':True},
            data_sha256=plan['data_sha256'][dataset], adapter={'files':all_files[name]} if name != 'base' else None))
        model_scores[name] = dict(n=len(rows), strict_vector=[0]*len(rows),
                                  predictions_sha256=sha(root/'math_predictions.jsonl'))
    for prefix, selected in [('confirmation_', oldnames), ('supplement_', newnames)]:
        write(f/f'audits/scoring_{prefix}{dataset}.json', dict(script_sha256=plan['strict_script_sha256'],
            dataset_sha256=plan['data_sha256'][dataset], prediction_root=str(f/'results/evaluation'/dataset),
            models={n:model_scores[n] for n in selected}, parser_comparison_warnings=[]))
contract['dataset_sha256'] = plan['data_sha256']['ood_minerva']
write(f/'audits/minerva_numeric_contract.json', contract)
plan['numeric_contract_sha256'] = sha(f/'audits/minerva_numeric_contract.json')
write(f/'audits/expanded_unaveraged_supplement_plan.json', plan)
primary = dict(families=oldfamilies, evaluation_protocol=protocol, adapter_files_sha256=all_files)
write(f/'results/confirmation_plan.json', primary)
write(f/'audits/expanded_unaveraged_supplement_frozen.json', dict(
    plan_sha256=sha(f/'audits/expanded_unaveraged_supplement_plan.json'), analysis_sha256=sha(source),
    primary_plan_sha256=sha(f/'results/confirmation_plan.json'), artifacts={n:{'files':all_files[n]} for n in newnames}))
for name in ['expanded_unaveraged_supplement_status', 'final_collection_status']:
    write(f/f'audits/{name}.json', dict(all_complete=True, synthetic=True))
write(f/'audits/expanded_supplement_gpu1_protocol_replay.json', dict(passed=True, synthetic=True))
shutil.copy2(S/'audits/final_statistics_contract.json', f/'audits/final_statistics_contract.json')
numeric_old = {}
for name in oldnames:
    path = f/'results/evaluation/ood_minerva'/name/'math_predictions.jsonl'
    numeric_old[name] = dict(predictions_sha256=sha(path), primary_vector=[0]*272, strict_vector=[0]*272,
        sensitivity_vectors={str(t):dict(primary_vector=[0]*272, strict_vector=[0]*272) for t in contract['sensitivity_rtols']})
write(f/'audits/scoring_minerva_numeric_corrected.json', dict(
    plan_sha256=sha(f/'results/confirmation_plan.json'), models=numeric_old, synthetic=True))
code = "import sys,runpy;sys.path.insert(0,sys.argv[1]);import common;from pathlib import Path;common.S=Path(sys.argv[2]);runpy.run_path(sys.argv[3],run_name='__main__')"
with (f/'run.log').open('w') as log:
    subprocess.run([sys.executable, '-c', code, str(S/'scripts'), str(f), str(source)],
                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
result = json.loads((f/'results/expanded_unaveraged_supplement_analysis.json').read_text())
expected = 191/272/3*100
for mode, value in result['results'].items():
    extra = value['datasets']['ood_minerva']['families']['teacher7_expanded_epoch4']
    assert extra['per_seed_correct'] == [191]*5, (mode, extra)
    assert abs(value['ood_macro']['teacher7_expanded_epoch4']['delta_pp']-expected) < 1e-10
    assert abs(value['ood_macro_contrasts']['teacher7_expanded_avg1234_minus_teacher7_expanded_epoch4']['delta_pp']+expected) < 1e-10
    assert value['ood_macro_contrasts']['teacher7_avg1234_minus_teacher7_epoch4']['delta_pp'] == 0
    assert value['supplemental_minerva_sensitivity']['legacy_saved_scoring']['ood_macro']['delta_pp'] == 0
    assert not value['descriptive_criteria']['math_all_seeds_positive']
write(S/'audits/expanded_supplement_analysis_integration.json', dict(time=time.time(), passed=True,
    analysis_script_sha256=sha(source), test_script_sha256=sha(__file__), fixture=str(f),
    expected_numeric_macro_pp=expected, checks='Actual analyzer on21 synthetic models:191 numeric rescues, signed factorial contrasts, legacy0, zeroMATH failure, both score modes.',
    scope='Synthetic CPU fixture only, not an actual confirmation plan, model output, GPU replay, or model efficacy result.'))
print('Supplementary analyzer synthetic integration passed.')
