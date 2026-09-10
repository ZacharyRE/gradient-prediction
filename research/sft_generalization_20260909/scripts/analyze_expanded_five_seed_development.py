"""Report both saved endpoints of all five expanded runs; no new GPU work/selection."""
import ast
import subprocess
import numpy as np
from common import *

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
seeds = [43, 44, 45, 46, 47]
groups = {
    'averaged': [f'avg_teacher7_expansion_combined_1234_s{x}_epoch4' for x in seeds],
    'unaveraged': [f'teacher7_expansion_combined_lr5e5_s{x}_epoch4' for x in seeds],
}
names = ['base'] + sum(groups.values(), [])
root = S/'results/evaluation/dev'
assert all((root/n/'math_summary.json').exists() for n in names)
out = S/'results/expanded_five_seed_development.json'
assert not out.exists()
auditpath = S/'audits/scoring_expanded_five_seed_development.json'
if not auditpath.exists():
    with (S/'logs/strict_expanded_five_seed_development.log').open('w') as log:
        subprocess.run([sys.executable, str(S/'scripts/score_sensitivity.py'),
                        '--dataset', 'dev', '--tag', 'expanded_five_seed_development',
                        '--models', *names], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=max(1, DEADLINE-time.time()))
audit = json.loads(auditpath.read_text())
assert set(audit['models']) == set(names)
assert audit['dataset_sha256'] == sha(S/'data/dev.jsonl')
assert audit['script_sha256'] == sha(S/'scripts/score_sensitivity.py')
bridge = json.loads((S/'audits/evaluator_version_bridge.json').read_text())
assert bridge['passed']
base = read(root/'base/math_predictions.jsonl')
base_manifest = json.loads((root/'base/math_manifest.json').read_text())
vectors = {'raw': {}, 'strict': {}}
identities = {}
for name in names:
    path = root/name/'math_predictions.jsonl'
    rows = read(path)
    assert len(rows) == 500
    assert [r['sample_hash'] for r in rows] == [r['sample_hash'] for r in base]
    assert audit['models'][name]['predictions_sha256'] == sha(path)
    manifest = json.loads((root/name/'math_manifest.json').read_text())
    assert manifest['script_sha256'] in bridge['allowed_development_script_sha256']
    for key in ['model', 'data_sha256', 'schema', 'engine', 'batch_invariant',
                'max_tokens', 'temperature', 'chunk_size']:
        assert manifest[key] == base_manifest[key], (name, key)
    identities[name] = dict(predictions_sha256=sha(path),
                            manifest_sha256=sha(root/name/'math_manifest.json'))
    vectors['raw'][name] = np.array([r['correct'] for r in rows], int)
    vectors['strict'][name] = np.array(audit['models'][name]['strict_vector'], int)

# Use the already tested paired seed/question statistic without importing its CLI.
source = S/'scripts/analyze_confirmation.py'
contract = json.loads((S/'audits/final_statistics_contract.json').read_text())
assert contract['passed'] and contract['analysis_script_sha256'] == sha(source)
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compare')
namespace = {'np': np}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
compare = namespace['compare']
results = {}
for mode, vv in vectors.items():
    b = vv['base']
    gg = {g: np.stack([vv[n] for n in nn]) for g, nn in groups.items()}
    results[mode] = dict(base_correct=int(b.sum()),
        families={g: dict(per_seed_correct=v.sum(1).tolist(), vs_base=compare(v-b))
                  for g, v in gg.items()},
        averaged_minus_unaveraged=compare(gg['averaged']-gg['unaveraged']))
write(out, dict(time=time.time(), script_sha256=sha(__file__), seeds=seeds, n=500,
    groups=groups, identities=identities, scoring_audit_sha256=sha(auditpath),
    statistics_source_sha256=sha(source),
    replication_plan_sha256=sha(S/'audits/teacher7_expansion_seed_replication_plan.json'),
    parser_warning_count=len(audit['parser_comparison_warnings']), results=results,
    scope='Descriptive development analysis of both already fixed endpoints, assembled after '
          'four seed results were known. Unadjusted intervals, no new success criterion or '
          'primary selection change. Expanded unaveraged replicas are not in the fixed final '
          'benchmark family manifest and do not receive an OOD/general-efficacy claim here.'))
print(json.dumps(results, indent=2))
