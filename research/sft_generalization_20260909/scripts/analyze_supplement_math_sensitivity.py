"""Apply the same previously fixed MATH exclusions to the supplementary family."""
import ast
import numpy as np
from common import *
p = S/'audits/expanded_supplement_math_sensitivity_plan.json'; plan = json.loads(p.read_text())
supp_path = S/'audits/expanded_unaveraged_supplement_plan.json'; supp = json.loads(supp_path.read_text())
assert plan['supplement_plan_sha256'] == sha(supp_path)
analysis_path = S/'results/expanded_unaveraged_supplement_analysis.json'
analysis = json.loads(analysis_path.read_text()); assert analysis['plan_sha256'] == sha(supp_path)
assert plan['primary_plan_sha256'] == sha(S/'results/confirmation_plan.json')
source = S/'scripts/analyze_confirmation.py'; assert sha(source) == supp['analysis_script_sha256']
node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'compare')
ns = {'np': np}; exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
compare = ns['compare']; dataset = plan['dataset']; root = S/'results/evaluation'/dataset
names = ['base']+list(supp['models'].values())
old = json.loads((S/f'audits/scoring_confirmation_{dataset}.json').read_text())
new = json.loads((S/f'audits/scoring_supplement_{dataset}.json').read_text())
strict_scores = dict(old['models'], **new['models']); raw = {}; strict = {}
for name in names:
    path = root/name/'math_predictions.jsonl'; rows = read(path); assert len(rows) == 5000
    assert sha(path) == analysis['identities'][dataset][name]['predictions_sha256']
    assert sha(path) == strict_scores[name]['predictions_sha256']
    raw[name] = np.array([r['correct'] for r in rows], int)
    strict[name] = np.array(strict_scores[name]['strict_vector'], int)
results = {}
for mode, vectors in [('raw', raw), ('strict', strict)]:
    b = vectors['base']; group = np.stack([vectors[n] for n in names[1:]])
    results[mode] = {}
    for label, rule in plan['schemes'].items():
        keep = rule.get('keep_indices', [i for i in range(5000) if i not in rule.get('exclude_indices', [])])
        results[mode][label] = dict(n=len(keep), base_correct=int(b[keep].sum()),
            per_seed_correct=group[:, keep].sum(1).tolist(), vs_base=compare((group-b)[:, keep]))
write(S/'results/expanded_supplement_math_sensitivity.json', dict(time=time.time(),
    plan_sha256=sha(p), analysis_sha256=sha(analysis_path), script_sha256=sha(__file__),
    results=results, scope=plan['scope']))
print(json.dumps(results, indent=2))
