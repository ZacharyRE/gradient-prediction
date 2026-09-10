"""CPU mathematical checks for frozen-analysis pairing/seed/bootstrap behavior."""
import ast
import numpy as np
from common import *
path=S/'scripts/analyze_confirmation.py';module=ast.parse(path.read_text())
functions=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ['compare','macro_compare']]
assert len(functions)==2
namespace={'np':np,'seeds':[43,44,45,46,47]}
exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(path),'exec'),namespace)
compare=namespace['compare'];macro=namespace['macro_compare'];checks=[]
for label,fn,args,expected in [
 ('identical_paired_models',compare,(np.zeros((5,8),int),),0),
 ('all_seed_all_question_rescue',compare,(np.ones((5,8),int),),100),
 ('all_seed_all_question_loss',compare,(-np.ones((5,8),int),),-100),
 ('equal_macro_weight_despite_unequal_n',macro,([np.ones((5,3),int),-np.ones((5,8),int),np.zeros((5,17),int)],),0)]:
 r=fn(*args);assert abs(r['delta_pp']-expected)<1e-12
 for key in ['question_ci95','seed_question_ci95','per_seed_delta_pp']:assert np.allclose(r[key],expected), (label,key,r[key])
 assert r['every_seed_positive']==(expected>0);checks.append(dict(name=label,passed=True,result=r))
d=np.array([[-1]*8,[-1]*8,[0]*8,[1]*8,[1]*8])
r=compare(d);assert r['question_ci95']==[0.,0.] and r['seed_question_ci95'][0]<0<r['seed_question_ci95'][1]
checks.append(dict(name='shared_questions_do_not_hide_training_seed_variability',passed=True,result=r))
# Opposite effects at every seed across two tasks must cancel when each bootstrap uses the same seed draw.
r=macro([d[:,:3],-d,np.zeros((5,17),int)])
for key in ['question_ci95','seed_question_ci95','per_seed_delta_pp']:assert np.allclose(r[key],0),(key,r[key])
checks.append(dict(name='same_seed_draw_shared_across_macro_tasks',passed=True,result=r))
v=np.array([[0,1,0,-1,1,0,-1,1]]*5);assert compare(v)==compare(v)
checks.append(dict(name='fixed_bootstrap_reproducibility',passed=True))
write(S/'audits/final_statistics_contract.json',dict(time=time.time(),script_sha256=sha(__file__),analysis_script_sha256=sha(path),checks=checks,passed=True,scope='Synthetic invariants for actual extracted statistics functions, not validation of unseen final data or a proof of bootstrap coverage with five seeds. No final model outputs consumed; no analysis code changed.'))
print(json.dumps(dict(passed=True,n_checks=len(checks))))
