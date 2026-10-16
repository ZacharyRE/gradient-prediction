"""Finish the predeclared final protocol as soon as blind training is complete."""
from common import *
import sys

def run(script,args=()):
    subprocess.run([sys.executable,str(EXP/'scripts'/script),*map(str,args)],check=True,cwd=ROOT)

def main():
    plan_path=EXP/'configs/surrogate_final_recipe_plan.json';plan=json.loads(plan_path.read_text());required=[Path(p) for g in plan['entry_groups'] for p in g['sources']]
    while not all(p.exists() for p in required):
        if deadline(7200):raise RuntimeError('Insufficient final reserve; no partial scientific freeze')
        time.sleep(30)
    run('assemble_surrogate_final_entries.py')
    configs=['surrogate_final_recipe_plan.json','surrogate_analysis_spec.json','surrogate_cost_policy.json','development_selection_rule.json','secondary_analysis_plan.json','behavior_common_selection_plan.json','blind_matched_jobs.json','final_candidate_oracle_jobs.json','final_candidate_zero_jobs.json','final_fullpool_control_jobs.json','final_fullpool_tuned_cv_jobs.json','final_fullpool_cv8_graph_jobs.json','final_fullpool_state_jobs.json','final_fullpool_graph_oracle_jobs.json']
    recipe_paths=[EXP/'configs'/p for p in configs]+[EXP/'results/behavior_common_selection.json']
    run('surrogate_freeze.py',['--entries',EXP/'configs/surrogate_final_entries.json','--recipe-configs',*recipe_paths,'--purpose','Frozen transfer, downstream quality, candidate-selection and cost evaluation after bounded development; no selection outcomes used for method choice.','--behavior-models',*plan['behavior_models']])
    run('build_surrogate_readout_queue.py');run('build_surrogate_timing_queue.py')
    children=[]
    for name in ['readouts','timing']:
        logpath=EXP/'logs'/f'queue_surrogate_final_{name}.log'
        with logpath.open('a') as out:
            child=subprocess.Popen([sys.executable,str(EXP/'scripts/run_queue.py'),'--queue',str(EXP/'configs'/f'queue_surrogate_final_{name}.json')],env=dict(os.environ,BENCH_FINAL_VALIDATION='1',BENCH_RUN_ID=EXP.name),cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(dict(name=name,pid=child.pid,process_start=process_start(child.pid),log=str(logpath)))
    write(EXP/'results/surrogate_final_launch.json',dict(epoch=time.time(),children=children));log(children)

if __name__=='__main__':main()
