"""Materialize the timing plan independently of final quality outcomes."""
from common import *
import sys
from surrogate_freeze import FREEZE

def main():
    frozen=json.loads(FREEZE.read_text());plan_path=EXP/'configs/surrogate_final_recipe_plan.json';plan=json.loads(plan_path.read_text());assert frozen['recipes'][str(plan_path)]['sha256']==sha(plan_path)
    entries=[r['entry'] for r in frozen['entries'].values()];tp=plan['timing_plan'];selected=[]
    for warm in tp['starts']:
        for subset,recipes in [(tp['candidate_subset'],tp['candidate_recipes']),('update',tp['fullpool_recipes'])]:
            for recipe in recipes:
                found=[e for e in entries if e.get('warmup_seed')==warm and e.get('subset')==subset and all(e.get(k)==v for k,v in recipe.items())];assert len(found)==1,(warm,subset,recipe);selected+=found
    queue=[]
    def add(e,repeat,flops=False):
        name=f'surrogate_final_{"flops" if flops else "timing"}_{e["warmup_seed"]}_{e["subset"]}_{e["method"]}_{e["steps"]}_r{repeat}'
        args=[sys.executable,str(EXP/'scripts/replay_reuse.py'),'--reference-adapter',e['adapter'],'--repeat',str(repeat)]
        if flops:args+=['--flops']
        queue.append(dict(name=name,command=[sys.executable,str(EXP/'scripts/stage.py'),'--name',name,*args]))
    # Rotate the registered ordering across rounds; do not run three repeats consecutively.
    for repeat in range(1,tp['repetitions']+1):
        offset=(repeat-1)*7;ordered=selected[offset:]+selected[:offset]
        for e in ordered:add(e,repeat)
    for e in selected:
        if e['warmup_seed'] in tp['flops_starts']:add(e,1,True)
    queue.append(dict(name='surrogate_final_summarize_timing',command=[sys.executable,str(EXP/'scripts/summarize_reuse_timings.py')]))
    target=EXP/'configs/queue_surrogate_final_timing.json';assert not target.exists();write(target,queue)
    write(EXP/'results/surrogate_timing_queue_receipt.json',dict(freeze_sha256=sha(FREEZE),plan_sha256=sha(plan_path),queue_sha256=sha(target),reference_entries=selected,timing_stages=len(selected)*tp['repetitions'],flop_stages=sum(e['warmup_seed'] in tp['flops_starts'] for e in selected),scope=tp['qualification']))
    log(dict(references=len(selected),stages=len(queue)))

if __name__=='__main__':main()
