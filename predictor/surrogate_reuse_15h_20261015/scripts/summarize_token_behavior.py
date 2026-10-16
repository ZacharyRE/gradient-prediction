"""Compare existing KD models with current and unchanged-initial teacher behavior."""
from common import *

def main():
    plan=json.loads((EXP/'configs/token_behavior_readout_plan.json').read_text());out=EXP/'results/behavior'/plan['name']
    assert json.loads((out/'complete.json').read_text())['complete']
    protocol=json.loads((out/'protocol.json').read_text());assert protocol['full_original_prompts'] and protocol['gold_class_never_enters_model_input']
    entries=json.loads(Path(plan['entries']).read_text());source=read(data('mnli','accuracy_dev'));ids=[r['sample_id'] for r in source];gold=np.array(['ABC'.index(r['gold']) for r in source])
    summary=json.loads((out/'summary.json').read_text());teachers={};predictions={};initial={}
    def load_logits(path):
        records=read(path);assert [r['sample_id'] for r in records]==ids
        logits=np.array([r['logits'] for r in records]);assert np.isfinite(logits).all() and logits.shape==(len(ids),3)
        labels=logits.argmax(-1);assert [r['correct'] for r in records]==(labels==gold).tolist()
        return logits
    for e in entries:
        teachers[e['tag']]=load_logits(out/f'{e["tag"]}__teacher.jsonl')
        if e['steps']==0:initial[e['warmup_seed']]=teachers[e['tag']]
        for model in plan['predictors']:
            name='mnli_critic_'+model;predictions[e['tag'],model]=load_logits(out/f'{e["tag"]}__{name}.jsonl')
    rows=[]
    for model in plan['predictors']:
        for step in [0,32,128,256,512]:
            ee=[e for e in entries if e['steps']==step];assert sorted(e['warmup_seed'] for e in ee)==[301,311,321]
            per=[]
            for e in ee:
                teacher=teachers[e['tag']];pred=predictions[e['tag'],model];base=initial[e['warmup_seed']]
                t=teacher.argmax(-1);p=pred.argmax(-1);b=base.argmax(-1)
                r=summary[e['tag']]['mnli_critic_'+model]
                per.append(dict(warmup_seed=e['warmup_seed'],accuracy_pct=float(np.mean(p==gold)*100),teacher_accuracy_pct=float(np.mean(t==gold)*100),initial_accuracy_pct=float(np.mean(b==gold)*100),teacher_agreement_pct=float(np.mean(p==t)*100),initial_teacher_agreement_pct=float(np.mean(b==t)*100),agreement_delta_pp=float(np.mean((p==t).astype(float)-(b==t))*100),readout_seconds=r['seconds'],teacher_seconds=r['teacher_forced_seconds']))
            row=dict(model=model,steps=step,per_start=per)
            row.update({k:float(np.mean([r[k] for r in per])) for k in per[0] if k!='warmup_seed'})
            rows.append(row)
    result=dict(complete=True,plan_sha256=sha(EXP/'configs/token_behavior_readout_plan.json'),predictor_freeze_sha256=sha(EXP/'provenance/token_behavior_predictor_freeze.json'),input_audit_sha256=sha(EXP/'results/token_behavior_input_audit.json'),rows=rows,
                scope='Post-freeze development diagnostic only:15 existing warmup/Oracle states and513 development inputs. ForcedABC class behavior, not generation. Initial-teacher predictions are retained as a state-insensitive baseline. Readout times are single-process components; prefix preparation and offline fitting remain additional; first batch includes one-time causal/label-invariance checks.')
    write(EXP/'results/token_behavior_development_summary.json',result);log(dict(rows=len(rows),teacher_states=len(entries),examples=len(ids)))

if __name__=='__main__':main()
