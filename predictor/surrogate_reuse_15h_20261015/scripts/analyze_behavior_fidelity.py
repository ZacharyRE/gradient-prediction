"""Assess prediction of changed agent decisions against unchanged initial behavior."""
from common import *
from surrogate_freeze import FREEZE
from analyze_selection import paired_interval
from collections import defaultdict
import csv

def main():
    frozen=json.loads(FREEZE.read_text());split='selection_holdout';root=EXP/'results/behavior/surrogate_final_behavior_selection_holdout';assert json.loads((root/'complete.json').read_text())['complete'];protocol=json.loads((root/'protocol.json').read_text());assert protocol['freeze']['freeze_sha256']==sha(FREEZE)
    rows=read(data('mnli',split));gold=np.array(['ABC'.index(r['gold']) for r in rows]);groups=[r['group_id'] for r in rows];starts=[302,303,304]
    def load_logits(path):
        rr=read(path);assert len(rr)==len(rows) and all(a['sample_id']==b['sample_id'] and a['gold']==b['gold'] for a,b in zip(rr,rows));v=np.array([r['logits'] for r in rr]);assert v.shape==(len(rows),3) and np.isfinite(v).all();return v
    anchors={w:load_logits(root/f'initial_teacher_w{w}.jsonl') for w in starts};collected=defaultdict(lambda:defaultdict(list));warmup_checks=[]
    for item in frozen['entries'].values():
        e=item['entry'];warm=e['warmup_seed'];teacher=load_logits(root/f'{e["tag"]}__teacher.jsonl');initial=anchors[warm];teacher_label=teacher.argmax(1);initial_label=initial.argmax(1);teacher_centered=teacher-teacher.mean(1,keepdims=True);initial_centered=initial-initial.mean(1,keepdims=True)
        for path in frozen['behavior_predictors']:
            model=Path(path).parent.name;pred=load_logits(root/f'{e["tag"]}__{model}.jsonl');label=pred.argmax(1);pred_centered=pred-pred.mean(1,keepdims=True)
            if not e['steps']:
                warmup_checks.append(dict(warmup_seed=warm,model=model,logits_exact_initial=np.array_equal(pred,initial)));assert warmup_checks[-1]['logits_exact_initial'];continue
            subset=e['subset'];family='fullpool' if subset=='update' else 'random' if subset.startswith('candidate_') else 'genre';key=json.dumps(dict(family=family,method=e['method'],kind=e['kind'],steps=e['steps'],lr=e['lr'],behavior_model=model),sort_keys=True)
            metrics=dict(surrogate_accuracy=(label==gold).astype(float),teacher_accuracy=(teacher_label==gold).astype(float),initial_accuracy=(initial_label==gold).astype(float),surrogate_teacher_agreement=(label==teacher_label).astype(float),initial_teacher_agreement=(initial_label==teacher_label).astype(float),surrogate_logit_mse=((pred_centered-teacher_centered)**2).mean(1),initial_logit_mse=((initial_centered-teacher_centered)**2).mean(1),teacher_changes_class=(teacher_label!=initial_label).astype(float),surrogate_changes_class=(label!=initial_label).astype(float))
            metrics['accuracy_delta_vs_initial']=metrics['surrogate_accuracy']-metrics['initial_accuracy'];metrics['agreement_delta_vs_initial']=metrics['surrogate_teacher_agreement']-metrics['initial_teacher_agreement'];metrics['logit_mse_reduction_vs_initial']=metrics['initial_logit_mse']-metrics['surrogate_logit_mse']
            collected[key][warm].append(metrics)
    results=[]
    for key,perstart in sorted(collected.items()):
        assert sorted(perstart)==starts;expected=1 if json.loads(key)['family']=='fullpool' else 5;assert all(len(perstart[w])==expected for w in starts)
        names=list(perstart[starts[0]][0]);metrics={name:paired_interval(np.stack([np.mean([r[name] for r in perstart[w]],axis=0) for w in starts]),groups) for name in names}
        results.append(dict(**json.loads(key),metrics=metrics))
    result=dict(complete=True,freeze_sha256=sha(FREEZE),protocol_sha256=sha(root/'protocol.json'),rows=results,warmup_identity_checks=warmup_checks,scope='Forced-format ABC behavior, not free-generation accuracy. Five fixed candidate adapters averaged within each start, or one fullpool adapter. Paired start/source-group bootstrap; three starts. The unchanged initial teacher is a mandatory state-independent behavior baseline. Positive agreement or logit-MSE reduction means better prediction of the changed teacher than unchanged initial logits.',source_sha256=source_signature(__file__))
    write(EXP/'results/final_behavior_fidelity.json',result)
    flat=[]
    for r in results:
        row={k:v for k,v in r.items() if k!='metrics'}
        for k,v in r['metrics'].items():row.update({k:v['mean'],k+'_ci_low':v['ci95'][0],k+'_ci_high':v['ci95'][1]})
        flat.append(row)
    with (EXP/'results/final_behavior_fidelity.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
    log(dict(groups=len(results),initial_identity_checks=len(warmup_checks)))

if __name__=='__main__':main()
