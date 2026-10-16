"""Compare behavior models on a common development objective and initial-agent control."""
from common import *
from collections import defaultdict

def main():
    fits=[]
    for path in sorted((EXP/'models').glob('mnli_behavior_*/summary.json')):
        summary=json.loads(path.read_text())
        if not summary.get('complete'):continue
        history=json.loads((path.parent/'history.json').read_text());cfg=json.loads((path.parent/'config.json').read_text())
        row=next(r for r in history if r['epoch']==summary['best_epoch'])
        score=row['normalized_logit_mse']+row['normalized_state_change_mse']
        fits.append(dict(model=path.parent.name,selected_epoch=summary['best_epoch'],within_fit_joint_score=score,fit_cache=Path(cfg.get('cache','critic_hidden')).name,fit_state_count=len(cfg.get('states',[])) or 8,normalized_logit_mse=row['normalized_logit_mse'],normalized_state_change_mse=row['normalized_state_change_mse'],teacher_class_agreement=row['teacher_class_agreement'],student_accuracy=row.get('student_accuracy',float(np.mean([s['student_accuracy'] for s in row.get('states',[])])) if row.get('states') else None),initial_teacher_baseline=summary.get('initial_teacher_baseline'),fit_seconds=summary['fit_seconds'],feature_preparation_seconds=summary['feature_preparation_seconds'],summary_sha256=sha(path),history_sha256=sha(path.parent/'history.json')))
    pilots=[]
    for protocol_path in sorted((EXP/'results/behavior').glob('*/protocol.json')):
        protocol=json.loads(protocol_path.read_text())
        if protocol['args']['split']!='accuracy_dev':continue
        path=protocol_path.parent/'summary.json'
        if not path.exists():continue
        entries=json.loads(Path(protocol['args']['entries']).read_text());entries={e['tag']:e for e in entries}
        groups=defaultdict(list)
        for tag,models in json.loads(path.read_text()).items():
            entry=entries[tag]
            for model,metrics in models.items():
                key=json.dumps(dict(model=model,method=entry['method'],steps=entry['steps']),sort_keys=True)
                groups[key].append(dict(warmup_seed=entry['warmup_seed'],tag=tag,metrics=metrics))
        for key,rows in sorted(groups.items()):
            fields=['accuracy','teacher_forced_accuracy','teacher_class_agreement','initial_teacher_accuracy','initial_teacher_class_agreement','logit_mse','initial_teacher_logit_mse','initial_teacher_vs_current_teacher_agreement','seconds','teacher_forced_seconds']
            means={field:float(np.mean([r['metrics'][field] for r in rows])) for field in fields if all(field in r['metrics'] for r in rows)}
            if 'initial_teacher_logit_mse' in means:
                means['logit_mse_reduction_vs_initial']=means['initial_teacher_logit_mse']-means['logit_mse']
                means['accuracy_delta_vs_initial']=means['accuracy']-means['initial_teacher_accuracy']
            pilots.append(dict(**json.loads(key),evaluation=protocol_path.parent.name,starts=len(rows),warmup_seeds=[r['warmup_seed'] for r in rows],means=means,per_start=rows,complete=(protocol_path.parent/'complete.json').exists()))
    write(EXP/'results/behavior_development_summary.json',dict(fits=fits,pilot_groups=pilots,scope='Predictor-dev fit selection and exploratory accuracy-dev readouts only. Within-fit scores sum normalized logit and within-example state-change MSE regardless of training loss, but must not be compared across different state populations/normalizers. Use behavior_common_selection.json for the common12-state cross-model choice. Forced ABC classification is not generated-answer accuracy. Positive logit-MSE reduction means better than unchanged initial-teacher logits; initial accuracy alone does not establish learning. Readout times exclude shared prefix cache and paid initial-teacher anchors, which remain in each protocol.'))
    log(dict(fits=len(fits),pilot_groups=len(pilots)))

if __name__=='__main__':main()
