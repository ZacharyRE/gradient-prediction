"""Independent fixed-stop replay; exact weight checks qualify timing/FLOP claims."""
import time as _time
SCRIPT_START=_time.perf_counter()
SCRIPT_EPOCH=_time.time()
from common import *
import argparse,importlib,sys,contextlib

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference-adapter',required=True);ap.add_argument('--module',choices=['reuse_train','train_tangent'],default='reuse_train');ap.add_argument('--repeat',type=int,default=1);ap.add_argument('--flops',action='store_true');ap.add_argument('--audit-backend',choices=['eager_static']);ap.add_argument('--measurement-scope',choices=['standalone','selection_pipeline'],default='standalone');a=ap.parse_args()
    reference=Path(a.reference_adapter);cfg=json.loads((reference.parent/'config.json').read_text());entry=next(e for e in json.loads((reference.parent/'entries.json').read_text()) if Path(e['adapter'])==reference)
    job=cfg['job'];assert job.get('schedule','constant')=='constant' or job['steps']==entry['steps'],'A shorter scheduled horizon changes the training recipe'
    replay_id=str(time.time_ns());name=f'replay_{job["name"]}_{replay_id}';job=dict(job,name=name,steps=entry['steps'],save=[entry['steps']],diagnostics=False)
    path=EXP/'configs'/f'{name}.json';write(path,[job]);os.environ['BENCH_FINAL_VALIDATION']='1'
    from time_recipe import assert_clean_gpu
    assert_clean_gpu();module=importlib.import_module(a.module);suffix='_standalone';task=entry['task'];seed_id=cfg['update_seed'];warm=cfg['warmup_seed'];subset=cfg['subset']
    argv=[a.module,'--warmup-seed',str(warm),'--subset',subset,'--seed',str(seed_id),'--jobs',str(path),'--suffix',suffix]
    if a.module=='reuse_train':
        argv+=['--task',task,'--cache-policy',cfg.get('cache_policy','all'),'--teacher-backend',cfg.get('teacher_backend','native'),'--graph-bucket',str(cfg.get('graph_bucket',64)),'--max-graphs',str(cfg.get('max_graphs',4))]
        if not cfg.get('graph_initial_audit',True):argv+=['--skip-graph-audit']
    else:
        argv+=['--skip-initial-audit']
        if cfg.get('anchor_adapter'):argv+=['--anchor-adapter',cfg['anchor_adapter']]
    if a.audit_backend:
        assert a.module=='reuse_train' and cfg.get('teacher_backend')=='graph' and not a.flops
        argv[argv.index('--teacher-backend')+1]=a.audit_backend
    sys.argv=argv;counter=None;coverage=None;phases=[]
    if a.flops:
        from flop_accounting import NativeFlopCounterMode,FUSED_MAPPING,OperatorCoverage,coverage_record
        counter=NativeFlopCounterMode(display=False,custom_mapping=FUSED_MAPPING);coverage=OperatorCoverage()
        if a.module=='reuse_train' and cfg.get('teacher_backend','native')!='native':
            from graph_teacher import GraphTeacher
            GraphTeacher.flop_counter=counter
        for attr in ['TeacherPrefixCache','TangentSuffix']:
            if not hasattr(module,attr):continue
            original=getattr(module,attr)
            def wrapper(*args,_original=original,_phase=attr,**kwargs):
                before=counter.get_total_flops();out=_original(*args,**kwargs);phases.append(dict(phase=_phase,matrix_flops=counter.get_total_flops()-before));return out
            setattr(module,attr,wrapper)
    tick=time.perf_counter()
    with counter if counter is not None else contextlib.nullcontext(),coverage if coverage is not None else contextlib.nullcontext():module.main()
    import torch
    torch.cuda.synchronize();pipeline_seconds=time.perf_counter()-tick;script_seconds=time.perf_counter()-SCRIPT_START;finished_epoch=time.time();assert_clean_gpu()
    directory=EXP/'models'/f'{task}__w{warm}_{subset}_{name}_u{seed_id}{suffix}';summary=json.loads((directory/'summary.json').read_text());history=json.loads((directory/'history.json').read_text());actual_cfg=json.loads((directory/'config.json').read_text());assert summary['complete'] and len(history)==entry['steps']
    actual_adapter=directory/f'step-{entry["steps"]}'
    from safetensors.torch import load_file
    actual=load_file(str(actual_adapter/'adapter_model.safetensors'));expected=load_file(str(reference/'adapter_model.safetensors'));assert actual.keys()==expected.keys()
    bitwise=all(torch.equal(actual[k],expected[k]) for k in actual);error=max(float((actual[k]-expected[k]).abs().max()) for k in actual);relative=(sum(float((actual[k].double()-expected[k].double()).square().sum()) for k in actual)/max(1e-30,sum(float(expected[k].double().square().sum()) for k in actual)))**.5
    result=dict(reference_entry=entry,reference_config_sha256=sha(reference.parent/'config.json'),reference_sha256=sha(reference/'adapter_model.safetensors'),actual_adapter=str(actual_adapter),actual_sha256=sha(actual_adapter/'adapter_model.safetensors'),weights_bitwise_equal=bitwise,max_absolute_weight_difference=error,relative_weight_difference=relative,repeat=a.repeat,module=a.module,mode='flops' if a.flops else 'timing',timing_valid_for_reference=bitwise and not a.flops,online_seconds=summary['online_seconds'],preparation_seconds=summary['cache_seconds'],model_setup_seconds=actual_cfg.get('model_setup_seconds'),data_encoding_seconds=actual_cfg.get('data_encoding_seconds'),pipeline_seconds=pipeline_seconds,script_seconds=script_seconds,started_epoch=SCRIPT_EPOCH,finished_epoch=finished_epoch,source_sha256=source_signature(__file__),cache_policy=actual_cfg.get('cache_policy','all'),scope='Single fresh process: imports, model/data/predictor loading, preparation, updates and final save. Shared warmup, inherited predictor preparation, evaluation and post-run weight comparison excluded. Profiled times are not speed measurements.')
    result.update(reference_teacher_backend=cfg.get('teacher_backend','native'),actual_teacher_backend=actual_cfg.get('teacher_backend','native'))
    if a.audit_backend:
        result.update(mode='backend_equivalence_audit',timing_valid_for_reference=False,scope='Exact full-horizon graph versus same-padding eager-static weight audit. Timing is not a graph speed measurement.')
    if counter is not None:
        raw_total=counter.get_total_flops();receipt=summary.get('teacher_backend_receipt');adjusted=raw_total
        if receipt and 'executed_teacher_matrix_flops' in receipt:
            adjusted=raw_total-receipt['instrumented_teacher_matrix_flops']+receipt['executed_teacher_matrix_flops'];assert adjusted>=0
            result['teacher_backend_flop_accounting']=receipt
        result.update(total_matrix_flops=adjusted,raw_instrumented_matrix_flops=raw_total,preparation_phase_flops=phases,coverage=coverage_record(coverage,counter))
    result['measurement_scope']=a.measurement_scope
    output=EXP/'results'/('pipeline_training' if a.measurement_scope=='selection_pipeline' else 'backend_equivalence' if a.audit_backend else 'reuse_flops' if a.flops else 'reuse_timing')/f'{entry["tag"]}_r{a.repeat}_{replay_id}.json';write(output,result);log({k:v for k,v in result.items() if k not in ['coverage','reference_entry']})
    if not bitwise:raise RuntimeError('Replay changed final weights; record retained, timing is not qualified for this reference')

if __name__=='__main__':main()
