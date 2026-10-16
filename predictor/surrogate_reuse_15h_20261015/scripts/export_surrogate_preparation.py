"""Inventory historical preparation receipts; keep reused dependencies explicit.

These component sums are not clean, cold-start benchmarks. Teacher initialization,
imports and some serialization costs have different historical timer boundaries.
"""
from common import *

def main():
    protocol=json.loads((EXP/'configs/protocol.json').read_text());source=Path(protocol['source_study']);components={};models=[]
    def add(name,seconds,path,scope):
        components[name]=dict(seconds=float(seconds),receipt=str(path),receipt_sha256=sha(path),scope=scope)
    inherited=source/'models/mnli_predictor_globalfit_p123';s=json.loads((inherited/'summary.json').read_text());cpath=source/'cache/mnli/complete.json';cache=json.loads(cpath.read_text())
    add('original_predictor_labels',sum(r['seconds'] for r in cache['audits'] if r['split'] in ['predictor_train','predictor_dev']),cpath,'Training and predictor-dev gradient labels; separate gradient_test excluded')
    add('original_predictor_fit',s['seconds'],inherited/'summary.json','Historical fit including checkpoint selection')
    models.append(dict(predictor=inherited.name,predictor_sha256=s['best_sha256'],components=['original_predictor_labels','original_predictor_fit'],source_warmup_seeds=[201]))
    for cache_name in ['critic_v1','critic_hidden']:
        path=EXP/'cache'/cache_name/'complete.json'
        if not path.exists():continue
        receipt=json.loads(path.read_text());add(cache_name+'_labels',sum(a['seconds'] for st in receipt['states'] for a in st['audits']),path,'Training and predictor-dev label collection across all8 teacher states; same512 training examples repeated across states')
        for st in receipt['states']:
            state=st['state'];key=f'teacher_trajectory_w{state["warmup_seed"]}'
            if not state['step']:continue
            adapter=Path(state['adapter']);entries_path=adapter.parent/'entries.json';entries=json.loads(entries_path.read_text());entry=next(e for e in entries if Path(e['adapter'])==adapter)
            seconds=entry['online_seconds']+entry.get('cache_seconds',0)
            if key not in components or seconds>components[key]['seconds']:add(key,seconds,entries_path,'Longest required teacher trajectory once, including recorded cache; do not sum intermediate checkpoints')
    factor_path=EXP/'cache/factor_v1/complete.json'
    if factor_path.exists():add('factor_cache',json.loads(factor_path.read_text())['total_seconds'],factor_path,'Training-only bases and covariance-cache preparation')
    for fast_cache_name in ['critic_fast_v1','critic_fast_hidden']:
        fast_path=EXP/'cache'/fast_cache_name/'complete.json'
        if not fast_path.exists():continue
        receipt=json.loads(fast_path.read_text());add(fast_cache_name+'_labels',sum(a['seconds'] for st in receipt['states'] for a in st['audits']),fast_path,'Four additional high-LR Oracle states; same fitting/development example IDs')
        for item in receipt['states']:
            adapter=Path(item['state']['adapter']);entries_path=adapter.parent/'entries.json';entry=next(e for e in json.loads(entries_path.read_text()) if Path(e['adapter'])==adapter)
            key='teacher_fast_trajectory_w301';seconds=entry['online_seconds']+entry.get('cache_seconds',0)
            if key not in components or seconds>components[key]['seconds']:add(key,seconds,entries_path,'Additional LR0.001 trajectory through512. Separate from original LR0.0003 trajectory; charge both for expanded coverage.')
    diversity_path=EXP/'cache/critic_diversity2048/complete.json'
    if diversity_path.exists():
        receipt=json.loads(diversity_path.read_text())
        add('critic_diversity2048_labels',sum(a['seconds'] for st in receipt['states'] for a in st['audits']),diversity_path,'Post-freeze development collection:2048 distinct original predictor_train rows across12 previously declared teacher states; predictor_dev unchanged')
    for path in sorted((EXP/'models').glob('mnli_*/summary.json')):
        if '__w' in path.parent.name or not (path.parent/'best.pt').exists():continue
        s=json.loads(path.read_text());cfg=json.loads((path.parent/'config.json').read_text());name=path.parent.name;keys=[];warmups=[201,301]
        if name.startswith('mnli_critic_'):
            cname=Path(cfg['cache']).name;keys=[cname+'_labels','teacher_trajectory_w201','teacher_trajectory_w301'];fit=s['seconds']+s.get('mean_normalization_preparation_seconds',0.)
            if cname=='critic_coverage_v1':keys=['critic_v1_labels','critic_fast_v1_labels','teacher_trajectory_w201','teacher_trajectory_w301','teacher_fast_trajectory_w301']
            if cname=='critic_diversity2048':keys+=['teacher_fast_trajectory_w301']
        elif name.startswith('mnli_behavior_'):
            keys=['critic_hidden_labels','teacher_trajectory_w201','teacher_trajectory_w301'];fit=s['fit_seconds']+s['feature_preparation_seconds']
            if Path(cfg.get('cache','critic_hidden')).name=='critic_coverage_hidden':keys+=['critic_fast_hidden_labels','teacher_fast_trajectory_w301']
        elif name.startswith('mnli_factor_scale_'):
            # The selected epoch0 template is a dependency. Its historical fit
            # cost is recorded separately, not silently treated as zero.
            keys=['critic_v1_labels','teacher_trajectory_w201','teacher_trajectory_w301','factor_cache','fit_mnli_factor_u128v256'];fit=s['fit_seconds']+s['shared_preparation_seconds']
        elif name.startswith('mnli_factor_'):
            keys=['critic_v1_labels','teacher_trajectory_w201','teacher_trajectory_w301','factor_cache'];fit=s['seconds']
        elif name.startswith('mnli_position_'):
            scope=cfg['scope'];key='position_'+scope+'_train_labels'
            add(key,s['training_label_collection_seconds'],path,'Only training-label subset used by predeclared template construction; validation diagnostics are research overhead')
            keys=[key];fit=s['seconds'];warmups=[301] if scope=='w301_step0' else [201,301]
            if scope!='w301_step0':keys+=['teacher_trajectory_w201','teacher_trajectory_w301']
        else:continue
        key='fit_'+name;add(key,fit,path,'Historical fit/feature-preparation timer; reused label and trajectory costs are separate');keys.append(key)
        models.append(dict(predictor=name,predictor_sha256=s['best_sha256'],components=keys,source_warmup_seeds=warmups))
    warmups={}
    for w in [201,301]:
        path=source/'models/mnli__warmup_lr0.001_s201/summary.json' if w==201 else EXP/'models/mnli_warmup_w301/summary.json'
        d=json.loads(path.read_text());warmups[w]=dict(seconds=d.get('online_seconds',d.get('seconds')),receipt=str(path),receipt_sha256=sha(path),scope='Historical source warmup, distinct from a new target warmup; original timer boundaries apply')
    for model in models:
        assert all(k in components for k in model['components']),model
        model['recorded_preparation_seconds_excluding_source_warmups']=sum(components[k]['seconds'] for k in set(model['components']))
        model['recorded_source_warmup_seconds']=sum(warmups[w]['seconds'] for w in model['source_warmup_seeds'])
        model['recorded_component_sum_seconds']=model['recorded_preparation_seconds_excluding_source_warmups']+model['recorded_source_warmup_seconds']
    target_warmups={}
    for w in [302,303,304]:
        path=EXP/'models'/f'mnli_warmup_w{w}'/'summary.json'
        if not path.exists():continue
        d=json.loads(path.read_text());assert d['warmup_steps']==32 and d['batch_size']==32 and d['lr']==.001 and d['full_inputs']
        target_warmups[w]=dict(warmup_seed=w,warmup_steps=d['warmup_steps'],batch_size=d['batch_size'],lr=d['lr'],online_seconds=d['online_seconds'],recorded_wall_seconds=d['wall_seconds'],adapter_sha256=d['entry']['adapter_sha256'],data_sha256=d['data_sha256'],receipt=str(path),receipt_sha256=sha(path),scope='Shared target warmup excluded from incremental comparisons. One historical run per target start; wall timer includes setup/data encoding/save but excludes imports before main. Do not treat these as three timing repetitions of one start or add duplicated setup without qualification.')
    write(EXP/'results/surrogate_preparation_components.json',dict(components=components,source_warmups=warmups,target_warmups=target_warmups,predictors=models,scope='Historical receipt inventory, not a clean cold-start benchmark or minimum achievable construction cost. Models share dependencies: do not sum rows. Imports, model loading and unrecorded I/O remain additional. Current study research overhead includes unsuccessful fits, redundant cache collection, validation diagnostics and candidate search. Offline FLOPs are not inferred from elapsed time.',source_sha256=source_signature(__file__)))
    log(dict(predictors=len(models),components=len(components)))

if __name__=='__main__':main()
