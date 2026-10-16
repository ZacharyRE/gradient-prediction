"""Matched candidate-data rollouts from a declared starting adapter.

Oracle and CV share the exact frozen-prefix backend. Cache construction is
charged once per rollout; cross-candidate sharing is reported separately.
"""
from research_teacher_cache import *
import predictor_core as pc
from fast_backend import configure

def cached_rows(cache,ds):
    rows=[]
    for i in range(len(ds)):
        if i not in cache.rows:rows.append(None);continue
        labels=torch.tensor(ds[i]['labels'][1:]+[-100],device='cuda')
        rows.append(dict(index=i,x=cache.rows[i]['x'].detach().bfloat16(),residual=cache.rows[i]['residual'],active=labels>=0,n=ds[i]['n'],label_ids=labels))
    return rows

def training_order(job,ds,seed_id):
    rng=np.random.default_rng(seed_id)
    if job.get('dataset_sampling')=='iid':return [int(i) for _ in range(job['steps']) for i in rng.choice(len(ds),32,replace=False)]
    order=[]
    while len(order)<job['steps']*32:order.extend(map(int,rng.permutation(len(ds))))
    return order

def needed_prefix_indices(jobs,ds,seed_id):
    """Plan from fixed data/sampling seeds; no model output or gradient is read."""
    needed=set()
    for job in jobs:
        if job.get('table_bootstrap',job['kind']=='full_table'):needed.update(range(len(ds)));continue
        crng=np.random.default_rng(seed_id+10000);order=training_order(job,ds,seed_id)
        stratifier=None
        if job['kind']=='stratified':
            from class_covariance import StratifiedGradients
            stratifier=StratifiedGradients(ds,seed_id)
        for st in range(1,job['steps']+1):
            ids=order[(st-1)*32:st*32];k=job.get('burnin_k',32) if st<=job.get('burnin_steps',0) else job.get('k',32)
            positions=list(map(int,crng.choice(32,k,replace=False)))
            if k==32:positions=list(range(32))
            if job['kind'] in ['oracle','cv','prediction','neural_hybrid']:needed.update(ids)
            elif job['kind'] in ['held_gradient','zero_gradient']:
                if (st-1)%job.get('true_every',4)==0:needed.update(ids)
            elif job.get('refresh_every') and st%job['refresh_every']==0:needed.update(ids)
            elif stratifier is not None:
                positions,_=stratifier.sample(ids,k);needed.update(ids[j] for j in positions)
            elif job['kind'] in ['subset','table','full_table','transport','class_ema','class_cov']:needed.update(ids[j] for j in positions)
            else:raise ValueError(f'No prefix planning rule for {job["kind"]}')
    return sorted(needed)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',default='mnli');ap.add_argument('--warmup-seed',type=int,required=True);ap.add_argument('--subset',default='update');ap.add_argument('--seed',type=int,default=401);ap.add_argument('--jobs',required=True);ap.add_argument('--suffix',default='');ap.add_argument('--cache-policy',choices=['all','needed'],default='all');ap.add_argument('--teacher-backend',choices=['native','graph','eager_static'],default='native');ap.add_argument('--graph-bucket',type=int,default=64);ap.add_argument('--max-graphs',type=int,default=4);ap.add_argument('--skip-graph-audit',action='store_true');a=ap.parse_args()
    jobs=json.loads(Path(a.jobs).read_text());warm=json.loads((EXP/'configs'/f'{a.task}_warmup_w{a.warmup_seed}.json').read_text())
    if a.teacher_backend=='graph' and len(jobs)>1:
        # Graph pools and retained autograd hooks can outlive a recipe. Use an
        # independent CUDA context per recipe; charge its own setup and cache.
        import sys
        aggregate=[];start=time.time();receipts=[]
        for job in jobs:
            digest=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:20]
            single=EXP/'configs/isolated_graph_jobs'/f'{digest}.json';write(single,[job])
            argv=list(sys.argv[1:]);argv[argv.index('--jobs')+1]=str(single)
            tick=time.time();subprocess.run([sys.executable,str(Path(__file__).resolve()),*argv],check=True)
            out=EXP/'models'/f'{a.task}__w{a.warmup_seed}_{a.subset}_{job["name"]}_u{a.seed}{a.suffix}'
            summary=json.loads((out/'summary.json').read_text());assert summary['complete']
            aggregate.extend(summary['entries']);receipts.append(dict(job=job['name'],process_wall_seconds=time.time()-tick))
        write(EXP/'results'/f'entries_{a.task}_w{a.warmup_seed}_{a.subset}_u{a.seed}{a.suffix}.json',aggregate)
        write(EXP/'results'/f'isolated_graph_processes_w{a.warmup_seed}_{a.subset}_u{a.seed}{a.suffix}.json',dict(recipes=receipts,wall_seconds=time.time()-start,reason='Fresh CUDA context per recipe after multi-recipe graph lifetime failure'))
        return
    source_sha=source_signature(__file__);snapshot=EXP/'provenance/source_versions'/source_sha;snapshot.mkdir(parents=True,exist_ok=True)
    for file in CODE_ROOT.glob('*.py'):
        if not (snapshot/file.name).exists():shutil.copy2(file,snapshot/file.name)
    start_all=time.time();m,tok,mod=setup('1p5b',warm['adapter'],rng=a.seed);A,B,s=weights(mod);initial=[v.detach().clone() for v in [A,B]];basever=versions(m);cap=Capture(mod,m) if any(j.get('cal_every') or j.get('cal_on_refresh') for j in jobs) else Capture(mod);model_setup_seconds=time.time()-start_all
    tick=time.perf_counter();ds=Dataset(a.task,a.subset,tok);dev=Dataset(a.task,'accuracy_dev',tok) if any(j.get('diagnostics',True) for j in jobs) else None;data_encoding_seconds=time.perf_counter()-tick
    configure(True,8192);tick=clock();cache_ids=needed_prefix_indices(jobs,ds,a.seed) if a.cache_policy=='needed' else None;planning_seconds=clock()-tick
    teacher=TeacherPrefixCache(m,tok,mod,cap,ds,ids=cache_ids,micro=32)
    backend=None
    if a.teacher_backend!='native':
        import sys
        from graph_teacher import GraphTeacher
        backend=GraphTeacher(m,tok,mod,cap,ds,teacher,a.teacher_backend,a.graph_bucket,a.max_graphs,not a.skip_graph_audit);backend.install(sys.modules[__name__])
    tick=clock();cached=cached_rows(teacher,ds) if any(j['kind'] in ['cv','prediction','neural_hybrid','table','full_table'] for j in jobs) else [];metadata_seconds=clock()-tick
    reserve_worker_cache();entries=[]
    for job in jobs:
        name=f'{a.task}__w{a.warmup_seed}_{a.subset}_{job["name"]}_u{a.seed}{a.suffix}';out=EXP/'models'/name
        if (out/'summary.json').exists():
            summary=json.loads((out/'summary.json').read_text());prior=json.loads((out/'config.json').read_text());assert summary['complete'] and prior['job']==job and prior.get('cache_policy','all')==a.cache_policy and prior.get('teacher_backend','native')==a.teacher_backend and (backend is None or prior['graph_bucket']==a.graph_bucket and prior['max_graphs']==a.max_graphs),'Cannot resume a changed recipe/backend/cache policy';entries.extend(summary['entries']);continue
        if deadline(5400):raise RuntimeError('Reserve time for final validation')
        out.mkdir(exist_ok=True);seed(a.seed)
        with torch.no_grad():A.copy_(initial[0]);B.copy_(initial[1])
        if backend is not None:backend.begin_job(job)
        tick=clock();path=Path(job.get('predictor_path',predictor_path(a.task,123)))
        p=load_predictor(path)[0] if job['kind'] in ['cv','prediction','neural_hybrid'] else None;predictor_load_seconds=clock()-tick
        for r in cached:
            if r is not None:r['want_h']=p is not None and getattr(p,'input_field','y')=='h'
        online=None
        online_init_seconds=0.
        if job['kind'] in ['transport','class_ema']:
            from online_transport import ClassSurrogate
            tick=clock()
            online=ClassSurrogate(mod,ds.rows,job['kind'],job.get('forget',.99),job.get('ridge',10.))
            online_init_seconds=clock()-tick
        if job['kind']=='table':
            from gradient_table import GradientTable
            tick=clock();online=GradientTable(m,tok,mod,cap,ds,cached,job);online_init_seconds=clock()-tick
        if job['kind']=='full_table':
            from full_gradient_table import FullGradientTable
            tick=clock();online=FullGradientTable(m,tok,mod,cap,ds,cached,job);online_init_seconds=clock()-tick
        if job['kind'] in ['class_cov','stratified']:
            from class_covariance import FrozenClassCovariance,StratifiedGradients
            tick=clock();online=FrozenClassCovariance(ds,path) if job['kind']=='class_cov' else StratifiedGradients(ds,a.seed);online_init_seconds=clock()-tick
        opt=torch.optim.AdamW([A,B],lr=job['lr'],weight_decay=0)
        if job.get('cal_scope')=='head':
            for param_name,param in p.named_parameters():param.requires_grad_(param_name.startswith('head.'))
        calopt=torch.optim.AdamW([v for v in p.parameters() if v.requires_grad],lr=job.get('cal_lr',1e-5),weight_decay=.01) if job.get('cal_every') or job.get('cal_on_refresh') else None
        order=training_order(job,ds,a.seed)
        crng=np.random.default_rng(a.seed+10000);calrng=np.random.default_rng(a.seed+20000);replay=[];coefs=[1.,1.];history=[];ee=[];stored_gradient=None
        prep=teacher.seconds+planning_seconds+(metadata_seconds+predictor_load_seconds if p is not None else metadata_seconds if job['kind'] in ['table','full_table'] else 0.)+online_init_seconds
        config=dict(task=a.task,warmup_seed=a.warmup_seed,subset=a.subset,update_seed=a.seed,job=job,full_inputs=True,warmup_sha256=warm['adapter_sha256'],data_sha256=sha(data(a.task,a.subset)),predictor_sha256=sha(path) if p is not None else None,source_sha256=source_sha,teacher_cache_seconds=teacher.seconds,metadata_seconds=metadata_seconds,predictor_load_seconds=predictor_load_seconds,online_initialization_seconds=online_init_seconds,initialization_true_gradient_examples=getattr(online,'bootstrap_examples',0),prep_seconds=prep)
        config['online_surrogate_sha256']=getattr(online,'source_sha256',None)
        config.update(cache_policy=a.cache_policy,cached_examples=len(teacher.rows),prefix_planning_seconds=planning_seconds,cache_indices_sha256=hashlib.sha256(json.dumps(sorted(teacher.rows)).encode()).hexdigest(),model_setup_seconds=model_setup_seconds,data_encoding_seconds=data_encoding_seconds)
        config.update(teacher_backend=a.teacher_backend,graph_bucket=a.graph_bucket,max_graphs=a.max_graphs,graph_initial_audit=not a.skip_graph_audit)
        write(out/'config.json',config);started=time.time()
        for st in range(1,job['steps']+1):
            if deadline(900):raise RuntimeError('Compute deadline')
            schedule=job.get('schedule','constant');fraction=(st-1)/max(1,job['steps']-1);floor=job.get('lr_floor',.1)
            if schedule=='cosine':multiplier=floor+(1-floor)*.5*(1+np.cos(np.pi*fraction))
            elif schedule=='linear':multiplier=floor+(1-floor)*(1-fraction)
            elif schedule=='constant':multiplier=1.
            else:raise ValueError(schedule)
            for group in opt.param_groups:group['lr']=job['lr']*multiplier
            ids=order[(st-1)*32:st*32];burnin=st<=job.get('burnin_steps',0);effective_k=job.get('burnin_k',32) if burnin else job.get('k',32);positions=list(map(int,crng.choice(32,effective_k,replace=False)))
            if effective_k==32:positions=list(range(32))
            step_job=dict(job,coefficient=0.) if burnin and job['kind'] in ['cv','class_cov'] else job
            if job.get('predictor_stop_step') is not None:
                assert job['kind']=='cv' and not any(job.get(key) for key in ['cal_every','cal_on_refresh','refresh_every','lambda_mode'])
                assert isinstance(job['predictor_stop_step'],int) and job['predictor_stop_step']>=0
                if st>job['predictor_stop_step']:step_job=dict(job,kind='subset')
            if job['kind'] in ['neural_hybrid','held_gradient','zero_gradient']:
                tick=clock();n=sum(ds[i]['n'] for i in ids);true_step=(st-1)%job.get('true_every',4)==0;pred_seconds=0.;label_seconds=0.
                if true_step:
                    t=clock();_,truth,_=collect(m,tok,mod,cap,ds,ids,micro=32,keep=False,verify_reconstruction=False);label_seconds=clock()-t
                    gradients=tuple(g/n for g in truth);stored_gradient=tuple(g.detach().clone() for g in gradients)
                elif p is not None:
                    t=clock();rr=predictor_features(p,mod,cached,ids);pred=per_example_predictions(p,rr,A,B,s);gradients=tuple(job.get('pred_scale',1.)*g.sum(0)/n for g in pred);pred_seconds=clock()-t
                elif job['kind']=='held_gradient':gradients=tuple(g.clone() for g in stored_gradient)
                else:gradients=tuple(torch.zeros_like(w) for w in [A,B])
                opt.zero_grad(set_to_none=True)
                for w,g in zip([A,B],gradients):w.grad=g.detach()
                norm=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
                row=dict(step=st,online_seconds=clock()-tick,true_gradient_examples=32 if true_step else 0,calibration_seconds=0.,gradient_norm=norm,prefix_prediction_seconds=pred_seconds,label_seconds=label_seconds,true_step=true_step)
            elif online is not None:row=online.step(m,tok,mod,cap,ds,ids,positions,opt,step_job,st)
            elif job['kind']=='prediction':
                tick=clock();rr=predictor_features(p,mod,cached,ids);pred=per_example_predictions(p,rr,A,B,s);n=sum(ds[i]['n'] for i in ids)
                opt.zero_grad(set_to_none=True)
                for w,g in zip([A,B],pred):w.grad=(g.sum(0)/n).detach()
                norm=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
                row=dict(step=st,online_seconds=clock()-tick,calibration_seconds=0.,true_gradient_examples=0,gradient_norm=norm)
            else:row=update(m,tok,mod,cap,p,ds,ids,positions,opt,step_job,st,replay,calopt,calrng,coefs,cached if p is not None else None)
            row.update(indices=ids,positions=positions,learning_rate=opt.param_groups[0]['lr'])
            if job.get('predictor_stop_step') is not None:row['predictor_active']=st<=job['predictor_stop_step']
            history.append(row)
            row['true_positions']=row.get('stratified_positions',list(range(32)) if row['true_gradient_examples']==32 else [] if row['true_gradient_examples']==0 else positions)
            if st in job.get('save',[32,128,256]):
                dest=out/f'step-{st}';tick=clock();m.save_pretrained(dest);save_seconds=clock()-tick
                calibrated_path=None;calibrated_save_seconds=0.
                if calopt is not None:
                    calibrated_path=dest/'predictor.pt';tick=clock();save_predictor(p,calibrated_path);calibrated_save_seconds=clock()-tick
                tick=clock();dev_ce=ce(m,tok,dev,ids=list(range(min(128,len(dev)))),micro=32) if job.get('diagnostics',True) else None;diag_seconds=clock()-tick
                entry=dict(task=a.task,tag=f'{name}__step{st}',adapter=str(dest),method=job['name'],kind=job['kind'],k=job.get('k',32),lr=job['lr'],seed=a.seed,warmup_seed=a.warmup_seed,subset=a.subset,predictor_seed=123 if p is not None and 'predictor_path' not in job else None,predictor_path=str(path) if p is not None else None,steps=st,online_seconds=sum(r['online_seconds'] for r in history),cache_seconds=prep,dev_ce=dev_ce,adapter_sha256=sha(dest/'adapter_model.safetensors'),save_seconds=save_seconds,diagnostic_seconds=diag_seconds)
                if calibrated_path is not None:entry.update(calibrated_predictor=str(calibrated_path),calibrated_predictor_sha256=sha(calibrated_path),calibrated_predictor_save_seconds=calibrated_save_seconds)
                ee.append(entry);write(out/'entries.json',ee);log(entry)
            if st%32==0:write(out/'history.json',history)
        assert versions(m)==basever
        summary=dict(complete=True,entries=ee,wall_seconds=time.time()-started,online_seconds=sum(r['online_seconds'] for r in history),cache_seconds=prep,true_gradient_examples=sum(r['true_gradient_examples'] for r in history),calibration_seconds=sum(r['calibration_seconds'] for r in history),base_unchanged=True)
        summary['initialization_true_gradient_examples']=getattr(online,'bootstrap_examples',0)
        if backend is not None:summary['teacher_backend_receipt']=backend.receipt()
        write(out/'history.json',history);write(out/'summary.json',summary);entries.extend(ee)
        del p,opt,calopt;gc.collect()
    if backend is not None:backend.close_job()
    teacher.close();cap.close()
    write(EXP/'results'/f'entries_{a.task}_w{a.warmup_seed}_{a.subset}_u{a.seed}{a.suffix}.json',entries)
    log(dict(complete=True,entries=len(entries),wall_seconds=time.time()-start_all))

if __name__=='__main__':main()
