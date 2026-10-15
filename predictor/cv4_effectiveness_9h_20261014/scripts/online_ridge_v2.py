"""Fast online gradient surrogate with delayed recursive ridge updates and CV correction.

Features use invariant frozen-prefix X, with known TRAINING labels. The true
correction labels update the predictor only AFTER the current adapter update.
No large predictor fitting loop or extra validation backpropagation is required.
"""
from research_v3 import *
from fast_backend import configure
ENTRY_SHA=sha(Path(__file__))
CODE_SHA=source_signature(Path(__file__))

class OnlineRidge:
    def __init__(self,dim,outdim,ridge,forget):
        self.W=torch.zeros(dim,outdim,device='cuda')
        self.P=torch.eye(dim,device='cuda',dtype=torch.float64)/ridge
        self.forget=forget
    @torch.no_grad()
    def predict(self,x):return x@self.W
    @torch.no_grad()
    def update(self,x,y):
        xd=x.double();prior=self.P/self.forget
        px=prior@xd.T
        small=torch.eye(len(x),device='cuda',dtype=torch.float64)+xd@px
        gain=torch.linalg.solve(small,px.T).T
        self.W.add_(gain.float()@(y-x@self.W))
        self.P=prior-gain@(xd@prior)
        self.P=(self.P+self.P.T)/2
        if not torch.isfinite(self.W).all() or not torch.isfinite(self.P).all():raise FloatingPointError('RLS became nonfinite')

@torch.no_grad()
def feature_matrix(rows,source_rows,projection,classes):
    output=[];dim=projection.shape[1]
    for row in rows:
        if dim:
            x=row['x'].float().cuda();where=torch.where(row['active'])[0]
            last=int(where[0]) if len(where) else len(x)-1
            mean=x[:last+1].mean(0);point=x[last]
            parts=[]
            for vector in [mean,point]:
                vector=F.layer_norm(vector,(len(vector),))@projection
                vector=vector/vector.square().mean().sqrt().clamp_min(1e-6)/(dim**.5)
                parts.append(vector)
            single=torch.cat([torch.ones(1,device='cuda'),*parts])
        else:single=torch.ones(1,device='cuda')
        phi=torch.zeros(len(single)*len(classes),device='cuda')
        label=classes[source_rows[row['index']]['gold']]
        phi[label*len(single):(label+1)*len(single)]=single
        output.append(phi)
    return torch.stack(output)

@torch.no_grad()
def individual_truth(rows,A,B,s):
    d=pack(rows);ga,gb=raw(d['x'],d['g'],A,B,s)
    return torch.cat([ga.flatten(1),gb.flatten(1)],1)

def run_ridge(task,jobs,seed_id):
    snapshot=EXP/'provenance/source_versions'/CODE_SHA;snapshot.mkdir(parents=True,exist_ok=True)
    for script in (EXP/'scripts').glob('*.py'):
        if not (snapshot/script.name).exists():shutil.copy2(script,snapshot/script.name)
    warm=json.loads((EXP/'configs'/f'{task}_warmup.json').read_text())
    m,tok,mod=setup('1p5b',warm['adapter'],rng=seed_id);A,B,s=weights(mod);original=[v.detach().clone() for v in [A,B]];basever=versions(m);cap=Capture(mod)
    ds=Dataset(task,'update',tok);dev=Dataset(task,'accuracy_dev',tok);audit=Dataset(task,'gradient_test',tok)
    configure(True,8192)
    needs_prefix=any(j.get('projection_dim',16)>0 for j in jobs)
    cached,cache_seconds=cache_features(m,tok,mod,cap,ds) if needs_prefix else ([{'index':i} for i in range(len(ds))],0.)
    classes={v:i for i,v in enumerate(sorted(set(r['gold'] for r in ds.rows)))}
    features_by_dim={};projections={};entries=[]
    for job in jobs:
        name=f'{task}__{job["name"]}_u{seed_id}_p0';out=EXP/'models'/name
        if (out/'summary.json').exists():entries.extend(json.loads((out/'summary.json').read_text())['entries']);continue
        if deadline(5400):raise RuntimeError('Final validation reserve reached')
        out.mkdir(exist_ok=True);seed(seed_id)
        with torch.no_grad():A.copy_(original[0]);B.copy_(original[1])
        dim=job.get('projection_dim',16)
        if dim not in features_by_dim:
            g=torch.Generator(device='cpu').manual_seed(20261014)
            proj=(torch.randn(A.shape[1],dim,generator=g)/(A.shape[1]**.5)).cuda()
            started=clock();phi=feature_matrix(cached,ds.rows,proj,classes)
            projections[dim]=proj;features_by_dim[dim]=(phi,clock()-started)
        phi,feature_seconds=features_by_dim[dim];proj=projections[dim]
        ridge=OnlineRidge(phi.shape[1],A.numel()+B.numel(),job.get('ridge',10.),job.get('forget',.99))
        offline_seconds=0.;offline_examples=0
        if job.get('offline_init'):
            assert json.loads((SOURCE/'cache'/task/'complete.json').read_text())['start_sha256']==warm['adapter_sha256']
            started=clock();offline_ds=read(SOURCE/'data'/task/'predictor_train.jsonl')
            for file in sorted((SOURCE/'cache'/task/'predictor_train').glob('shard*.pt')):
                rows=torch.load(file,map_location='cpu',weights_only=False)
                with torch.no_grad():
                    x=feature_matrix(rows,offline_ds,proj,classes);y=individual_truth(rows,A,B,s)
                    saved=ridge.forget;ridge.forget=1.;ridge.update(x,y);ridge.forget=saved
                offline_examples+=len(rows)
                if offline_examples>=job.get('offline_examples',512):break
            offline_seconds=clock()-started
        opt=torch.optim.AdamW([A,B],lr=job['lr'],weight_decay=0)
        rng=np.random.default_rng(seed_id);order=[]
        while len(order)<job['steps']*32:order.extend(map(int,rng.permutation(len(ds))))
        crng=np.random.default_rng(seed_id+10000);history=[];ee=[];started=time.time()
        write(out/'config.json',dict(task=task,job=job,update_seed=seed_id,predictor_seed=0,code_sha256=ENTRY_SHA,source_bundle_sha256=CODE_SHA,warmup_sha256=warm['adapter_sha256'],full_inputs=True,classes=classes,feature_dim=phi.shape[1],predictor_parameters=ridge.W.numel(),gradient_labels='Only current uniformly sampled correction examples, plus explicitly declared offline initialization if requested',prediction_before_current_labels=True))
        for st in range(1,job['steps']+1):
            if deadline(3600):raise RuntimeError('Final hour reserve')
            ids=order[(st-1)*32:st*32];positions=list(map(int,crng.choice(32,job.get('k',4),replace=False)));subset=[ids[i] for i in positions]
            tick=clock();pred=ridge.predict(phi[ids]);c=job.get('coefficient',1.);n=sum(ds[i]['n'] for i in ids)
            t=clock();rows,truth,_=collect(m,tok,mod,cap,ds,subset,micro=job.get('k',4),keep=True,verify_reconstruction=False);label_seconds=clock()-t
            with torch.no_grad():
                gradients=(flat(truth)*(32/len(positions))+c*(pred.sum(0)-(32/len(positions))*pred[positions].sum(0)))/n
                targets=individual_truth(rows,A,B,s)
            opt.zero_grad(set_to_none=True);A.grad=gradients[:A.numel()].reshape_as(A);B.grad=gradients[A.numel():].reshape_as(B)
            norm=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
            t=clock();ridge.update(phi[subset],targets);cal_seconds=clock()-t
            row=dict(step=st,online_seconds=clock()-tick,label_seconds=label_seconds,calibration_seconds=cal_seconds,true_gradient_examples=len(subset),gradient_norm=norm,indices=ids,positions=positions)
            history.append(row)
            if st in job.get('save',[32,64,128,256]):
                dest=out/f'step-{st}';m.save_pretrained(dest)
                t=clock();row['dev_ce']=ce(m,tok,dev,ids=list(range(128)),micro=16)
                # Audit labels are strictly excluded from ridge fitting and timing.
                rr,_,_=collect(m,tok,mod,cap,audit,list(range(32)),micro=16,keep=True,verify_reconstruction=False)
                with torch.no_grad():
                    xp=feature_matrix(rr,audit.rows,proj,classes);pp=ridge.predict(xp);tt=individual_truth(rr,A,B,s)
                    pc=pp-pp.mean(0);tc=tt-tt.mean(0)
                    row['gradient_audit']=dict(batch_lora=measures(pp.sum(0),tt.sum(0)),centered_correlation=float((pc*tc).sum()/(pc.norm()*tc.norm()).clamp_min(1e-20)),cv_residual_variance_ratio=float((tc-c*pc).square().sum()/tc.square().sum().clamp_min(1e-20)))
                row['audit_seconds']=clock()-t
                e=dict(task=task,tag=f'{name}__step{st}',adapter=str(dest),method=job['name'],lr=job['lr'],seed=seed_id,predictor_seed=0,steps=st,online_seconds=sum(r['online_seconds'] for r in history),cache_seconds=(cache_seconds if dim else 0.)+feature_seconds,offline_fit_seconds=offline_seconds,offline_gradient_examples=offline_examples,dev_ce=row['dev_ce'],adapter_sha256=sha(dest/'adapter_model.safetensors'))
                ee.append(e);write(out/'entries.json',ee)
            if st%16==0 or st==1:
                write(out/'history.json',history);log(dict(run=name,step=st,online_seconds=sum(r['online_seconds'] for r in history),dev_ce=row.get('dev_ce')))
        assert versions(m)==basever
        torch.save(dict(W=ridge.W.cpu(),P=ridge.P.cpu(),projection=proj.cpu(),classes=classes),out/'ridge-final.pt')
        write(out/'history.json',history);write(out/'summary.json',dict(complete=True,entries=ee,online_seconds=sum(r['online_seconds'] for r in history),wall_seconds=time.time()-started,cache_seconds=(cache_seconds if dim else 0.)+feature_seconds,offline_fit_seconds=offline_seconds,offline_gradient_examples=offline_examples,true_gradient_examples=sum(r['true_gradient_examples'] for r in history),base_unchanged=True))
        entries.extend(ee);write(EXP/'results'/f'{task}_ridge_entries_u{seed_id}.json',entries)
        del ridge,opt;gc.collect()
    cap.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--seed',type=int,default=101);a=ap.parse_args()
    run_ridge(a.task,json.loads(Path(a.jobs).read_text()),a.seed)

if __name__=='__main__':main()
