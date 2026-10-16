"""Exact mathematical simplification of the selected epoch-zero matrix predictor.

Only class means and current LoRA factors are needed. No neural feature pass is
performed. The correction uses the imbalance in class counts of the random
true-gradient subset. A stratified true-gradient sampler is a separate control.
"""
from predictor_core import *
from gpu_features import collect_gpu
from updates_legacy import clock
from torch.nn.attention import sdpa_kernel,SDPBackend

class FrozenClassCovariance:
    def __init__(self,ds,path):
        obj=torch.load(path,map_location='cpu',weights_only=False);state=obj['state'];assert obj['config']['kind']=='factor_surrogate'
        assert not torch.count_nonzero(state['body.4.weight']) and not torch.count_nonzero(state['body.4.bias']),'Simplification only valid for zero residual head'
        self.U=state['U'].cuda();self.V=state['V'].cuda();self.means=state['class_mean'].cuda();self.classes=np.array(['ABC'.index(r['gold']) for r in ds.rows]);self.source_sha256=sha(path);self.lengths=ds.lengths

    @torch.no_grad()
    def correction(self,mod,ids,positions):
        A,B,s=weights(mod);classes=self.classes[ids];counts=np.bincount(classes,minlength=3)-(len(ids)/len(positions))*np.bincount(classes[positions],minlength=3)
        C=(torch.tensor(counts,device=A.device,dtype=A.dtype)[:,None,None]*self.means).sum(0)
        return s*((B.T@self.U)@C@self.V.T),s*(self.U@C@(self.V.T@A.T))

    @torch.no_grad()
    def per_example_correction(self,mod,ids,positions):
        # Retain the original factor predictor's batching and reduction order.
        # This isolates removing its zero-head MLP from changing FP32 sums.
        A,B,s=weights(mod);pa=torch.empty(len(ids),*A.shape,device=A.device,dtype=A.dtype);pb=torch.empty(len(ids),*B.shape,device=B.device,dtype=B.dtype)
        metadata=[dict(x=range(self.lengths[i])) for i in ids]
        for ix in sample_chunks(metadata):
            cls=torch.tensor(self.classes[[ids[j] for j in ix]],device=A.device);C=self.means[cls]
            pa[ix]=s*((B.T@self.U)[None]@C@self.V.T);pb[ix]=s*(self.U[None]@C@(self.V.T@A.T))
        return tuple(p.sum(0)-(len(ids)/len(positions))*p[positions].sum(0) for p in [pa,pb])

    def step(self,m,tok,mod,cap,ds,ids,positions,opt,job,step):
        tick=clock();A,B,s=weights(mod);n=sum(ds[i]['n'] for i in ids);k=len(positions)
        correction_fn=self.per_example_correction if job.get('class_cov_reduction')=='per_example' else self.correction
        t=clock();correction=correction_fn(mod,ids,positions) if job.get('coefficient',1.)!=0 else [torch.zeros_like(A),torch.zeros_like(B)];pred_seconds=clock()-t
        t=clock();_,truth,_=collect_gpu(m,tok,mod,cap,ds,[ids[j] for j in positions],micro=32,keep=False,verify_reconstruction=False);label_seconds=clock()-t
        gradients=tuple(((len(ids)/k)*g+job.get('coefficient',1.)*c)/n for g,c in zip(truth,correction))
        opt.zero_grad(set_to_none=True)
        for w,g in zip([A,B],gradients):w.grad=g.detach()
        norm=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
        return dict(step=step,online_seconds=clock()-tick,true_gradient_examples=k,calibration_seconds=0.,prefix_prediction_seconds=pred_seconds,label_seconds=label_seconds,gradient_norm=norm,source_predictor_sha256=self.source_sha256)

class StratifiedGradients:
    def __init__(self,ds,seed):self.classes=np.array(['ABC'.index(r['gold']) for r in ds.rows]);self.rng=np.random.default_rng(seed+10000)

    def sample(self,ids,k):
        classes=self.classes[ids];counts=np.bincount(classes,minlength=3);present=np.flatnonzero(counts)
        assert k>=len(present) and k<=len(ids)
        allocation=np.zeros(3,dtype=int);allocation[present]=1
        while allocation.sum()<k:
            # Allocate the remaining draws by largest proportional deficit,
            # with deterministic class-index ties; all within-class draws uniform.
            deficit=counts*k/len(ids)-allocation;deficit[allocation>=counts]=-np.inf;allocation[int(np.argmax(deficit))]+=1
        positions=[];weights=[]
        for c in present:
            selected=self.rng.choice(np.flatnonzero(classes==c),int(allocation[c]),replace=False)
            positions.extend(map(int,selected));weights.extend([float(counts[c]/allocation[c])]*len(selected))
        return positions,weights

    def step(self,m,tok,mod,cap,ds,ids,positions,opt,job,step):
        import predictor_core as pc
        tick=clock();A,B,s=weights(mod);positions,weights0=self.sample(ids,len(positions));selected=[ids[j] for j in positions];N=sum(ds[i]['n'] for i in ids);opt.zero_grad(set_to_none=True)
        # Pack classes together; split only for the shared token budget.
        cursor=0;loss_value=0.
        for ix in pc.chunks(ds,selected,max_n=32,token_budget=8192):
            bb=pc.batch(ds,ix,tok);weight=torch.tensor(weights0[cursor:cursor+len(ix)],device=A.device);cursor+=len(ix)
            with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.MATH):
                base=m.get_base_model();hidden=base.model(input_ids=bb['input_ids'],attention_mask=bb['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
                labels=bb['labels'][:,1:];valid=labels!=-100;logits=base.lm_head(hidden[:,:-1][valid]);token_losses=F.cross_entropy(logits.float(),labels[valid],reduction='none');token_weights=weight[:,None].expand_as(labels)[valid];loss=(token_losses*token_weights).sum()/N
            loss.backward();loss_value+=float(loss.detach())
        norm=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
        return dict(step=step,online_seconds=clock()-tick,true_gradient_examples=len(selected),calibration_seconds=0.,gradient_norm=norm,stratified_positions=positions,stratified_weights=weights0,loss=loss_value,estimator='Class-stratified batch sum, unbiased before clipping/Adam')
