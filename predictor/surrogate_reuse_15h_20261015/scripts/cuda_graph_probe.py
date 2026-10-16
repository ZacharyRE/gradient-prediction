"""Optional common-backend latency probe; no adaptation recipe is changed.

Capture a static padded teacher-gradient call, retaining every real input token.
If supported, compare with the same eager callable, changed weights and batches,
and the native cached teacher. Capture time is reported separately. Unsupported
capture is recorded as a negative engineering result, not a passing audit.
"""
from reuse_train import *
from torch.nn.attention import sdpa_kernel,SDPBackend

class StaticTeacherGradient:
    def __init__(self,m,tok,mod,cap,ds,cache,ids,length,loss_form='sum'):
        self.m=m;self.tok=tok;self.mod=mod;self.cap=cap;self.ds=ds;self.cache=cache;self.batch_size=len(ids);self.length=length
        A,B,s=weights(mod);self.params=[A,B];self.grads=[torch.zeros_like(w) for w in self.params];D=A.shape[1];n=len(ids);T=max(ds[i]['n'] for i in range(len(ds)))
        self.inputs_embeds=torch.zeros(n,length,D,device='cuda');self.positions=torch.arange(length,device='cuda')[None]
        self.active={k:torch.zeros(n,length,D,device='cuda',dtype=cache.rows[ids[0]][k].dtype) for k in ['x','residual']}
        self.mask=torch.zeros(n,1,length,length,device='cuda');self.target_positions=torch.zeros(n,T,device='cuda',dtype=torch.long);self.labels=torch.full((n,T),-100,device='cuda',dtype=torch.long);self.normalizer=torch.tensor(1.,device='cuda');self.mean_multiplier=torch.tensor(1.,device='cuda');self.loss_form=loss_form
        self.causal=torch.ones(length,length,device='cuda',dtype=torch.bool).triu(1);self.load(ids);self.loss=None;self.graph=None

    def load(self,ids):
        assert len(ids)==self.batch_size and all(len(self.ds[i]['input_ids'])<=self.length for i in ids)
        positions=[];labels=[];lengths=[];T=self.labels.shape[1]
        for j,i in enumerate(ids):
            row=self.ds[i];L=len(row['input_ids']);lengths.append(L)
            for k in self.active:self.active[k][j].zero_();self.active[k][j,:L].copy_(self.cache.rows[i][k])
            shifted=row['labels'][1:]+[-100];pos=[q for q,v in enumerate(shifted) if v!=-100];lab=[shifted[q] for q in pos];positions.append(pos+[0]*(T-len(pos)));labels.append(lab+[-100]*(T-len(lab)))
        self.target_positions.copy_(torch.tensor(positions,device='cuda'));self.labels.copy_(torch.tensor(labels,device='cuda'));self.supervised_tokens=sum(self.ds[i]['n'] for i in ids);self.set_normalizer(self.supervised_tokens)
        valid=torch.arange(self.length,device='cuda')[None,:]<torch.tensor(lengths,device='cuda')[:,None]
        self.mask.zero_();self.mask.masked_fill_(self.causal[None,None]|~valid[:,None,None,:],float('-inf'));self.cache.active=self.active

    def set_normalizer(self,n):
        self.normalizer.fill_(n);self.mean_multiplier.fill_(self.supervised_tokens/n)

    def eager(self):
        # Release the previous autograd graph before building another one on
        # a different stream; Capture's output hook also retains a graph edge.
        self.loss=None
        if hasattr(self.cap,'y'):self.cap.y=self.cap.y.detach()
        for w,g in zip(self.params,self.grads):g.zero_();w.grad=g
        self.cache.active=self.active
        with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.MATH):
            base=self.m.get_base_model();h=base.model(inputs_embeds=self.inputs_embeds,attention_mask={'full_attention':self.mask},position_ids=self.positions,use_cache=False,return_dict=True).last_hidden_state
            selected=h.gather(1,self.target_positions[...,None].expand(-1,-1,h.shape[-1]));logits=base.lm_head(selected)
            if self.loss_form=='native_mean':self.loss=F.cross_entropy(logits.float().flatten(0,1),self.labels.flatten(),reduction='mean',ignore_index=-100)*self.mean_multiplier
            else:self.loss=F.cross_entropy(logits.float().flatten(0,1),self.labels.flatten(),reduction='sum',ignore_index=-100)/self.normalizer
        self.loss.backward()
        self.output_x=self.cap.x.detach();self.output_y=self.cap.y.detach();self.output_g=self.cap.y.grad.detach()
        self.output_h=self.cap.h.detach() if self.cap.residual_norm is not None else None

    def capture(self,stream=None):
        stream=stream or torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.eager()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        # Graph-private pools cannot use the worker's unused default-pool padding.
        # The cooperative GPU lock and holder remain active throughout this stage.
        torch.cuda.empty_cache();before_reserved=torch.cuda.memory_reserved();tick=clock();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.eager()
        self.graph_reserved_bytes=max(0,torch.cuda.memory_reserved()-before_reserved)
        return clock()-tick

    def replay(self):
        for w,g in zip(self.params,self.grads):w.grad=g
        self.graph.replay()

def main():
    path=EXP/'results/cuda_graph_probe.json';result=dict(complete=False,supported=False,full_inputs=True,purpose='Common teacher backend probe, not a predictor-only speedup')
    try:
        warm=json.loads((EXP/'configs/mnli_warmup_w301.json').read_text());m,tok,mod=setup('1p5b',warm['adapter'],rng=811);cap=Capture(mod);ds=Dataset('mnli','predictor_dev',tok);configure(True,8192);cache=TeacherPrefixCache(m,tok,mod,cap,ds,micro=32)
        assert not m.get_base_model().model.has_sliding_layers
        A,B,s=weights(mod);initial=(A.detach().clone(),B.detach().clone());records=[]
        for batch_size in [4,16,32]:
            ids=list(range(batch_size));other=list(range(32,32+batch_size));L=((max(ds.lengths[i] for i in ids+other)+7)//8)*8
            static=StaticTeacherGradient(m,tok,mod,cap,ds,cache,ids,L);static.eager();eager=[g.clone() for g in static.grads];loss=float(static.loss);capture_seconds=static.capture();reserve_worker_cache();static.replay();torch.cuda.synchronize();same=measures(flat(static.grads),flat(eager));assert same['relative_l2']<1e-6,same
            # The graph must consume changed data AND current parameter values.
            with torch.no_grad():A.add_(.0001);B.mul_(1.001)
            static.load(other);static.eager();changed=[g.clone() for g in static.grads];static.replay();torch.cuda.synchronize();change_check=measures(flat(static.grads),flat(changed));assert change_check['relative_l2']<1e-6,change_check
            samples={}
            for mode in ['eager_static','graph_core','graph_with_load']:
                times=[]
                for repeat in range(12):
                    tick=clock()
                    if mode=='graph_with_load':static.load(ids if repeat%2==0 else other)
                    static.eager() if mode=='eager_static' else static.replay()
                    times.append(clock()-tick)
                samples[mode]=dict(median_seconds=float(np.median(times)),mean_seconds=float(np.mean(times)))
            with torch.no_grad():A.copy_(initial[0]);B.copy_(initial[1])
            static.load(ids);static.eager();fixed=[g.clone() for g in static.grads]
            import predictor_core as pc
            native_times=[]
            for _ in range(4):
                tick=clock();bb=pc.batch(ds,ids,tok);m.zero_grad(set_to_none=True);loss=pc.forward(m,bb).loss;loss.backward();native_times.append(clock()-tick)
            native_check=measures(flat(fixed),flat([A.grad,B.grad]));assert native_check['cosine']>.99 and native_check['relative_l2']<.05,native_check
            records.append(dict(batch_size=batch_size,padded_length=L,real_lengths=[ds.lengths[i] for i in ids+other],capture_seconds=capture_seconds,graph_vs_eager=same,changed_inputs_and_weights_check=change_check,native_cached_teacher_check=native_check,native_eager_mean_seconds=float(np.mean(native_times)),timings=samples));log(records[-1]);del static;gc.collect();torch.cuda.empty_cache()
        cache.close();cap.close();result.update(complete=True,supported=True,records=records,limitations='Fixed-shape probe only. Bucket capture, input packing, memory, adaptation equivalence, and end-to-end rollout timing require separate validation before use.')
    except Exception as error:
        import traceback
        result.update(complete=True,supported=False,error=repr(error),traceback=traceback.format_exc());log(dict(cuda_graph_probe_failed=repr(error)))
    write(path,result)

if __name__=='__main__':main()
