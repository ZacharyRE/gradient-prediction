"""Common cached-suffix backend, with full inputs and paid lazy CUDA captures.

Graph and eager-static modes execute the same padded shapes and loss arithmetic.
The original native backend remains an independent control. No GPU graph is
shared across independently timed training recipes.
"""
from predictor_core import *
from updates_legacy import clock
import gc

class GraphTeacher:
    def __init__(self,m,tok,mod,cap,ds,cache,mode,bucket=64,max_graphs=12,audit=True):
        # Import lazily to avoid a reuse_train/cuda_graph_probe import cycle.
        from cuda_graph_probe import StaticTeacherGradient
        self.factory=StaticTeacherGradient;self.m=m;self.tok=tok;self.mod=mod;self.cap=cap;self.ds=ds;self.cache=cache;self.mode=mode;self.bucket=bucket;self.max_graphs=max_graphs;self.audit=audit;self.graphs={};self.records={};self.stream=torch.cuda.Stream()
        self.native_collect=collect;self.native_oracle=oracle_step;self.counter=getattr(type(self),'flop_counter',None)
        assert mode in ['graph','eager_static'] and bucket>=8 and bucket%8==0

    def begin_job(self,job):
        assert job['kind'] in ['oracle','subset','cv','prediction','class_cov','neural_hybrid','held_gradient','zero_gradient','table','full_table']
        self.close_job();self.records={};self.job=job

    def close_job(self):
        torch.cuda.synchronize();self.m.zero_grad(set_to_none=True)
        if hasattr(self.cap,'y'):self.cap.y=self.cap.y.detach()
        self.last_static=None
        for static in self.graphs.values():static.loss=None
        self.graphs.clear();self.cache.active=None;gc.collect();torch.cuda.empty_cache()

    def compute(self,ids,normalizer):
        counted_start=self.counter.get_total_flops() if self.counter is not None else 0
        L=((max(self.ds.lengths[i] for i in ids)+self.bucket-1)//self.bucket)*self.bucket;key=(len(ids),L);record=self.records.setdefault(str(key),dict(batch_size=len(ids),padded_length=L,calls=0,capture_seconds=0.,capture_warmup_backward_examples=0,audit_backward_examples=0,graph=False))
        if key not in self.graphs:
            tick=clock();static=self.factory(self.m,self.tok,self.mod,self.cap,self.ds,self.cache,ids,L,loss_form='native_mean');static.set_normalizer(normalizer)
            used_pool=sum(getattr(v,'graph_reserved_bytes',0) for v in self.graphs.values());estimated_pool=len(ids)*L*2*1024**2
            # Bound private graph pools while retaining space for a full eager
            # fallback. Every real token is still processed; only padding grows.
            allowed=len(ids)*L<=4096 and used_pool+estimated_pool<=12*1024**3 and sum(v.graph is not None for v in self.graphs.values())<self.max_graphs
            if self.job.get('graph_capture_batch_sizes') is not None:allowed=allowed and len(ids) in self.job['graph_capture_batch_sizes']
            record.update(capture_allowed_by_memory_policy=allowed,graph_pool_bytes_before=used_pool,estimated_new_pool_bytes=estimated_pool)
            if self.mode=='graph' and allowed:
                expected=None
                if self.audit or self.counter is not None:
                    before=self.counter.get_total_flops() if self.counter is not None else 0;static.eager()
                    if self.counter is not None:record['matrix_flops_per_call']=self.counter.get_total_flops()-before
                    if self.audit:expected=[g.clone() for g in static.grads]
                    else:record['profiling_only_extra_eager_calls']=1
                # Multiple live graphs retain AccumulateGrad edges. This local
                # PyTorch API redirects stale stream edges during capture only.
                torch.autograd.graph.set_override_stale_capture_stream(True)
                try:static.capture(self.stream)
                finally:torch.autograd.graph.set_override_stale_capture_stream(False)
                record['capture_warmup_backward_examples']=3*len(ids);record['capture_traced_backward_examples']=len(ids);record['graph']=True
                if self.audit:
                    static.replay();torch.cuda.synchronize();error=measures(flat(static.grads),flat(expected));assert error['relative_l2']==0.,error;record['graph_vs_same_shape_eager']=error;record['audit_backward_examples']=2*len(ids)
                record['graph_reserved_bytes']=static.graph_reserved_bytes
            record['capture_seconds']=clock()-tick;self.graphs[key]=static
        static=self.graphs[key];static.load(ids);static.set_normalizer(normalizer)
        before=self.counter.get_total_flops() if self.counter is not None else 0
        static.replay() if static.graph is not None else static.eager()
        if self.counter is not None and static.graph is None:
            counted=self.counter.get_total_flops()-before
            if 'matrix_flops_per_call' in record:assert record['matrix_flops_per_call']==counted
            record['matrix_flops_per_call']=counted
        result=tuple(g.detach().clone() for g in static.grads);loss=float(static.loss.detach());record['calls']+=1
        assert all(torch.isfinite(g).all() for g in result)
        if self.counter is not None:record['instrumented_matrix_flops']=record.get('instrumented_matrix_flops',0)+self.counter.get_total_flops()-counted_start
        self.last_static=static;return result,loss

    def collect(self,m,tok,mod,cap,ds,ids,micro=32,keep=True,verify_reconstruction=True):
        assert ds is self.ds and not verify_reconstruction
        A,B,s=weights(mod);truth=[torch.zeros_like(A),torch.zeros_like(B)];tick=clock();samples=[]
        for ix in chunks(ds,ids,max_n=micro,token_budget=8192):
            pair,_=self.compute(ix,1.)
            for acc,value in zip(truth,pair):acc.add_(value)
            if keep:
                static=self.last_static
                for j,i in enumerate(ix):
                    L=len(ds[i]['input_ids']);labels=torch.tensor(ds[i]['labels'][1:]+[-100],device=A.device)
                    row=dict(index=i,x=static.output_x[j,:L].bfloat16().clone(),y=static.output_y[j,:L].bfloat16().clone(),g=static.output_g[j,:L].float().clone(),active=labels>=0,label_ids=labels,n=ds[i]['n'])
                    if static.output_h is not None:row['h']=static.output_h[j,:L].bfloat16().clone()
                    samples.append(row)
        m.zero_grad(set_to_none=True)
        return samples,truth,dict(seconds=clock()-tick,backend=self.mode,full_inputs=True)

    def oracle_step(self,m,tok,mod,ds,ids,opt,micro=32):
        assert ds is self.ds;A,B,s=weights(mod);truth=[torch.zeros_like(A),torch.zeros_like(B)];N=sum(ds[i]['n'] for i in ids);loss=0.
        for ix in chunks(ds,ids,max_n=micro,token_budget=8192):
            pair,value=self.compute(ix,N);loss+=value
            for acc,g in zip(truth,pair):acc.add_(g)
        opt.zero_grad(set_to_none=True)
        for w,g in zip([A,B],truth):w.grad=g
        norm=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
        return dict(loss=loss,gradient_norm=norm)

    def receipt(self):
        result=dict(mode=self.mode,bucket=self.bucket,max_graphs=self.max_graphs,full_inputs=True,shapes=self.records,capture_and_initialization_seconds=sum(r['capture_seconds'] for r in self.records.values()),additional_capture_backward_examples=sum(r['capture_warmup_backward_examples']+r['audit_backward_examples'] for r in self.records.values()),capture_cost_included_in_preparation_and_online_seconds=True,all_recipes_receive_same_backend=True,memory_policy='Capture at most4096 padded batch tokens and12GiB estimated total graph-private pools; other shapes use identical eager-static arithmetic without truncation')
        if self.counter is not None:
            result['instrumented_teacher_matrix_flops']=sum(r['instrumented_matrix_flops'] for r in self.records.values())
            result['executed_teacher_matrix_flops']=sum(r['matrix_flops_per_call']*(r['calls']+(r['capture_warmup_backward_examples']+r['audit_backward_examples'])//r['batch_size']) for r in self.records.values())
            result['flop_scope']='Per-shape eager operator arithmetic multiplied by actual training, warmup and audit executions. CUDA capture tracing and profiling-only eager probes are excluded; opaque graph replays are restored explicitly.'
        return result

    def install(self,trainer):
        import research_teacher_cache as rt
        import class_covariance as cc
        import gradient_table as gt
        import full_gradient_table as ft
        trainer.collect=self.collect;rt.collect=self.collect;rt.oracle_step=self.oracle_step;cc.collect_gpu=self.collect;gt.collect_gpu=self.collect;ft.collect_gpu=self.collect
