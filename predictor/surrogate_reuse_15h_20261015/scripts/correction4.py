"""Four-label control-variate update; all sampling is over batch positions."""
from predictor_core import *
from updates_legacy import clock


def corrected_pair(predicted, truth_subset, positions, supervised_tokens):
    """Linear unbiased estimator before clipping/Adam, for a uniform subset."""
    n=predicted[0].shape[0];k=len(positions)
    assert n>0 and 0<k<=n and len(set(positions))==k
    assert all(0<=i<n for i in positions) and supervised_tokens>0
    return tuple((p.sum(0)+(n/k)*(t-p[positions].sum(0)))/supervised_tokens
                 for p,t in zip(predicted,truth_subset))


@torch.no_grad()
def per_example_predictions(p,rows,A,B,s):
    """Each prediction occurs once, with grouping independent of subset choice."""
    result_a=torch.empty((len(rows),*A.shape),device=A.device,dtype=A.dtype)
    result_b=torch.empty((len(rows),*B.shape),device=B.device,dtype=B.dtype)
    for positions in sample_chunks(rows):
        d=pack([rows[i] for i in positions])
        condition_lora_state(p,d,A,B,s)
        # Predictor inference may use BF16; reconstruct raw LoRA gradients in
        # FP32 so the correction estimator does not inherit BF16 accumulation.
        if hasattr(p,'predict_factors'):
            with torch.autocast('cuda',enabled=False):pa,pb=p.predict_factors(d,A,B,s)
        else:
            g=p(d)
            with torch.autocast('cuda',enabled=False):pa,pb=raw(d['x'],g,A,B,s)
        result_a[positions]=pa;result_b[positions]=pb
    return result_a,result_b


def correction_step(method,m,tok,mod,cap,p,ds,ids,positions,opt):
    assert method in ['cv4','oracle4','oracle4_matched']
    A,B,s=weights(mod);start=clock();cal=[ids[i] for i in positions]
    n=sum(ds[i]['n'] for i in ids)
    row=dict(true_gradient_examples=len(cal),true_gradient_tokens=sum(ds[i]['n'] for i in cal),
             true_gradient_input_tokens=sum(ds.lengths[i] for i in cal),
             main_batch_input_tokens=sum(ds.lengths[i] for i in ids),
             supervised_normalization_tokens=n,calibration_epochs=0,
             estimator='unbiased_raw_gradient_before_clipping_and_adam')
    if method=='cv4':
        assert p is not None and not p.training
        tick=clock()
        rows=features(m,tok,mod,cap,ds,ids)
        assert [r['index'] for r in rows]==ids
        predicted=per_example_predictions(p,rows,A,B,s)
        row['prefix_prediction_seconds']=clock()-tick
    tick=clock()
    _,truth,_=collect(m,tok,mod,cap,ds,cal,keep=False)
    row['label_seconds']=clock()-tick
    tick=clock()
    if method=='cv4':
        gradients=corrected_pair(predicted,truth,positions,n)
        row['model_input_tokens']=row['main_batch_input_tokens']+row['true_gradient_input_tokens']
    else:
        gradients=tuple(t*(len(ids)/len(positions))/n for t in truth)
        row['model_input_tokens']=row['true_gradient_input_tokens']
    opt.zero_grad(set_to_none=True)
    for parameter,g in zip([A,B],gradients):
        assert torch.isfinite(g).all(),'Nonfinite corrected gradient'
        parameter.grad=g.detach()
    row['gradient_norm']=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True))
    opt.step();m.zero_grad(set_to_none=True)
    row['prediction_update_seconds']=row.get('prefix_prediction_seconds',0)+clock()-tick
    row['online_seconds']=clock()-start
    return row
