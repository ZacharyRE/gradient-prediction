from research_v3 import *
from gpu_features import features_gpu,collect_gpu
from fast_backend import configure

def main():
    warm=json.loads((EXP/'configs/paws_warmup.json').read_text())
    m,tok,mod=setup('1p5b',warm['adapter']);cap=Capture(mod,m);configure(True,8192)
    ds=Dataset('paws','update',tok);ids=list(range(32))
    cpu=features(m,tok,mod,cap,ds,ids,micro=32);gpu=features_gpu(m,tok,mod,cap,ds,ids,micro=32)
    differences={k:max(float((a[k].cuda().float()-b[k].float()).abs().max()) for a,b in zip(cpu,gpu)) for k in ['x','y','h','active','label_ids']}
    assert all(v==0 for v in differences.values()),differences
    p,_=load_predictor(predictor_path('paws',123));A,B,s=weights(mod)
    aa=per_example_predictions(p,cpu,A,B,s);bb=per_example_predictions(p,gpu,A,B,s)
    error=float((flat(aa)-flat(bb)).abs().max());assert error==0,error
    cr,ct,_=collect(m,tok,mod,cap,ds,ids[:8],micro=8,keep=True)
    gr,gt,_=collect_gpu(m,tok,mod,cap,ds,ids[:8],micro=8,keep=True)
    label_error=max(float((c['g'].cuda()-g['g']).abs().max()) for c,g in zip(cr,gr))
    truth_error=float((flat(ct)-flat(gt)).abs().max())
    assert label_error==truth_error==0,(label_error,truth_error)
    write(EXP/'results/gpu_feature_audit.json',dict(passed=True,max_absolute_feature_differences=differences,raw_gradient_prediction_difference=error,teacher_label_max_difference=label_error,teacher_gradient_max_difference=truth_error,full_inputs=True,scope='Identical batch shape and capture points; CPU round-trip removed only. Separate microbatch-size changes can still change BF16 rounding.'))
    log(dict(passed=True,gradient_prediction_difference=error));cap.close()

if __name__=='__main__':main()
