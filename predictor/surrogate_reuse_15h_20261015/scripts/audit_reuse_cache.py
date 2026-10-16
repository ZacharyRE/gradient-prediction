"""Check current-state H reconstruction and true gradients with frozen-prefix reuse."""
from reuse_train import *
from safetensors.torch import load_file

def main():
    warm=json.loads((EXP/'configs/mnli_warmup_w301.json').read_text())
    m,tok,mod=setup('1p5b',warm['adapter'],rng=811);cap=Capture(mod,m);A,B,s=weights(mod)
    ds=Dataset('mnli','predictor_dev',tok);configure(True,8192)
    teacher=TeacherPrefixCache(m,tok,mod,cap,ds,micro=32);cached=cached_rows(teacher,ds)
    for r in cached:r['want_h']=True
    ids=[0,9,44,87];reports=[]
    # Resolve the recorded adapter instead of relying on a hand-written tag.
    entries=json.loads((EXP/'results/entries_mnli_w301_update_u401.json').read_text())
    late=next(Path(e['adapter']) for e in entries if e['kind']=='oracle' and e['lr']==.0003 and e['steps']==256)
    for label,path in [('warmup',Path(warm['adapter'])),('step256',late)]:
        vals=load_file(str(path/'adapter_model.safetensors'))
        with torch.no_grad():
            A.copy_(next(v for k,v in vals.items() if k.endswith('lora_A.weight')));B.copy_(next(v for k,v in vals.items() if k.endswith('lora_B.weight')))
        direct=current_features(mod,cached,ids);via_model=features_gpu(m,tok,mod,cap,ds,ids,micro=4)
        equality={k:all(torch.equal(a[k],b[k]) for a,b in zip(direct,via_model)) for k in ['x','y','h','label_ids']}
        assert all(equality.values()),equality
        teacher.enabled=True;rr,truth,audit=collect_gpu(m,tok,mod,cap,ds,ids,micro=4,keep=True)
        teacher.enabled=False;native=features_gpu(m,tok,mod,cap,ds,ids,micro=4)
        _,native_truth,_=collect_gpu(m,tok,mod,cap,ds,ids,micro=4,keep=True)
        feature_metrics={k:measures(torch.cat([r[k].flatten() for r in direct]),torch.cat([r[k].flatten() for r in native])) for k in ['x','y','h']}
        gradient_metrics=measures(flat(truth),flat(native_truth))
        assert gradient_metrics['cosine']>.99 and gradient_metrics['relative_l2']<.08,gradient_metrics
        reports.append(dict(state=label,cached_h_bitwise_equal_to_executed_model=equality,native_feature_rounding=feature_metrics,native_gradient_rounding=gradient_metrics,reconstruction=audit))
        teacher.enabled=True
    teacher.close();cap.close();write(EXP/'results/reuse_cache_audit.json',dict(passed=True,full_inputs=True,states=reports,interpretation='Cache matches its executed teacher exactly; native differences are separately measured BF16 batch-shape rounding. All compared methods use the same cache backend.'));log(reports)

if __name__=='__main__':main()
