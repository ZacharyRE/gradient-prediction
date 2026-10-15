from research_v3 import *
import cached_head as ch

def main():
    gpu_guard()
    rows=load_samples(SOURCE/'cache/paws/predictor_dev')[:4]
    p,_=load_predictor(predictor_path('paws',123));p=ch.configure_predictor(p);p.eval()
    reserve_worker_cache()
    rows=ch.encode_rows(p,rows);checks=[]
    for changed in [False,True]:
        if changed:
            with torch.no_grad():p.head[1].weight.add_(.001*torch.randn_like(p.head[1].weight))
        with torch.no_grad():
            full=p.__dict__['original_forward'](ch.BASE_PACK(rows));cached=p(ch.pack_cached(rows))
        error=float((full-cached).abs().max());assert error<1e-6,error
        checks.append(dict(head_changed=changed,max_absolute_output_error=error))
    assert all(name.startswith('head.')==parameter.requires_grad for name,parameter in p.named_parameters())
    write(EXP/'results/cached_head_audit.json',dict(passed=True,checks=checks,note='Exact encoder caching at the same warmup inputs; stale body features during later LoRA changes are intentional and corrected by CV.'))
    log(checks)

if __name__=='__main__':main()
