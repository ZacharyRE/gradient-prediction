from research_h import *
from fast_backend import configure

def main():
    warm=json.loads((EXP/'configs/paws_warmup.json').read_text())
    m,tok,mod=setup('1p5b',warm['adapter']);cap=Capture(mod,m);A,B,s=weights(mod);configure(True,8192)
    ds=Dataset('paws','update',tok);ds.items=ds.items[:4];ds.rows=ds.rows[:4];ds.lengths=ds.lengths[:4]
    cache,_=cache_features(m,tok,mod,cap,ds);results=[]
    for change in [False,True]:
        if change:
            with torch.no_grad():A.add_(.001*torch.randn_like(A));B.add_(.001*torch.randn_like(B))
        exact=features(m,tok,mod,cap,ds,list(range(4)),micro=4);estimated=current_features(mod,cache,list(range(4)))
        error=max(float((a['h'].float().cuda()-b['h'].float()).norm()/a['h'].float().norm().clamp_min(1e-20)) for a,b in zip(exact,estimated))
        assert error<.01,error
        results.append(dict(perturbed=change,max_h_relative_l2=error))
    write(EXP/'results/h_cache_audit.json',dict(passed=True,checks=results,scope='Full input preserved; FP32 LoRA delta added to BF16 cached H, approximate under autocast rounding.'))
    log(results);cap.close()

if __name__=='__main__':main()
