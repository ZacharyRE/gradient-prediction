from research_fastio import *
from fast_backend import configure
from teacher_prefix_cache import TeacherPrefixCache

def main():
    warm=json.loads((EXP/'configs/paws_warmup.json').read_text())
    m,tok,mod=setup('1p5b',warm['adapter']);cap=Capture(mod,m);configure(True,8192)
    ds=Dataset('paws','update',tok);ids=list(range(8));A,B,s=weights(mod)
    cache=TeacherPrefixCache(m,tok,mod,cap,ds,ids=ids,micro=8)
    checks=[]
    for perturbation in [0.,.001]:
        if perturbation:
            with torch.no_grad():A.add_(torch.randn_like(A)*perturbation);B.add_(torch.randn_like(B)*perturbation)
        for micro in [8,4]:
            cache.enabled=False
            _,truth,_=collect(m,tok,mod,cap,ds,ids,micro=micro,keep=False,verify_reconstruction=False)
            full=flat(truth).clone()
            cache.enabled=True
            _,truth,_=collect(m,tok,mod,cap,ds,ids,micro=micro,keep=False,verify_reconstruction=False)
            measured=measures(flat(truth),full)
            if micro==8:assert measured['relative_l2']<1e-6,measured
            else:assert measured['relative_l2']<.03 and measured['cosine']>.999,measured
            checks.append(dict(perturbation=perturbation,teacher_micro=micro,**measured))
    # The shortcut must be disabled for every other dataset automatically.
    other=Dataset('paws','gradient_test',tok)
    collect(m,tok,mod,cap,other,list(range(4)),micro=4,keep=False,verify_reconstruction=False)
    assert cache.active is None
    cache.close();cap.close()
    write(EXP/'results/teacher_prefix_audit.json',dict(passed=True,checks=checks,cache_seconds=cache.seconds,cache_bytes=cache.bytes,full_inputs=True,scope='Same cache/teacher batch shape should match; different BF16 batch shapes may round differently. Two LoRA states; nontraining datasets bypass cache. Cache stores residual in original dtype.'))
    log(dict(passed=True,checks=checks))

if __name__=='__main__':main()
