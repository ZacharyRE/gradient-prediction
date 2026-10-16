from research import *
from fast_backend import configure

def main():
    task='paws';warm=json.loads((EXP/'configs'/f'{task}_warmup.json').read_text())
    m,tok,mod=setup('1p5b',warm['adapter']);A,B,s=weights(mod);cap=Capture(mod)
    ds=Dataset(task,'update',tok);records=[]
    head=m.get_base_model().lm_head;original_head=head.forward
    def fp32_head(x):
        with torch.autocast('cuda',enabled=False):return F.linear(x.float(),head.weight.float())
    def full_fp32(m,b):
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with torch.autocast('cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):return m(**b,use_cache=False)
    def sparse_fp32(m,b):
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from types import SimpleNamespace
        with torch.autocast('cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            base=m.get_base_model();out=base.model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False,return_dict=True)
            labels=b['labels'][:,1:];valid=labels!=-100
            logits=base.lm_head(out.last_hidden_state[:,:-1][valid])
            return SimpleNamespace(loss=F.cross_entropy(logits,labels[valid]))
    for precision in ['normal_bf16','fp32_head_no_tf32','full_fp32_no_tf32']:
        if precision=='fp32_head_no_tf32':
            head.forward=fp32_head;torch.backends.cuda.matmul.allow_tf32=False
        for ids in [[0,1,2,3],[16,21,25,29],[100,111,130,190]]:
            result=[]
            for use_fast in [False,True]:
                configure(use_fast)
                import predictor_core as pc
                if precision=='full_fp32_no_tf32':pc.forward=sparse_fp32 if use_fast else full_fp32
                t=clock();rows,truth,recon=collect(m,tok,mod,cap,ds,ids,keep=True)
                elapsed=clock()-t
                with torch.no_grad():loss=float(pc.forward(m,batch(ds,ids,tok)).loss)
                result.append(dict(loss=loss,seconds=elapsed,truth=[x.clone() for x in truth],reconstruction=recon))
            comparison=measures(flat(result[1]['truth']),flat(result[0]['truth']))
            delta=abs(result[1]['loss']-result[0]['loss'])
            for r in result:r.pop('truth')
            records.append(dict(precision=precision,indices=ids,original=result[0],supervised_logits_only=result[1],loss_absolute_difference=delta,gradient=comparison))
            log(records[-1])
    head.forward=original_head;torch.backends.cuda.matmul.allow_tf32=True
    record=dict(records=records,reason_for_extended_audit='First BF16 check had 1.33% gradient difference despite matching loss; isolate GEMM shape-dependent precision using an FP32 vocabulary head.',passed=False)
    write(EXP/'results/fast_backend_audit.json',record)
    assert max(r['loss_absolute_difference'] for r in records)<1e-5
    assert max(r['gradient']['relative_l2'] for r in records if r['precision']=='full_fp32_no_tf32')<.0001
    assert max(r['gradient']['relative_l2'] for r in records)<.05
    record.update(passed=True,limitation='Same mathematical supervised-token loss; normal BF16 kernels can differ numerically with GEMM shape. Both oracle and predictor must be evaluated under this backend; no bitwise-equivalence claim.')
    write(EXP/'results/fast_backend_audit.json',record);cap.close()

if __name__=='__main__':main()
