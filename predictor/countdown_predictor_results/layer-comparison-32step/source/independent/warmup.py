"""Original Countdown2 warmup, applied to one independently initialized layer."""
from single_layer import *

def main():
    out=EXP/'models'/f'teacher_l{LAYER:02d}';out.mkdir(exist_ok=False)
    m,tok,mods=setup(fresh=True,rng=101)
    params=[w for mod in mods.values() for w in weights(mod)[:2]]
    from safetensors.torch import load_file
    original=load_file(str(EXP/'provenance/original_layer8_step0/adapter_model.safetensors'))
    for factor,w in zip(['A','B'],params):
        key=next(k for k in original if f'lora_{factor}.weight' in k)
        assert torch.equal(w.detach().cpu(),original[key]), 'Single-layer initialization no longer matches original seed101'
    assert torch.count_nonzero(params[1])==0
    before=fingerprints(m);initial={l:[w.detach().clone() for w in weights(mod)[:2]] for l,mod in mods.items()}
    ds=CompletionDataset(data_path('warmup'),tok,m.config.max_position_embeddings)
    opt=torch.optim.AdamW(params,lr=5e-4,weight_decay=.01)
    cfg=json.loads((EXP/'configs/experiment.json').read_text())['warmup']
    write(out/'config.json',dict(**cfg,layers=list(mods),input_truncation=False,trainable_parameters=sum(w.numel() for w in params),data_sha256=hashlib.sha256(data_path('warmup').read_bytes()).hexdigest()))
    m.save_pretrained(out/'step-0');hist=[];order=[];start=time.time()
    for step in range(1,33):
        if len(order)<32:order=list(np.random.permutation(len(ds)))
        ix,order=order[:32],order[32:];opt.zero_grad(set_to_none=True)
        denom=sum(sum(t!=-100 for t in ds[i]['labels'][1:]) for i in ix);loss_sum=0
        for j in range(0,len(ix),8):
            bb=batch(ds,ix[j:j+8],tok);n=int((bb['labels'][:,1:]!=-100).sum())
            loss=forward(m,bb).loss;(loss*n/denom).backward();loss_sum+=float(loss.detach())*n
        norm=nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
        row=dict(step=step,loss=loss_sum/denom,grad_norm=float(norm),sample_indices=[int(i) for i in ix],seconds=time.time()-start)
        hist.append(row);write(out/'history.json',hist);print(json.dumps(row),flush=True)
    assert before==fingerprints(m)
    delta={str(l):{f:float((w-old).norm()) for f,w,old in zip(['A','B'],weights(mod)[:2],initial[l])} for l,mod in mods.items()}
    assert all(v>0 for d in delta.values() for v in d.values())
    m.save_pretrained(out/'step-32')
    torch.save(opt.state_dict(),out/'optimizer-final.pt')
    write(out/'audit.json',dict(complete=True,base_unchanged=True,layers=list(mods),steps=32,parameter_change_norms=delta,elapsed=time.time()-start))
if __name__=='__main__':main()
