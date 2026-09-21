from predictor import *


def setup(scope,adapter=None,rng=101):
    if scope=='single':
        m,tok,mod=setup_model(adapter or EXP.parent.parent/'predictor_countdown2/models/teacher/step-32',rng=rng)
        return m,tok,{8:mod}
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import get_peft_model,LoraConfig,PeftModel
    seed(rng);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).cuda()
    if adapter:m=PeftModel.from_pretrained(m,str(adapter),is_trainable=True)
    else:
        m=get_peft_model(m,LoraConfig(r=64,lora_alpha=64,lora_dropout=0,target_modules=['o_proj'],task_type='CAUSAL_LM',bias='none'))
        from safetensors.torch import load_file
        old=load_file(str(EXP.parent.parent/'predictor_countdown2/models/teacher/step-32/adapter_model.safetensors'))
        params=dict(m.named_parameters())
        with torch.no_grad():
            for name,v in old.items():
                name=name.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight')
                assert name in params;params[name].copy_(v)
    m.config.use_cache=False;m.eval()
    mods={int(n.split('.layers.')[1].split('.')[0]):mod for n,mod in m.named_modules() if hasattr(mod,'lora_A')}
    assert sorted(mods)==list(range(24))
    assert all(('lora_' in n)==p.requires_grad for n,p in m.named_parameters())
    return m,tok,mods


class Captures:
    def __init__(self,mods):
        self.x={};self.y={};self.handles=[]
        for layer,mod in mods.items():
            def hook(m,args,out,layer=layer):
                self.x[layer]=args[0].detach();self.y[layer]=out
                if out.requires_grad:out.retain_grad()
            self.handles.append(mod.register_forward_hook(hook))
    def close(self):
        for h in self.handles:h.remove()


def collect(m,tok,mods,cap,ds,ids):
    samples={l:[] for l in mods};true={l:[torch.zeros_like(w) for w in weights(mod)[:2]] for l,mod in mods.items()}
    for st in range(0,len(ids),8):
        chunk=ids[st:st+8];bb=batch(ds,chunk,tok);m.zero_grad(set_to_none=True)
        n=int((bb['labels'][:,1:]!=-100).sum());(forward(m,bb).loss*n).backward()
        for layer,mod in mods.items():
            A,B,scale=weights(mod);true[layer][0].add_(A.grad);true[layer][1].add_(B.grad)
            xx=cap.x[layer];yy=cap.y[layer].detach();gg=cap.y[layer].grad.detach()
            for j,i in enumerate(chunk):
                length=len(ds[i]['input_ids']);active=torch.tensor(ds[i]['labels'][1:]+[-100])>=0
                samples[layer].append(dict(index=int(i),x=xx[j,:length].cpu().bfloat16(),y=yy[j,:length].cpu().bfloat16(),g=gg[j,:length].cpu().float(),active=active,n=int(active.sum())))
    m.zero_grad(set_to_none=True)
    return samples,true


def fingerprints(m):
    return {n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in m.named_parameters() if not p.requires_grad}


def pred(p,d):
    return torch.cat([infer(p,d,ii) for ii in torch.arange(len(d['x']),device='cuda').split(8)])


def diagnose(p,d,A,B,scale):
    with torch.no_grad():
        g=pred(p,d);pg=braw(d['x'],g,A.detach(),B.detach(),scale);tg=braw(d['x'],d['g'],A.detach(),B.detach(),scale)
        pp=torch.cat([x.flatten(1) for x in pg],1);tt=torch.cat([x.flatten(1) for x in tg],1)
        row=dict(activation={k:float(v.mean()) for k,v in metrics(g,d['g']).items()},lora={k:float(v.mean()) for k,v in metrics(pp,tt).items()},batch_lora=measures(pp.sum(0),tt.sum(0)))
        ind=sum(float(((a-b).square().sum((1,2))/b.square().sum((1,2)).clamp_min(1e-12)).mean()) for a,b in zip(pg,tg))/2
        row['selection_score']=ind+row['batch_lora']['relative_l2']**2
    return row
