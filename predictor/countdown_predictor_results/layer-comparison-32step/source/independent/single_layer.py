from predictor import *


def setup(scope='single',adapter=None,rng=101,fresh=False):
    assert scope=='single'
    source=None if fresh else adapter or EXP/'models'/f'teacher_l{LAYER:02d}'/'step-32'
    m,tok,mod=setup_model(adapter=source,layer=LAYER,rng=rng)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad)==114688
    return m,tok,{LAYER:mod}


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
