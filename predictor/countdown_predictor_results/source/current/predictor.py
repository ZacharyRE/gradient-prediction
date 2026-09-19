from core import *

class HiddenPredictor(nn.Module):
    def __init__(self,source='y',aux='mask_pos',width=512,depth=2,architecture='sequence',dim=896):
        super().__init__();self.source=source;self.aux=aux;self.architecture=architecture
        self.register_buffer('target_scale',torch.tensor(1.))
        din=dim*(2 if source=='xy' else 1)
        # Preserve raw scale as well as normalized features; same inputs/parameters across aux ablations.
        self.norm=nn.LayerNorm(din);self.proj=nn.Linear(din,width)
        self.extra=nn.Linear(5,width,bias=False)
        if architecture=='sequence':
            self.body=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,8,width*2,0.,'gelu',
                batch_first=True,norm_first=True),depth,enable_nested_tensor=False)
        else:self.body=nn.Sequential(*sum([[nn.Linear(width,width),nn.GELU()] for _ in range(depth)],[]))
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,dim))
    def forward(self,x,y,active,pos,valid):
        h=x if self.source=='x' else y if self.source in ['y','h'] else torch.cat([x,y],-1)
        h=h.float();rms=h.square().mean(-1).clamp_min(1e-12).sqrt().log()
        mask=active.float() if 'mask' in self.aux else torch.zeros_like(pos)
        p=pos if 'pos' in self.aux else torch.zeros_like(pos)
        extras=torch.stack([rms,mask,p,p*p,torch.sin(torch.pi*p)],-1)
        h=self.proj(self.norm(h))+self.extra(extras)
        if self.architecture=='sequence':h=self.body(h,src_key_padding_mask=~valid)
        else:h=self.body(h)
        return self.head(h)*self.target_scale*valid[...,None]

def load_predictor(path):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    p=HiddenPredictor(**obj['config']).cuda();p.load_state_dict(obj['state']);p.eval();return p,obj

def load_samples(cache,split):
    out=[]
    for f in sorted((Path(cache)/split).glob('shard*.pt')):out.extend(torch.load(f,map_location='cpu',weights_only=False))
    assert out,(cache,split)
    return out

def pack(samples,device='cuda'):
    n=len(samples);L=max(len(s['x']) for s in samples);D=samples[0]['x'].shape[-1]
    x=torch.zeros(n,L,D,device=device,dtype=torch.bfloat16);y=torch.zeros_like(x);g=torch.zeros(n,L,D,device=device)
    active=torch.zeros(n,L,device=device,dtype=torch.bool);valid=torch.zeros_like(active);pos=torch.zeros(n,L,device=device)
    counts=torch.tensor([s['n'] for s in samples],device=device)
    h=torch.zeros(n,L,D,device=device) if 'h' in samples[0] else None
    for i,s in enumerate(samples):
        l=len(s['x']);x[i,:l]=s['x'].to(device);y[i,:l]=s['y'].to(device);g[i,:l]=s['g'].to(device)
        active[i,:l]=s['active'].to(device);valid[i,:l]=True;pos[i,:l]=torch.arange(l,device=device)/max(1,l-1)
        if h is not None:h[i,:l]=s['h'].to(device)
    return dict(x=x,y=y,g=g,active=active,valid=valid,pos=pos,counts=counts,**({'h':h} if h is not None else {}))

def infer(p,d,ix):return p(d['x'][ix],d['h' if p.source=='h' else 'y'][ix],d['active'][ix],d['pos'][ix],d['valid'][ix])
def braw(x,g,A,B,scale):
    x=x.float();g=g.float()
    return scale*(g@B).transpose(-1,-2)@x,scale*g.transpose(-1,-2)@(x@A.transpose(-1,-2))

def metrics(p,t):
    p=p.flatten(1).float();t=t.flatten(1).float();tn=t.norm(dim=1);pn=p.norm(dim=1)
    return dict(cosine=F.cosine_similarity(p,t,dim=1),relative_l2=(p-t).norm(dim=1)/tn.clamp_min(1e-12),norm_ratio=pn/tn.clamp_min(1e-12))

def summarize(v):return {k:dict(mean=float(a.mean()),median=float(a.median()),std=float(a.std())) for k,a in v.items()}

@torch.no_grad()
def evaluate(p,d,meta,return_vectors=False):
    p.eval();A=meta['A'].cuda();B=meta['B'].cuda();scale=meta['scale'];allp=[];allt=[];act=[];regional={'prompt':[],'supervised':[]};gp=[];gt=[]
    for ix in torch.arange(len(d['x']),device='cuda').split(16):
        g=infer(p,d,ix);t=d['g'][ix];act.append(metrics(g,t))
        for name,mask in [('prompt',d['valid'][ix]&~d['active'][ix]),('supervised',d['active'][ix])]:
            regional[name].append(metrics(g*mask[...,None],t*mask[...,None]))
        pp=braw(d['x'][ix],g,A,B,scale);tt=braw(d['x'][ix],t,A,B,scale)
        allp.append(torch.cat([v.flatten(1) for v in pp],1)/d['counts'][ix,None]);allt.append(torch.cat([v.flatten(1) for v in tt],1)/d['counts'][ix,None])
        gp.append(torch.stack([(g*t).sum(),g.square().sum(),t.square().sum(),(g-t).square().sum()]))
    pp=torch.cat(allp);tt=torch.cat(allt);total=torch.stack(gp).sum(0)
    merge=lambda rows:{k:torch.cat([r[k] for r in rows]) for k in rows[0]}
    per=dict(activation=merge(act),lora=metrics(pp,tt),A=metrics(pp[:,:A.numel()],tt[:,:A.numel()]),B=metrics(pp[:,A.numel():],tt[:,A.numel():]))
    per.update({k:merge(v) for k,v in regional.items()})
    result={k:summarize(v) for k,v in per.items()}
    result.update(n=len(pp),activation_global=dict(cosine=float(total[0]/(total[1]*total[2]).sqrt()),relative_l2=float((total[3]/total[2]).sqrt()),norm_ratio=float((total[1]/total[2]).sqrt())),
        mean_gradient=measures((pp*d['counts'][:,None]).sum(0)/d['counts'].sum(),(tt*d['counts'][:,None]).sum(0)/d['counts'].sum()),
        selection_score=float((per['A']['relative_l2'].square().mean()+per['B']['relative_l2'].square().mean())/2))
    if return_vectors:return result,pp.cpu(),tt.cpu(),{k:{kk:vv.cpu() for kk,vv in v.items()} for k,v in per.items()}
    return result
