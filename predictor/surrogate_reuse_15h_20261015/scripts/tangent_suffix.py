"""Local surrogate of the frozen suffix's ABC logit differences.

Linearize at an actual teacher state with two suffix VJPs. Transport through
current LoRA product B A exactly. This is an approximation in activation space,
not a neural-network predictor fitted on evaluation labels.
"""
from predictor_core import *
from updates_legacy import clock
from torch.nn.attention import sdpa_kernel,SDPBackend

class TangentSuffix:
    def __init__(self,m,tok,mod,cap,ds,cache,micro=16):
        self.rows={};self.class_tokens=torch.tensor([tok.encode(c,add_special_tokens=False)[0] for c in 'ABC'],device='cuda')
        A,B,s=weights(mod);self.A0=A.detach().clone();self.B0=B.detach().clone();self.scale=s;self.ds=ds;self.cache=cache
        head=m.get_base_model().lm_head.weight[self.class_tokens].detach();tick=clock();self.backwards=0
        for ids in chunks(ds,list(range(len(ds))),max_n=micro,token_budget=8192):
            bb=batch_current(ds,ids,tok);m.zero_grad(set_to_none=True)
            shifted=bb['labels'][:,1:];class_mask=(shifted[...,None]==self.class_tokens[None,None,:]);assert torch.all(class_mask.any(-1).sum(1)==1)
            where=class_mask.any(-1).long().argmax(1);classes=class_mask.long().sum(1).argmax(1)
            with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.MATH):
                hidden=m.get_base_model().model(input_ids=bb['input_ids'],attention_mask=bb['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
                logits=hidden[torch.arange(len(ids),device='cuda'),where]@head.T
            differences=logits.float()[:,:2]-logits.float()[:,2,None]
            jac=[]
            for c in range(2):jac.append(torch.autograd.grad(differences[:,c].sum(),cap.y,retain_graph=c==0)[0].detach());self.backwards+=len(ids)
            for j,i in enumerate(ids):
                L=len(ds[i]['input_ids']);self.rows[i]=dict(jac=torch.stack([v[j,:L] for v in jac]).bfloat16().clone(),logits=differences[j].detach().clone(),cls=int(classes[j]),length=L)
            del hidden,logits,differences,jac,bb;m.zero_grad(set_to_none=True)
        self.seconds=clock()-tick;self.bytes=sum(r['jac'].numel()*r['jac'].element_size() for r in self.rows.values())

    @torch.no_grad()
    def predict(self,ids,mod):
        A,B,s=weights(mod);pa=torch.zeros(len(ids),*A.shape,device=A.device);pb=torch.zeros(len(ids),*B.shape,device=B.device);loss=0.
        for st in range(0,len(ids),16):
            selected=ids[st:st+16];L=max(self.rows[i]['length'] for i in selected);x=torch.zeros(len(selected),L,A.shape[1],device=A.device,dtype=torch.float32);J=torch.zeros(len(selected),2,L,B.shape[0],device=A.device)
            for j,i in enumerate(selected):
                n=self.rows[i]['length'];x[j,:n]=self.cache.rows[i]['x'];J[j,:,:n]=self.rows[i]['jac']
            delta=s*((x@A.T)@B.T-(x@self.A0.T)@self.B0.T)
            diff=torch.stack([self.rows[i]['logits'] for i in selected])+torch.einsum('bcld,bld->bc',J,delta)
            logits=torch.cat([diff,torch.zeros(len(selected),1,device=A.device)],1);classes=torch.tensor([self.rows[i]['cls'] for i in selected],device=A.device)
            residual=logits.softmax(-1)-F.one_hot(classes,3);g=torch.einsum('bc,bcld->bld',residual[:,:2],J)
            ga,gb=raw(x,g,A,B,s);pa[st:st+len(selected)]=ga;pb[st:st+len(selected)]=gb
            loss+=float(F.cross_entropy(logits,classes,reduction='sum'))
        return (pa,pb),loss/len(ids)

def batch_current(ds,ids,tok):
    # Resolve dynamically, because TeacherPrefixCache installs this wrapper.
    import predictor_core as pc
    return pc.batch(ds,ids,tok)

def exact_class_gradients(m,tok,mod,cap,ds,ids,class_tokens,micro=32):
    A,B,s=weights(mod);truth=[torch.zeros_like(A),torch.zeros_like(B)];loss_sum=0.
    head=m.get_base_model().lm_head.weight[class_tokens].detach()
    for ix in chunks(ds,ids,max_n=micro,token_budget=8192):
        bb=batch_current(ds,ix,tok);m.zero_grad(set_to_none=True);shifted=bb['labels'][:,1:];mask=shifted[...,None]==class_tokens[None,None,:];assert torch.all(mask.any(-1).sum(1)==1)
        where=mask.any(-1).long().argmax(1);classes=mask.long().sum(1).argmax(1)
        with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.MATH):
            hidden=m.get_base_model().model(input_ids=bb['input_ids'],attention_mask=bb['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
            logits=hidden[torch.arange(len(ix),device='cuda'),where]@head.T
            loss=F.cross_entropy(logits.float(),classes,reduction='sum')
        loss.backward();loss_sum+=float(loss.detach());truth[0]+=A.grad;truth[1]+=B.grad
    m.zero_grad(set_to_none=True);return truth,loss_sum
