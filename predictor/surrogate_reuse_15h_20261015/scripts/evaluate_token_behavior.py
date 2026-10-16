"""Read existing token critics as causal ABC behavior surrogates on dev only."""
from common import *
import sys

def main():
    plan_path=EXP/'configs/token_behavior_readout_plan.json';plan=json.loads(plan_path.read_text())
    assert plan['split']=='accuracy_dev' and sha(plan['entries'])==plan['entries_sha256']
    assert time.time()<plan['cutoff_epoch']
    from token_critic import TokenCritic
    import evaluate_behavior as evaluator
    import torch
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODELS['1p5b'],local_files_only=True)
    class_tokens=[tok.encode(c,add_special_tokens=False) for c in 'ABC'];assert all(len(t)==1 for t in class_tokens)
    class_tokens=[t[0] for t in class_tokens];audits=[]
    class TokenBehavior(TokenCritic):
        requires_initial_teacher=False
        def __init__(self,**cfg):
            super().__init__(**cfg)
            self.class_positions=[cfg['label_tokens'].index(t) for t in class_tokens]
            self._audited=False
        @property
        def class_tokens(self):return self.label_tokens.new_tensor(class_tokens)
        def features(self,d,A,B,s):return d
        def class_logits(self,d):
            where=(d['label_ids'][...,None]==self.class_tokens).any(-1)
            assert torch.all(where.sum(1)==1)
            positions=where.long().argmax(1)
            yy=self.token_logits(d,d['h'].float())
            return yy[torch.arange(len(positions),device=yy.device),positions][:,self.class_positions],positions
        def forward(self,d):
            logits,positions=self.class_logits(d)
            if not self._audited:
                # Gold identity is absent: the template is always boxed A.
                # Labels used only to locate the common answer position may
                # change to B; future hidden states must not influence it.
                alt=dict(d);alt['label_ids']=d['label_ids'].clone()
                alt['label_ids'][alt['label_ids']==class_tokens[0]]=class_tokens[1]
                alt['h']=d['h'].clone()
                future=torch.arange(d['h'].shape[1],device=d['h'].device)[None,:]>positions[:,None]
                alt['h'][future]=-alt['h'][future]+.123
                changed,_=self.class_logits(alt)
                error=float((changed-logits).abs().max())
                assert torch.allclose(changed,logits,rtol=0,atol=1e-6),error
                audits.append(dict(config=self.cfg,label_and_future_invariance_max_abs=error,passed=True))
                self._audited=True
            return logits
    evaluator.BehaviorSurrogate=TokenBehavior
    original_deadline=evaluator.deadline
    evaluator.deadline=lambda reserve=0:time.time()>=plan['cutoff_epoch'] or original_deadline(reserve)
    predictors=[str(EXP/'models'/f'mnli_critic_{n}/best.pt') for n in plan['predictors']]
    for path in predictors:
        summary=json.loads((Path(path).parent/'summary.json').read_text());assert summary['complete'] and summary['best_sha256']==sha(path)
    class CheckpointView:
        def __getattr__(self,name):return getattr(torch,name)
        def load(self,path,*args,**kwargs):
            obj=torch.load(path,*args,**kwargs)
            if str(path) in predictors:
                assert 'state' in obj and 'state_dict' not in obj
                obj=dict(obj,state_dict=obj['state'])
            return obj
    # Existing critic checkpoints name their unchanged tensors "state"; the
    # behavior reader expects "state_dict". Normalize only this serialization
    # field, without saving or modifying the checkpoint/model computation.
    evaluator.torch=CheckpointView()
    write(EXP/'provenance/token_behavior_predictor_freeze.json',dict(epoch=time.time(),plan_sha256=sha(plan_path),predictor_sha256={p:sha(p) for p in predictors},scope='Existing completed KD checkpoints fixed before this development behavior readout. No fitting or final-heldout evaluation.'))
    sys.argv=['evaluate_behavior','--entries',plan['entries'],'--predictors',*predictors,'--name',plan['name'],'--split','accuracy_dev','--teacher']
    evaluator.main()
    assert len(audits)==len(predictors)
    write(EXP/'results/token_behavior_input_audit.json',dict(passed=True,audits=audits,full_original_prompts=True,gold_class_never_enters_model_input=True,scope='Strict existing checkpoint loads; fixed common answer template; first batch per model is invariant to label-identity and future-hidden changes. Output is forced-formatABC, not free-generation behavior. The one-time checks are included in the first small-model timer.'))

if __name__=='__main__':main()
