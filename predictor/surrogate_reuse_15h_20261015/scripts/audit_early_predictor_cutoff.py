"""Check the prespecified early-CV intervention against its unchanged prefix."""
from common import *
from safetensors.torch import load_file
import torch

def main():
    rows=[]
    for warm,seed_id in [(301,401),(311,411),(321,421)]:
        root=EXP/'models'/f'mnli__w{warm}_update_state256_cv16_until32_lr0.001_u{seed_id}_earlycutoff'
        reference=EXP/'models'/f'mnli__w{warm}_update_state256_cv16_lr0.001_u{seed_id}_fullpoolconfirm'
        assert json.loads((root/'summary.json').read_text())['complete']
        history=json.loads((root/'history.json').read_text());other=json.loads((reference/'history.json').read_text())
        assert len(history)==len(other)==512
        for a,b in zip(history,other):
            assert all(a[k]==b[k] for k in ['step','indices','positions','learning_rate'])
            assert a['true_gradient_examples']==16
            assert a['predictor_active']==(a['step']<=32)
            if a['step']>32:assert a['prefix_prediction_seconds']==0.
        left=load_file(str(root/'step-32/adapter_model.safetensors'));right=load_file(str(reference/'step-32/adapter_model.safetensors'))
        assert left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
        rows.append(dict(warmup_seed=warm,first32_weights_bitwise_equal=True,sampling_and_lr_identical=True,later_prediction_seconds_zero=True,true_gradient_examples=16*512,intervention_history_sha256=sha(root/'history.json'),reference_history_sha256=sha(reference/'history.json')))
    write(EXP/'results/early_predictor_cutoff_audit.json',dict(passed=True,rows=rows,scope='Exact first32 weights; unchanged sampling/LR; later teacher-only dispatch. Does not assert that the final adapters or downstream quality should agree.'))
    log(dict(passed=True,starts=len(rows)))

if __name__=='__main__':main()
