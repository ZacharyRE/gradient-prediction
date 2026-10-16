"""Expose selected checkpoint architectures without publishing their tensors."""
from common import *
import torch
from predictor_core import Predictor
from state_predictor import StatePredictor
from behavior_surrogate import AnchoredBehaviorSurrogate


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Metadata inspection must stay on CPU'
    frozen=json.loads((EXP/'configs/surrogate_final_freeze.json').read_text())
    gradient={r['entry']['predictor_path'] for r in frozen['entries'].values() if r['entry'].get('predictor_path')}
    paths=sorted(gradient|set(frozen['behavior_predictors']));rows=[]
    for path in paths:
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        cfg=checkpoint['config'];kind=cfg.get('kind','gradient_predictor')
        factory={'gradient_predictor':Predictor,'state_predictor':StatePredictor,'anchored_behavior':AnchoredBehaviorSurrogate}[kind]
        model=factory(**cfg)
        model.load_state_dict(checkpoint['state_dict'] if kind=='anchored_behavior' else checkpoint['state'],strict=True)
        rows.append(dict(name=Path(path).parent.name,role='gradient' if path in gradient else 'behavior',kind=kind,config=cfg,parameter_count=sum(p.numel() for p in model.parameters()),buffer_elements=sum(b.numel() for b in model.buffers()),checkpoint_bytes=Path(path).stat().st_size,checkpoint_sha256=sha(path),strict_state_load_passed=True))
        del model,checkpoint
    result=dict(complete=True,freeze_sha256=sha(EXP/'configs/surrogate_final_freeze.json'),models=rows,scope='CPU constructor and strict checkpoint-state validation. Parameter counts exclude buffers; checkpoint bytes include metadata/buffers and are not inference memory measurements. No checkpoint tensors are exported.',source_sha256=source_signature(__file__))
    write(EXP/'results/selected_predictor_architectures.json',result)
    log(rows)


if __name__=='__main__':
    main()
