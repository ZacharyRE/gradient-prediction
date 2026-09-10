"""Bitwise implementation control for the separate anchored trainer at lambda0."""
import torch,difflib
from safetensors.torch import load_file
from common import *
torch.set_num_threads(2)
original=S/'results/training/teacher_all_lr5e5_s43';replay=S/'results/training/teacher_all_kl0_lr5e5_s43'
assert (original/'complete.json').exists() and (replay/'complete.json').exists()
a=json.loads((original/'manifest.json').read_text());b=json.loads((replay/'manifest.json').read_text())
assert a['data_sha']==b['data_sha'] and b['arguments']['kl_coefficient']==0
assert b['forked_original_train_sha256']==sha(S/'scripts/train.py')
assert a['script_sha']==sha(original/'train_source.py')
source_diff=''.join(difflib.unified_diff((original/'train_source.py').read_text().splitlines(True),(S/'scripts/train.py').read_text().splitlines(True),fromfile='original_teacher7_training_source',tofile='current_original_CE_trainer'))
(S/'audits/anchor_zero_legacy_source_diff.txt').write_text(source_diff)
args_a={k:v for k,v in a['arguments'].items() if k!='name'};args_b={k:v for k,v in b['arguments'].items() if k not in ['name','kl_coefficient']}
args_a.setdefault('resume_from',None);assert args_a['resume_from'] is None and args_b['resume_from'] is None;assert args_a==args_b
exposure_equal=json.loads((original/'exposure.json').read_text())==json.loads((replay/'exposure.json').read_text());assert exposure_equal
out=[]
for label in ['initial_adapter','epoch1','epoch2','epoch3','epoch4']:
 p=original/label/'adapter_model.safetensors';q=replay/label/'adapter_model.safetensors';x=load_file(str(p));y=load_file(str(q));assert set(x)==set(y)
 bad=[k for k in x if not torch.equal(x[k],y[k])]
 out.append(dict(checkpoint=label,tensors=len(x),different_tensors=bad,original_sha256=sha(p),zero_kl_sha256=sha(q)))
hist_a=json.loads((original/'history.json').read_text());hist_b=json.loads((replay/'history.json').read_text())
history_equal=[{k:v for k,v in h.items() if k!='time'} for h in hist_a]==[{k:v for k,v in h.items() if k!='time'} for h in hist_b]
passed=all(not r['different_tensors'] for r in out) and exposure_equal and history_equal
write(S/'audits/anchor_zero_replay_result.json',dict(time=time.time(),passed=passed,checkpoints=out,exposure_sequence_equal=exposure_equal,history_except_timestamps_equal=history_equal,config_equal_except_name_and_zero_kl_parameter=True,legacy_resume_default_normalized_to_none=True,original_training_source_sha256=a['script_sha'],current_original_source_sha256=b['forked_original_train_sha256'],source_diff_sha256=sha(S/'audits/anchor_zero_legacy_source_diff.txt'),source_difference='Original teacher7 run predates added optional resume support; both compared runs start from scratch. Actual tensors, exposure and numerical history remain the deciding checks.',scope='Verifies separate trainer reduces exactly to original CE implementation when KL is0; nonzero-KL GPU initial/reference invariance has separate per-run audit.'))
print(json.dumps(dict(passed=passed,history_equal=history_equal,different_tensors=[r['different_tensors'] for r in out])))
assert passed
