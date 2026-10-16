"""Finish pinned, previously queued native controls after mandatory validation.

Only admission timing changes; the original training source remains untouched.
New rollouts stop55minutes before the user deadline, leaving ten further minutes
for readouts and45minutes for reporting/cleanup. This is development, not a
final-validation replay, and does not set BENCH_FINAL_VALIDATION.
"""
from common import *
from complete_kd_readout import completion_deadline
import sys

def checked_plan(name):
    path=EXP/'configs/registered_controls_completion_policy.json';policy=json.loads(path.read_text())
    assert name in policy['stages'],'Unregistered control stage'
    audit=EXP/'provenance/optional_fit_validation_audit.json'
    assert sha(audit)==policy['validation_audit_sha256'] and json.loads(audit.read_text())['passed']
    cfg=json.loads((EXP/'configs/protocol.json').read_text())
    assert policy['training_cutoff_epoch']<=cfg['deadline_epoch']-3300
    assert policy['readout_cutoff_epoch']<=cfg['deadline_epoch']-2700
    assert policy['training_cutoff_epoch']<=policy['readout_cutoff_epoch']-600
    assert os.environ.get('BENCH_FINAL_VALIDATION')!='1'
    item=policy['stages'][name];argv=item['training_argv']
    assert argv[argv.index('--warmup-seed')+1]=='301'
    assert '--teacher-backend' not in argv
    jobs=Path(argv[argv.index('--jobs')+1]);assert sha(jobs)==item['jobs_sha256']
    assert sha(EXP/'configs/mnli_warmup_w301.json')==policy['warmup_sha256']
    for split,want in policy['data_sha256'].items():assert sha(data('mnli',split))==want
    return policy,item,cfg

def main():
    assert len(sys.argv)==2
    name=sys.argv[1];policy,item,cfg=checked_plan(name)
    assert time.time()<policy['training_cutoff_epoch']
    import reuse_train
    assert sha(reuse_train.__file__)==policy['training_source_sha256']
    reuse_train.deadline=lambda reserve=0:completion_deadline(time.time(),cfg['deadline_epoch'],policy['training_cutoff_epoch'],reserve)
    sys.argv=['reuse_train',*item['training_argv']]
    reuse_train.main()

if __name__=='__main__':main()
