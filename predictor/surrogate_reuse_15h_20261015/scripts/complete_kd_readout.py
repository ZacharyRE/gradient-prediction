"""Complete only the nine registered KD readouts after required validation.

The original training implementation stays byte-identical. This wrapper replaces
its obsolete90-minute method-search reserve with an explicit45-minute completion
reserve, only for pinned jobs after all three registered fits finish. The actual
user deadline and stage GPU cutoff remain enforced. No final-validation flag is
used to relabel these exploratory development rollouts.
"""
from common import *
import sys

def completion_deadline(now,user_deadline,cutoff,reserve=0):
    assert cutoff<=user_deadline-2700
    return now>=cutoff or now+(0 if reserve==5400 else reserve)>=user_deadline

def policy():
    p=EXP/'configs/kd_readout_completion_policy.json';v=json.loads(p.read_text())
    assert sha(EXP/'configs/distill_rollout_jobs.json')==v['jobs_sha256']
    assert sha(EXP/'configs/distill_fit_jobs.json')==v['fit_jobs_sha256']
    audit=EXP/'provenance/optional_fit_validation_audit.json'
    assert sha(audit)==v['validation_audit_sha256']
    a=json.loads(audit.read_text());assert a['passed'] and a['workflow_repetitions']==3 and a['timing_references']==60
    assert a['freeze_sha256']==sha(EXP/'configs/surrogate_final_freeze.json')
    cfg=json.loads((EXP/'configs/protocol.json').read_text())
    assert v['cutoff_epoch']<=cfg['deadline_epoch']-2700 and v['cutoff_epoch']<=cfg['compute_cutoff_epoch']-1800
    assert v['last_launch_epoch']<=v['cutoff_epoch']-900
    assert os.environ.get('BENCH_FINAL_VALIDATION')!='1'
    return v,sha(p),cfg

def train():
    v,policy_sha,cfg=policy()
    assert time.time()<v['cutoff_epoch']
    frozen=json.loads((EXP/'provenance/kd_readout_freeze.json').read_text())
    assert frozen['policy_sha256']==policy_sha
    for p,want in frozen['predictor_sha256'].items():assert sha(p)==want
    import reuse_train
    assert sha(reuse_train.__file__)==v['training_source_sha256']
    # Only the bookkeeping time check changes. All model/sampling/update code
    # executes through the pinned original main function.
    reuse_train.deadline=lambda reserve=0:completion_deadline(time.time(),cfg['deadline_epoch'],v['cutoff_epoch'],reserve)
    sys.argv=['reuse_train','--warmup-seed','301','--jobs',str(EXP/'configs/distill_rollout_jobs.json'),'--suffix','_latekd']
    reuse_train.main()

def main():
    if sys.argv[1:]==['--train']:train();return
    assert not sys.argv[1:]
    v,policy_sha,cfg=policy();fit=EXP/'results/stage_fit_distill_critic.json'
    while not fit.exists():
        if time.time()>=v['last_launch_epoch']:
            write(EXP/'results/kd_readout_completion.json',dict(complete=False,reason='Registered readout launch cutoff reached',epoch=time.time()));return
        time.sleep(5)
    assert json.loads(fit.read_text())['returncode']==0
    assert time.time()<v['last_launch_epoch']
    jobs=json.loads((EXP/'configs/distill_rollout_jobs.json').read_text());assert len(jobs)==9
    predictors=sorted({r['predictor_path'] for r in jobs});assert len(predictors)==3
    identities={}
    for p in predictors:
        summary=json.loads((Path(p).parent/'summary.json').read_text());assert summary['complete'] and summary['best_sha256']==sha(p)
        identities[p]=sha(p)
    freeze=dict(epoch=time.time(),policy_sha256=policy_sha,predictor_sha256=identities,
                jobs_sha256=v['jobs_sha256'],scope='Three completed KD predictors frozen before their registered development rollouts/readouts; no heldout-method promotion.')
    path=EXP/'provenance/kd_readout_freeze.json'
    if path.exists():assert json.loads(path.read_text())['predictor_sha256']==identities
    else:write(path,freeze)
    stages=[('late_kd_rollout',[sys.executable,str(EXP/'scripts/complete_kd_readout.py'),'--train']),
            ('late_kd_eval',[sys.executable,str(EXP/'scripts/evaluate.py'),'--model','1p5b','--entries',str(EXP/'results/entries_mnli_w301_update_u401_latekd.json'),'--name','late_kd','--split','accuracy_dev','--max-new-tokens','256'])]
    for name,args in stages:
        assert time.time()<v['cutoff_epoch']
        receipt=EXP/'results'/f'stage_{name}.json'
        if receipt.exists() and json.loads(receipt.read_text())['returncode']==0:continue
        subprocess.run([sys.executable,str(EXP/'scripts/stage.py'),'--name',name,*args],check=True,cwd=ROOT)
    write(EXP/'results/kd_readout_completion.json',dict(complete=True,epoch=time.time(),policy_sha256=policy_sha,freeze_sha256=sha(path),scope='Nine unchanged development recipes and their downstream readouts completed under the explicit post-validation reserve.'))

if __name__=='__main__':main()
