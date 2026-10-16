"""Explicit completion budget for registered development fits after validation.

No policy means the original fitting reserve remains in force. This changes only
when a fit must stop, never its data, optimizer, epochs or checkpoint selection.
"""
from common import *


def registered_fit_cutoff(jobs_path, cache):
    policy_path = EXP/'configs/optional_fit_completion_policy.json'
    if not policy_path.exists():
        return None
    policy = json.loads(policy_path.read_text())
    if str(Path(jobs_path).resolve()) != policy['jobs_path']:
        return None
    assert sha(jobs_path) == policy['jobs_sha256'], 'Registered fitting recipes changed'
    assert cache == policy['cache'] == 'critic_hidden'
    jobs = json.loads(Path(jobs_path).read_text())
    assert {r['name'] for r in jobs} == set(policy['job_names'])
    assert set(policy['job_names']) == {'token_kd128', 'token_kd128_nograd', 'token_kd256'}
    audit_path = EXP/'results/final_evidence_audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['passed'] and audit['timing_references'] == 60
    assert audit['timing_repetitions'] == 3 and audit['flop_profiles'] == 20
    assert audit['workflow_conditions'] == 6 and audit['workflow_repetitions'] == 3
    assert audit['freeze_sha256'] == sha(EXP/'configs/surrogate_final_freeze.json')
    protocol = json.loads((EXP/'configs/protocol.json').read_text())
    cutoff = policy['fit_cutoff_epoch']
    assert policy['registered_epoch'] < cutoff <= protocol['deadline_epoch'] - 2700
    assert cutoff <= protocol['compute_cutoff_epoch'] - 1800
    assert time.time() < cutoff, 'Post-validation fit completion window has closed'
    return dict(cutoff_epoch=cutoff, policy_sha256=sha(policy_path),
                validation_audit_sha256=sha(audit_path),
                scope='Only the three previously registered KD fits; unchanged scientific recipes. Mandatory validation already passed. At least45minutes remain before the authorized study deadline and30minutes before the GPU cutoff.')
