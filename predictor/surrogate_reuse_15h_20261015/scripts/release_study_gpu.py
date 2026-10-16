"""Release this study's idle GPU reservation after required validation passes.

Default inspection never signals a process. --release requires the final evidence
audit, takes the stage mutex, and signals only verified study controllers through
pidfds. It refuses to interrupt an active stage or terminate an unrelated process.
"""
from common import *
import argparse
import fcntl
import signal

CONTROLLERS = {
    'run_queue.py', 'stage.py', 'await_surrogate_analysis.py',
    'await_surrogate_audits.py', 'await_selection_pipeline.py',
    'resume_optional_research.py', 'await_diversity.py', 'complete_kd_readout.py',
    'finish_registered_controls.py',
    'await_token_behavior.py',
}
SUPERVISORS = {'monitor_gpu.py', 'reserve_gpu.py'}


def identity(pid):
    proc = Path('/proc') / str(pid)
    try:
        if proc.stat().st_uid != os.getuid():
            return None
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        env = (proc / 'environ').read_bytes().split(b'\0')
        if ('BENCH_RUN_ID=' + EXP.name).encode() not in env:
            legacy_path = EXP / 'provenance/resource_release_legacy_controllers.json'
            legacy = json.loads(legacy_path.read_text())['controllers'] if legacy_path.exists() else []
            if not any(r['pid'] == pid and r['process_start'] == stat[19]
                       and r['uid'] == os.getuid()
                       and r['cmdline_sha256'] == sha(proc / 'cmdline') for r in legacy):
                return None
        argv = [v.decode() for v in (proc / 'cmdline').read_bytes().split(b'\0') if v]
        scripts = [Path(v) for v in argv if v.endswith('.py')]
        eligible = [p for p in scripts if p.parent == EXP / 'scripts'
                    and p.name in CONTROLLERS | SUPERVISORS]
        if len(eligible) != 1:
            return None
        script = eligible[0].name
        if script == 'reserve_gpu.py' and b'BENCH_GPU_RESERVER=1' not in env:
            return None
        return dict(pid=pid, process_start=stat[19], script=script)
    except (OSError, UnicodeError):
        return None


def owned_processes():
    return [r for p in Path('/proc').iterdir() if p.name.isdigit()
            if (r := identity(int(p.name))) is not None]


def gpu_contexts(uuid):
    text = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    return [dict(pid=int(parts[1]), memory_mib=parts[2].strip())
            for line in text.splitlines() if len(parts := line.split(',')) == 3
            and parts[0].strip() == uuid]


def stop_verified(record):
    assert identity(record['pid']) == record, 'Process identity changed before stop'
    fd = os.pidfd_open(record['pid'])
    try:
        assert identity(record['pid']) == record, 'Process identity changed after pidfd open'
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        until = time.monotonic() + 20
        while identity(record['pid']) == record and time.monotonic() < until:
            time.sleep(.2)
        assert identity(record['pid']) != record, 'Owned process did not exit after SIGTERM'
    finally:
        os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', action='store_true')
    args = ap.parse_args()
    cfg = json.loads((EXP / 'configs/protocol.json').read_text())
    holder = json.loads((EXP / 'results/gpu_reservation.json').read_text())
    current = owned_processes()
    if not args.release:
        log(dict(inspection_only=True, owned_controllers=current,
                 gpu_contexts=gpu_contexts(cfg['gpu_uuid']),
                 final_evidence_audit_present=(EXP / 'results/final_evidence_audit.json').exists()))
        return
    audit_path = EXP / 'results/final_evidence_audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['passed'] and audit['timing_references'] == 60
    assert audit['timing_repetitions'] == 3 and audit['workflow_repetitions'] == 3
    lock = (EXP / 'results/stage_execution.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected_holder = dict(pid=holder['pid'], process_start=str(holder['process_start']),
                           script='reserve_gpu.py')
    assert identity(holder['pid']) == expected_holder
    assert all(r['pid'] == holder['pid'] for r in gpu_contexts(cfg['gpu_uuid'])), 'GPU stage/context remains active'
    stopped = []
    # Parent launchers exit first; then remove any newly spawned waiting stage.
    for row in current:
        if row['script'] in CONTROLLERS - {'stage.py'} and identity(row['pid']) == row:
            stop_verified(row); stopped.append(row)
    for row in owned_processes():
        if row['script'] == 'stage.py':
            stop_verified(row); stopped.append(row)
    assert not any(r['script'] in CONTROLLERS for r in owned_processes())
    assert all(r['pid'] == holder['pid'] for r in gpu_contexts(cfg['gpu_uuid']))
    for row in owned_processes():
        if row['script'] == 'monitor_gpu.py':
            stop_verified(row); stopped.append(row)
    stop_verified(expected_holder); stopped.append(expected_holder)
    until = time.monotonic() + 15
    while any(r['pid'] == holder['pid'] for r in gpu_contexts(cfg['gpu_uuid'])) and time.monotonic() < until:
        time.sleep(.5)
    assert not any(r['pid'] == holder['pid'] for r in gpu_contexts(cfg['gpu_uuid']))
    assert not owned_processes()
    receipt = dict(released=True, epoch=time.time(), gpu_index=cfg['gpu_index'],
                   final_evidence_audit_sha256=sha(audit_path),
                   stopped_script_counts={name:sum(r['script']==name for r in stopped)
                                          for name in sorted({r['script'] for r in stopped})},
                   scope='Verified owned controllers and reservation exited; no active study stage was interrupted. PID identity was checked before pidfd-based SIGTERM. Other users may acquire the GPU after release.')
    write(EXP / 'provenance/resource_release_private.json', dict(receipt=receipt, stopped=stopped))
    write(EXP / 'results/resource_release.json', receipt)
    log(receipt)


if __name__ == '__main__':
    main()
