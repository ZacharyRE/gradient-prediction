"""Stop only the queue after its already-running last bounded pilot completes."""
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
name='teacher_common_lr5e5_s43'
planpath=S/'audits/exploration_pause_after_matched_sources_plan.json'
assert not planpath.exists() and not (S/'results/training'/name/'manifest.json').exists()
write(planpath,dict(time=time.time(),last_training_job=name,script_sha256=sha(__file__),
    queue_sha256=sha(S/'results/train_queue.json'),
    rationale='Set STOP_TRAINER only once the last already-planned exact-question generator-source pilot has started. '
              'The current training child completes all four epochs normally, then its unchanged queue exits. '
              'Reserve GPU1 for candidate replication or final native evaluation after fixed pilot review. '
              'No training process is signaled and no current training is interrupted.'))
while time.time()<DEADLINE and not (S/'STOP_DIAGNOSTICS').exists():
    path=S/'results/training'/name/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        assert manifest['arguments']['name']==name and manifest['arguments']['stop']==4
        assert manifest['arguments']['objective']=='token_mean' and manifest['arguments']['seed']==43
        assert manifest['script_sha']==sha(S/'scripts/train.py')
        (S/'STOP_TRAINER').touch()
        write(S/'audits/exploration_pause_after_matched_sources_dispatch.json',dict(
            time=time.time(),training_manifest_sha256=sha(path),name=name,
            already_complete=(path.parent/'complete.json').exists(),
            action='STOP_TRAINER set; current training finishes naturally; no process signaled.'))
        print('STOP_TRAINER set after last bounded pilot started; current child continues to completion.',flush=True)
        break
    time.sleep(3)
