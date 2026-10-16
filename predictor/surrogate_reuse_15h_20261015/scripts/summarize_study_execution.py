"""Summarize recorded execution intervals without calling them GPU kernel time."""
from common import *

def main():
    records={}
    def keep(r):
        if not all(k in r for k in ['name','started','finished','returncode','reason']):return
        if not isinstance(r['started'],(int,float)) or not isinstance(r['finished'],(int,float)):return
        assert r['finished']>=r['started']
        key=(r['name'],r['started'],r['finished']);records[key]={k:r[k] for k in ['name','started','finished','returncode','reason']}
    attempts=EXP/'results/stage_attempts.jsonl'
    if attempts.exists():
        for row in read(attempts):keep(row)
    for path in (EXP/'results').glob('stage_*.json'):
        try:keep(json.loads(path.read_text()))
        except (ValueError,TypeError):continue
    for path in (EXP/'logs').glob('queue*.log'):
        for line in path.read_text(errors='replace').splitlines():
            if not line.startswith('{"name":'):continue
            try:keep(json.loads(line))
            except (ValueError,TypeError):continue
    intervals=sorted((r['started'],r['finished']) for r in records.values());merged=[]
    for start,finish in intervals:
        if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],finish)
        else:merged.append([start,finish])
    total=sum(b-a for a,b in merged);deadline_cfg=json.loads((EXP/'configs/run_deadline.json').read_text());plan=json.loads((EXP/'configs/surrogate_final_recipe_plan.json').read_text())
    final_progress={g['name']:dict(completed_manifests=sum(Path(f).exists() for f in g['sources']),required_manifests=len(g['sources'])) for g in plan['entry_groups']}
    result=dict(created_epoch=time.time(),recorded_stage_attempts=len(records),successful_stage_attempts=sum(r['returncode']==0 for r in records.values()),nonzero_stage_attempts=sum(r['returncode']!=0 for r in records.values()),recorded_stage_active_wall_seconds=total,final_training_manifest_completion=final_progress,authorized_deadline_epoch=deadline_cfg['deadline_epoch'],scope='Union of available stage start/finish receipts, including process setup, evaluation, fitting, failed attempts and profiling. This is not GPU kernel time, not total reservation time and not total FLOPs. An interrupted attempt lacking a completion receipt may be absent; original queue logs and local provenance remain the detailed record. Operational PID and foreign-process information is not exported.',source_sha256=source_signature(__file__))
    protocol=json.loads((EXP/'configs/protocol.json').read_text())
    result['authorized_seconds']=protocol['deadline_epoch']-protocol['start_epoch']
    result['study_elapsed_seconds_at_summary']=time.time()-protocol['start_epoch']
    release_path=EXP/'results/resource_release.json'
    if release_path.exists():
        result['resource_release']=json.loads(release_path.read_text())
    resource_path=EXP/'results/resource_observation_audit.json'
    if resource_path.exists():
        resource=json.loads(resource_path.read_text())
        result['sampled_resource_accounting']={k:resource[k] for k in ['updated','gpu_index','observation_count','observed_only_receipted_owned_processes','scope']}
        result['sampled_resource_accounting']['unmatched_observation_count']=len(resource['unmatched_process_observations'])
    write(EXP/'results/study_execution_summary.json',result);log(result)

if __name__=='__main__':main()
