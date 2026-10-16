"""Collect only completed registered controls that lack a full dev readout."""
from common import *

def main():
    policy=json.loads((EXP/'configs/registered_controls_completion_policy.json').read_text())
    covered=set();n=len(read(data('mnli','accuracy_dev')));wanted_sha=sha(data('mnli','accuracy_dev'))
    for path in (EXP/'results/eval').glob('*/summary.json'):
        protocol=json.loads((path.parent/'protocol.json').read_text())
        if protocol['args']['split']!='accuracy_dev' or protocol['input_sha256'].get('mnli')!=wanted_sha:continue
        if not protocol['full_inputs'] or protocol['max_new_tokens']!=256:continue
        covered.update(k for k,v in json.loads(path.read_text()).items() if v['n']==n)
    entries=[];completed=[];missing=[];seen=set();already_scored=0
    for name,item in policy['stages'].items():
        argv=item['training_argv'];jobs_path=Path(argv[argv.index('--jobs')+1]);assert sha(jobs_path)==item['jobs_sha256']
        arg=lambda k,default:argv[argv.index(k)+1] if k in argv else default
        warm=int(arg('--warmup-seed','301'));seed_id=int(arg('--seed','401'));subset=arg('--subset','update');suffix=arg('--suffix','')
        for job in json.loads(jobs_path.read_text()):
            root=EXP/'models'/f'mnli__w{warm}_{subset}_{job["name"]}_u{seed_id}{suffix}'
            p=root/'summary.json'
            if not p.exists() or not json.loads(p.read_text()).get('complete'):
                missing.append(dict(stage=name,method=job['name'],status='not completed'));continue
            cfg=json.loads((root/'config.json').read_text());assert cfg['job']==job
            summary=json.loads(p.read_text());completed.append(dict(stage=name,method=job['name'],checkpoint_count=len(summary['entries']),summary_sha256=sha(p)))
            for e in summary['entries']:
                key=e['tag']+'__'+e['task']
                if key in seen:continue
                seen.add(key)
                if key in covered:already_scored+=1;continue
                assert sha(Path(e['adapter'])/'adapter_model.safetensors')==e['adapter_sha256']
                entries.append(e)
    digest=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()[:16]
    path=EXP/'results'/f'entries_optional_completed_{digest}.json';write(path,entries)
    result=dict(epoch=time.time(),completed_recipes=completed,uncompleted_recipes=missing,
                already_scored_entries=already_scored,unscored_completed_entries=len(entries),
                entries_path=str(path),evaluation_name=f'optional_completed_{digest}',
                scope='Only fully completed previously registered recipes. Existing full-input256-new-token accuracy_dev readouts are reused. Missing or partial recipes are not labeled measured failures.')
    write(EXP/'results/optional_completed_readout_inventory.json',result)
    log({k:result[k] for k in ['unscored_completed_entries','already_scored_entries','entries_path','evaluation_name']})

if __name__=='__main__':main()
