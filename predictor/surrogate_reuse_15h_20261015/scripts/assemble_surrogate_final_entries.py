"""Assemble a completed scientific plan; no outcome or accuracy is consulted."""
from common import *
from surrogate_freeze import entry_key,FREEZE
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',default=str(EXP/'configs/surrogate_final_recipe_plan.json'));ap.add_argument('--check-only',action='store_true');a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());assert plan['complete'],'Finish recipe selection before assembling final entries'
    selected={};sources={};missing=[];memberships={}
    def keep(entry,group):
        key=entry_key(entry)
        if key in selected:assert selected[key]==entry,'Conflicting duplicate entry'
        selected[key]=entry;memberships.setdefault(key,[]).append(group)
    for group in plan['entry_groups']:
        count=0
        for source in group['sources']:
            path=Path(source)
            if not path.exists():missing.append(str(path));continue
            sources[str(path)]=sha(path);entries=json.loads(path.read_text())
            for selector in group['selectors']:
                matches=[e for e in entries if all(e.get(k)==v for k,v in selector.items())]
                assert len(matches)==1,(group['name'],str(path),selector,len(matches))
                e=matches[0];root=Path(e['adapter']).parent;cfg=json.loads((root/'config.json').read_text());summary=json.loads((root/'summary.json').read_text())
                assert summary['complete'] and any(x==e for x in summary['entries'])
                assert cfg['warmup_seed']==e['warmup_seed'] and cfg['update_seed']==e['seed'] and cfg['subset']==e['subset']
                assert cfg['job']['name']==e['method'] and cfg['job']['kind']==e['kind'] and cfg['job']['lr']==e['lr']
                assert sha(Path(e['adapter'])/'adapter_model.safetensors')==e['adapter_sha256']
                keep(e,group['name']);count+=1
        if not any(str(Path(s)) in missing for s in group['sources']):assert count==group['expected_entries'],(group['name'],count)
    for warm in plan['heldout_warmups']:
        path=EXP/'configs'/f'mnli_warmup_w{warm}.json';e=json.loads(path.read_text());assert e['steps']==0 and e['warmup_seed']==warm and sha(Path(e['adapter'])/'adapter_model.safetensors')==e['adapter_sha256'];sources[str(path)]=sha(path);keep(e,'shared_warmup_baseline')
    status=dict(complete=not missing,available_entries=len(selected),missing_sources=missing,plan_sha256=sha(a.plan),sources=sources)
    write(EXP/'results/surrogate_entry_assembly_status.json',status);log({k:v for k,v in status.items() if k!='sources'})
    if a.check_only or missing:return
    assert not FREEZE.exists(),'Do not rewrite entries after final freeze'
    entries=[selected[k] for k in sorted(selected)];assert len(entries)==plan['expected_unique_entries']
    path=EXP/'configs/surrogate_final_entries.json'
    if path.exists():assert json.loads(path.read_text())==entries
    else:write(path,entries)
    write(EXP/'results/surrogate_entry_memberships.json',dict(plan_sha256=sha(a.plan),entries_sha256=sha(path),memberships=memberships,sources=sources,scope='Scientific roles only; selection uses declared recipe fields, never downstream outcomes. Shared reference adapters appear once.'))

if __name__=='__main__':main()
