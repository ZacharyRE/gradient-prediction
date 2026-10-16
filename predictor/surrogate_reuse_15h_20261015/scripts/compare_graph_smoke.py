"""Require graph replay and the same eager-static padded recipe to match weights."""
from common import *
from safetensors.torch import load_file
import torch

jobs=json.loads((EXP/'configs/graph_smoke_jobs.json').read_text());results=[]
for job in jobs:
    values={};records={}
    for mode in ['graph','eager_static']:
        directory=EXP/'models'/f'mnli__w301_candidate_0_{job["name"]}_u401_graph_smoke_{mode}';values[mode]=load_file(str(directory/'step-24/adapter_model.safetensors'));records[mode]=json.loads((directory/'summary.json').read_text())
    a,b=values['graph'],values['eager_static'];assert a.keys()==b.keys();same=all(torch.equal(a[k],b[k]) for k in a);difference=max(float((a[k]-b[k]).abs().max()) for k in a)
    results.append(dict(recipe=job['name'],weights_bitwise_equal=same,max_weight_difference=difference,records=records))
report=dict(passed=all(r['weights_bitwise_equal'] for r in results),full_inputs=True,results=results,scope='24 updates with changed batches/weights, lazy multiple-shape capture, common padding and optimizer; not equivalence to different native padding shapes')
write(EXP/'results/graph_rollout_audit.json',report);log(dict(passed=report['passed'],results=[{k:v for k,v in r.items() if k!='records'} for r in results]));assert report['passed']
