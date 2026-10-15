"""Require a saved, timestamped recipe manifest before any new test evaluation."""
from common import *

def ensure_global_freeze(task,entries):
    path=EXP/'configs/final_freeze.json'
    assert path.exists(),'Freeze selected methods and comparison scope before reading final tests'
    cfg=json.loads(path.read_text());allowed={e['tag']:e for e in cfg['entries'] if e['task']==task}
    assert cfg['selection_split']=='accuracy_dev' and cfg['frozen_epoch']<time.time()
    for e in entries:
        assert e['tag'] in allowed,e['tag']
        frozen=allowed[e['tag']]
        assert e['adapter']==frozen['adapter']
        assert sha(Path(e['adapter'])/'adapter_model.safetensors')==frozen['adapter_sha256']
    return dict(freeze_sha256=sha(path),frozen_epoch=cfg['frozen_epoch'],task=task,entries=len(entries))
