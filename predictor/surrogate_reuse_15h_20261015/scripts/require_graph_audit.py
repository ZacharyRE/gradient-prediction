from common import *
path=EXP/'results/graph_rollout_audit.json'
while not path.exists():
    if deadline(5400):raise RuntimeError('Validation reserve reached while waiting for graph audit')
    time.sleep(1)
assert json.loads(path.read_text())['passed'],'Graph/eager-static rollout audit failed'
