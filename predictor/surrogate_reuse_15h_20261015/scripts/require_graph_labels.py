from common import *
path=EXP/'results/graph_labels_audit.json'
while not path.exists():
    if deadline(5400):raise RuntimeError('Validation reserve while waiting for retained-gradient audit')
    time.sleep(1)
assert json.loads(path.read_text())['passed']
