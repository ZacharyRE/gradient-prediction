from common import *
files=json.loads((EXP/'configs/needed_entry_sources.json').read_text());entries=[]
for f in files:entries.extend(json.loads(Path(f).read_text()))
write(EXP/'results/entries_needed.json',entries)
