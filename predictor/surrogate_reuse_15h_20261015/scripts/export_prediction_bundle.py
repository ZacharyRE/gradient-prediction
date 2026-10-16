"""Compact, pickle-free correctness arrays for independent statistical reproduction.

Full prompts and generated text remain in the original local records. This
bundle does not support independent regrading; it reproduces aggregate analysis.
"""
from analyze_selection import *

def main():
    frozen=json.loads(FREEZE.read_text());assert DECISIONS.exists(),'Candidate choices must already be frozen'
    tags=sorted(frozen['entries']);arrays={};metadata=dict(freeze_sha256=sha(FREEZE),decision_sha256=sha(DECISIONS),sources={},sample_ids={},correct_counts={},data_sha256={},scope='Reproduces fixed-choice accuracy, regret and paired confidence intervals. Full generated text is not included, so this is not an independent grading audit.')
    for split in ['selection_dev','selection_holdout']:
        rows,values,sources=read_frozen_generated(split,frozen);matrix=np.stack([values[tag] for tag in tags]);assert np.isin(matrix,[0,1]).all();arrays[split]=matrix.astype(bool)
        metadata['sources'][split]=sources;metadata['sample_ids'][split]=[r['sample_id'] for r in rows];metadata['correct_counts'][split]=list(map(int,matrix.sum(1)));metadata['data_sha256'][split]=sha(data('mnli',split))
    path=EXP/'results/frozen_correctness.npz';np.savez_compressed(path,tags=np.array(tags),metadata=np.array(json.dumps(metadata,sort_keys=True)),**arrays)
    for split in arrays:
        _,actual,_=read_frozen_generated(split,frozen,path);assert all(np.array_equal(actual[t],arrays[split][i]) for i,t in enumerate(tags))
    write(EXP/'results/frozen_correctness_audit.json',dict(passed=True,entries=len(tags),sha256=sha(path),compressed_bytes=path.stat().st_size,freeze_sha256=sha(FREEZE),decision_sha256=sha(DECISIONS),scope=metadata['scope']))
    log(dict(passed=True,entries=len(tags),compressed_bytes=path.stat().st_size))

if __name__=='__main__':main()
