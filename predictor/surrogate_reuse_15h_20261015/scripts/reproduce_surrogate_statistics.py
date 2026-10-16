"""Rebuild all final statistics from published compact evidence; CPU only.

Requires reconstructed selection_dev.jsonl and selection_holdout.jsonl.
No base model, predictor, adapter tensors, CUDA device or original absolute paths.
The compact bundle supports statistical reproduction, not independent text grading.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study-dir',type=Path,default=Path(__file__).resolve().parents[1],help='Published code directory containing scripts/ and configs/')
    ap.add_argument('--report-dir',type=Path,required=True,help='Report directory containing results/frozen_correctness.npz and final_selection_analysis.json')
    ap.add_argument('--data-dir',type=Path,required=True,help='Reconstructed mnli directory containing the two complete selection JSONL files')
    ap.add_argument('--output',type=Path,required=True,help='New output directory; existing paths are never overwritten')
    a=ap.parse_args();tick=time.time();study=a.study_dir.resolve();report=a.report_dir.resolve();out=a.output.resolve();data=a.data_dir.resolve()
    assert not out.exists(),'Use a fresh output directory'
    configs=study/'configs';scripts=study/'scripts';frozen=json.loads((configs/'surrogate_final_freeze.json').read_text());assert frozen['complete']
    for name,want in frozen['code'].items():assert digest(scripts/name)==want,f'Frozen analysis dependency differs: {name}'
    for split in ['selection_dev','selection_holdout']:assert digest(data/f'{split}.jsonl')==frozen['data_sha256']['mnli'][split],f'Different full input data: {split}'
    bundle=report/'results/frozen_correctness.npz';reference=report/'results/final_selection_analysis.json';audit=json.loads((report/'results/frozen_correctness_audit.json').read_text())
    assert audit['passed'] and digest(bundle)==audit['sha256'] and audit['freeze_sha256']==digest(configs/'surrogate_final_freeze.json')
    assert audit['decision_sha256']==digest(configs/'surrogate_selection_decisions.json')
    original=json.loads(reference.read_text());assert original['complete']
    for folder in ['configs','results','data/mnli']:(out/folder).mkdir(parents=True)
    for name in ['surrogate_final_freeze.json','surrogate_selection_decisions.json','surrogate_analysis_spec.json']:shutil.copy2(configs/name,out/'configs'/name)
    for split in ['selection_dev','selection_holdout']:shutil.copy2(data/f'{split}.jsonl',out/'data/mnli'/f'{split}.jsonl')
    cmd=[sys.executable,str(scripts/'analyze_selection.py'),'--analysis-spec',str(out/'configs/surrogate_analysis_spec.json'),'--prediction-bundle',str(bundle)]
    env=dict(os.environ,BENCH_EXP_DIR=str(out),BENCH_SOURCE_DIR=str(scripts),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with (out/'analysis.log').open('w') as log:subprocess.run(cmd,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
    actual=json.loads((out/'results/final_selection_analysis.json').read_text());fields=['complete','freeze_sha256','decisions_sha256','spec','groups','per_start','direct_quality','direct_paired_comparisons','units','important'];checks={k:original[k]==actual[k] for k in fields};assert all(checks.values()),checks
    receipt=dict(passed=True,seconds=time.time()-tick,source_result_sha256=digest(reference),bundle_sha256=digest(bundle),checked_fields=checks,command=cmd,scope='Published frozen recipes/choices/spec, reconstructed complete input rows and pickle-free correctness bundle. Exact equality of all numerical quality, ranking, regret and paired bootstrap outputs, in a new relocated layout. No model weights, GPU or original absolute artifact paths are needed. Raw generated responses are absent, so this does not independently regrade text.')
    (out/'reproduction_audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(passed=True,output=str(out),seconds=receipt['seconds'],checked_fields=list(checks))),flush=True)

if __name__=='__main__':main()
