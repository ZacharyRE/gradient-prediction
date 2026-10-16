"""Export compact row metadata and rebuild byte-identical full MNLI inputs."""
from common import *
import argparse,gzip,io
import pyarrow.parquet as pq

def problem(row):
    return 'Premise:\n'+row['premise']+'\n\nHypothesis:\n'+row['hypothesis']+'\n\nA) entailment\nB) neutral\nC) contradiction'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--export',action='store_true');ap.add_argument('--metadata',type=Path);ap.add_argument('--raw-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path);a=ap.parse_args()
    metadata=a.metadata or EXP/'provenance/data_reproduction.json.gz'
    if a.export:
        paths=[p for p in sorted((EXP/'data/mnli').glob('*.jsonl')) if p.stem not in ['test','gradient_test','oracle_confirmation']];splits={}
        for p in paths:
            rr=read(p);keys=list(rr[0]);assert all(list(r)==keys for r in rr)
            splits[p.stem]=dict(file_sha256=sha(p),key_order=keys,rows=[dict(fields={k:v for k,v in r.items() if k!='problem'},problem_sha256=hashlib.sha256(r['problem'].encode()).hexdigest()) for r in rr])
        sources={s for v in splits.values() for r in v['rows'] for s in [r['fields']['source_split']]}
        record=dict(task='mnli',full_inputs=True,raw_source_sha256={s:sha(a.raw_dir/f'{s}-00000-of-00001.parquet') for s in sources},splits=splits,purpose='Reconstruct exact full evaluation/training inputs from original public MNLI parquet and fixed row metadata. No generated outcomes included.')
        metadata.parent.mkdir(exist_ok=True,parents=True)
        metadata.write_bytes(gzip.compress(json.dumps(record,ensure_ascii=False,separators=(',',':')).encode(),mtime=0))
    record=json.loads(gzip.decompress(metadata.read_bytes()));raw={}
    for split,expected in record['raw_source_sha256'].items():
        path=a.raw_dir/f'{split}-00000-of-00001.parquet';assert sha(path)==expected,'Different raw dataset revision';raw[split]=pq.read_table(path).to_pylist()
    reports=[]
    for split,info in record['splits'].items():
        rows=[]
        for item in info['rows']:
            r=dict(item['fields']);source=raw[r['source_split']][r['source_index']];text=problem(source);assert hashlib.sha256(text.encode()).hexdigest()==item['problem_sha256']
            assert r['gold']=='ABC'[source['label']];r['problem']=text;rows.append({k:r[k] for k in info['key_order']})
        payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode();assert hashlib.sha256(payload).hexdigest()==info['file_sha256'],split
        if a.output_dir:
            dest=a.output_dir/f'{split}.jsonl';dest.parent.mkdir(exist_ok=True,parents=True)
            if dest.exists():assert sha(dest)==info['file_sha256'],'Refuse to overwrite different data'
            else:dest.write_bytes(payload)
        reports.append(dict(split=split,n=len(rows),sha256=info['file_sha256'],byte_identical=True))
    receipt=dict(passed=True,full_inputs=True,metadata_sha256=sha(metadata),metadata_compressed_bytes=metadata.stat().st_size,splits=reports,raw_source_sha256=record['raw_source_sha256'])
    if a.export:write(EXP/'provenance/data_reproduction_audit.json',receipt)
    log(receipt)

if __name__=='__main__':main()
