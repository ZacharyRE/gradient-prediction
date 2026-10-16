"""Run the public CPU reproduction entry point in a fresh relocated directory."""
from common import *
import tempfile,sys

def main():
    subprocess.run([sys.executable,str(CODE_ROOT/'audit_generated_records.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),check=True)
    with tempfile.TemporaryDirectory(prefix='surrogate-statistics-replay-') as temp:
        out=Path(temp)/'reproduced'
        command=[sys.executable,str(CODE_ROOT/'reproduce_surrogate_statistics.py'),'--study-dir',str(EXP),'--report-dir',str(EXP),'--data-dir',str(EXP/'data/mnli'),'--output',str(out)]
        subprocess.run(command,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),check=True)
        record=json.loads((out/'reproduction_audit.json').read_text());assert record['passed'];record['source_sha256']=source_signature(__file__)
    write(EXP/'results/surrogate_analysis_reproduction_audit.json',record);log(record)

if __name__=='__main__':main()
