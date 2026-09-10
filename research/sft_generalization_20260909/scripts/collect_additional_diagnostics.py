"""CPU-only dependency runner; prepares evidence, never approves reviews or writes report."""
import subprocess
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
scripts=['prepare_final_warning_review.py','prepare_final_transition_review.py',
         'analyze_expanded_supplement.py','prepare_supplement_warning_review.py',
         'analyze_supplement_math_sensitivity.py','analyze_final_math_strata.py',
         'analyze_math_validation_transfer.py','prepare_minerva_unit_review.py',
         'analyze_native_target_controls.py']
hashes={n:sha(S/'scripts'/n) for n in scripts}
audit=S/'audits/additional_diagnostic_collection.json';assert not audit.exists()
done=[];calls=[]
def get(p):
    path=S/p
    return json.loads(path.read_text()) if path.exists() else {}
def publish(status,**extra):
    write(audit,dict(time=time.time(),status=status,all_complete=status=='prepared_for_manual_review',
        done=done,calls=calls,script_sha256=sha(__file__),dependency_script_sha256=hashes,
        scope='CPU preparation only; no GPU work, no score policy changes, no automatic review approval or report verdict.',**extra))
def run(n):
    if n in done:return
    assert sha(S/'scripts'/n)==hashes[n]
    command=[sys.executable,str(S/'scripts'/n)]
    with (S/f'logs/additional_{Path(n).stem}.log').open('w') as log:
        p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                         timeout=max(1,DEADLINE-60-time.time()))
    calls.append(dict(time=time.time(),script=n,returncode=p.returncode))
    if p.returncode:
        publish('error',failed_script=n)
        raise SystemExit(p.returncode)
    done.append(n);publish('waiting_or_collecting')
publish('waiting_or_collecting')
while time.time()<DEADLINE-60:
    main=get('audits/final_collection_status.json').get('all_complete',False)
    supp=get('audits/expanded_unaveraged_supplement_status.json')
    native=get('audits/native_target_control_status.json')
    supp_terminal=supp.get('all_complete') or supp.get('status','').startswith('not_started') or supp.get('status')=='incomplete_error_or_cutoff'
    native_terminal=native.get('all_complete') or native.get('status','').startswith('not_started') or native.get('status')=='incomplete_error_or_cutoff'
    if main:
        run('prepare_final_warning_review.py');run('prepare_final_transition_review.py')
    if main and supp_terminal:
        if supp.get('all_complete'):
            run('analyze_expanded_supplement.py');run('prepare_supplement_warning_review.py')
            run('analyze_supplement_math_sensitivity.py')
        run('analyze_final_math_strata.py');run('analyze_math_validation_transfer.py')
        run('prepare_minerva_unit_review.py')
    if native.get('all_complete'):run('analyze_native_target_controls.py')
    if main and supp_terminal and native_terminal:
        publish('prepared_for_manual_review',supplement_complete=bool(supp.get('all_complete')),
                native_control_complete=bool(native.get('all_complete')))
        break
    publish('waiting_or_collecting');time.sleep(10)
else:publish('global_deadline_incomplete')
