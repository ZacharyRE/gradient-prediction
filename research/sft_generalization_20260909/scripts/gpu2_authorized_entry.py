"""Run an unchanged frozen inference script with the user-authorized GPU2 guard."""
import runpy
import common

def main():
    p=common.S/'audits/gpu2_additional_execution_amendment.json'
    a=common.json.loads(p.read_text());ah=common.sha(p)
    assert a['maximum_simultaneous_gpus']==3 and a['allowed_physical_gpus']==['1','2','3']
    assert common.sha(__file__)==a['entry_script_sha256']
    target=common.Path(common.sys.argv[1]).resolve()
    assert target.parent==common.S/'scripts' and target.name in a['unchanged_inference_scripts']
    assert common.sha(target)==a['unchanged_inference_scripts'][target.name]
    def authorized_guard():
        assert common.os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
        assert common.sha(p)==ah and common.time.time()<common.DEADLINE
        assert not (common.S/'STOP_GPU2_ADDITIONAL').exists()
    authorized_guard()
    common.gpu_guard=authorized_guard
    common.sys.argv=common.sys.argv[1:]
    runpy.run_path(str(target),run_name='__main__')

if __name__=='__main__':main()
