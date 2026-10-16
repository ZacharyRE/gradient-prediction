"""Check replayed activation gradients and cloned calibration/table labels."""
from reuse_train import *
from graph_teacher import GraphTeacher
from cuda_graph_probe import StaticTeacherGradient

def main():
    warm=json.loads((EXP/'configs/mnli_warmup_w301.json').read_text());m,tok,mod=setup('1p5b',warm['adapter'],rng=811);cap=Capture(mod,m);ds=Dataset('mnli','predictor_dev',tok);configure(True,8192);cache=TeacherPrefixCache(m,tok,mod,cap,ds,micro=32);A,B,s=weights(mod);records=[]
    backend=GraphTeacher(m,tok,mod,cap,ds,cache,'graph',64,4,True);backend.begin_job(dict(kind='subset',k=4));ids=list(range(4))+list(range(32,36));rows,truth,_=backend.collect(m,tok,mod,cap,ds,ids,micro=4,keep=True,verify_reconstruction=False)
    first_snapshot=[r['g'].clone() for r in rows[:4]]
    for start in [0,4]:
        ix=ids[start:start+4];L=((max(ds.lengths[i] for i in ix)+63)//64)*64;static=StaticTeacherGradient(m,tok,mod,cap,ds,cache,ix,L,loss_form='native_mean');static.set_normalizer(1);static.eager()
        for j,i in enumerate(ix):
            r=rows[start+j];length=len(ds[i]['input_ids'])
            # Stored X/Y/H intentionally use BF16, exactly like native labels;
            # H is produced in FP32, so compare after the declared storage cast.
            for key,value in [('x',static.output_x),('y',static.output_y),('g',static.output_g),('h',static.output_h)]:assert torch.equal(r[key],value[j,:length].to(r[key].dtype)),(i,key)
        records.append(dict(indices=ix,all_retained_fields_bitwise_equal=True))
    with torch.no_grad():A.add_(.0001);B.mul_(1.001)
    changed,_,_=backend.collect(m,tok,mod,cap,ds,list(range(4)),micro=4,keep=True,verify_reconstruction=False)
    assert all(torch.equal(r['g'],v) for r,v in zip(rows[:4],first_snapshot)),'Stored labels alias mutable replay buffers'
    L=((max(ds.lengths[i] for i in range(4))+63)//64)*64
    static=StaticTeacherGradient(m,tok,mod,cap,ds,cache,list(range(4)),L,loss_form='native_mean');static.set_normalizer(1);static.eager()
    for j,r in enumerate(changed):assert torch.equal(r['g'],static.output_g[j,:len(r['g'])].float())
    ga,gb=raw(static.output_x,static.output_g,A,B,s);reconstruction=measures(flat([ga.sum(0),gb.sum(0)]),flat(static.grads));assert reconstruction['relative_l2']<.02
    report=dict(passed=True,records=records,changed_weights_gradient_labels_match=True,retained_labels_do_not_alias_replay_buffers=True,native_factor_gradient_reconstruction=reconstruction,full_inputs=True,source_sha256=source_signature(__file__))
    backend.close_job();cache.close();cap.close();write(EXP/'results/graph_labels_audit.json',report);log(report)

if __name__=='__main__':main()
