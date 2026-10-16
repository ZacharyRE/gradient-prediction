from online_ridge import *

def main():
    gpu_guard();seed(42);reserve_worker_cache()
    x=torch.randn(24,7,device='cuda');y=torch.randn(24,11,device='cuda');reg=2.;records=[]
    for tf32 in [True,False]:
        torch.backends.cuda.matmul.allow_tf32=tf32
        model=OnlineRidge(7,11,reg,1.)
        for i in range(0,24,4):model.update(x[i:i+4],y[i:i+4])
        target=torch.linalg.solve(x.double().T@x.double()+reg*torch.eye(7,device='cuda',dtype=torch.float64),x.double().T@y.double())
        error=float((model.W.double()-target).norm()/target.norm())
        inverse=torch.linalg.inv(x.double().T@x.double()+reg*torch.eye(7,device='cuda',dtype=torch.float64))
        inv_error=float((model.P-inverse).norm()/inverse.norm())
        records.append(dict(tf32=tf32,recursive_vs_closed_form_relative_l2=error,inverse_relative_l2=inv_error))
        assert error<(1e-3 if tf32 else 1e-5) and inv_error<1e-10,(error,inv_error)
    result=dict(passed=True,records=records)
    write(EXP/'results/ridge_numerical_audit.json',result);log(result)

if __name__=='__main__':main()
