import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from path_bounds import prepare,solve
from query_gap_distillation import query_gaps,envelopes

class QueryGapTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261108)
        self.logits=torch.randn(7,3,dtype=torch.float64,requires_grad=True)
        self.d=np.array([[.7,.2],[.3,.8]])
        self.a=np.array([[.2,.8],[.7,.3]])
        self.lambdas=(-3.,-1.,1.,3.)

    def test_values(self):
        p=self.logits.softmax(-1)[:,:2]
        out=query_gaps(p,self.d,self.a,self.lambdas)
        args=prepare(p.detach().numpy(),self.d,self.a);base=solve(*args)
        wanted=[(solve(*args,lam=x)[0]-base[0])/(x*len(p)) for x in self.lambdas]
        np.testing.assert_allclose(out.detach().numpy(),wanted,atol=1e-12,rtol=0.)

    def test_finite_difference_gradients_full_and_window(self):
        for weights in [None,np.array([0.,0.,1.,1.,1.,0.,0.])]:
            function=lambda logits:query_gaps(logits.softmax(-1)[:,:2],self.d,self.a,self.lambdas,weights)
            self.assertTrue(torch.autograd.gradcheck(function,(self.logits,),eps=1e-6,atol=2e-6,rtol=1e-4))

    def test_uniform_envelope_transfer(self):
        rng=np.random.default_rng(20261112)
        for _ in range(100):
            teacher=torch.from_numpy(rng.normal(size=4));student=teacher+torch.from_numpy(rng.normal(size=4)*.01)
            delta=float((teacher-student).abs().max())
            for eta in [0.,.0001,.001,.005,.01,.05,.5,10.]:
                t=envelopes(teacher,self.lambdas,eta);s=envelopes(student,self.lambdas,eta)
                self.assertLessEqual(max(abs(float(a-b)) for a,b in zip(t,s)),delta+1e-12)

    def test_float32_cpu_and_cuda(self):
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        values=[]
        for device in devices:
            x=self.logits.detach().float().to(device).requires_grad_()
            gap=query_gaps(x.softmax(-1)[:,:2],self.d,self.a,self.lambdas)
            gap.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            values.append(gap.detach().cpu().numpy())
        if len(values)==2:np.testing.assert_allclose(*values,atol=2e-7,rtol=0.)
        print('devices_checked='+','.join(devices),flush=True)

if __name__=='__main__':unittest.main(verbosity=2)
