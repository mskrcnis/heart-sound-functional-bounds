import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from query_training_common import *

class TrainingChecks(unittest.TestCase):
    def test_upstream_capture_matches_saved_potentials(self):
        with np.load(ROOT/'phase10/results/development_inputs_v1/00.npz',allow_pickle=False) as z:
            graphs=Potentials()(z['posterior'])
            for n,(p,d,a) in zip(NAMES,graphs):
                for suffix,value in [('posterior',p),('duration',d),('transition',a)]:
                    np.testing.assert_array_equal(value,z[n+'_'+suffix])

    def test_recomputed_graph_torch_numpy_values_and_gradient(self):
        torch.manual_seed(17);x=torch.randn(66,5,requires_grad=True);p=x.softmax(-1)
        capture=Potentials();g,s=targets_numpy(p.detach().numpy(),capture)
        tg=targets_torch(p,capture,'query');ts=targets_torch(p,capture,'score')
        np.testing.assert_allclose(tg.detach().numpy(),g,atol=2e-13,rtol=0.)
        self.assertAlmostEqual(float(ts),s,places=13)
        (tg.square().mean()+ts.square()).backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(float(x.grad.abs().sum()),0.)

    def test_metrics_identity_and_batch_padding(self):
        torch.manual_seed(9);p=torch.randn(75,5).softmax(-1).numpy()
        g,s=targets_numpy(p,Potentials());m=metrics(p,p,g,s,g,s)
        for k in ['primary_bound_mae','emission_kl','score_mae']:self.assertEqual(m[k],0.)
        self.assertTrue(all(v['agrees'] for v in m['sign_agreement'].values()))
        model=PhysiologicalStudent().eval()
        rows=[dict(features=np.random.default_rng(7).normal(size=(40,T)).astype('float32')) for T in [51,73]]
        x,lengths=batch_features(rows,'cpu')
        with torch.no_grad():
            batch=model(x,lengths)
            alone=model(x[:1,:,:51],lengths[:1])
        torch.testing.assert_close(batch[:1,:,:51],alone,atol=2e-6,rtol=1e-5)
        self.assertEqual(sum(p.numel() for p in model.parameters()),4877)

if __name__=='__main__':unittest.main(verbosity=2)
