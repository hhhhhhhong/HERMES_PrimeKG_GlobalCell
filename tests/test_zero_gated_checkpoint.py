"""Stochastic full-fit forward, gradients and updates must match without it."""
import unittest

import torch

from global_cell.common import author_imports
from global_cell.network import train_epoch, evaluate
from global_cell.synergy import add_training_synergy, rows_sha256
from models.zero_gated_hgnn import ZeroGatedHypergraphModel
from models.zero_gated_checkpoint import CheckpointedZeroGatedModel
from test_global_cell import fixture


class CheckpointEquivalence(unittest.TestCase):
    def test_real_synergy_dropout_gradients_updates_and_rng(self):
        torch.set_num_threads(2)
        author_imports()
        torch.manual_seed(0)
        graph=fixture()
        for i in range(10):graph['groups']['fixture_relation_'+str(i)]=graph['groups']['drug_target']
        graph['n_cell']=2
        graph['summary']=dict(relation_counts={k:len(v) for k,v in graph['groups'].items()})
        rows=torch.tensor([[0.,1.,2.,1.],[1.,0.,3.,0.],[0.,1.,3.,1.]])
        graph=add_training_synergy(graph,rows,fold=0,expected_train_sha256=rows_sha256(rows))
        cache=dict(num_drug=2,num_cell=2,cell_input_dim=3,inputs=dict(drug_x=torch.randn(6,75),
            drug_adj=torch.tensor([[0,1,1,2,3,4,4,5],[1,0,2,1,4,3,5,4]]),
            ibatch=torch.tensor([0,0,0,1,1,1]),cells=torch.randn(2,3)))
        coder=torch.randn(3,768)
        torch.manual_seed(0);a=ZeroGatedHypergraphModel(cache,graph,coder)
        torch.manual_seed(0);b=CheckpointedZeroGatedModel(cache,graph,coder)
        for name,value in a.state_dict().items():torch.testing.assert_close(b.state_dict()[name],value,rtol=0,atol=0)
        optimizers=[torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=.01) for m in [a,b]]
        for step in range(3):
            rng=torch.get_rng_state()
            la,ga=train_epoch(a,optimizers[0],rows,rows[:,3],chunk=2)
            end_a=torch.get_rng_state()
            torch.set_rng_state(rng)
            lb,gb=train_epoch(b,optimizers[1],rows,rows[:,3],chunk=2)
            self.assertAlmostEqual(la,lb,places=7)
            self.assertTrue(torch.equal(torch.get_rng_state(),end_a))
            for name,p in a.named_parameters():
                q=dict(b.named_parameters())[name]
                self.assertEqual(p.grad is None,q.grad is None)
                if p.grad is not None:torch.testing.assert_close(q.grad,p.grad,rtol=1e-5,atol=1e-8)
                torch.testing.assert_close(q,p,rtol=1e-5,atol=1e-8)
        torch.testing.assert_close(evaluate(a,rows),evaluate(b,rows),rtol=1e-5,atol=1e-7)


if __name__=='__main__':unittest.main()
