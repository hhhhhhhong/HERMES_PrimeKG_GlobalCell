"""Verify shared-graph gradient accumulation and epoch-based early stopping."""
import unittest
import torch
from dual_branch.stable_training import train_epoch
from dual_branch.backbone_repair_training import AucEarlyStop


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.modal=torch.nn.Linear(2,3)
        self.refiner=torch.nn.Linear(3,3)
        self.head=torch.nn.Linear(6,1)
        self.register_buffer('x',torch.tensor([[.1,.2],[.4,.3],[.2,.9],[.8,.1]]))
        self.calls=0
    def base_features(self): return self.modal(self.x)
    def knowledge_table(self,base):
        self.calls+=1
        return self.refiner(base)
    def forward(self,rows,table,base):
        return self.head(torch.cat((base[rows],table[rows]),-1)).squeeze(-1).sigmoid(),None
    def components(self): return {'all':list(self.parameters())}


class RepairTests(unittest.TestCase):
    def test_shared_ancestor_gradient_matches_full_loss(self):
        torch.manual_seed(0)
        model=Toy();reference=Toy();reference.load_state_dict(model.state_dict())
        opt=torch.optim.SGD(model.parameters(),lr=.01)
        refopt=torch.optim.SGD(reference.parameters(),lr=.01)
        rows=torch.arange(4);labels=torch.tensor([0.,1.,1.,0.])
        train_epoch(model,opt,rows,labels,chunk=2,max_grad_norm=100.)
        base=reference.base_features();table=reference.knowledge_table(base)
        loss=torch.nn.functional.binary_cross_entropy(reference(rows,table,base)[0],labels)
        loss.backward();refopt.step()
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a,b,atol=1e-7,rtol=0)
        self.assertEqual(model.calls,1)

    def test_patience_epochs_and_minimum(self):
        s=AucEarlyStop(100,300)
        self.assertEqual(s.observe(1,.8),(True,False))
        self.assertEqual(s.observe(100,.8),(False,False))
        self.assertEqual(s.observe(250,.81),(True,False))
        self.assertEqual(s.observe(300,.80),(False,False))
        self.assertEqual(s.observe(350,.80),(False,True))
        self.assertEqual(s.observe(355,.82),(True,False))
        with self.assertRaises(ValueError): s.observe(360,float('nan'))

if __name__=='__main__': unittest.main()
