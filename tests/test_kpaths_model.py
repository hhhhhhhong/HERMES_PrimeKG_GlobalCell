"""Ordered relation and padding semantics independent of real-label training."""
import unittest
import torch
from dual_branch.kpaths_model import OrderedPathEncoder


class PathEncoderTests(unittest.TestCase):
    def test_direction_changes_encoding_and_real_weights_learn(self):
        torch.manual_seed(0)
        encoder=OrderedPathEncoder(5,8)
        nodes=torch.randn(5,8,requires_grad=True)
        ids=torch.tensor([[0,2,1,0,0],[0,2,1,0,0]])
        rel=torch.tensor([[1,2,0,0],[3,4,0,0]])
        out=encoder(nodes,ids,rel,torch.tensor([2,2]))
        self.assertGreater(float((out[0]-out[1]).abs().max()),1e-5)
        out.sum().backward()
        self.assertGreater(float(encoder.relation.weight.grad.abs().sum()),0)
        self.assertGreater(float(encoder.sequence.weight_ih_l0.grad.abs().sum()),0)

    def test_padding_and_sentinel_have_no_effect(self):
        torch.manual_seed(1)
        encoder=OrderedPathEncoder(5,8)
        nodes=torch.randn(5,8)
        ids=torch.tensor([[0,0,0,0,0],[0,2,1,0,0]])
        rel=torch.tensor([[0,0,0,0],[1,2,0,0]])
        hops=torch.tensor([0,2])
        original=encoder(nodes,ids,rel,hops)
        self.assertEqual(float(original[0].abs().max()),0)
        ids[1,3:]=4;rel[1,2:]=4
        torch.testing.assert_close(encoder(nodes,ids,rel,hops),original,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
