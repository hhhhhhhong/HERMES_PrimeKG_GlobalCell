import unittest
import torch
from dual_branch.entity_token_adapter import EntityAdaptedTokenInteraction, audit_entity_tokens, install_entity_token_adapter
from dual_branch.multimodal_fusion import JointTokenInteraction


class EntityTokenAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)
        self.readout = EntityAdaptedTokenInteraction(dim=16, heads=4, dropout=0.).eval()
        self.x = torch.randn(5,3,16)

    def test_exchange_equivariance_and_no_cross_sample_attention(self):
        expected = self.readout(self.x)
        p = [1,0,2]
        torch.testing.assert_close(self.readout(self.x[:,p]),expected[:,p],rtol=0,atol=2e-6)
        separate = torch.cat([self.readout(x[None]) for x in self.x])
        torch.testing.assert_close(separate,expected,rtol=0,atol=2e-6)
        changed = self.x.clone(); changed[1:] *= 100
        torch.testing.assert_close(self.readout(changed)[:1],expected[:1],rtol=0,atol=2e-6)

    def test_per_token_scale_shift_and_real_entity_interaction(self):
        adapted = self.readout.adapt(self.x)
        scaled = self.x*torch.tensor([.01,100.,3.])[None,:,None]+torch.tensor([10.,-5.,7.])[None,:,None]
        # Finite eps means LN+GELU is NOT exactly scale invariant. Check the
        # intended contract: calibrated RMS, plus additive-shift invariance.
        shifted = self.x+torch.tensor([10.,-5.,7.])[None,:,None]
        torch.testing.assert_close(self.readout.adapt(shifted),adapted,rtol=0,atol=5e-6)
        scaled_adapted = self.readout.adapt(scaled)
        self.assertTrue(torch.isfinite(scaled_adapted).all())
        torch.testing.assert_close(scaled_adapted.square().mean(-1).sqrt(),torch.ones(5,3),rtol=0,atol=2e-3)
        torch.testing.assert_close(adapted.mean(-1),torch.zeros(5,3),rtol=0,atol=2e-6)
        torch.testing.assert_close(adapted.square().mean(-1).sqrt(),torch.ones(5,3),rtol=0,atol=2e-3)
        changed=self.x.clone();changed[:,2]+=torch.linspace(-1,1,16)
        self.assertGreater(float((self.readout(changed)[:,:2]-self.readout(self.x)[:,:2]).abs().max()),1e-8)
        audit=audit_entity_tokens(self.readout,self.x)
        self.assertEqual(len(audit['attention_mean_by_head']),4)

    def test_gradients_reach_all_inputs_both_adapters_and_qkv(self):
        x=self.x.requires_grad_(True)
        (self.readout(x)*torch.randn_like(x)).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(bool((x.grad.norm(dim=-1)>0).all()))
        for module in [self.readout.drug_adapter,self.readout.cell_adapter,self.readout.encoder.self_attn]:
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm())>0 for p in module.parameters()))

    def test_cli_factory_is_process_local_and_restored_on_error(self):
        import sys
        from unittest.mock import patch
        from dual_branch import adapted_readout_cli, hypothesis_cli, hypothesis_evidence
        from dual_branch.entity_token_adapter import EntityAdaptedHypothesisModel
        original = hypothesis_evidence.HypothesisEvidenceModel
        def verify_factory():
            self.assertIs(hypothesis_evidence.HypothesisEvidenceModel,EntityAdaptedHypothesisModel)
            raise RuntimeError('intentional test exception')
        with patch.object(sys,'argv',['adapter_cli','--device=cpu']), patch.object(hypothesis_cli,'main',verify_factory):
            with self.assertRaisesRegex(RuntimeError,'intentional test exception'):
                adapted_readout_cli.main()
        self.assertIs(hypothesis_evidence.HypothesisEvidenceModel,original)

    def test_installer_preserves_core_weights_rng_and_reload(self):
        parent=torch.nn.Module();parent.joint_transformer=JointTokenInteraction(dim=16,heads=4,dropout=0.)
        before={k:v.clone() for k,v in parent.joint_transformer.state_dict().items()}
        rng=torch.get_rng_state().clone()
        install_entity_token_adapter(parent)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        after=parent.joint_transformer.state_dict()
        for key,value in before.items():torch.testing.assert_close(after[key],value,rtol=0,atol=0)
        self.assertEqual(set(after)-set(before),{'drug_adapter.1.weight','cell_adapter.1.weight'})
        other=EntityAdaptedTokenInteraction(dim=16,heads=4,dropout=0.)
        other.load_state_dict(after,strict=True)
        parent.eval();other.eval()
        torch.testing.assert_close(other(self.x),parent.joint_transformer(self.x),rtol=0,atol=0)
        with self.assertRaises(ValueError):install_entity_token_adapter(parent)
        with self.assertRaises(ValueError):other(torch.zeros(5,4,16))

if __name__ == '__main__':unittest.main()
