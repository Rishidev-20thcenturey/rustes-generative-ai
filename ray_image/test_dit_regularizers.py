"""Tests for the flag-gated DiT v5 regularisers: router similarity loss and SAGE.

These need torch. They are skipped when torch is not installed, so the Monet tests still run
on a machine without it.
"""

import unittest

try:
    import torch
    HAVE_TORCH = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_TORCH = False

if HAVE_TORCH:
    from ray_image.dit_v5_moe import RAYDiT_v5, router_similarity_loss, sage_loss


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class RouterSimilarityLossTests(unittest.TestCase):
    def test_identical_experts_cost_more_than_disjoint(self):
        torch.manual_seed(0)
        n, e, k = 64, 4, 2
        # Disjoint: each token strongly prefers experts (0,1) or (2,3), never mixing groups.
        probs_disjoint = torch.full((n, e), 1e-3)
        idx_disjoint = torch.zeros(n, k, dtype=torch.long)
        for i in range(n):
            pair = (0, 1) if i % 2 == 0 else (2, 3)
            probs_disjoint[i, pair[0]] = 0.5
            probs_disjoint[i, pair[1]] = 0.5
            idx_disjoint[i] = torch.tensor(pair)
        # Collapsed: all tokens pick the same pair, so experts 0 and 1 are always co-selected.
        probs_same = torch.full((n, e), 1e-3)
        probs_same[:, 0] = 0.5
        probs_same[:, 1] = 0.5
        idx_same = torch.tensor([[0, 1]]).repeat(n, 1)
        loss_disjoint = router_similarity_loss(probs_disjoint, idx_disjoint, e)
        loss_same = router_similarity_loss(probs_same, idx_same, e)
        self.assertTrue(torch.isfinite(loss_disjoint))
        self.assertGreater(loss_same.item(), loss_disjoint.item())

    def test_gradient_reaches_probs_only(self):
        torch.manual_seed(1)
        logits = torch.randn(32, 6, requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        idx = torch.topk(logits.detach(), 2, dim=-1).indices
        loss = router_similarity_loss(probs, idx, 6)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_top_k_one_has_no_off_diagonal_mass(self):
        # With k=1 no two experts are ever picked together, so the off-diagonal weight is zero.
        torch.manual_seed(2)
        probs = torch.softmax(torch.randn(16, 4), dim=-1)
        idx = probs.argmax(dim=-1, keepdim=True)
        loss = router_similarity_loss(probs, idx, 4)
        self.assertTrue(torch.isfinite(loss))
        # Only the diagonal weight is non-zero, so the loss is the diagonal share of P'.
        m = torch.zeros(16, 4).scatter_(1, idx, 1.0)
        diag = (m.t() @ m).diagonal()
        weight_diag = diag / (diag.sum().clamp_min(1e-6) * 4)
        expected = (weight_diag * (probs.t() @ probs).diagonal()).sum() / 16
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class SageLossTests(unittest.TestCase):
    def test_zero_when_unconditional_lies_in_conditional_subspace(self):
        torch.manual_seed(3)
        n, d = 256, 16
        f_c = torch.randn(n, d)
        # f_u is a linear combination of the first m rows of the conditional subspace.
        # With k_split=2, m = d // 2 = 8 columns are drawn at random from f_c, so the subspace of
        # f_u must sit inside the span of ALL of f_c. Use a rank-8 f_c to make that exact.
        basis = torch.linalg.qr(torch.randn(d, 8)).Q
        f_c = torch.randn(n, 8) @ basis.t()
        f_u = torch.randn(n, 8) @ basis.t()
        loss = sage_loss(f_c, f_u, k_split=2)
        self.assertLess(loss.item(), 1e-5)

    def test_positive_for_orthogonal_unconditional(self):
        torch.manual_seed(4)
        n, d = 256, 16
        f_c = torch.zeros(n, d)
        f_c[:, :8] = torch.randn(n, 8)
        f_u = torch.zeros(n, d)
        f_u[:, 8:] = torch.randn(n, 8)
        loss = sage_loss(f_c, f_u, k_split=2)
        self.assertAlmostEqual(loss.item(), 1.0, places=4)

    def test_gradient_flows_to_unconditional_only(self):
        torch.manual_seed(5)
        f_c = torch.randn(128, 16, requires_grad=True)
        f_u = torch.randn(128, 16, requires_grad=True)
        sage_loss(f_c, f_u, k_split=2).backward()
        self.assertIsNone(f_c.grad)
        self.assertGreater(f_u.grad.abs().sum().item(), 0.0)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class CaptureTests(unittest.TestCase):
    def _tiny(self):
        torch.manual_seed(6)
        return RAYDiT_v5(
            latent_channels=4, latent_size=16, dim=32, depth=2, heads=4, patch=4,
            cond_dim=32, num_experts=4, top_k=2, expert_hidden_dim=16, num_shared_experts=1,
        )

    def _inputs(self, batch=2):
        z = torch.randn(batch, 4, 16, 16)
        t = torch.rand(batch)
        text = torch.randn(batch, 5, 32)
        return z, t, text

    def test_default_forward_is_unchanged_and_records_nothing(self):
        model = self._tiny().train()
        z, t, text = self._inputs()
        pred, lb, zl = model(z, t, text, checkpoint_blocks=False)
        self.assertEqual(pred.shape, z.shape)
        self.assertFalse(hasattr(model, "last_captures"))
        for block in model.blocks:
            self.assertFalse(block.moe.record_routing)
            self.assertIsNone(block.moe.last_out)

    def test_capture_returns_per_layer_tensors(self):
        model = self._tiny().train()
        z, t, text = self._inputs()
        model(z, t, text, capture=True)
        self.assertEqual(len(model.last_captures), 2)
        for probs, idx, out in model.last_captures:
            self.assertEqual(probs.shape, (2 * 16, 4))
            self.assertEqual(idx.shape, (2 * 16, 2))
            self.assertEqual(out.shape, (2 * 16, 32))
        for block in model.blocks:
            self.assertFalse(block.moe.record_routing)


if __name__ == "__main__":
    unittest.main()
