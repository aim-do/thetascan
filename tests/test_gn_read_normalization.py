from __future__ import annotations

import importlib.util
import unittest

import torch

from thetascan import (
    GNConfig,
    RoPEConfig,
    TemporalConfig,
    ThetaScan,
    ThetaScanConfig,
)
from thetascan._core.config import ThetaScanConfig as CoreConfig
from thetascan._core.modules.norms import rmsnorm
from thetascan._core.ops import engine
from thetascan._core.ops.engine import Streams
from thetascan._core.ops.interface import (
    Accumulator,
    FadeFast,
    FadeStale,
    NullAccumulator,
)


class GNReadNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    @staticmethod
    def _public_config(
        normalization: str = "both_feature_mass",
        temporal: str = "sum",
        rope: str = "none",
        nonlinearity: str = "relu2",
    ) -> ThetaScanConfig:
        return ThetaScanConfig(
            d_model=16,
            n_heads=2,
            memory_multiplier=2,
            family="gn",
            gn=GNConfig(
                nonlinearity=nonlinearity,
                jacobian_steps=1,
                read_normalization=normalization,
            ),
            temporal=TemporalConfig(mode=temporal),
            rope=RoPEConfig(mode=rope),
        )

    @staticmethod
    def _core_inputs(backend: str, ema: bool = False, depth: int = 1):
        torch.manual_seed(13)
        dtype = torch.float64
        batch, heads, time, dim = 1, 2, 7, 4
        kwargs = dict(
            d_model=heads * dim,
            n_heads=heads,
            head_dim=dim,
            mem_mult=2,
            depth=depth,
            backend=backend,
            rope="none",
            qk_norm=True,
            read_norm=True,
            read_norm_w1=True,
        )
        if ema:
            kwargs.update(accumulation="ema_gate", decay_gate="scalar")
        cfg = CoreConfig(**kwargs)
        hidden = cfg.mem_hidden
        weights = [
            (
                torch.randn(heads, hidden, dim, dtype=dtype) * 0.3,
                torch.randn(heads, dim, hidden, dtype=dtype) * 0.3,
                torch.randn(heads, hidden, dim, dtype=dtype) * 0.3,
            )
            for _ in range(depth)
        ]
        key = torch.randn(batch, heads, time, dim, dtype=dtype)
        query = torch.randn(batch, heads, time, dim, dtype=dtype)
        value = torch.randn(batch, heads, time, dim, dtype=dtype)
        key = key / key.norm(dim=-1, keepdim=True)
        query = query / query.norm(dim=-1, keepdim=True)
        decay_d = decay_m = None
        if ema:
            decay_m = -torch.nn.functional.softplus(
                torch.randn(batch, heads, time, 1, dtype=dtype) * 0.3
            )
            # A deliberately different d-lane catches accidental use of the old
            # literal-W1 lane. The normalized hidden read is keyed in m-space.
            decay_d = decay_m * 0.37
        return cfg, weights, key, value, query, decay_d, decay_m

    def test_public_axis_maps_to_two_explicit_core_flags(self) -> None:
        expected = {
            "none": (False, False),
            "w2_feature_mass": (True, False),
            "both_feature_mass": (True, True),
        }
        for mode, flags in expected.items():
            with self.subTest(mode=mode):
                core = self._public_config(normalization=mode)._to_core_config()
                self.assertEqual((core.read_norm, core.read_norm_w1), flags)

    def test_both_feature_mass_adds_no_trainable_parameters(self) -> None:
        torch.manual_seed(5)
        baseline = ThetaScan(self._public_config(normalization="none"))
        torch.manual_seed(5)
        normalized = ThetaScan(self._public_config(normalization="both_feature_mass"))
        base_signature = [(name, tuple(p.shape)) for name, p in baseline.named_parameters()]
        norm_signature = [(name, tuple(p.shape)) for name, p in normalized.named_parameters()]
        self.assertEqual(base_signature, norm_signature)
        self.assertEqual(
            sum(p.numel() for p in baseline.parameters()),
            sum(p.numel() for p in normalized.parameters()),
        )

    def test_public_sum_ema_bank_and_all_rope_modes_forward_backward(self) -> None:
        for temporal in ("sum", "ema", "bank"):
            for rope in ("none", "partial", "full"):
                with self.subTest(temporal=temporal, rope=rope):
                    torch.manual_seed(17)
                    mixer = ThetaScan(self._public_config(temporal=temporal, rope=rope))
                    x = torch.randn(2, 6, 16, requires_grad=True)
                    y = mixer(x)
                    y.square().mean().backward()
                    self.assertTrue(torch.isfinite(y).all())
                    self.assertIsNotNone(x.grad)
                    self.assertTrue(torch.isfinite(x.grad).all())

        thresholded = ThetaScan(
            self._public_config(nonlinearity="relu2_threshold", temporal="ema")
        )
        thresholded(torch.randn(1, 5, 16)).square().mean().backward()

    def test_invalid_public_and_private_combinations_are_rejected(self) -> None:
        invalid_gn = (
            GNConfig(read_normalization="unknown"),
            GNConfig(read_normalization="feature_mass"),
            GNConfig(nonlinearity="silu", read_normalization="w2_feature_mass"),
            GNConfig(jacobian_steps=2, read_normalization="both_feature_mass"),
        )
        for gn in invalid_gn:
            with self.subTest(gn=gn), self.assertRaises(ValueError):
                gn.validate()

        with self.assertRaises(ValueError):
            CoreConfig(read_norm_w1=True)
        with self.assertRaises(ValueError):
            CoreConfig(read_norm=True, write_rule="gn", nonlin="silu")
        with self.assertRaises(ValueError):
            CoreConfig(
                read_norm=True,
                read_norm_w1=True,
                accumulation="ema_gate",
                decay_gate="channel",
                backend="naive",
                )

    def test_direct_two_stage_formula_and_corrected_w2_query(self) -> None:
        dtype = torch.float64
        cfg = CoreConfig(
            d_model=2,
            n_heads=1,
            head_dim=2,
            mem_mult=1,
            depth=1,
            backend="naive",
            rope="none",
            qk_norm=False,
            read_norm=True,
            read_norm_w1=True,
        )
        w1 = torch.tensor([[[1.2, -0.4], [0.3, 1.1]]], dtype=dtype)
        w2 = torch.tensor([[[0.7, -0.2], [0.1, 0.9]]], dtype=dtype)
        wg = torch.zeros_like(w1)
        key = torch.tensor(
            [[[[0.8, 0.2], [-0.1, 1.0], [0.7, -0.5]]]], dtype=dtype
        )
        query = torch.tensor(
            [[[[0.4, 0.9], [1.0, -0.2], [-0.3, 0.8]]]], dtype=dtype
        )
        g = torch.tensor(
            [[[[0.5, -0.3], [-0.2, 0.9], [0.7, 0.4]]]], dtype=dtype
        )
        lam = torch.tensor(
            [[[[0.4, -0.1], [-0.6, 0.8], [0.2, 0.5]]]], dtype=dtype
        )

        def feature(x: torch.Tensor) -> torch.Tensor:
            a = torch.einsum("hmd,bhtd->bhtm", w1, x)
            return torch.relu(rmsnorm(a, cfg.eps)[0]).square()

        feature_key = feature(key)
        streams = Streams(la1=[(key, g)], la2=[(feature_key, lam)])
        actual = engine.dual_read(
            [(w1, w2, wg)], cfg, query, streams, Accumulator("naive")
        )

        expected = []
        wrong_w2_query = []
        last_normalized_c1 = last_literal_c1 = None
        for t in range(query.shape[2]):
            qt = query[:, :, t]
            a0 = torch.einsum("hmd,bhd->bhm", w1, qt)
            phi_q = torch.relu(rmsnorm(a0, cfg.eps)[0]).square()
            phi_i = feature_key[:, :, : t + 1]
            mass = phi_i.sum(dim=2)
            weights1 = torch.einsum("bhik,bhk->bhi", phi_i, phi_q)
            c1 = torch.einsum("bhi,bhiv->bhv", weights1, g[:, :, : t + 1])
            c1 = c1 / ((phi_q * mass).sum(-1, keepdim=True) + cfg.eps)
            hidden = torch.relu(rmsnorm(a0 + c1, cfg.eps)[0]).square()
            weights2 = torch.einsum("bhik,bhk->bhi", phi_i, hidden)
            c2 = torch.einsum("bhi,bhiv->bhv", weights2, lam[:, :, : t + 1])
            c2 = c2 / ((hidden * mass).sum(-1, keepdim=True) + cfg.eps)
            expected.append(qt + torch.einsum("hdm,bhm->bhd", w2, hidden) + c2)

            wrong_weights = torch.einsum("bhik,bhk->bhi", phi_i, phi_q)
            wrong_c2 = torch.einsum(
                "bhi,bhiv->bhv", wrong_weights, lam[:, :, : t + 1]
            )
            wrong_c2 = wrong_c2 / ((phi_q * mass).sum(-1, keepdim=True) + cfg.eps)
            wrong_w2_query.append(
                qt + torch.einsum("hdm,bhm->bhd", w2, hidden) + wrong_c2
            )
            last_normalized_c1 = c1
            last_literal_c1 = torch.einsum(
                "bhid,bhid->bhi", query[:, :, t : t + 1].expand_as(key[:, :, : t + 1]),
                key[:, :, : t + 1],
            )
            last_literal_c1 = torch.einsum(
                "bhi,bhiv->bhv", last_literal_c1, g[:, :, : t + 1]
            )

        expected = torch.stack(expected, dim=2)
        wrong_w2_query = torch.stack(wrong_w2_query, dim=2)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        self.assertGreater((actual - wrong_w2_query).abs().max().item(), 1e-4)
        self.assertGreater(
            (last_normalized_c1 - last_literal_c1).abs().max().item(), 1e-3
        )

        zero_streams = Streams(
            la1=[(key, g)],
            la2=[(torch.zeros_like(feature_key), lam)],
        )
        zero_read = engine.dual_read(
            [(w1, w2, wg)], cfg, query, zero_streams, Accumulator("naive")
        )
        self.assertTrue(torch.isfinite(zero_read).all())

        # Direct EMA recurrence: N1, N2 and z must all receive exactly the same
        # alpha_t before the current write is added.
        log_alpha = torch.tensor([[[[-0.2], [-0.7], [-0.4]]]], dtype=dtype)
        ema_actual = engine.dual_read(
            [(w1, w2, wg)],
            cfg,
            query,
            streams,
            Accumulator(
                "naive", decay_d=log_alpha * 0.11, decay_m=log_alpha
            ),
        )
        mass = torch.zeros(1, 1, 2, dtype=dtype)
        n1 = torch.zeros(1, 1, 2, 2, dtype=dtype)
        n2 = torch.zeros(1, 1, 2, 2, dtype=dtype)
        ema_expected = []
        for t in range(query.shape[2]):
            alpha = log_alpha[:, :, t].exp()
            mass = mass * alpha + feature_key[:, :, t]
            n1 = n1 * alpha.unsqueeze(-1) + torch.einsum(
                "bhv,bhk->bhvk", g[:, :, t], feature_key[:, :, t]
            )
            n2 = n2 * alpha.unsqueeze(-1) + torch.einsum(
                "bhv,bhk->bhvk", lam[:, :, t], feature_key[:, :, t]
            )
            qt = query[:, :, t]
            a0 = torch.einsum("hmd,bhd->bhm", w1, qt)
            phi_q = torch.relu(rmsnorm(a0, cfg.eps)[0]).square()
            c1 = torch.einsum("bhvk,bhk->bhv", n1, phi_q)
            c1 = c1 / ((phi_q * mass).sum(-1, keepdim=True) + cfg.eps)
            hidden = torch.relu(rmsnorm(a0 + c1, cfg.eps)[0]).square()
            c2 = torch.einsum("bhvk,bhk->bhv", n2, hidden)
            c2 = c2 / ((hidden * mass).sum(-1, keepdim=True) + cfg.eps)
            ema_expected.append(
                qt + torch.einsum("hdm,bhm->bhd", w2, hidden) + c2
            )
        torch.testing.assert_close(
            ema_actual,
            torch.stack(ema_expected, dim=2),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_sum_and_ema_match_independent_materialized_oracle(self) -> None:
        for ema in (False, True):
            for backend in ("naive", "quad", "cumsum"):
                with self.subTest(ema=ema, backend=backend):
                    cfg, weights, key, value, query, decay_d, decay_m = \
                        self._core_inputs(backend, ema)
                    streams, _, _ = engine.write_streams(weights, cfg, key, value)
                    acc = Accumulator(backend, decay_d=decay_d, decay_m=decay_m)
                    actual = engine.dual_read(weights, cfg, query, streams, acc)
                    expected = engine.naive_read(
                        weights,
                        cfg,
                        key,
                        value,
                        query,
                        decay_d=decay_d,
                        decay_m=decay_m,
                    )
                    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)

        # Stream indexing and the shared mass are per memory block, not global.
        cfg, weights, key, value, query, _, _ = self._core_inputs("quad", depth=2)
        streams, _, _ = engine.write_streams(weights, cfg, key, value)
        actual = engine.dual_read(weights, cfg, query, streams, Accumulator("quad"))
        expected = engine.naive_read(weights, cfg, key, value, query)
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)

    def test_fade_fast_and_stale_match_full_two_stage_oracle(self) -> None:
        for mode in ("fast", "stale"):
            for backend in ("naive", "quad", "cumsum"):
                with self.subTest(mode=mode, backend=backend):
                    cfg, weights, key, value, query, _, _ = self._core_inputs(backend)
                    cfg.read_fade = True
                    cfg.read_fade_mode = mode
                    cfg.validate()
                    streams, _, _ = engine.write_streams(weights, cfg, key, value)
                    heads, time = cfg.n_heads, key.shape[2]
                    eta = torch.tensor([0.31, 0.57], dtype=key.dtype)
                    log_alpha = torch.tensor([-0.23, -0.61], dtype=key.dtype)
                    decay = log_alpha.view(1, heads, 1, 1).expand(1, heads, time, 1)
                    slow = Accumulator(backend)
                    fast = Accumulator(backend, decay_d=decay, decay_m=decay)
                    actual = engine.dual_read(weights, cfg, query, streams, slow)
                    if mode == "fast":
                        recency = engine.dual_read(
                            weights, cfg, query, streams, FadeFast(fast, log_alpha)
                        )
                        actual = actual + eta.view(1, heads, 1, 1) * (recency - actual)
                    else:
                        stale = engine.dual_read(
                            weights,
                            cfg,
                            query,
                            streams,
                            FadeStale(slow, fast, log_alpha),
                        )
                        base = engine.dual_read(
                            weights, cfg, query, streams, NullAccumulator()
                        )
                        actual = actual - eta.view(1, heads, 1, 1) * (stale - base)
                    expected = engine.naive_read(
                        weights, cfg, key, value, query, fade=(eta, log_alpha)
                    )
                    tol = 1e-7 if mode == "stale" and backend == "cumsum" else 1e-9
                    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("fla") is not None,
        "FLA parity requires CUDA and flash-linear-attention",
    )
    def test_fla_matches_naive_for_both_feature_mass(self) -> None:
        # This remains skipped on CPU CI but runs automatically in an FLA GPU job.
        # EMA is separate from sum because mass_cum itself dispatches through FLA.
        for ema in (False, True):
            with self.subTest(temporal="ema" if ema else "sum"):
                cfg, weights, key, value, query, decay_d, decay_m = \
                    self._core_inputs("naive", ema=ema)
                weights = [tuple(t.float().cuda() for t in block) for block in weights]
                key, value, query = (t.float().cuda() for t in (key, value, query))
                decay_d = None if decay_d is None else decay_d.float().cuda()
                decay_m = None if decay_m is None else decay_m.float().cuda()
                streams, _, _ = engine.write_streams(weights, cfg, key, value)
                expected = engine.dual_read(
                    weights,
                    cfg,
                    query,
                    streams,
                    Accumulator("naive", decay_d=decay_d, decay_m=decay_m),
                )
                actual = engine.dual_read(
                    weights,
                    cfg,
                    query,
                    streams,
                    Accumulator("fla", decay_d=decay_d, decay_m=decay_m),
                )
                torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)

        for mode in ("fast", "stale"):
            with self.subTest(temporal=f"fade-{mode}"):
                cfg, weights, key, value, query, _, _ = self._core_inputs("naive")
                cfg.read_fade = True
                cfg.read_fade_mode = mode
                cfg.validate()
                weights = [tuple(t.float().cuda() for t in block) for block in weights]
                key, value, query = (t.float().cuda() for t in (key, value, query))
                streams, _, _ = engine.write_streams(weights, cfg, key, value)
                eta = torch.tensor([0.31, 0.57], device="cuda")
                log_alpha = torch.tensor([-0.23, -0.61], device="cuda")
                decay = log_alpha.view(1, 2, 1, 1).expand(1, 2, key.shape[2], 1)

                def fade_read(backend: str) -> torch.Tensor:
                    slow = Accumulator(backend)
                    fast = Accumulator(backend, decay_d=decay, decay_m=decay)
                    result = engine.dual_read(weights, cfg, query, streams, slow)
                    if mode == "fast":
                        recency = engine.dual_read(
                            weights,
                            cfg,
                            query,
                            streams,
                            FadeFast(fast, log_alpha),
                        )
                        return result + eta.view(1, 2, 1, 1) * (recency - result)
                    stale = engine.dual_read(
                        weights,
                        cfg,
                        query,
                        streams,
                        FadeStale(slow, fast, log_alpha),
                    )
                    base = engine.dual_read(
                        weights, cfg, query, streams, NullAccumulator()
                    )
                    return result - eta.view(1, 2, 1, 1) * (stale - base)

                torch.testing.assert_close(
                    fade_read("fla"), fade_read("naive"), atol=5e-4, rtol=5e-4
                )


if __name__ == "__main__":
    unittest.main()
