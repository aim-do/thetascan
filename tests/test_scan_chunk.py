"""Exactness, gradient and footprint coverage for the chunked scan backend."""
from __future__ import annotations

import math
import unittest
from unittest import mock

import torch

from thetascan import ThetaScan, ThetaScanConfig
from thetascan._core.ops import scan_chunk, scan_naive, scan_quad
from thetascan._core.ops.interface import (Accumulator, FadeFast, FadeStale,
                                           NullAccumulator, ema_cumsum)

CHUNKS = (1, 3, 8, 64, 128)


def _streams(T: int = 37, Dk: int = 5, Dv: int = 4, B: int = 2, H: int = 3):
    torch.manual_seed(20260725)
    q = torch.randn(B, H, T, Dk, dtype=torch.float64)
    k = torch.randn(B, H, T, Dk, dtype=torch.float64)
    v = torch.randn(B, H, T, Dv, dtype=torch.float64)
    log_alpha = torch.tensor([-0.05, -0.4, -2.0], dtype=torch.float64)
    return q, k, v, log_alpha


def _as_stream(log_alpha: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    B, H, T = like.shape[:3]
    return log_alpha.view(1, H, 1, 1).expand(B, H, T, 1)


class SavedBytes:
    """Sum the bytes autograd retains for backward inside the block."""

    def __init__(self) -> None:
        self.total = 0
        self.largest = 0

    def __enter__(self) -> "SavedBytes":
        def pack(tensor: torch.Tensor) -> torch.Tensor:
            size = tensor.numel() * tensor.element_size()
            self.total += size
            self.largest = max(self.largest, size)
            return tensor

        self._hooks = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._hooks.__exit__(*exc)


class ChunkedScanExactnessTests(unittest.TestCase):
    def test_plain_sum_matches_the_sequential_oracle(self) -> None:
        q, k, v, _ = _streams()
        expected = scan_naive.linattn(q, k, v)
        for chunk in CHUNKS:
            with self.subTest(chunk=chunk):
                torch.testing.assert_close(
                    scan_chunk.linattn(q, k, v, chunk=chunk), expected,
                    rtol=1e-12, atol=1e-12,
                )

    def test_static_retention_matches_the_sequential_oracle(self) -> None:
        q, k, v, log_alpha = _streams()
        expected = scan_naive.linattn(q, k, v, _as_stream(log_alpha, q))
        for chunk in CHUNKS:
            with self.subTest(chunk=chunk):
                torch.testing.assert_close(
                    scan_chunk.linattn(q, k, v, chunk=chunk, log_alpha=log_alpha),
                    expected, rtol=1e-12, atol=1e-12,
                )

    def test_per_token_retention_stream_matches_the_sequential_oracle(self) -> None:
        """The general scalar stream is the same recurrence, tile-local."""
        q, k, v, _ = _streams()
        torch.manual_seed(11)
        decay = -torch.rand(*q.shape[:3], 1, dtype=torch.float64)
        expected = scan_naive.linattn(q, k, v, decay)
        for chunk in CHUNKS:
            with self.subTest(chunk=chunk):
                torch.testing.assert_close(
                    scan_chunk.linattn(q, k, v, decay, chunk=chunk),
                    expected, rtol=1e-12, atol=1e-12,
                )

    def test_static_and_stream_forms_agree(self) -> None:
        q, k, v, log_alpha = _streams()
        stream = _as_stream(log_alpha, q)
        for chunk in CHUNKS:
            with self.subTest(chunk=chunk):
                torch.testing.assert_close(
                    scan_chunk.linattn(q, k, v, chunk=chunk, log_alpha=log_alpha),
                    scan_chunk.linattn(q, k, v, stream, chunk=chunk),
                    rtol=1e-12, atol=1e-12,
                )

    def test_single_tile_reproduces_quad_bit_for_bit(self) -> None:
        """At chunk >= T the local and global cumulative logs coincide."""
        q, k, v, log_alpha = _streams()
        T = q.shape[2]
        stream = _as_stream(log_alpha, q)
        for chunk in (T, T + 1, 4096):
            with self.subTest(chunk=chunk):
                self.assertTrue(torch.equal(
                    scan_chunk.linattn(q, k, v, chunk=chunk),
                    scan_quad.linattn(q, k, v),
                ))
                self.assertTrue(torch.equal(
                    scan_chunk.linattn(q, k, v, stream, chunk=chunk),
                    scan_quad.linattn(q, k, v, stream),
                ))

    def test_a_short_final_tile_is_handled(self) -> None:
        """T deliberately indivisible by every chunk exercised above."""
        q, k, v, log_alpha = _streams(T=37)
        self.assertNotEqual(37 % 8, 0)
        torch.testing.assert_close(
            scan_chunk.linattn(q, k, v, chunk=8, log_alpha=log_alpha),
            scan_naive.linattn(q, k, v, _as_stream(log_alpha, q)),
            rtol=1e-12, atol=1e-12,
        )

    def test_reduced_precision_stays_close_to_the_oracle(self) -> None:
        q, k, v, log_alpha = _streams(T=64)
        expected = scan_naive.linattn(q, k, v, _as_stream(log_alpha, q))
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                actual = scan_chunk.linattn(
                    q.to(dtype), k.to(dtype), v.to(dtype), chunk=16,
                    log_alpha=log_alpha.to(dtype),
                )
                self.assertEqual(actual.dtype, dtype)
                scale = expected.abs().max()
                error = (actual.to(torch.float64) - expected).abs().max()
                tolerance = 1e-5 if dtype is torch.float32 else 5e-2
                self.assertLess(error / scale, tolerance)

    def test_bfloat16_chunk_512_static_bank_has_no_position_aliases(self) -> None:
        """A 512-wide bank tile must form its exponents before BF16 rounding."""
        alpha = 0.9
        log_alpha = torch.tensor([alpha]).log()
        tile, _ = scan_chunk._static_weights(
            log_alpha, 512, torch.bfloat16, torch.device("cpu")
        )
        subdiagonal = torch.diagonal(tile[0], offset=-1)
        expected_weight = torch.exp(log_alpha).to(torch.bfloat16)
        self.assertTrue(torch.equal(
            subdiagonal,
            expected_weight.expand_as(subdiagonal),
        ))

        # Pin the end-to-end recurrence too: the old BF16-index construction
        # oscillated by roughly one full unit after position 256.
        length = 600
        stream = torch.ones(1, 1, length, 1, dtype=torch.bfloat16)
        actual = scan_chunk.linattn(
            stream, stream, stream, chunk=512, log_alpha=log_alpha
        ).float().flatten()
        position = torch.arange(1, length + 1, dtype=torch.float64)
        expected = (
            (1.0 - alpha ** position) / (1.0 - alpha)
        ).float()
        self.assertLess(float((actual - expected).abs().max()), 0.1)

    def test_bfloat16_chunk_512_dynamic_ema_has_no_log_cumsum_aliases(self) -> None:
        """A per-token EMA must differ phases in FP32, then round its weights."""
        alpha = 0.982
        log_alpha = torch.tensor(alpha).log()
        decay = torch.full(
            (1, 1, 512, 1), float(log_alpha), dtype=torch.bfloat16
        )
        tile, _ = scan_chunk._tile_weights(
            decay,
            tiles=1,
            width=512,
            heads=1,
            static=None,
            dtype=torch.bfloat16,
        )
        subdiagonal = torch.diagonal(tile[0, 0, 0], offset=-1)
        expected_weight = torch.exp(decay.reshape(-1)[0].float()).to(
            torch.bfloat16
        )
        self.assertTrue(torch.equal(
            subdiagonal,
            expected_weight.expand_as(subdiagonal),
        ))

        length = 600
        stream = torch.ones(1, 1, length, 1, dtype=torch.bfloat16)
        dynamic = decay[:, :, :1].expand(1, 1, length, 1)
        actual = scan_chunk.linattn(
            stream, stream, stream, dynamic, chunk=512
        ).float().flatten()
        # The input log-retention is BF16, while its exponent is evaluated in
        # FP32.  Use that effective alpha rather than exponentiating in BF16 a
        # second time.
        rounded_alpha = math.exp(float(decay.reshape(-1)[0]))
        position = torch.arange(1, length + 1, dtype=torch.float64)
        expected = (
            (1.0 - rounded_alpha ** position) / (1.0 - rounded_alpha)
        ).float()
        self.assertLess(float((actual - expected).abs().max()), 0.35)

    def test_bfloat16_chunk_512_static_mass_has_no_position_aliases(self) -> None:
        """The normalized bank denominator must use the numerator's FP32 phases."""
        alpha = 0.9
        log_alpha = torch.tensor([alpha]).log()
        length = 600
        stream = torch.ones(1, 1, length, 1, dtype=torch.bfloat16)
        actual = ema_cumsum(
            stream, log_alpha, chunk=512
        ).float().flatten()
        position = torch.arange(1, length + 1, dtype=torch.float64)
        expected = (
            (1.0 - alpha ** position) / (1.0 - alpha)
        ).float()
        # The old BF16 arange path missed this by >1.0 around position 256.
        self.assertLess(float((actual - expected).abs().max()), 0.1)

    def test_rejects_channel_decay_and_a_degenerate_tile(self) -> None:
        q, k, v, _ = _streams()
        channel = -torch.rand_like(k)
        with self.assertRaises(NotImplementedError):
            scan_chunk.linattn(q, k, v, channel)
        with self.assertRaises(ValueError):
            scan_chunk.linattn(q, k, v, chunk=0)
        with self.assertRaises(ValueError):
            scan_chunk.linattn(
                q, k, v, chunk=8, log_alpha=torch.zeros(7, dtype=torch.float64)
            )


class ChunkedScanCausalityTests(unittest.TestCase):
    """No output may depend on a later position, not even by rounding.

    The tile axis carries an exclusive scan of per-tile updates. Forming it by
    subtracting each tile's own term from the inclusive scan is algebraically
    exact and numerically is not: the state entering a tile then depends on that
    tile's own tokens at the level of rounding, which makes the scan causal only
    to within round-off. Nothing else notices -- every oracle comparison has a
    tolerance far above it -- so it is tested by perturbation.
    """

    def _leak(self, chunk: int, cut: int, **kwargs) -> float:
        q, k, v, _ = _streams(T=24, B=1, H=2)
        base = scan_chunk.linattn(q, k, v, chunk=chunk, **kwargs)
        for stream in (k, v):
            stream[:, :, cut:] += 2.0
        moved = scan_chunk.linattn(q, k, v, chunk=chunk, **kwargs)
        return float((moved - base)[:, :, :cut].abs().max())

    def test_a_later_key_or_value_never_moves_an_earlier_output(self) -> None:
        for chunk in (3, 8, 24, 128):
            for cut in (1, 9, 16):
                with self.subTest(chunk=chunk, cut=cut):
                    self.assertEqual(self._leak(chunk, cut), 0.0)

    def test_the_retained_views_are_causal_too(self) -> None:
        _, _, _, log_alpha = _streams(T=24, B=1, H=2)
        alpha = log_alpha[:2]
        for chunk in (3, 8):
            with self.subTest(chunk=chunk):
                self.assertEqual(
                    self._leak(chunk, 9, log_alpha=alpha), 0.0
                )


class ChunkedScanGradientTests(unittest.TestCase):
    def _grads(self, fn):
        q, k, v, log_alpha = _streams()
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v)]
        alpha = log_alpha.clone().requires_grad_(True)
        fn(*leaves, alpha).square().sum().backward()
        return [x.grad for x in leaves] + [alpha.grad]

    def test_gradients_match_the_quadratic_backend(self) -> None:
        reference = self._grads(
            lambda q, k, v, a: scan_quad.linattn(q, k, v, _as_stream(a, q))
        )
        for chunk in (3, 8, 64):
            actual = self._grads(
                lambda q, k, v, a, c=chunk: scan_chunk.linattn(
                    q, k, v, chunk=c, log_alpha=a
                )
            )
            for name, want, got in zip(
                ("q", "k", "v", "log_alpha"), reference, actual
            ):
                with self.subTest(chunk=chunk, wrt=name):
                    torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-9)

    def test_the_retention_gradient_is_not_dropped(self) -> None:
        """A tile-local retention must still reach the shared parameter."""
        grads = self._grads(
            lambda q, k, v, a: scan_chunk.linattn(q, k, v, chunk=8, log_alpha=a)
        )
        self.assertIsNotNone(grads[-1])
        self.assertTrue((grads[-1].abs() > 0).all())

    def test_bfloat16_chunk_512_retention_gradients_match_closed_form(self) -> None:
        """FP32 phase construction must preserve static and dynamic gradients."""
        length = 512
        stream = torch.ones(1, 1, length, 1, dtype=torch.bfloat16)
        cases = (("static", 0.9), ("dynamic", 0.982))
        for mode, alpha in cases:
            with self.subTest(mode=mode):
                log_alpha = torch.tensor(
                    [math.log(alpha)],
                    dtype=torch.float32,
                    requires_grad=True,
                )
                if mode == "static":
                    output = scan_chunk.linattn(
                        stream,
                        stream,
                        stream,
                        chunk=512,
                        log_alpha=log_alpha,
                    )
                else:
                    decay = log_alpha.to(torch.bfloat16).view(
                        1, 1, 1, 1
                    ).expand(1, 1, length, 1)
                    output = scan_chunk.linattn(
                        stream, stream, stream, decay, chunk=512
                    )
                output[0, 0, -1, 0].float().backward()

                expected = sum(
                    power * alpha ** power
                    for power in range(1, length + 1)
                )
                self.assertIsNotNone(log_alpha.grad)
                actual = float(log_alpha.grad)
                self.assertTrue(torch.isfinite(log_alpha.grad).all())
                self.assertGreater(actual, 0.0)
                self.assertLess(abs(actual - expected) / expected, 5e-3)


class ChunkedScanFootprintTests(unittest.TestCase):
    """The point of the backend is what autograd has to keep alive."""

    SHAPE = dict(B=1, H=2, T=512, Dk=64, Dv=33)

    def _saved(self, fn) -> SavedBytes:
        shape = self.SHAPE
        torch.manual_seed(5)
        q = torch.randn(shape["B"], shape["H"], shape["T"], shape["Dk"])
        k = torch.randn_like(q)
        v = torch.randn(shape["B"], shape["H"], shape["T"], shape["Dv"])
        for tensor in (q, k, v):
            tensor.requires_grad_(True)
        with SavedBytes() as saved:
            fn(q, k, v)
        return saved

    def test_the_largest_retained_tensor_is_tiled_not_quadratic(self) -> None:
        """Retained bytes must scale with ``T * chunk``, never with ``T * T``.

        Every tile is issued in one batched call, so the largest retained tensor
        is the whole ``[B,H,T/chunk,chunk,chunk]`` score rather than a single
        ``[B,H,chunk,chunk]`` tile.  That is the same total -- the loop retained
        one tile per iteration -- in one allocation instead of ``T/chunk`` of
        them, and it is still linear in sequence length, which is the property
        this backend exists for.
        """
        shape = self.SHAPE
        chunk = 64
        log_alpha = torch.full((shape["H"],), -0.1)
        quad = self._saved(
            lambda q, k, v: scan_quad.linattn(
                q, k, v, log_alpha.view(1, -1, 1, 1).expand(*q.shape[:3], 1)
            )
        )
        chunked = self._saved(
            lambda q, k, v: scan_chunk.linattn(
                q, k, v, chunk=chunk, log_alpha=log_alpha
            )
        )
        score_bytes = shape["B"] * shape["H"] * shape["T"] * shape["T"] * 4
        tiled_bytes = shape["B"] * shape["H"] * shape["T"] * chunk * 4
        self.assertGreaterEqual(quad.largest, score_bytes)
        self.assertLessEqual(chunked.largest, tiled_bytes)
        self.assertLess(chunked.largest, score_bytes)
        self.assertLess(chunked.total, quad.total)

    def test_the_footprint_grows_linearly_in_sequence_length(self) -> None:
        """Doubling T must not quadruple what backward retains."""
        log_alpha = torch.full((self.SHAPE["H"],), -0.1)

        def measure(T: int) -> int:
            self.SHAPE = {**self.SHAPE, "T": T}
            return self._saved(
                lambda q, k, v: scan_chunk.linattn(
                    q, k, v, chunk=64, log_alpha=log_alpha
                )
            ).total

        base = measure(512)
        doubled = measure(1024)
        self.SHAPE = {**self.SHAPE, "T": 512}
        self.assertLess(doubled, 2.4 * base)


class ChunkedBackendIntegrationTests(unittest.TestCase):
    """Every promoted preset must agree with the quadratic reference."""

    PRESETS = (
        "gn_reference_v0_1",
        "gn_expanded_reference_v0_1",
        "kernel_expanded_reference_v0_1",
    )

    def _mixer(self, preset: str, backend: str, scan_chunk_size: int = 16):
        config = ThetaScanConfig.from_dict({
            "preset": preset, "d_model": 64, "n_heads": 2,
            "runtime": {"backend": backend, "scan_chunk": scan_chunk_size},
        })
        torch.manual_seed(4242)
        return ThetaScan(config).double()

    def test_presets_match_quad_within_float64_tolerance(self) -> None:
        torch.manual_seed(7)
        x = torch.randn(2, 40, 64, dtype=torch.float64)
        for preset in self.PRESETS:
            for chunk in (16, 64):
                with self.subTest(preset=preset, scan_chunk=chunk):
                    reference = self._mixer(preset, "quad")(x)
                    actual = self._mixer(preset, "chunk", chunk)(x)
                    torch.testing.assert_close(
                        actual, reference, rtol=1e-10, atol=1e-10
                    )

    def test_auto_selects_chunk_for_every_preset_on_cpu(self) -> None:
        torch.manual_seed(7)
        x = torch.randn(1, 24, 64, dtype=torch.float64)
        for preset in self.PRESETS:
            with self.subTest(preset=preset):
                mixer = self._mixer(preset, "auto")
                mixer(x)
                self.assertEqual(mixer._core._backend, "chunk")

    def test_a_single_tile_reproduces_quad_for_the_whole_mixer(self) -> None:
        torch.manual_seed(7)
        x = torch.randn(2, 40, 64, dtype=torch.float64)
        for preset in self.PRESETS:
            with self.subTest(preset=preset):
                torch.testing.assert_close(
                    self._mixer(preset, "chunk", 4096)(x),
                    self._mixer(preset, "quad")(x),
                    rtol=1e-13, atol=1e-13,
                )

    def test_preset_gradients_match_quad(self) -> None:
        torch.manual_seed(7)
        x = torch.randn(2, 24, 64, dtype=torch.float64)

        def grads(backend: str, chunk: int = 8):
            mixer = self._mixer("kernel_expanded_reference_v0_1", backend, chunk)
            inp = x.clone().requires_grad_(True)
            mixer(inp).square().sum().backward()
            named = {n: p.grad for n, p in mixer.named_parameters()
                     if p.grad is not None}
            return inp.grad, named

        want_x, want = grads("quad")
        got_x, got = grads("chunk")
        torch.testing.assert_close(got_x, want_x, rtol=1e-9, atol=1e-9)
        self.assertEqual(set(got), set(want))
        for name in want:
            with self.subTest(parameter=name):
                torch.testing.assert_close(
                    got[name], want[name], rtol=1e-8, atol=1e-8
                )


if __name__ == "__main__":
    unittest.main()


class MultiViewScanTests(unittest.TestCase):
    """Sharing the score tile must not change any view's result."""

    def _views(self):
        q, k, v, log_alpha = _streams(T=40, Dk=5, Dv=4, B=2, H=3)
        second = torch.tensor([-0.2, -0.8, -1.5], dtype=torch.float64)
        return q, k, v, (None, log_alpha, second)

    def test_shared_tile_is_bitwise_identical_to_separate_scans(self) -> None:
        q, k, v, retentions = self._views()
        for chunk in (3, 8, 40, 128):
            shared = scan_chunk.linattn_views(q, k, v, retentions, chunk)
            for index, retention in enumerate(retentions):
                with self.subTest(chunk=chunk, view=index):
                    separate = scan_chunk.linattn(
                        q, k, v, chunk=chunk, log_alpha=retention
                    )
                    self.assertTrue(torch.equal(shared[index], separate))

    def test_each_view_matches_the_sequential_oracle(self) -> None:
        q, k, v, retentions = self._views()
        shared = scan_chunk.linattn_views(q, k, v, retentions, 8)
        for index, retention in enumerate(retentions):
            stream = None if retention is None else _as_stream(retention, q)
            with self.subTest(view=index):
                torch.testing.assert_close(
                    shared[index], scan_naive.linattn(q, k, v, stream),
                    rtol=1e-12, atol=1e-12,
                )

    def test_gradients_reach_every_retention(self) -> None:
        q, k, v, retentions = self._views()
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v)]
        alphas = [
            None if r is None else r.clone().requires_grad_(True)
            for r in retentions
        ]
        outs = scan_chunk.linattn_views(*leaves, tuple(alphas), 8)
        sum(out.square().sum() for out in outs).backward()
        for leaf in leaves:
            self.assertTrue(torch.isfinite(leaf.grad).all())
        for index, alpha in enumerate(alphas):
            if alpha is None:
                continue
            with self.subTest(view=index):
                self.assertIsNotNone(alpha.grad)
                self.assertTrue((alpha.grad.abs() > 0).all())

    def test_one_view_is_the_plain_scan(self) -> None:
        q, k, v, _ = self._views()
        self.assertEqual(len(scan_chunk.linattn_views(q, k, v, (), 8)), 0)
        self.assertTrue(torch.equal(
            scan_chunk.linattn_views(q, k, v, (None,), 8)[0],
            scan_chunk.linattn(q, k, v, chunk=8),
        ))

    def test_mixed_static_and_stream_views_agree(self) -> None:
        """A per-token stream view may share a tile with a static one."""
        q, k, v, log_alpha = _streams(T=24)
        stream = _as_stream(log_alpha, q)
        shared = scan_chunk.linattn_views(q, k, v, (None, log_alpha, stream), 8)
        torch.testing.assert_close(shared[1], shared[2], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            shared[0], scan_naive.linattn(q, k, v), rtol=1e-12, atol=1e-12
        )


class GroupedReadEquivalenceTests(unittest.TestCase):
    """A grouped multi-view read must equal one read per view.

    Grouping shares the score tile while retaining one independent state per
    distinct temporal retention.  Its retention-slot bookkeeping is also part
    of the compiled graph, so these tests pin both the algebra and sharing.
    """

    PRESETS = (
        "gn_reference_v0_1",
        "gn_expanded_reference_v0_1",
        "kernel_expanded_reference_v0_1",
    )

    def _fixture(self, preset: str, backend: str):
        from thetascan._core.ops import engine

        config = ThetaScanConfig.from_dict({
            "preset": preset, "d_model": 64, "n_heads": 2,
            "runtime": {"backend": backend, "scan_chunk": 16},
        })
        torch.manual_seed(2718)
        core = ThetaScan(config).double()._core
        cfg = core.cfg
        shape = (2, cfg.n_heads, 24, cfg.head_dim)
        key = torch.randn(*shape, dtype=torch.float64)
        value = torch.randn(*shape, dtype=torch.float64)
        query = torch.randn(*shape, dtype=torch.float64)
        weights = core._weights()
        streams, _, _ = engine.write_streams(weights, cfg, key, value)
        return engine, cfg, weights, query, streams

    def _views(self, backend: str, cfg, batch: int, heads: int, T: int):
        alphas = (
            torch.tensor([-0.15, -0.6], dtype=torch.float64),
            torch.tensor([-0.05, -1.2], dtype=torch.float64),
        )
        slow = Accumulator(backend, chunk=cfg.scan_chunk)
        views = [slow]
        for alpha in alphas:
            stream = alpha.view(1, heads, 1, 1).expand(batch, heads, T, 1)
            fast = Accumulator(
                backend, decay_d=stream, decay_m=stream,
                chunk=cfg.scan_chunk, static_log_alpha=alpha,
            )
            views.append(FadeFast(fast, alpha, chunk=cfg.scan_chunk))
            views.append(FadeStale(slow, fast, alpha, cfg.scan_chunk))
        views.append(NullAccumulator())
        return views

    def test_grouped_read_matches_one_read_per_view(self) -> None:
        for preset in self.PRESETS:
            for backend in ("chunk", "quad"):
                with self.subTest(preset=preset, backend=backend):
                    engine, cfg, weights, query, streams = self._fixture(
                        preset, backend
                    )
                    views = self._views(
                        backend, cfg, query.shape[0], query.shape[1],
                        query.shape[2],
                    )
                    grouped = engine.dual_read_views(
                        weights, cfg, query, streams, views
                    )
                    self.assertEqual(len(grouped), len(views))
                    for index, view in enumerate(views):
                        separate = engine.dual_read(
                            weights, cfg, query, streams, view
                        )
                        torch.testing.assert_close(
                            grouped[index], separate, rtol=1e-12, atol=1e-12
                        )

    def test_grouping_issues_one_scan_call_for_all_views(self) -> None:
        """The sharing must actually happen, not merely be available."""
        from thetascan._core.ops import interface

        engine, cfg, weights, query, streams = self._fixture(
            "kernel_expanded_reference_v0_1", "chunk"
        )
        views = self._views(
            "chunk", cfg, query.shape[0], query.shape[1], query.shape[2]
        )
        sizes: list[int] = []
        original = interface.scan_views

        def counting(group, *args, **kwargs):
            sizes.append(len(group))
            return original(group, *args, **kwargs)

        with mock.patch.object(interface, "scan_views", counting):
            engine.dual_read_views(weights, cfg, query, streams, views)
        # Kernel memory shares its whole query pipeline, so every view arrives in
        # one grouped call rather than one call each.
        self.assertEqual(max(sizes), len(views))

    def test_a_zero_memory_view_does_not_pick_the_backend(self) -> None:
        """A view that scans nothing must not decide the execution context."""
        from thetascan._core.ops import interface

        query = torch.randn(1, 2, 8, 3, dtype=torch.float64)
        key = torch.randn(1, 2, 8, 3, dtype=torch.float64)
        value = torch.randn(1, 2, 8, 4, dtype=torch.float64)
        views = (NullAccumulator(), Accumulator("naive"))
        outs = interface.scan_views(views, query, key, value, "m")
        self.assertTrue(torch.equal(outs[0], torch.zeros_like(outs[0])))
        torch.testing.assert_close(
            outs[1], scan_naive.linattn(query, key, value),
            rtol=1e-12, atol=1e-12,
        )


class CompileStabilityTests(unittest.TestCase):
    """The forward must trace once, not once per step.

    A guard on mutating module state -- a lazily cached backend, a container of
    freshly allocated tensors read from a closure -- makes TorchDynamo invalidate
    its cache every step, recompile until it hits its limit, and then fall back
    to eager. That costs far more than any backend choice, and it is invisible
    unless something counts traces. `backend="eager"` exercises the tracer
    without needing a C++ toolchain.
    """

    def _traces(self, temporal: dict, blend: float) -> int:
        import torch._dynamo as dynamo

        config = ThetaScanConfig.from_dict({
            "preset": "kernel_expanded_reference_v0_1",
            "d_model": 64, "n_heads": 2,
            "temporal": temporal,
            "runtime": {"backend": "chunk", "scan_chunk": 16},
        })
        torch.manual_seed(11)
        mixer = ThetaScan(config)
        with torch.no_grad():
            if mixer._core.fade_eta is not None:
                mixer._core.fade_eta.fill_(blend)
        dynamo.reset()
        dynamo.utils.counters.clear()
        compiled = torch.compile(mixer, backend="eager", dynamic=False)
        x = torch.randn(1, 32, 64)
        for _ in range(6):
            compiled(x).square().mean().backward()
            compiled.zero_grad(set_to_none=True)
        return dynamo.utils.counters["stats"].get("unique_graphs", 0)

    def test_every_temporal_mode_traces_once(self) -> None:
        for label, temporal, blend in (
            ("sum", {"mode": "sum"}, 0.0),
            ("ema", {"mode": "ema"}, 0.0),
            ("bank", {}, 0.3),
        ):
            with self.subTest(temporal=label):
                self.assertEqual(self._traces(temporal, blend), 1)

    def test_retained_modes_are_one_full_forward_graph(self) -> None:
        """EMA/bank may not silently fall through a Dynamo graph break.

        The trace-count test catches recompilation across steps.  ``fullgraph``
        complements it by making any break inside the first retained forward a
        hard error; running backward also exercises the differentiable
        retention path used by training.
        """
        import torch._dynamo as dynamo

        for label, temporal, blend in (
            ("ema", {"mode": "ema"}, 0.0),
            ("bank", {}, 0.3),
        ):
            with self.subTest(temporal=label):
                config = ThetaScanConfig.from_dict({
                    "preset": "kernel_expanded_reference_v0_1",
                    "d_model": 64,
                    "n_heads": 2,
                    "temporal": temporal,
                    "runtime": {"backend": "chunk", "scan_chunk": 16},
                })
                torch.manual_seed(11)
                mixer = ThetaScan(config)
                with torch.no_grad():
                    if mixer._core.fade_eta is not None:
                        mixer._core.fade_eta.fill_(blend)
                dynamo.reset()
                compiled = torch.compile(
                    mixer, backend="eager", dynamic=False, fullgraph=True
                )
                x = torch.randn(1, 32, 64)
                compiled(x).square().mean().backward()
