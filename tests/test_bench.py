"""The benchmark must time the right computation: shapes match the configs, and ONNX Runtime's
2-bit MatMulNBits really computes a ternary matrix product with our packing."""

import numpy as np
import pytest

from ternavla.bench.lm_graph import default_zero_point, pack_nbits, ternary_codes
from ternavla.shapes import BITCPM_05B, BITCPM_1B, BITNET_2B4T, BUDGETS


def test_parameter_counts_match_published_models():
    # BitNet 2B4T: 2,412,820,480 total minus the 128,256 x 2,560 tied embedding.
    # The remaining 440,320 are norm weights: per layer input/post-attention norms plus BitNet's
    # two sub-layer norms (2560 + 2560 + 2560 + 6912), and the final norm.
    norms = 30 * (3 * 2_560 + 6_912) + 2_560
    assert BITNET_2B4T.ternary_params + BITNET_2B4T.embedding_params + norms == 2_412_820_480
    assert round(BITCPM_1B.ternary_params / 1e9, 2) == 1.32     # card: 1.32 B non-embedding
    assert 0.35e9 < BITCPM_05B.ternary_params < 0.37e9


def test_token_budgets():
    assert BUDGETS["S"].total == 43
    assert BUDGETS["M"].total == BUDGETS["M-lite"].total == 91


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_pack_layout_lowest_bits_first(bits):
    codes = (np.arange(32).reshape(1, 32) % (1 << bits)).astype(np.uint8)
    b = pack_nbits(codes, bits, 32)
    per = 8 // bits
    assert b.shape == (1, 1, 32 * bits // 8)
    first = b[0, 0, 0]
    for i in range(per):
        assert (first >> (bits * i)) & ((1 << bits) - 1) == codes[0, i]


@pytest.mark.parametrize("accuracy_level", [0, 4])
@pytest.mark.parametrize("block_size", [32, 128])
def test_matmulnbits_2bit_is_a_ternary_matmul(accuracy_level, block_size):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(1)
    k, n, t = 512, 384, 37
    codes = ternary_codes(rng, n, k, 2)
    w = codes.astype(np.float32) - default_zero_point(2)          # {-1, 0, +1}, shape (N, K)
    assert set(np.unique(w)) <= {-1.0, 0.0, 1.0}
    scale = 0.05
    x = rng.standard_normal((t, k)).astype(np.float32)
    ref = x @ (w.T * scale)

    node = helper.make_node("MatMulNBits", ["x", "B", "S"], ["y"], domain="com.microsoft",
                            K=k, N=n, bits=2, block_size=block_size, accuracy_level=accuracy_level)
    graph = helper.make_graph(
        [node], "t", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [t, k])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [t, n])],
        initializer=[numpy_helper.from_array(pack_nbits(codes, 2, block_size), "B"),
                     numpy_helper.from_array(np.full((n, k // block_size), scale, np.float32), "S")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                                    helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 9
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    y = sess.run(None, {"x": x})[0]
    rel = np.abs(y - ref).max() / np.abs(ref).max()
    # fp32 path is exact up to summation order; the int8-activation path is within int8 error.
    assert rel < (1e-5 if accuracy_level == 0 else 2e-2), rel
