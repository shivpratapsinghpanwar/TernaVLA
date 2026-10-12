"""The language-model part of one action step, rebuilt at real shapes for timing.

One layer of the stack, per token row (attention scores are omitted: at <= 128 tokens they are
< 1 % of the work; every projection that holds ternary weights is included):

    h  = norm(x);  qkv = h @ W_qkv;  a = norm(qkv[:, :q_dim]);  x = x + a @ W_o
    h  = norm(x);  g, u = split(h @ W_gate_up);  m = norm(act(g) * u);  x = x + m @ W_down

``act`` is relu² for BitNet and SiLU for the Llama-style BitCPM models; the inner norms stand for
BitNet's sub-layer norms and keep random-weight activations at a sane scale (no denormals).

Backends:
- ``torch-fp32``   F.linear with fp32 ternary values (what an unoptimised PyTorch deployment does)
- ``torch-int8``   per-token int8 activations x int8 ternary weights via ``torch._int_mm``
- ``ort-fp32``     ONNX Runtime MatMul, fp32 weights
- ``ort-nbits{b}`` ONNX Runtime ``com.microsoft.MatMulNBits`` with b-bit packed weights and
                   ``accuracy_level=4`` (activations quantised to int8 inside the kernel); b=2
                   stores ternary weights exactly as codes {1, 2, 3} around zero point 2.
"""

from __future__ import annotations

import math

import numpy as np

from ..shapes import LMShape

# --------------------------------------------------------------------------------------------
# Weight packing for MatMulNBits
# --------------------------------------------------------------------------------------------


def default_zero_point(bits: int) -> int:
    return 1 << (bits - 1)


def pack_nbits(codes: np.ndarray, bits: int, block_size: int) -> np.ndarray:
    """codes (N, K) uint8 in [0, 2**bits) -> MatMulNBits B of shape (N, k_blocks, blob_size).

    Values are packed along K, lowest bits first (the operator's documented layout).
    """
    n, k = codes.shape
    assert k % block_size == 0, "K must be a multiple of block_size"
    per = 8 // bits
    c = codes.reshape(n, k // per, per).astype(np.uint8)
    packed = np.zeros((n, k // per), dtype=np.uint8)
    for i in range(per):
        packed |= (c[:, :, i] << np.uint8(bits * i)).astype(np.uint8)
    return packed.reshape(n, k // block_size, block_size * bits // 8)


def ternary_codes(rng: np.random.Generator, n: int, k: int, bits: int) -> np.ndarray:
    """Random ternary weights {-1, 0, +1} expressed as unsigned codes around the default zero point."""
    t = rng.integers(-1, 2, size=(n, k), dtype=np.int8)
    return (t.astype(np.int16) + default_zero_point(bits)).astype(np.uint8)


# --------------------------------------------------------------------------------------------
# ONNX graph
# --------------------------------------------------------------------------------------------


def build_onnx(shape: LMShape, depth: int, backend: str, block_size: int = 32, seed: int = 0):
    """ModelProto for ``depth`` layers of ``shape`` with the given backend ('fp32' or 'nbitsB')."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(seed)
    nodes, inits = [], []
    ln_cache: dict[int, tuple[str, str]] = {}

    def ln_params(dim: int) -> tuple[str, str]:
        if dim not in ln_cache:
            s, b = f"ln_s_{dim}", f"ln_b_{dim}"
            inits.append(numpy_helper.from_array(np.ones(dim, np.float32), s))
            inits.append(numpy_helper.from_array(np.zeros(dim, np.float32), b))
            ln_cache[dim] = (s, b)
        return ln_cache[dim]

    def norm(x: str, dim: int, out: str) -> str:
        s, b = ln_params(dim)
        nodes.append(helper.make_node("LayerNormalization", [x, s, b], [out], axis=-1, epsilon=1e-5))
        return out

    def const_i64(name: str, vals: list[int]) -> str:
        inits.append(numpy_helper.from_array(np.array(vals, np.int64), name))
        return name

    def matmul(x: str, k: int, n: int, out: str, tag: str) -> str:
        scale = 1.0 / math.sqrt(k)
        if backend == "fp32":
            w = rng.integers(-1, 2, size=(k, n), dtype=np.int8).astype(np.float32) * scale
            inits.append(numpy_helper.from_array(w, f"W_{tag}"))
            nodes.append(helper.make_node("MatMul", [x, f"W_{tag}"], [out]))
        elif backend.startswith("nbits"):
            bits = int(backend[5:])
            codes = ternary_codes(rng, n, k, bits)
            b = pack_nbits(codes, bits, block_size)
            sc = np.full((n, k // block_size), scale, np.float32)
            inits.append(numpy_helper.from_array(b, f"B_{tag}"))
            inits.append(numpy_helper.from_array(sc, f"S_{tag}"))
            nodes.append(helper.make_node(
                "MatMulNBits", [x, f"B_{tag}", f"S_{tag}"], [out], domain="com.microsoft",
                K=k, N=n, bits=bits, block_size=block_size, accuracy_level=4))
        else:
            raise ValueError(backend)
        return out

    H, I, Q = shape.hidden, shape.intermediate, shape.q_dim
    zero = const_i64("c0", [0])
    one = const_i64("c1", [1])
    qd = const_i64("cq", [Q])
    i1 = const_i64("ci", [I])
    i2 = const_i64("ci2", [2 * I])

    x = "x"
    for l in range(depth):
        p = f"l{l}_"
        h = norm(x, H, p + "h")
        qkv = matmul(h, H, shape.qkv_out, p + "qkv", p + "qkv")
        nodes.append(helper.make_node("Slice", [qkv, zero, qd, one], [p + "a0"]))
        a = norm(p + "a0", Q, p + "a")
        o = matmul(a, Q, H, p + "o", p + "o")
        nodes.append(helper.make_node("Add", [x, o], [p + "x1"]))
        h2 = norm(p + "x1", H, p + "h2")
        gu = matmul(h2, H, 2 * I, p + "gu", p + "gu")
        nodes.append(helper.make_node("Slice", [gu, zero, i1, one], [p + "g"]))
        nodes.append(helper.make_node("Slice", [gu, i1, i2, one], [p + "u"]))
        if shape.act == "relu2":
            nodes.append(helper.make_node("Relu", [p + "g"], [p + "r"]))
            nodes.append(helper.make_node("Mul", [p + "r", p + "r"], [p + "act"]))
        else:
            nodes.append(helper.make_node("Sigmoid", [p + "g"], [p + "sg"]))
            nodes.append(helper.make_node("Mul", [p + "g", p + "sg"], [p + "act"]))
        nodes.append(helper.make_node("Mul", [p + "act", p + "u"], [p + "m0"]))
        m = norm(p + "m0", I, p + "m")
        d = matmul(m, I, H, p + "d", p + "down")
        nodes.append(helper.make_node("Add", [p + "x1", d], [p + "x2"]))
        x = p + "x2"

    nodes.append(helper.make_node("Identity", [x], ["y"]))
    graph = helper.make_graph(
        nodes, f"{shape.key}_{backend}_d{depth}",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["T", H])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["T", H])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                                    helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 9
    return model


def ort_session(model, threads: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])


# --------------------------------------------------------------------------------------------
# PyTorch pipeline
# --------------------------------------------------------------------------------------------


class TorchStack:
    """The same layer stack in PyTorch, fp32 or int8 (torch._int_mm)."""

    def __init__(self, shape: LMShape, depth: int, mode: str, seed: int = 0):
        import torch

        self.torch = torch
        self.shape, self.depth, self.mode = shape, depth, mode
        g = torch.Generator().manual_seed(seed)
        self.layers = []
        for _ in range(depth):
            layer = {}
            for name, k, n in shape.linears():
                t = torch.randint(-1, 2, (k, n), generator=g, dtype=torch.int8)
                scale = 1.0 / math.sqrt(k)
                if mode == "fp32":
                    layer[name] = (t.float().mul_(scale).t().contiguous(), None)   # (N, K) for F.linear
                elif mode == "int8":
                    layer[name] = (t.contiguous(), scale)                            # (K, N) for _int_mm
                else:
                    raise ValueError(mode)
            self.layers.append(layer)

    def _mm(self, x, w):
        torch = self.torch
        weight, scale = w
        if self.mode == "fp32":
            return torch.nn.functional.linear(x, weight)
        s = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / 127.0
        q = (x / s).round_().clamp_(-128, 127).to(torch.int8)
        return torch._int_mm(q, weight).float().mul_(s * scale)

    def __call__(self, x):
        torch = self.torch
        F = torch.nn.functional
        sh = self.shape
        with torch.inference_mode():
            for layer in self.layers:
                h = F.layer_norm(x, (sh.hidden,))
                a = F.layer_norm(self._mm(h, layer["qkv"])[:, : sh.q_dim], (sh.q_dim,))
                x = x + self._mm(a, layer["o"])
                gu = self._mm(F.layer_norm(x, (sh.hidden,)), layer["gate_up"])
                g, u = gu[:, : sh.intermediate], gu[:, sh.intermediate:]
                act = torch.relu(g).square() if sh.act == "relu2" else F.silu(g)
                m = F.layer_norm(act * u, (sh.intermediate,))
                x = x + self._mm(m, layer["down"])
        return x


# --------------------------------------------------------------------------------------------
# The fp32 periphery: projector and action head
# --------------------------------------------------------------------------------------------


class Periphery:
    """Projector (pooled vision patches -> LM width) and the 13-query MLPResNet action head."""

    def __init__(self, hidden: int, pool: int, queries: int = 13, steps_per_query: int = 4,
                 joints: int = 17, vision_width: int = 768):
        import torch
        from torch import nn

        self.torch = torch
        in_dim = vision_width * pool * pool
        self.projector = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 2048), nn.GELU(),
                                       nn.Linear(2048, hidden)).eval()
        blocks = [nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU())
                  for _ in range(2)]
        self.head_in = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU()).eval()
        self.blocks = nn.ModuleList(blocks).eval()
        self.head_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, steps_per_query * joints)).eval()
        self.queries, self.hidden = queries, hidden
        self.image_tokens = (256 // (pool * pool))
        self.in_dim = in_dim
        self.params = sum(p.numel() for m in (self.head_in, self.blocks, self.head_out)
                          for p in m.parameters())

    def project(self, patches):
        with self.torch.inference_mode():
            return self.projector(patches)

    def head(self, h):
        with self.torch.inference_mode():
            z = self.head_in(h)
            for blk in self.blocks:
                z = z + blk(z)
            return self.head_out(z)
