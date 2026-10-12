"""SigLIP2-base-patch16-256 vision tower at real size (random weights) in PyTorch and ONNX Runtime."""

from __future__ import annotations

from pathlib import Path

from ..shapes import VISION


def siglip_vision(seed: int = 0):
    import torch
    from transformers import SiglipVisionConfig, SiglipVisionModel

    torch.manual_seed(seed)
    cfg = SiglipVisionConfig(image_size=VISION["image_size"], patch_size=VISION["patch_size"],
                             hidden_size=VISION["hidden"], intermediate_size=VISION["intermediate"],
                             num_hidden_layers=VISION["layers"], num_attention_heads=VISION["heads"])
    cfg._attn_implementation = "eager"   # exportable everywhere; sdpa is timed separately
    model = SiglipVisionModel(cfg).eval()
    return model


class _Tower:
    """Wraps the HF model so export and timing see one tensor in, the patch tokens out."""

    def __init__(self, model):
        import torch

        class M(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, pixel_values):
                return self.m(pixel_values=pixel_values).last_hidden_state

        self.module = M(model).eval()


def export_onnx(model, path: Path) -> Path:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tower = _Tower(model).module
    x = torch.randn(1, 3, VISION["image_size"], VISION["image_size"])
    with torch.inference_mode():
        try:
            torch.onnx.export(tower, (x,), str(path), input_names=["pixel_values"],
                              output_names=["patches"], opset_version=17, dynamo=False)
        except TypeError:
            torch.onnx.export(tower, (x,), str(path), input_names=["pixel_values"],
                              output_names=["patches"], opset_version=17)
    return path


def quantize_int8(src: Path, dst: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=True)
    return dst


def ort_vision_session(path: Path, threads: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
