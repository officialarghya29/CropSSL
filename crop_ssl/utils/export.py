"""
Model Export for Production Deployment.

Supports ONNX export with dynamic batching and input shapes, plus a
mobile-oriented path (fixed 224x224 input, static int8-quantized graph)
tuned for on-device runtimes such as onnxruntime-mobile / TFLite-style
deployments.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    opset_version: int = 14,
    dynamic_axes: Optional[dict] = None,
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
) -> str:
    """Export model to ONNX format.

    Args:
        model: Trained model to export.
        save_path: Path to save ONNX file.
        input_shape: Example input shape.
        opset_version: ONNX opset version.
        dynamic_axes: Dynamic axis configuration.
        input_names: Names for input tensors.
        output_names: Names for output tensors.

    Returns:
        Path to saved ONNX file.
    """
    model.eval()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*input_shape)

    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    if input_names is None:
        input_names = ["input"]

    if output_names is None:
        output_names = ["output"]

    # dynamo=False keeps the legacy (torchscript-based) exporter, which works
    # without the optional `onnxscript` dependency on all torch 2.x versions.
    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"Model exported to {save_path}")
    print(f"  Input shape: {input_shape}")
    print(f"  Opset version: {opset_version}")

    return str(save_path)


def verify_onnx(
    onnx_path: str,
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    atol: float = 1e-5,
) -> bool:
    """Verify ONNX model matches PyTorch output.

    Args:
        onnx_path: Path to ONNX file.
        model: Original PyTorch model.
        input_shape: Input shape for verification.
        atol: Absolute tolerance.

    Returns:
        True if outputs match within tolerance.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Skipping verification.")
        return False  # honest: we did NOT verify without onnxruntime

    model.eval()
    dummy_input = torch.randn(*input_shape)

    # PyTorch output
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    # ONNX output
    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(
        None, {"input": dummy_input.numpy()}
    )[0]

    match = abs(torch_output - onnx_output).max() < atol
    print(f"ONNX verification: {'PASS' if match else 'FAIL'}")
    print(f"  Max diff: {abs(torch_output - onnx_output).max():.8f}")
    return match


def export_ssl_backbone(
    ssl_model: nn.Module,
    save_path: str,
    backbone_type: str = "teacher",
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
) -> str:
    """Export SSL model backbone for deployment.

    Args:
        ssl_model: Pre-trained SSL model.
        save_path: Path to save ONNX file.
        backbone_type: 'teacher' or 'student' for DINOv2/MoCo.
        input_shape: Input shape.

    Returns:
        Path to saved ONNX file.
    """
    ssl_model.eval()

    # Extract backbone
    if hasattr(ssl_model, "teacher_backbone") and backbone_type == "teacher":
        backbone = ssl_model.teacher_backbone
    elif hasattr(ssl_model, "student_backbone"):
        backbone = ssl_model.student_backbone
    elif hasattr(ssl_model, "encoder"):
        backbone = ssl_model.encoder
    elif hasattr(ssl_model, "query_encoder"):
        backbone = ssl_model.query_encoder
    else:
        backbone = ssl_model

    # Create wrapper that returns features
    class BackboneWrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            return self.backbone.forward_features(x)

    wrapper = BackboneWrapper(backbone)

    return export_to_onnx(
        wrapper, save_path, input_shape=input_shape,
        input_names=["input"], output_names=["features"],
    )


def export_to_onnx_mobile(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    opset_version: int = 14,
    quantize: bool = True,
    verify: bool = True,
) -> dict:
    """Export a mobile-deployment ONNX model (fp32 + optional static int8).

    The mobile path differs from :func:`export_to_onnx` in three ways that
    matter on-device:

    1. **Static input shape** — no dynamic batch axis. Android runtimes
       (onnxruntime-mobile, NNAPI EP) pre-allocate for fixed shapes.
    2. **Opset >= 14** — required for the QuantizeLinear/QLinearConv ops
       used by static int8 quantization.
    3. **Post-training static quantization** with per-tensor scales and
       zero-points on weights and activations, which converts the graph to
       integer arithmetic (QDQ nodes) usable by mobile integer backends.

    Args:
        model: Trained model to export.
        save_path: Base path. Produces ``<save_path>`` (fp32) and, when
            ``quantize`` is True, ``<save_path>.int8.onnx``.
        input_shape: Fixed input shape (batch must stay fixed for mobile).
        opset_version: ONNX opset (>= 14 for int8 QDQ ops).
        quantize: Also produce a static int8-quantized copy.
        verify: Check the fp32 export numerically against PyTorch when
            onnxruntime is available.

    Returns:
        Dict with keys ``fp32_path``, ``int8_path`` (None if not produced),
        ``fp32_size_kb``, ``int8_size_kb`` (None), ``verified`` (bool),
        ``max_diff`` (float, fp32 vs PyTorch; 0.0 if not verifiable),
        ``quantized_ops`` (bool; whether the int8 graph really uses
        QuantizeLinear/QLinearConv ops).
    """
    if opset_version < 14:
        raise ValueError(
            "Mobile export requires opset >= 14 for int8 QDQ quantization ops."
        )
    if len(input_shape) != 4 or input_shape[0] != 1:
        raise ValueError(
            "Mobile export expects a fixed single-image shape like "
            f"(1, 3, {input_shape[-2] if len(input_shape) == 4 else 224}, 224); "
            f"got {tuple(input_shape)}."
        )

    model.eval()
    save_path = str(save_path)

    # --- fp32 static-shape export ---------------------------------------
    export_to_onnx(
        model,
        save_path,
        input_shape=input_shape,
        opset_version=opset_version,
        dynamic_axes=None,  # static shapes: mobile runtimes pre-allocate
        input_names=["input"],
        output_names=["logits"],
    )

    verified = False
    max_diff = 0.0
    if verify:
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime not installed. Skipping mobile fp32 verification.")
        else:
            dummy = torch.randn(*input_shape)
            with torch.no_grad():
                ref = model(dummy).numpy()
            sess = ort.InferenceSession(save_path)
            out = sess.run(None, {"input": dummy.numpy()})[0]
            max_diff = float(abs(ref - out).max())
            verified = max_diff < 1e-4
            print(
                f"Mobile fp32 verification: {'PASS' if verified else 'FAIL'}"
                f" (max diff {max_diff:.2e})"
            )

    fp32_size = Path(save_path).stat().st_size / 1024.0

    result = {
        "fp32_path": save_path,
        "int8_path": None,
        "fp32_size_kb": round(fp32_size, 1),
        "int8_size_kb": None,
        "verified": verified,
        "max_diff": max_diff,
        "quantized_ops": False,
    }

    # --- static int8 quantization ----------------------------------------
    if not quantize:
        return result

    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError:
        print(
            "onnxruntime.quantization not available; "
            "skipping int8 quantization (fp32 model is still exported)."
        )
        return result

    class _RandomCalibrationReader(CalibrationDataReader):
        """Feeds random samples for activation-range calibration.

        Calibrating on random inputs yields a conservative dynamic range;
        for maximum accuracy on a real crop set, pass a representative
        calibration reader holding ~200 field images.
        """

        def __init__(self, shape, n_samples: int = 64):
            super().__init__()
            self._shape = tuple(shape)
            self._n = int(n_samples)
            self._idx = 0

        def get_next(self):
            if self._idx >= self._n:
                return None
            self._idx += 1
            torch.manual_seed(self._idx)
            return {"input": torch.randn(*self._shape).numpy()}

    int8_path = save_path + ".int8.onnx"
    try:
        quantize_static(
            model_input=save_path,
            model_output=int8_path,
            calibration_data_reader=_RandomCalibrationReader(input_shape),
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
            per_channel=False,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
        )
    except Exception as exc:  # quantization is best-effort; fp32 is the fallback
        print(f"int8 quantization failed ({exc}); keeping fp32 export only.")
        return result

    int8_size = Path(int8_path).stat().st_size / 1024.0
    result.update(
        int8_path=int8_path,
        int8_size_kb=round(int8_size, 1),
        quantized_ops=_graph_has_quant_ops(int8_path),
    )
    print(
        f"Mobile int8 model: {int8_path} "
        f"({result['int8_size_kb']} KB vs fp32 {result['fp32_size_kb']} KB, "
        f"quant ops present: {result['quantized_ops']})"
    )
    return result


def _graph_has_quant_ops(onnx_path: str) -> bool:
    """Check (with the hard dependency `onnx` only) that a graph really
    contains QuantizeLinear/QLinearConv/QLinearMatMul ops — i.e. it is a
    genuine integer graph, not just a renamed fp32 one."""
    import onnx

    graph = onnx.load(onnx_path).graph
    ops = {node.op_type for node in graph.node}
    return bool(ops & {"QuantizeLinear", "QLinearConv", "QLinearMatMul", "ConvInteger"})


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters.

    Returns:
        Dict with total, trainable, and frozen parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": trainable / max(total, 1) * 100,
        "frozen_pct": frozen / max(total, 1) * 100,
    }


def model_summary(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
) -> str:
    """Generate model summary string.

    Args:
        model: Model to summarize.
        input_shape: Input shape for FLOPs calculation.

    Returns:
        Formatted summary string.
    """
    params = count_parameters(model)

    lines = [
        "=" * 50,
        "Model Summary",
        "=" * 50,
        f"Total parameters:     {params['total']:>12,}",
        f"Trainable parameters: {params['trainable']:>12,} ({params['trainable_pct']:.2f}%)",
        f"Frozen parameters:    {params['frozen']:>12,} ({params['frozen_pct']:.2f}%)",
        f"Input shape:          {str(input_shape):>12}",
        "=" * 50,
    ]

    return "\n".join(lines)
