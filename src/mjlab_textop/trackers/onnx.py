from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Loading torch first makes its bundled CUDA and cuDNN libraries available to
# ONNX Runtime. This is required when both libraries share the same CUDA runtime.
ort = importlib.import_module("onnxruntime")


@dataclass(frozen=True)
class OnnxModelSpec:
    """Expected interface of one dense float ONNX model."""

    input_name: str | None = None
    input_width: int | None = None
    output_name: str | None = None
    output_width: int = 29
    batch_size: int | None = None


class OnnxTensorModel:
    """Execute a single-input, single-output ONNX model on a torch device."""

    def __init__(
        self,
        model_file: str | Path,
        *,
        device: str | torch.device,
        spec: OnnxModelSpec,
    ) -> None:
        self.device = _resolve_device(device)
        self.spec = spec
        self.device_id: int | None = None
        self._binding: Any | None = None
        self._output: torch.Tensor | None = None
        inference_session = _inference_session_factory()

        if self.device.type == "cuda":
            _require_cuda_execution_provider()
            self.device_id = self.device.index
            assert self.device_id is not None
            stream = torch.cuda.current_stream(self.device_id)
            session_options = _cuda_session_options()
            session_kwargs: dict[str, Any] = {
                "providers": [
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": self.device_id,
                            "user_compute_stream": str(stream.cuda_stream),
                        },
                    )
                ]
            }
            if session_options is not None:
                session_kwargs["sess_options"] = session_options
            self.session = inference_session(str(model_file), **session_kwargs)
            self._binding = self.session.io_binding()
        else:
            self.session = inference_session(
                str(model_file),
                providers=["CPUExecutionProvider"],
            )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "Expected one ONNX input and one output, got "
                f"{len(inputs)} inputs and {len(outputs)} outputs"
            )

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        _validate_node(
            inputs[0],
            expected_name=spec.input_name,
            expected_width=spec.input_width,
            expected_batch_size=spec.batch_size,
            label="input",
        )
        _validate_node(
            outputs[0],
            expected_name=spec.output_name,
            expected_width=spec.output_width,
            expected_batch_size=spec.batch_size,
            label="output",
        )

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self._validate_value(value)
        if self.device.type == "cuda":
            return self._run_cuda(value)
        return self._run_cpu(value)

    def _validate_value(self, value: torch.Tensor) -> None:
        if value.ndim != 2:
            raise ValueError(f"ONNX input must be rank 2, got {tuple(value.shape)}")
        if self.spec.batch_size is not None and value.shape[0] != self.spec.batch_size:
            raise ValueError(
                f"ONNX model requires batch size {self.spec.batch_size}, "
                f"got {value.shape[0]}"
            )
        if (
            self.spec.input_width is not None
            and value.shape[1] != self.spec.input_width
        ):
            raise ValueError(
                f"ONNX model requires input width {self.spec.input_width}, "
                f"got {value.shape[1]}"
            )

    def _run_cpu(self, value: torch.Tensor) -> torch.Tensor:
        cpu_value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        output = self.session.run(
            [self.output_name],
            {self.input_name: cpu_value.numpy()},
        )[0]
        return torch.from_numpy(output).to(value.device)

    def _run_cuda(self, value: torch.Tensor) -> torch.Tensor:
        if value.device != self.device:
            raise RuntimeError(
                "CUDA ONNX model requires inputs on its policy device: "
                f"expected {self.device}, got {value.device}"
            )

        model_input = value.detach().to(dtype=torch.float32).contiguous()
        output_shape = (int(model_input.shape[0]), self.spec.output_width)
        if self._output is None or self._output.shape != output_shape:
            self._output = torch.empty(
                output_shape,
                dtype=torch.float32,
                device=self.device,
            )

        assert self.device_id is not None
        binding = self._binding
        assert binding is not None
        binding.clear_binding_inputs()
        binding.clear_binding_outputs()
        binding.bind_input(
            name=self.input_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(model_input.shape),
            buffer_ptr=model_input.data_ptr(),
        )
        binding.bind_output(
            name=self.output_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=output_shape,
            buffer_ptr=self._output.data_ptr(),
        )
        self.session.run_with_iobinding(binding)
        return self._output


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return resolved
    device_id = (
        resolved.index if resolved.index is not None else torch.cuda.current_device()
    )
    return torch.device("cuda", device_id)


def _inference_session_factory() -> Any:
    factory = getattr(ort, "InferenceSession", None)
    if callable(factory):
        return factory
    raise RuntimeError(
        "The imported ONNX Runtime module does not expose InferenceSession "
        f"({_onnxruntime_description()}). This is not a usable ONNX Runtime "
        "installation. Make sure a local module is not shadowing onnxruntime and "
        "that onnxruntime and onnxruntime-gpu are not installed together."
    )


def _require_cuda_execution_provider() -> None:
    get_available_providers = getattr(ort, "get_available_providers", None)
    if not callable(get_available_providers):
        return

    available_providers = get_available_providers()
    if "CUDAExecutionProvider" not in available_providers:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable in the imported ONNX Runtime "
            f"installation ({_onnxruntime_description()}). Available providers: "
            f"{available_providers}. Install the project's 'cu128' extra rather "
            "than the CPU-only 'cpu' extra."
        )


def _cuda_session_options() -> Any | None:
    session_options_factory = getattr(ort, "SessionOptions", None)
    if not callable(session_options_factory):
        warnings.warn(
            "The imported ONNX Runtime module does not expose SessionOptions "
            f"({_onnxruntime_description()}); continuing without explicitly "
            "disabling CPU execution-provider fallback.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    session_options = session_options_factory()
    session_options.add_session_config_entry(
        "session.disable_cpu_ep_fallback",
        "1",
    )
    return session_options


def _onnxruntime_description() -> str:
    version = getattr(ort, "__version__", "unknown version")
    location = getattr(ort, "__file__", "unknown location")
    return f"version {version!r}, imported from {location!r}"


def _validate_node(
    node: Any,
    *,
    expected_name: str | None,
    expected_width: int | None,
    expected_batch_size: int | None,
    label: str,
) -> None:
    if expected_name is not None and node.name != expected_name:
        raise ValueError(
            f"Unexpected ONNX {label} name: expected {expected_name!r}, "
            f"got {node.name!r}"
        )

    node_type = getattr(node, "type", None)
    if node_type is not None and node_type != "tensor(float)":
        raise ValueError(
            f"Unexpected ONNX {label} dtype: expected tensor(float), got {node_type}"
        )

    shape = getattr(node, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        return
    if expected_batch_size is not None and shape[0] != expected_batch_size:
        raise ValueError(
            f"Unexpected ONNX {label} batch dimension: "
            f"expected {expected_batch_size}, got {shape[0]}"
        )
    if expected_width is not None and shape[1] != expected_width:
        raise ValueError(
            f"Unexpected ONNX {label} width: expected {expected_width}, got {shape[1]}"
        )
