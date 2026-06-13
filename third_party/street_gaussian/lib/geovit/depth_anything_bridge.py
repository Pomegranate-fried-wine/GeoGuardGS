"""Bridge Depth Anything 3 predictions into camera-level GeoViT guidance."""

from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn


class GeoViTDepthBridge(nn.Module):
    """Run Depth Anything 3 on a camera image with per-camera caching.

    ``depth_anything_3.api.DepthAnything3`` is imported only when the bridge
    needs to instantiate a model.  A prebuilt model can also be supplied for
    tests or custom inference wrappers.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
        cache_key: str = "geovit_depth_anything_3",
        use_cache: bool = True,
        detach_cache: bool = True,
        include_confidence: bool = True,
        include_tokens: bool = False,
        inference_kwargs: Optional[Dict[str, Any]] = None,
        **model_kwargs: Any,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.device = torch.device(device) if device is not None else None
        self.cache_key = cache_key
        self.use_cache = use_cache
        self.detach_cache = detach_cache
        self.include_confidence = include_confidence
        self.include_tokens = include_tokens
        self.inference_kwargs = inference_kwargs or {}
        self.model_kwargs = model_kwargs
        self._last_image_shape: Optional[torch.Size] = None
        self._last_image_device: Optional[torch.device] = None
        self._last_image_dtype: Optional[torch.dtype] = None
        self.model_kwargs = model_kwargs

    def _load_model(self) -> Any:
        """Instantiate Depth Anything 3 on first use."""

        if self.model is None:
            from depth_anything_3.api import DepthAnything3

            if self.model_name is not None and hasattr(DepthAnything3, "from_pretrained"):
                self.model = DepthAnything3.from_pretrained(self.model_name, **self.model_kwargs)
            elif self.model_name is not None:
                self.model = DepthAnything3(self.model_name, **self.model_kwargs)
            else:
                self.model = DepthAnything3(**self.model_kwargs)

        if self.device is not None and hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self.model

    @staticmethod
    def _cache_store(viewpoint_cam: Any) -> Dict[str, Any]:
        """Return a camera-local cache dictionary without mutating guidance."""

        cache = getattr(viewpoint_cam, "_geovit_guidance_cache", None)
        if cache is None:
            cache = {}
            setattr(viewpoint_cam, "_geovit_guidance_cache", cache)
        return cache

    @staticmethod
    def _detach_value(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, dict):
            return {key: GeoViTDepthBridge._detach_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(GeoViTDepthBridge._detach_value(item) for item in value)
        return value

    @staticmethod
    def _to_torch(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
        """Convert numpy DA3 outputs to tensors while preserving nested aux data."""

        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            return tensor.to(device=device)
        if isinstance(value, dict):
            return {key: GeoViTDepthBridge._to_torch(item, device, dtype) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(GeoViTDepthBridge._to_torch(item, device, dtype) for item in value)
        return value

    @staticmethod
    def _ensure_single_camera_map(value: Any) -> Any:
        """Normalize single-camera depth/confidence maps to ``[1, H, W]``."""

        if torch.is_tensor(value) and value.ndim == 2:
            return value.unsqueeze(0)
        if torch.is_tensor(value) and value.ndim == 3 and value.shape[0] == 1:
            return value
        return value

    def _prepare_image(self, viewpoint_cam: Any) -> torch.Tensor:
        if not hasattr(viewpoint_cam, "original_image"):
            raise AttributeError("viewpoint_cam must expose an original_image tensor.")

        image = viewpoint_cam.original_image
        if not torch.is_tensor(image):
            image = torch.as_tensor(image)
        image = image.float().clamp(0.0, 1.0)
        if self.device is not None:
            image = image.to(self.device)

        self._last_image_shape = image.shape
        self._last_image_device = image.device
        self._last_image_dtype = image.dtype
        return image

    @staticmethod
    def _tensor_to_da3_images(image: torch.Tensor) -> Any:
        """Convert CHW/BCHW torch camera images to DA3 API numpy image input."""

        image_cpu = image.detach().clamp(0.0, 1.0).cpu()
        if image_cpu.ndim == 3:
            if image_cpu.shape[0] in {1, 3, 4}:
                image_cpu = image_cpu.permute(1, 2, 0)
            image_np = (image_cpu.numpy() * 255.0).round().astype(np.uint8)
            return [image_np]
        if image_cpu.ndim == 4:
            images = []
            for item in image_cpu:
                if item.shape[0] in {1, 3, 4}:
                    item = item.permute(1, 2, 0)
                images.append((item.numpy() * 255.0).round().astype(np.uint8))
            return images
        raise ValueError("original_image must be a CHW, HWC, BCHW, or BHWC tensor.")

        return image

    def _run_model(self, image: torch.Tensor) -> Any:
        model = self._load_model()

        with torch.no_grad():
            if hasattr(model, "inference"):
                return model.inference(
                    self._tensor_to_da3_images(image),
                    **self.inference_kwargs,
                )
            if hasattr(model, "infer"):
                return model.infer(image)
            if hasattr(model, "predict"):
                return model.predict(image)
            if hasattr(model, "infer_image"):
                return model.infer_image(image)
            if callable(model):
                return model(image)

        raise TypeError(
            "Depth Anything 3 model must expose inference/infer/predict/infer_image or be callable."
        )
        raise TypeError("Depth Anything 3 model must be callable or expose infer/predict/infer_image.")

    @staticmethod
    def _first_available(mapping: Mapping[str, Any], *keys: str) -> Optional[Any]:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return None

    @staticmethod
    def _first_attr(obj: Any, *keys: str) -> Optional[Any]:
        for key in keys:
            value = getattr(obj, key, None)
            if value is not None:
                return value
        return None

    @staticmethod
    def _tokens_from_aux(aux: Any) -> Optional[Any]:
        if aux is None:
            return None
        if not isinstance(aux, Mapping):
            return aux

        token_keys = [
            key
            for key in aux.keys()
            if key.startswith("feat_layer_") or key in {"tokens", "features", "embeddings"}
        ]
        if len(token_keys) == 0:
            return aux
        return {key: aux[key] for key in token_keys}

    def _normalize_result(self, raw_result: Any) -> Dict[str, Any]:
        if isinstance(raw_result, Mapping):
            relative_depth = self._first_available(
                raw_result,
                "relative_depth",
                "depth",
                "pred_depth",
                "prediction",
            )
            if relative_depth is None:
                raise KeyError("Depth Anything 3 result does not contain a depth prediction.")

            result = {"relative_depth": relative_depth}
            confidence = self._first_available(raw_result, "confidence", "conf", "uncertainty")
            tokens = self._first_available(raw_result, "tokens", "features", "embeddings")
            if tokens is None:
                tokens = self._tokens_from_aux(raw_result.get("aux"))
        elif isinstance(raw_result, (tuple, list)):
            if len(raw_result) == 0:
                raise ValueError("Depth Anything 3 returned an empty result.")
            result = {"relative_depth": raw_result[0]}
            confidence = raw_result[1] if len(raw_result) > 1 else None
            tokens = raw_result[2] if len(raw_result) > 2 else None
        elif hasattr(raw_result, "depth"):
            result = {"relative_depth": raw_result.depth}
            confidence = self._first_attr(raw_result, "conf", "confidence", "uncertainty")
            tokens = self._first_attr(raw_result, "tokens", "features", "embeddings")
            if tokens is None:
                tokens = self._tokens_from_aux(getattr(raw_result, "aux", None))
        else:
            result = {"relative_depth": raw_result}
            confidence = None
            tokens = None

        if self._last_image_device is not None and self._last_image_dtype is not None:
            result = self._to_torch(result, self._last_image_device, self._last_image_dtype)
            confidence = self._to_torch(confidence, self._last_image_device, self._last_image_dtype)
            tokens = self._to_torch(tokens, self._last_image_device, self._last_image_dtype)

        result["relative_depth"] = self._ensure_single_camera_map(result["relative_depth"])
        if self.include_confidence and confidence is not None:
            result["confidence"] = self._ensure_single_camera_map(confidence)
        if self.include_confidence and confidence is not None:
            result["confidence"] = confidence
        if self.include_tokens and tokens is not None:
            result["tokens"] = tokens
        return result

    def forward(self, viewpoint_cam: Any, refresh_cache: bool = False) -> Dict[str, Any]:
        """Infer DA3 guidance from ``viewpoint_cam.original_image``.

        The returned dictionary always contains ``relative_depth``.  It may also
        contain ``confidence`` and/or ``tokens`` when provided by the underlying
        Depth Anything 3 API and enabled in this bridge.
        """

        cache = self._cache_store(viewpoint_cam)
        if self.use_cache and not refresh_cache and self.cache_key in cache:
            return cache[self.cache_key]

        image = self._prepare_image(viewpoint_cam)
        result = self._normalize_result(self._run_model(image))
        if self.detach_cache:
            result = self._detach_value(result)

        if self.use_cache:
            cache[self.cache_key] = result
        return result
