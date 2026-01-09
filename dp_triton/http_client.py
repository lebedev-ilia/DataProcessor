from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TritonError(RuntimeError):
    def __init__(self, message: str, *, error_code: str = "triton_error") -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class TritonInferResult:
    output_name: str
    output: np.ndarray
    datatype: str = "FP32"


class TritonHttpClient:
    """
    Minimal Triton HTTP client (no external deps).
    Uses Triton HTTP v2 API.
    """

    def __init__(self, *, base_url: str, timeout_sec: float = 5.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout_sec = float(timeout_sec)

    def _get_json(self, path: str) -> Dict[str, Any]:
        url = self.base_url + path
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read()
                return json.loads(body.decode("utf-8")) if body else {}
        except urllib.error.HTTPError as e:
            raise TritonError(f"HTTP {e.code} GET {url}", error_code="triton_http_error") from e
        except Exception as e:
            raise TritonError(f"GET {url} failed: {e}", error_code="triton_unavailable") from e

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url + path
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read()
                return json.loads(body.decode("utf-8")) if body else {}
        except urllib.error.HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode("utf-8")
            except Exception:
                raw = ""
            raise TritonError(f"HTTP {e.code} POST {url} body={raw[:400]}", error_code="triton_http_error") from e
        except Exception as e:
            raise TritonError(f"POST {url} failed: {e}", error_code="triton_unavailable") from e

    def ready(self) -> bool:
        url = self.base_url + "/v2/health/ready"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                return int(getattr(resp, "status", 200) or 200) == 200
        except Exception:
            return False

    def infer(
        self,
        *,
        model_name: str,
        model_version: Optional[str],
        input_name: str,
        input_tensor: np.ndarray,
        output_name: str,
        datatype: str = "FP32",
    ) -> TritonInferResult:
        if not isinstance(input_tensor, np.ndarray):
            raise TritonError("input_tensor must be a numpy array", error_code="triton_bad_request")

        # Triton JSON expects lists for data (slow but OK for MVP).
        # Ensure Python scalars (not numpy types) in the payload.
        dt = str(datatype or "FP32").upper()
        if dt in ("INT64", "INT32", "INT16", "INT8", "UINT64", "UINT32", "UINT16", "UINT8"):
            data_list = input_tensor.reshape(-1).astype(np.int64).tolist()
        else:
            data_list = input_tensor.reshape(-1).astype(np.float32).tolist()
        shape = list(input_tensor.shape)

        version_path = f"/versions/{model_version}" if model_version else ""
        path = f"/v2/models/{model_name}{version_path}/infer"
        payload = {
            "inputs": [
                {
                    "name": input_name,
                    "shape": shape,
                    "datatype": datatype,
                    "data": data_list,
                }
            ],
            "outputs": [{"name": output_name}],
        }
        resp = self._post_json(path, payload)
        outs = resp.get("outputs") or []
        if not isinstance(outs, list) or not outs:
            raise TritonError("missing outputs in Triton response", error_code="triton_bad_response")
        out0 = None
        for o in outs:
            if isinstance(o, dict) and o.get("name") == output_name:
                out0 = o
                break
        if out0 is None:
            out0 = outs[0] if isinstance(outs[0], dict) else None
        if not isinstance(out0, dict):
            raise TritonError("invalid outputs format", error_code="triton_bad_response")

        out_shape = out0.get("shape")
        out_data = out0.get("data")
        out_dt = str(out0.get("datatype") or "FP32").upper()
        if not isinstance(out_shape, list) or not isinstance(out_data, list):
            raise TritonError("invalid output shape/data", error_code="triton_bad_response")

        def _dtype(dt: str):
            if dt in ("INT64",):
                return np.int64
            if dt in ("INT32",):
                return np.int32
            if dt in ("INT16",):
                return np.int16
            if dt in ("INT8",):
                return np.int8
            if dt in ("UINT64",):
                return np.uint64
            if dt in ("UINT32",):
                return np.uint32
            if dt in ("UINT16",):
                return np.uint16
            if dt in ("UINT8",):
                return np.uint8
            if dt in ("FP16",):
                return np.float16
            return np.float32

        arr = np.asarray(out_data, dtype=_dtype(out_dt)).reshape(tuple(int(x) for x in out_shape))
        return TritonInferResult(output_name=output_name, output=arr, datatype=out_dt)

    def infer_multi(
        self,
        *,
        model_name: str,
        model_version: Optional[str],
        input_name: str,
        input_tensor: np.ndarray,
        outputs: List[Tuple[str, str]],
        input_datatype: str = "FP32",
    ) -> Dict[str, TritonInferResult]:
        """
        Triton infer with multiple requested outputs.
        outputs: list of (output_name, output_datatype) expected by the caller.
        """
        if not outputs:
            raise TritonError("outputs list is empty", error_code="triton_bad_request")
        if not isinstance(input_tensor, np.ndarray):
            raise TritonError("input_tensor must be a numpy array", error_code="triton_bad_request")

        dt = str(input_datatype or "FP32").upper()
        if dt in ("INT64", "INT32", "INT16", "INT8", "UINT64", "UINT32", "UINT16", "UINT8"):
            data_list = input_tensor.reshape(-1).astype(np.int64).tolist()
        else:
            data_list = input_tensor.reshape(-1).astype(np.float32).tolist()
        shape = list(input_tensor.shape)

        version_path = f"/versions/{model_version}" if model_version else ""
        path = f"/v2/models/{model_name}{version_path}/infer"
        payload = {
            "inputs": [{"name": input_name, "shape": shape, "datatype": input_datatype, "data": data_list}],
            "outputs": [{"name": name} for name, _ in outputs],
        }
        resp = self._post_json(path, payload)
        outs = resp.get("outputs") or []
        if not isinstance(outs, list) or not outs:
            raise TritonError("missing outputs in Triton response", error_code="triton_bad_response")

        by_name: Dict[str, Dict[str, Any]] = {}
        for o in outs:
            if isinstance(o, dict) and isinstance(o.get("name"), str):
                by_name[str(o.get("name"))] = o

        results: Dict[str, TritonInferResult] = {}
        for out_name, _expected_dt in outputs:
            o = by_name.get(out_name)
            if not isinstance(o, dict):
                raise TritonError(f"missing output {out_name} in Triton response", error_code="triton_bad_response")
            out_shape = o.get("shape")
            out_data = o.get("data")
            out_dt = str(o.get("datatype") or _expected_dt or "FP32").upper()
            if not isinstance(out_shape, list) or not isinstance(out_data, list):
                raise TritonError(f"invalid output shape/data for {out_name}", error_code="triton_bad_response")

            def _dtype(dt: str):
                if dt in ("INT64",):
                    return np.int64
                if dt in ("INT32",):
                    return np.int32
                if dt in ("INT16",):
                    return np.int16
                if dt in ("INT8",):
                    return np.int8
                if dt in ("UINT64",):
                    return np.uint64
                if dt in ("UINT32",):
                    return np.uint32
                if dt in ("UINT16",):
                    return np.uint16
                if dt in ("UINT8",):
                    return np.uint8
                if dt in ("FP16",):
                    return np.float16
                return np.float32

            arr = np.asarray(out_data, dtype=_dtype(out_dt)).reshape(tuple(int(x) for x in out_shape))
            results[out_name] = TritonInferResult(output_name=out_name, output=arr, datatype=out_dt)
        return results

    def infer_two_inputs(
        self,
        *,
        model_name: str,
        model_version: Optional[str],
        input0_name: str,
        input0_tensor: np.ndarray,
        input1_name: str,
        input1_tensor: np.ndarray,
        output_name: str,
        datatype: str = "FP32",
    ) -> TritonInferResult:
        """
        Triton infer for models with TWO input tensors (common for optical flow: prev_frame + cur_frame).
        JSON payload (slow but OK for MVP).
        """
        if not isinstance(input0_tensor, np.ndarray) or not isinstance(input1_tensor, np.ndarray):
            raise TritonError("input tensors must be numpy arrays", error_code="triton_bad_request")

        dt = str(datatype or "FP32").upper()

        def _to_list(x: np.ndarray) -> list:
            if dt in ("INT64", "INT32", "INT16", "INT8", "UINT64", "UINT32", "UINT16", "UINT8"):
                return x.reshape(-1).astype(np.int64).tolist()
            return x.reshape(-1).astype(np.float32).tolist()

        version_path = f"/versions/{model_version}" if model_version else ""
        path = f"/v2/models/{model_name}{version_path}/infer"
        payload = {
            "inputs": [
                {"name": input0_name, "shape": list(input0_tensor.shape), "datatype": datatype, "data": _to_list(input0_tensor)},
                {"name": input1_name, "shape": list(input1_tensor.shape), "datatype": datatype, "data": _to_list(input1_tensor)},
            ],
            "outputs": [{"name": output_name}],
        }
        resp = self._post_json(path, payload)
        outs = resp.get("outputs") or []
        if not isinstance(outs, list) or not outs:
            raise TritonError("missing outputs in Triton response", error_code="triton_bad_response")
        out0 = None
        for o in outs:
            if isinstance(o, dict) and o.get("name") == output_name:
                out0 = o
                break
        if out0 is None:
            out0 = outs[0] if isinstance(outs[0], dict) else None
        if not isinstance(out0, dict):
            raise TritonError("invalid outputs format", error_code="triton_bad_response")

        out_shape = out0.get("shape")
        out_data = out0.get("data")
        if not isinstance(out_shape, list) or not isinstance(out_data, list):
            raise TritonError("invalid output shape/data", error_code="triton_bad_response")

        arr = np.asarray(out_data, dtype=np.float32).reshape(tuple(int(x) for x in out_shape))
        return TritonInferResult(output_name=output_name, output=arr)


