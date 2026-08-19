# Shim for broken vllm_ascend plugin probe.
# vLLM probes for this Huawei Ascend NPU plugin at import time; if a stale
# stub is found the import crashes. We register a proper fake module so the
# probe succeeds harmlessly. This must run before any TRL or vLLM import.
import importlib.machinery
import importlib.util
import sys
import types
from typing import Any

if importlib.util.find_spec("vllm_ascend") is None:
    _mod = types.ModuleType("vllm_ascend")
    _mod.__path__ = []
    _mod.__spec__ = importlib.machinery.ModuleSpec("vllm_ascend", None)
    sys.modules["vllm_ascend"] = _mod
    for _sub in (
        "distributed",
        "distributed.device_communicators",
        "distributed.device_communicators.pyhccl",
    ):
        _fqn = f"vllm_ascend.{_sub}"
        # Annotated `Any`: this is a hand-built stub module onto which we
        # deliberately graft attributes ModuleType does not declare.
        _m: Any = types.ModuleType(_fqn)
        _m.__spec__ = importlib.machinery.ModuleSpec(_fqn, None)
        if _sub.endswith("pyhccl"):
            _m.PyHcclCommunicator = None
        sys.modules[_fqn] = _m
