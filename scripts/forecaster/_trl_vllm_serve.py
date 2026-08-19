"""Tiny shim that fixes a TRL 0.24 bug where ``is_vllm_ascend_available``
returns a `(False, None)` tuple — truthy — and triggers a spurious
``vllm_ascend`` import in ``trl.scripts.vllm_serve``.

We monkey-patch the function to return a real bool, then re-invoke
``trl.scripts.vllm_serve`` as if it were called directly via
``python -m trl.scripts.vllm_serve``.
"""
from __future__ import annotations


def _patch_trl_ascend_flag() -> None:
    """Fix a TRL 0.24-era bug where ``_vllm_ascend_available`` was the raw
    tuple from ``_is_package_available`` (truthy when the package is
    missing). TRL 1.4+ already returns a real bool from
    ``is_vllm_ascend_available``, so the private attr may not exist —
    that's fine, the bug isn't present.
    """
    import trl.import_utils as iu

    if not hasattr(iu, "_vllm_ascend_available"):
        return
    raw = iu._vllm_ascend_available
    flag = bool(raw[0]) if isinstance(raw, tuple) else bool(raw)
    iu._vllm_ascend_available = flag
    iu.is_vllm_ascend_available = lambda: flag


def main() -> None:
    _patch_trl_ascend_flag()
    # Re-route to the official entrypoint, matching the parser the module
    # uses when invoked via ``python -m trl.scripts.vllm_serve``.
    from trl.scripts.vllm_serve import main as serve_main
    from trl.scripts.vllm_serve import make_parser

    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    serve_main(script_args)


if __name__ == "__main__":
    main()
