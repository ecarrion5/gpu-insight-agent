"""Safe execution of LLM-generated pandas code.

Main points:
NEVER exec() model output in your own process. Here I use defense in depth:

  1. AST validation BEFORE running anything: reject imports, dunder access, and
     dangerous builtin names. Fail fast, cheaply, without executing.
  2. Execution in a SEPARATE process with a hard timeout: a runaway or infinite loop
     gets terminated, and a crash can't take down agent.
  3. A restricted namespace: the code sees only pandas (pd), numpy (np), the dataframe
     (df), and a small allow-list of safe builtins. No open(), no __import__.

Caveat: this is a prottype-grade sandbox, good enough for our own model's 
output. It is not a hardened boundary for adversarial code. In production 
I'd run untrusted code in a container with no network, a read-only
filesystem, seccomp, and cgroup CPU/memory limits — or a hosted code-interpreter
sandbox — and pass the worker a data path rather than pickling the whole frame.
"""

import ast
import multiprocessing as mp
import pandas as pd
import numpy as np

_FORBIDDEN_NAMES = {
    "exec", "eval", "compile", "open", "__import__", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
}

# Small allow-list so ordinary pandas code still runs.
_SAFE_BUILTINS = {
    "len": len, "min": min, "max": max, "sum": sum, "sorted": sorted,
    "round": round, "abs": abs, "range": range, "enumerate": enumerate,
    "zip": zip, "str": str, "int": int, "float": float, "bool": bool,
    "dict": dict, "list": list, "tuple": tuple, "set": set,
}


def _validate_ast(code: str) -> None:
    """Static safety check. Raises ValueError on anything we won't allow."""
    tree = ast.parse(code, mode="exec")  # SyntaxError here is caught by the caller
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed in generated code")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"use of `{node.id}` is not allowed")
        # Blocks the classic sandbox-escape route: ().__class__.__bases__ ...
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("access to dunder attributes is not allowed")


def _run(code: str, df: pd.DataFrame, q) -> None:
    """Runs inside the child process. Puts (status, payload) on the queue."""
    safe_globals = {"__builtins__": _SAFE_BUILTINS}
    safe_locals = {"pd": pd, "np": np, "df": df}
    try:
        exec(code, safe_globals, safe_locals)  # AST-validated + isolated + restricted
        if "result" not in safe_locals:
            q.put(("error", "code did not define `result`"))
        else:
            q.put(("ok", repr(safe_locals["result"])[:2000]))
    except Exception as e:  # noqa: BLE001 - we want to report ANY failure back to the loop
        q.put(("error", f"{type(e).__name__}: {e}"))


def run_pandas(code: str, df: pd.DataFrame, timeout: float = 10.0) -> tuple[str, str]:
    """Validate, then execute in an isolated process with a hard timeout.

    Returns (status, payload) where status is 'ok' | 'error' | 'timeout'.
    """
    try:
        _validate_ast(code)
    except (ValueError, SyntaxError) as e:
        return ("error", f"rejected before execution: {e}")

    # 'spawn' is consistent across macOS/Windows/Linux. A Manager().Queue() proxy is
    # picklable, so it can be passed to a spawned child (a plain mp.Queue cannot).
    ctx = mp.get_context("spawn")
    with ctx.Manager() as mgr:
        q = mgr.Queue()
        p = ctx.Process(target=_run, args=(code, df, q))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()   # hard kill the runaway
            p.join()
            return ("timeout", f"execution exceeded {timeout}s")
        return q.get() if not q.empty() else ("error", "no result produced")
