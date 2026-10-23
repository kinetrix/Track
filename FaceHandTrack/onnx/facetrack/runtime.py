"""ONNX Runtime sessions with a per-model choice of accelerator.

ONNX Runtime is the one runtime the whole engine uses. It runs the same .onnx
files on macOS, Windows, Linux, Android, iOS and in the browser (onnxruntime-web);
only the "execution provider" (the accelerator backend) differs per platform:

    macOS / iOS      CoreMLExecutionProvider  (Neural Engine / GPU)
    Windows          DmlExecutionProvider     (any DirectX 12 GPU)
    Linux / Windows  CUDAExecutionProvider    (NVIDIA)
    Android          NnapiExecutionProvider / QNNExecutionProvider
    everywhere       CPUExecutionProvider     (always available, the fallback)

accel="auto" times every available provider on each model once (accelerators in a
separate process, so one that crashes on a graph is simply skipped) and keeps the
fastest; the choice is cached so later launches start immediately.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "facetrack")


def _coreml(units: str) -> dict:
    return {
        "ModelFormat": "MLProgram",  # newer Core ML format: more ops, fp16 on GPU / Neural Engine
        "MLComputeUnits": units,
        "ModelCacheDirectory": os.path.join(CACHE_DIR, "coreml_" + units),  # compile once, not every launch
    }


# Providers worth trying, with their options. Core ML is tried twice: letting it
# choose freely (often the GPU) and pinned to the Neural Engine, because either can win.
_CANDIDATES = {
    "coreml": ("CoreMLExecutionProvider", _coreml("ALL")),
    "coreml_ane": ("CoreMLExecutionProvider", _coreml("CPUAndNeuralEngine")),
    "cuda": ("CUDAExecutionProvider", {}),
    "directml": ("DmlExecutionProvider", {}),
    "cpu": ("CPUExecutionProvider", {}),
}


def _options(threads: int) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        so.intra_op_num_threads = threads
    so.log_severity_level = 3  # errors only
    return so


def _has_dynamic_inputs(path: str) -> bool:
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return any(not isinstance(d, int) for i in s.get_inputs() for d in i.shape)


def _session(path: str, key: str, threads: int) -> ort.InferenceSession:
    name, opts = _CANDIDATES[key]
    if key == "cpu":
        return ort.InferenceSession(path, sess_options=_options(threads), providers=["CPUExecutionProvider"])
    opts = dict(opts)
    if name == "CoreMLExecutionProvider":
        os.makedirs(opts["ModelCacheDirectory"], exist_ok=True)
        if _has_dynamic_inputs(path):
            # Core ML (GPU) can abort the whole process on variable-size inputs:
            # let it take only the fixed-size parts of such graphs.
            opts["RequireStaticInputShapes"] = "1"
    return ort.InferenceSession(path, sess_options=_options(threads), providers=[(name, opts), "CPUExecutionProvider"])


def _example_feeds(sess: ort.InferenceSession, dyn: int = 1):
    """Random inputs; variable dimensions are set to `dyn`."""
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) else dyn for d in i.shape]
        feeds[i.name] = np.random.rand(*shape).astype(np.float32)
    return feeds


def _time(sess, n=3) -> float:
    for dyn in (2, 1):  # exercise variable-size inputs with two sizes; time the last
        feeds = _example_feeds(sess, dyn)
        sess.run(None, feeds)  # warm-up (first call also compiles for accelerators)
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, feeds)
    return (time.perf_counter() - t0) / n


def _probe(path: str, key: str, threads: int):
    """Time one provider on one model in a separate process, so a provider that
    crashes on this graph (it happens: hard aborts inside GPU compilers) only
    costs that candidate, not the app. Returns seconds or None."""
    code = ("import sys, json; sys.path.insert(0, %r); from facetrack.runtime import _session, _time; "
            "print('PROBE', json.dumps(_time(_session(%r, %r, %d))))"
            % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path, key, threads))
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    for line in out.stdout.splitlines():
        if line.startswith("PROBE "):
            return float(line[6:])
    return None


class Runtime:
    """Creates one session per model. accel: "auto", "cpu", or a provider key
    from _CANDIDATES ("coreml", "cuda", "directml")."""

    def __init__(self, accel: str = "auto", verbose: bool = True):
        self.accel = accel
        self.verbose = verbose
        avail = set(ort.get_available_providers())
        self.available = [k for k, (n, _) in _CANDIDATES.items() if n in avail]
        self.chosen = {}  # model file -> provider key
        self._cache_path = os.path.join(CACHE_DIR, "backend_choice.json")
        try:
            with open(self._cache_path) as f:
                self._cache = json.load(f)
        except Exception:
            self._cache = {}

    def _fingerprint(self, path: str, threads: int) -> str:
        st = os.stat(path)
        raw = f"{os.path.basename(path)}|{st.st_size}|{int(st.st_mtime)}|{ort.__version__}|{threads}|{self.available}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def load(self, path: str, threads: int = 0, example=None) -> ort.InferenceSession:
        """example: unused, kept for API compatibility (probes build their own inputs)."""
        name = os.path.basename(path)
        if self.accel != "auto":
            key = self.accel if self.accel in self.available else "cpu"
            sess = _session(path, key, threads)
            self.chosen[name] = key
            return sess

        fp = self._fingerprint(path, threads)
        key = self._cache.get(fp)
        if key in self.available:
            self.chosen[name] = key
            return _session(path, key, threads)

        best, best_t, results = None, float("inf"), []
        for k in self.available:
            if k == "cpu":
                try:
                    t = _time(_session(path, k, threads))
                except Exception as e:
                    results.append(f"{k}: failed ({type(e).__name__})")
                    continue
            else:
                t = _probe(path, k, threads)
                if t is None:
                    results.append(f"{k}: failed")
                    continue
            results.append(f"{k}: {t * 1000:.1f} ms")
            if t < best_t:
                best, best_t = k, t
        if best is None:
            raise RuntimeError(f"no execution provider could run {name}")
        if self.verbose:
            print(f"[runtime] {name}: {', '.join(results)} -> {best}")
        self.chosen[name] = best
        self._cache[fp] = best
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f, indent=1)
        except OSError:
            pass
        return _session(path, best, threads)
