"""Preload NVIDIA shared libraries so TensorFlow can discover the GPU.

The pip-installed `nvidia-*` packages place their libraries inside the venv,
but TensorFlow’s runtime dynamic linker does not know about those paths.
Importing this module (before any TF / Keras import) makes the libraries
available globally via `ctypes.RTLD_GLOBAL`.
"""

import ctypes
import glob
import importlib.util
import os
import sys

# Map of nvidia package names to the library file(s) we need from each.
# Order matters roughly by dependency (base runtime first, then libs that
# depend on it).  We do multiple passes so inter-library deps are resolved.
_LIBS = {
    "cuda_runtime": ["libcudart.so.*"],
    "cublas":       ["libcublas.so.*", "libcublasLt.so.*"],
    "cuda_nvrtc":   ["libnvrtc.so.*"],
    "cudnn":        ["libcudnn.so.*"],
    "cufft":        ["libcufft.so.*"],
    "curand":       ["libcurand.so.*"],
    "cusolver":     ["libcusolver.so.*"],
    "cusparse":     ["libcusparse.so.*"],
    "nvjitlink":    ["libnvJitLink.so.*"],
    "nccl":         ["libnccl.so.*"],
}


def _find_nvidia_base() -> str | None:
    """Return the directory that contains all nvidia.* packages."""
    try:
        import nvidia
        # nvidia is a namespace package; __path__ points to the base dir
        return str(nvidia.__path__[0])
    except Exception:
        return None


def _preload() -> None:
    base = _find_nvidia_base()
    if base is None:
        # Nothing to do – maybe running on CPU only.
        return

    loaded: set[str] = set()
    for _attempt in range(5):
        progress = False
        for pkg_name, patterns in _LIBS.items():
            lib_dir = os.path.join(base, pkg_name, "lib")
            if not os.path.isdir(lib_dir):
                continue
            for pattern in patterns:
                for path in sorted(glob.glob(os.path.join(lib_dir, pattern))):
                    if path in loaded:
                        continue
                    try:
                        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                        loaded.add(path)
                        progress = True
                    except OSError:
                        # Dependency not ready yet; retry in next pass.
                        pass
        if not progress:
            break

    if loaded:
        # Expose the lib directories on LD_LIBRARY_PATH as well, for any
        # subprocesses that get spawned (e.g. multi-worker training).
        lib_dirs = set(os.path.dirname(p) for p in loaded)
        old = os.environ.get("LD_LIBRARY_PATH", "")
        new = ":".join(lib_dirs)
        if old:
            new = new + ":" + old
        os.environ["LD_LIBRARY_PATH"] = new


_preload()
