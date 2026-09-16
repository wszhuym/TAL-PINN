"""Single-file NTK/loss mechanism analysis for the static-shock Burgers case.

This is the only analysis driver required. It imports the three existing
training utility modules, while keeping experiment orchestration, empirical
NTK construction, data export, final-residual evaluation, and all figures in
this file. ``ntk_jax.py`` and the companion notebook are not required.

The controlled comparison contains three independently trained methods:

1. Vanilla PINN: uniform sampling in physical ``(t, x)``.
2. Adaptive-sampling PINN: uniform ``(t, xi)`` samples mapped to physical space.
3. TAL-PINN: the same adaptive samples plus the learned coordinate ``xi(t, x)``.

The reference solution is used only to construct one fixed oracle adaptive
coordinate. This is a mechanism diagnostic, not the autonomous multi-stage
TAL-PINN algorithm. NTKs are evaluated at the paired initial solution-network
state; final physical residuals are evaluated after the three training runs.

Notebook use
------------
Import the required functions and execute the experiment one step at a time::

    import NTK_loss_analysis_burgers1d_reference as analysis
    config = analysis.ExperimentConfig()
    geometry = analysis.prepare_reference_geometry("burgers_riemann.mat", config)

Calling ``main()`` or executing this file directly intentionally performs no
training, saving, plotting, or printing. The notebook controls every step.

The script accepts utility filenames both with and without the ``(1)`` suffix
that may be added when duplicate files are downloaded.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import inspect
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import jax

# The supplied Utilities and older soap-jax releases are naturally float32.
# Enforcing one training dtype prevents lax.cond branches inside SOAP from
# returning incompatible float32/float64 optimizer states. The symmetric NTK
# eigenproblems are converted to NumPy float64 below.
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
import scipy.io
from flax import serialization
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.flatten_util import ravel_pytree

matplotlib.use("Agg")
import matplotlib.pyplot as plt


Array = jax.Array
PyTree = Any
ApplyFn = Callable[[PyTree, Array], Array]
WeightMode = Literal["mul", "div"]
TRAIN_DTYPE = jnp.float32
SCRIPT_DIR = Path(__file__).resolve().parent


def _load_utility_module(alias: str, candidates: Sequence[str]) -> tuple[Any, Path]:
    """Load a utility module beside this script, including ``(1)`` filenames."""

    tried: list[str] = []
    for filename in candidates:
        path = SCRIPT_DIR / filename
        tried.append(str(path))
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(alias, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        return module, path
    raise FileNotFoundError(
        f"Could not locate the {alias} utility module. Tried: " + ", ".join(tried)
    )


# Prefer the explicitly supplied duplicate-suffix files when both versions are
# present. A clean directory with conventional filenames works as well.
vanilla_utils, VANILLA_UTILITY_PATH = _load_utility_module(
    "_burgers_vanilla_utils",
    (
        "Utilities_pinn_burgers1d_statistic.py",
        "Utilities_pinn_burgers1d_statistic.py",
    ),
)
adaptive_utils, ADAPTIVE_UTILITY_PATH = _load_utility_module(
    "_burgers_adaptive_utils",
    (
        "Utilities_pinn_adsamp_burgers1d.py",
        "Utilities_pinn_adsamp_burgers1d.py",
    ),
)
tal_utils, TAL_UTILITY_PATH = _load_utility_module(
    "_burgers_tal_utils",
    (
        "Utilities_talpinn_burgers1d_statistic.py",
        "Utilities_talpinn_burgers1d_statistic.py",
    ),
)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    nu: float = 1.0e-4
    time_stride: int = 2
    space_stride: int = 2

    hidden_width: int = 40
    hidden_depth: int = 2
    fourier_dim: int = 20

    optimizer: str = "soap"
    learning_rate: float = 3.0e-3
    weight_decay: float = 0.01
    w_ic: float = 10.0
    w_bc: float = 10.0

    coordinate_iters: int = 10_000
    coordinate_batch_size: int = 10_000
    coordinate_eval_every: int = 1_000

    train_iters: int = 10_000
    train_pde_points: int = 10_000
    train_ic_points: int = 1_000
    train_bc_points: int = 1_000
    train_eval_every: int = 100
    max_runtime: float = 10_000.0

    ntk_pde_points: int = 1_000
    ntk_ic_points: int = 100
    ntk_bc_points: int = 100
    ntk_chunk_size: int = 128

    residual_time_stride: int = 2
    residual_space_stride: int = 2
    residual_chunk_size: int = 1_024
    plot_sample_points: int = 4_000


def _resolve_input_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    beside_script = SCRIPT_DIR / candidate
    if beside_script.is_file():
        return beside_script.resolve()
    raise FileNotFoundError(
        f"Reference file {path!s} was not found in the working directory "
        f"or beside {Path(__file__).name}."
    )


def _resolve_output_prefix(prefix: str | Path) -> Path:
    path = Path(prefix).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _broadcast_coordinate(
    coordinate: np.ndarray,
    shape: tuple[int, int],
    *,
    role: str,
) -> np.ndarray:
    value = np.asarray(coordinate, dtype=np.float32).squeeze()
    nt, nx = shape

    if value.ndim == 1:
        expected = nx if role == "x" else nt
        if value.size != expected:
            raise ValueError(
                f"The {role}-coordinate vector has length {value.size}; "
                f"expected {expected} for solution shape {shape}."
            )
        if role == "x":
            return np.broadcast_to(value[None, :], shape).copy()
        return np.broadcast_to(value[:, None], shape).copy()

    if value.ndim != 2:
        raise ValueError(
            f"The {role}-coordinate must be a vector or matrix; got {value.shape}."
        )
    try:
        return np.broadcast_to(value, shape).copy()
    except ValueError as exc:
        raise ValueError(
            f"The {role}-coordinate shape {value.shape} cannot broadcast to {shape}."
        ) from exc


def _standardize_candidate(
    u: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if u.ndim != 2:
        raise ValueError(f"U must be two-dimensional after squeeze; got {u.shape}.")

    x_grid = _broadcast_coordinate(x, u.shape, role="x")
    t_grid = _broadcast_coordinate(t, u.shape, role="t")
    u_grid = np.asarray(np.real(u), dtype=np.float32).copy()
    tol = 100.0 * np.finfo(np.float32).eps

    spatial_span = x_grid[:, -1] - x_grid[:, 0]
    if np.all(spatial_span < -tol):
        x_grid = x_grid[:, ::-1]
        t_grid = t_grid[:, ::-1]
        u_grid = u_grid[:, ::-1]
    elif not np.all(spatial_span > tol):
        raise ValueError("X must increase or decrease monotonically along axis 1.")

    temporal_span = t_grid[-1, :] - t_grid[0, :]
    if np.all(temporal_span < -tol):
        x_grid = x_grid[::-1, :]
        t_grid = t_grid[::-1, :]
        u_grid = u_grid[::-1, :]
    elif not np.all(temporal_span > tol):
        raise ValueError("T must increase or decrease monotonically along axis 0.")

    if np.any(np.diff(x_grid, axis=1) <= 0.0):
        raise ValueError("Every row of X must be strictly increasing.")
    if np.any(np.diff(t_grid[:, 0]) <= 0.0):
        raise ValueError("T[:, 0] must be strictly increasing.")
    if not all(np.all(np.isfinite(value)) for value in (u_grid, x_grid, t_grid)):
        raise ValueError("U, X, and T must contain only finite values.")
    return t_grid, x_grid, u_grid


def standardize_reference_grid(
    T: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time-first ``(Nt, Nx)`` matrices from common MAT layouts."""

    u = np.asarray(U).squeeze()
    x = np.asarray(X).squeeze()
    t = np.asarray(T).squeeze()
    if u.ndim != 2:
        raise ValueError(f"U must describe a two-dimensional field; got {u.shape}.")

    candidates = [
        (u, x, t),
        (u.T, x.T if x.ndim == 2 else x, t.T if t.ndim == 2 else t),
    ]
    failures: list[str] = []
    for candidate in candidates:
        try:
            return _standardize_candidate(*candidate)
        except ValueError as exc:
            failures.append(str(exc))
    raise ValueError(
        "Could not infer a time-first reference-grid orientation. "
        + " | ".join(failures)
    )


def _stride_indices(size: int, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError("A grid stride must be positive.")
    indices = np.arange(0, size, stride, dtype=np.int64)
    if indices[-1] != size - 1:
        indices = np.append(indices, size - 1)
    return indices


def load_reference_grid(
    path: str | Path,
    *,
    time_stride: int,
    space_stride: int,
) -> tuple[Path, Array, Array, Array]:
    resolved = _resolve_input_path(path)
    raw = scipy.io.loadmat(resolved)
    missing = sorted({"U", "X", "T"}.difference(raw))
    if missing:
        raise KeyError(f"{resolved} is missing MAT variables: {missing}.")

    t_grid, x_grid, u_grid = standardize_reference_grid(
        raw["T"], raw["X"], raw["U"]
    )
    time_index = _stride_indices(u_grid.shape[0], time_stride)
    space_index = _stride_indices(u_grid.shape[1], space_stride)
    index = np.ix_(time_index, space_index)
    t_grid, x_grid, u_grid = t_grid[index], x_grid[index], u_grid[index]

    if min(u_grid.shape) < 3:
        raise ValueError("The strided reference grid must retain at least 3 x 3 nodes.")
    endpoints = (t_grid[0, 0], t_grid[-1, 0], x_grid[0, 0], x_grid[0, -1])
    if not np.allclose(endpoints, (0.0, 1.0, 0.0, 1.0), atol=1.0e-6):
        raise ValueError(
            "The reused Burgers Utilities assume [t, x] in [0, 1]^2; "
            f"reference endpoints are {endpoints}."
        )
    return (
        resolved,
        jnp.asarray(t_grid, dtype=TRAIN_DTYPE),
        jnp.asarray(x_grid, dtype=TRAIN_DTYPE),
        jnp.asarray(u_grid, dtype=TRAIN_DTYPE),
    )


def prepare_reference_geometry(
    data_path: str | Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    resolved, t_grid, x_grid, u_reference = load_reference_grid(
        data_path,
        time_stride=config.time_stride,
        space_stride=config.space_stride,
    )
    start = time.perf_counter()
    xi_grid, xi_of_x, x_adapt = tal_utils.compute_adaptive_mesh_1d(
        t_grid, x_grid, u_reference
    )
    x_adapt.block_until_ready()
    mesh_seconds = time.perf_counter() - start
    map_xi_to_x = adaptive_utils.create_map_xi_to_x(t_grid, xi_grid, x_adapt)
    return {
        "data_path": resolved,
        "T": t_grid,
        "X": x_grid,
        "U": u_reference,
        "xi_grid": xi_grid,
        "Xi_of_x": xi_of_x,
        "X_adapt": x_adapt,
        "map_xi_to_x": map_xi_to_x,
        "mesh_seconds": mesh_seconds,
    }


# ---------------------------------------------------------------------------
# Empirical NTK implementation (integrated; no ntk_jax.py dependency)
# ---------------------------------------------------------------------------


def _scalar_output(value: Array) -> Array:
    value = jnp.asarray(value)
    if value.size != 1:
        raise ValueError(f"A Burgers network output must be scalar; got {value.shape}.")
    return jnp.reshape(value, ())


def _validate_points(name: str, points: Array, dimension: int) -> Array:
    points = jnp.asarray(points, dtype=TRAIN_DTYPE)
    if points.ndim != 2 or points.shape[1] != dimension:
        raise ValueError(f"{name} must have shape (N, {dimension}); got {points.shape}.")
    return points


def _validate_rows(name: str, values: Array, n_rows: int, width: int = 1) -> Array:
    values = jnp.asarray(values, dtype=TRAIN_DTYPE)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape != (n_rows, width):
        raise ValueError(
            f"{name} must have shape ({n_rows}, {width}); got {values.shape}."
        )
    return values


def _weight_factor(weight: Array, mode: WeightMode, eps: float) -> Array:
    safe = jnp.maximum(jnp.abs(weight), jnp.asarray(eps, dtype=weight.dtype))
    if mode == "mul":
        return jnp.sqrt(safe)
    if mode == "div":
        return jax.lax.rsqrt(safe)
    raise ValueError("mode must be 'mul' or 'div'.")


def _residual_jacobian(
    params: PyTree,
    point_residual_fn: Callable[..., Array],
    batched_args: Sequence[Array],
    *,
    chunk_size: int,
) -> tuple[Array, Array, Array]:
    arrays = tuple(jnp.asarray(arg, dtype=TRAIN_DTYPE) for arg in batched_args)
    if not arrays:
        raise ValueError("At least one batched NTK argument is required.")
    n_points = int(arrays[0].shape[0])
    if n_points == 0 or any(int(arg.shape[0]) != n_points for arg in arrays):
        raise ValueError("All NTK arguments need the same nonzero leading dimension.")
    if chunk_size <= 0:
        raise ValueError("NTK chunk_size must be positive.")

    flat_params, unravel = ravel_pytree(params)

    def batch_residual(flat: Array, *args: Array) -> Array:
        current = unravel(flat)
        values = jax.vmap(
            lambda *point_args: jnp.atleast_1d(
                point_residual_fn(current, *point_args)
            )
        )(*args)
        return values

    def value_and_jacobian(flat: Array, *args: Array) -> tuple[Array, Array]:
        values = batch_residual(flat, *args)
        jacobian = jax.jacrev(batch_residual, argnums=0)(flat, *args)
        return values, jacobian

    evaluate = jax.jit(value_and_jacobian)
    residual_chunks: list[Array] = []
    jacobian_chunks: list[Array] = []
    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        chunk_args = tuple(arg[start:stop] for arg in arrays)
        residual_chunk, jacobian_chunk = evaluate(flat_params, *chunk_args)
        residual_chunks.append(residual_chunk)
        jacobian_chunks.append(jacobian_chunk)

    residual = jnp.concatenate(residual_chunks, axis=0).reshape(-1)
    jacobian = jnp.concatenate(jacobian_chunks, axis=0).reshape(
        residual.size, flat_params.size
    )
    kernel = jacobian @ jacobian.T
    return 0.5 * (kernel + kernel.T), residual, jacobian


def empirical_ntk_residual(
    apply_fn: ApplyFn,
    params: PyTree,
    residual_points: Array,
    nu: float,
    *,
    lifted: bool,
    geometry: Array | None,
    residual_weight: Array | None,
    residual_weight_mode: WeightMode,
    chunk_size: int,
) -> tuple[Array, Array, Array]:
    dimension = 3 if lifted else 2
    points = _validate_points("residual_points", residual_points, dimension)
    n_points = int(points.shape[0])
    nu_array = jnp.asarray(nu, dtype=TRAIN_DTYPE)

    if residual_weight is None:
        weights = jnp.ones((n_points, 1), dtype=TRAIN_DTYPE)
        use_weight = False
    else:
        weights = _validate_rows("residual_weight", residual_weight, n_points)
        use_weight = True

    if lifted:
        if geometry is None:
            raise ValueError("A lifted residual requires (xi_t, xi_x, xi_xx).")
        geometry_array = _validate_rows("geometry", geometry, n_points, width=3)
    else:
        if geometry is not None:
            raise ValueError("geometry is valid only for a lifted residual.")
        geometry_array = jnp.empty((n_points, 0), dtype=TRAIN_DTYPE)

    def vanilla_residual(current: PyTree, point: Array, weight: Array) -> Array:
        def solution(q: Array) -> Array:
            return _scalar_output(apply_fn(current, q))

        value = solution(point)
        gradient = jax.grad(solution)(point)
        hessian = jax.hessian(solution)(point)
        residual = gradient[0] + value * gradient[1] - nu_array * hessian[1, 1]
        if use_weight:
            residual = residual * _weight_factor(
                weight[0], residual_weight_mode, 1.0e-8
            )
        return residual

    def lifted_residual(
        current: PyTree,
        point: Array,
        derivatives: Array,
        weight: Array,
    ) -> Array:
        def solution(q: Array) -> Array:
            return _scalar_output(apply_fn(current, q))

        value = solution(point)
        gradient = jax.grad(solution)(point)
        hessian = jax.hessian(solution)(point)
        xi_t, xi_x, xi_xx = derivatives
        u_t = gradient[0] + gradient[2] * xi_t
        u_x = gradient[1] + gradient[2] * xi_x
        u_xx = (
            hessian[1, 1]
            + 2.0 * hessian[1, 2] * xi_x
            + hessian[2, 2] * xi_x**2
            + gradient[2] * xi_xx
        )
        residual = u_t + value * u_x - nu_array * u_xx
        if use_weight:
            residual = residual * _weight_factor(
                weight[0], residual_weight_mode, 1.0e-8
            )
        return residual

    if lifted:
        return _residual_jacobian(
            params,
            lifted_residual,
            (points, geometry_array, weights),
            chunk_size=chunk_size,
        )
    return _residual_jacobian(
        params,
        vanilla_residual,
        (points, weights),
        chunk_size=chunk_size,
    )


def empirical_ntk_data(
    apply_fn: ApplyFn,
    params: PyTree,
    points: Array,
    targets: Array,
    *,
    lifted: bool,
    chunk_size: int,
) -> tuple[Array, Array, Array]:
    dimension = 3 if lifted else 2
    points = _validate_points("data_points", points, dimension)
    targets = _validate_rows("data_targets", targets, int(points.shape[0]))

    def point_residual(current: PyTree, point: Array, target: Array) -> Array:
        return _scalar_output(apply_fn(current, point)) - target[0]

    return _residual_jacobian(
        params,
        point_residual,
        (points, targets),
        chunk_size=chunk_size,
    )


def _symmetric_spectrum(kernel: Array) -> np.ndarray:
    """Compute a stable descending spectrum on CPU in NumPy float64."""

    matrix = np.asarray(jax.device_get(kernel), dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    values = np.linalg.eigvalsh(matrix)[::-1].copy()
    scale = max(float(np.max(np.abs(values))), np.finfo(np.float64).tiny)
    values[np.abs(values) < scale * 1.0e-13] = 0.0
    return values


def assemble_dirichlet_ntk(
    *,
    apply_fn: ApplyFn,
    params: PyTree,
    residual_points: Array,
    nu: float,
    ic_points: Array,
    ic_targets: Array,
    bc_points: Array,
    bc_targets: Array,
    lifted: bool,
    geometry: Array | None,
    residual_weight: Array | None,
    residual_weight_mode: WeightMode,
    w_ic: float,
    w_bc: float,
    chunk_size: int,
) -> dict[str, Any]:
    kernel_residual, residual, jacobian_residual = empirical_ntk_residual(
        apply_fn,
        params,
        residual_points,
        nu,
        lifted=lifted,
        geometry=geometry,
        residual_weight=residual_weight,
        residual_weight_mode=residual_weight_mode,
        chunk_size=chunk_size,
    )
    _, residual_ic, jacobian_ic = empirical_ntk_data(
        apply_fn,
        params,
        ic_points,
        ic_targets,
        lifted=lifted,
        chunk_size=chunk_size,
    )
    _, residual_bc, jacobian_bc = empirical_ntk_data(
        apply_fn,
        params,
        bc_points,
        bc_targets,
        lifted=lifted,
        chunk_size=chunk_size,
    )

    scales = {
        "residual": 1.0 / np.sqrt(max(int(residual.size), 1)),
        "ic": np.sqrt(w_ic / max(int(residual_ic.size), 1)),
        "bc": np.sqrt(w_bc / max(int(residual_bc.size), 1)),
    }
    jacobian_residual_scaled = jacobian_residual * scales["residual"]
    jacobian_total = jnp.concatenate(
        (
            jacobian_residual_scaled,
            jacobian_ic * scales["ic"],
            jacobian_bc * scales["bc"],
        ),
        axis=0,
    )
    residual_total = jnp.concatenate(
        (
            residual * scales["residual"],
            residual_ic * scales["ic"],
            residual_bc * scales["bc"],
        )
    )
    kernel_total = jacobian_total @ jacobian_total.T
    kernel_total = 0.5 * (kernel_total + kernel_total.T)

    eigen_residual_unscaled = _symmetric_spectrum(kernel_residual)
    eigen_residual = eigen_residual_unscaled * scales["residual"] ** 2
    eigen_total = _symmetric_spectrum(kernel_total)
    initial_decay = float(
        jax.device_get(jnp.vdot(residual_total, kernel_total @ residual_total))
    )
    scaled_residual = residual * scales["residual"]
    return {
        "eigen_residual_unscaled": eigen_residual_unscaled,
        "eigen_residual": eigen_residual,
        "eigen_total": eigen_total,
        "initial_residual_loss": float(
            jax.device_get(jnp.vdot(scaled_residual, scaled_residual))
        ),
        "initial_total_loss": float(jax.device_get(jnp.vdot(residual_total, residual_total))),
        "initial_decay_quadratic": initial_decay,
        "n_residual_rows": int(residual.size),
        "n_ic_rows": int(residual_ic.size),
        "n_bc_rows": int(residual_bc.size),
    }


def spectral_summary(eigenvalues: np.ndarray) -> dict[str, float]:
    values = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    trace = float(np.sum(values))
    if values.size == 0 or values[0] <= 0.0:
        return {
            "trace": trace,
            "effective_rank": 0.0,
            "numerical_rank": 0.0,
            "condition": np.inf,
        }
    positive = values[values > 1.0e-12 * values[0]]
    probabilities = values / max(trace, np.finfo(np.float64).tiny)
    probabilities = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return {
        "trace": trace,
        "effective_rank": float(np.exp(entropy)),
        "numerical_rank": float(positive.size),
        "condition": np.inf if positive.size == 0 else float(positive[0] / positive[-1]),
    }


# ---------------------------------------------------------------------------
# Initialization, optimizer, coordinate fit, and training
# ---------------------------------------------------------------------------

def _floating_dtypes(tree: PyTree) -> set[str]:
    dtypes: set[str] = set()
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            dtypes.add(str(leaf.dtype))
    return dtypes


def _ensure_float32_tree(tree: PyTree, name: str) -> PyTree:
    converted = jax.tree_util.tree_map(
        lambda value: (
            value.astype(TRAIN_DTYPE)
            if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating)
            else value
        ),
        tree,
    )
    dtypes = _floating_dtypes(converted)
    if dtypes != {"float32"}:
        raise TypeError(f"{name} must contain only float32 floating leaves; got {dtypes}.")
    return converted


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def make_optimizer(config: ExperimentConfig) -> Any:
    """Create one optimizer while explicitly aligning SOAP internal dtypes."""

    if config.optimizer == "adamw":
        return optax.adamw(
            learning_rate=config.learning_rate,
            b1=0.95,
            b2=0.95,
            weight_decay=config.weight_decay,
        )
    if config.optimizer != "soap":
        raise ValueError("optimizer must be 'soap' or 'adamw'.")

    soap_fn = tal_utils.soap
    signature = inspect.signature(soap_fn)
    kwargs: dict[str, Any] = {
        "learning_rate": config.learning_rate,
        "b1": 0.95,
        "b2": 0.95,
        "weight_decay": config.weight_decay,
        "precondition_frequency": 10,
    }
    if "precondition_1d" in signature.parameters:
        kwargs["precondition_1d"] = False
    if "mu_dtype" in signature.parameters:
        kwargs["mu_dtype"] = TRAIN_DTYPE
    if "qr_dtype" in signature.parameters:
        kwargs["qr_dtype"] = TRAIN_DTYPE
    return soap_fn(**kwargs)


def check_optimizer_dtype(optimizer: Any, params: PyTree, name: str) -> None:
    """Trace enough zero-gradient steps to exercise SOAP's periodic branch."""

    params = _ensure_float32_tree(params, name)
    state = optimizer.init(params)
    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)

    @jax.jit
    def probe_periodic_branch(current_state: PyTree) -> PyTree:
        # SOAP is configured with precondition_frequency=10. Unrolling twelve
        # steps covers both the ordinary and preconditioner-update lax.cond
        # branches, which is where mixed QR/moment dtypes previously failed.
        for _ in range(12):
            _, current_state = optimizer.update(
                zero_grads,
                current_state,
                params=params,
            )
        return current_state

    try:
        result = probe_periodic_branch(state)
        for leaf in jax.tree_util.tree_leaves(result):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
    except TypeError as exc:
        raise RuntimeError(
            "Optimizer dtype self-check failed before training. This script forces "
            "parameters/data and, when supported, SOAP moments/QR matrices to float32. "
            f"Installed soap-jax version: {_package_version('soap-jax')}. "
            "If this is an old or locally modified SOAP build, either install a "
            "consistent soap-jax release or run with --optimizer adamw."
        ) from exc


def initialize_models(
    config: ExperimentConfig,
    key_vanilla: Array,
    key_adaptive: Array,
    key_tal: Array,
    key_coordinate: Array,
) -> tuple[dict[str, Any], dict[str, PyTree]]:

    layer_sizes = [config.hidden_width] * config.hidden_depth + [1]

    # Flax Module 本身不保存参数，因此二维网络结构可以共用；
    # 两次独立 init 得到的参数树是完全独立的。
    model_2d = vanilla_utils.FNN_fourier(
        layer_sizes=layer_sizes,
        fourier_dim=config.fourier_dim,
    )
    model_3d = tal_utils.FNN_fourier(
        layer_sizes=layer_sizes,
        fourier_dim=config.fourier_dim,
    )
    coordinate_model = tal_utils.FNN_fourier(
        layer_sizes=layer_sizes,
        fourier_dim=config.fourier_dim,
    )

    vanilla_variables = _ensure_float32_tree(
        model_2d.init(
            key_vanilla,
            jnp.ones((2,), dtype=TRAIN_DTYPE),
        ),
        "Vanilla parameters",
    )

    adaptive_variables = _ensure_float32_tree(
        model_2d.init(
            key_adaptive,
            jnp.ones((2,), dtype=TRAIN_DTYPE),
        ),
        "Adaptive parameters",
    )

    tal_variables = _ensure_float32_tree(
        model_3d.init(
            key_tal,
            jnp.ones((3,), dtype=TRAIN_DTYPE),
        ),
        "TAL parameters",
    )

    coordinate_variables = _ensure_float32_tree(
        coordinate_model.init(
            key_coordinate,
            jnp.ones((2,), dtype=TRAIN_DTYPE),
        ),
        "Coordinate parameters",
    )

    models = {
        "solution_2d": model_2d,
        "solution_3d": model_3d,
        "coordinate": coordinate_model,
    }

    initial_params = {
        "vanilla": vanilla_variables,
        "adaptive": adaptive_variables,
        "tal": tal_variables,
        "coordinate": coordinate_variables,
    }

    return models, initial_params


def coordinate_diagnostics(
    coordinate_model: Any,
    coordinate_params: PyTree,
    t_grid: Array,
    x_grid: Array,
    xi_target: Array,
) -> dict[str, float]:
    inputs = jnp.stack((t_grid.reshape(-1), x_grid.reshape(-1)), axis=-1)

    def scalar_coordinate(tx: Array) -> Array:
        return jnp.squeeze(coordinate_model.apply(coordinate_params, tx))

    prediction = jax.vmap(scalar_coordinate)(inputs)
    gradients = jax.vmap(jax.grad(scalar_coordinate))(inputs)
    error = prediction - xi_target.reshape(-1)
    values = jax.device_get(
        (
            jnp.mean(error**2),
            jnp.max(jnp.abs(error)),
            jnp.min(gradients[:, 1]),
            jnp.max(gradients[:, 1]),
        )
    )
    return {
        "mse": float(values[0]),
        "max_abs_error": float(values[1]),
        "min_xi_x": float(values[2]),
        "max_xi_x": float(values[3]),
    }


def fit_coordinate_network(
    config: ExperimentConfig,
    coordinate_model: Any,
    coordinate_initial: PyTree,
    geometry: Mapping[str, Any],
    key: Array,
) -> tuple[PyTree, Callable[[Array], Array], dict[str, float], float]:
    optimizer = make_optimizer(config)
    check_optimizer_dtype(optimizer, coordinate_initial, "coordinate parameters")
    loss_grad_fn = tal_utils.create_coor_loss_grad_fn(
        coordinate_model,
        lambda_pos=0.01,
        xi_x_min=0.05,
    )
    start = time.perf_counter()
    coordinate_params, _ = tal_utils.train_coordinate_net(
        coor_net=coordinate_model,
        coor_params=coordinate_initial,
        coor_optimizer=optimizer,
        coor_loss_grad_fn=loss_grad_fn,
        T_grid=geometry["T"],
        X_grid=geometry["X"],
        Xi_of_x=geometry["Xi_of_x"],
        key=key,
        max_iters=config.coordinate_iters,
        batch_size=config.coordinate_batch_size,
        eval_every=config.coordinate_eval_every,
    )
    seconds = time.perf_counter() - start
    coordinate_fn = tal_utils.create_coor_fn(coordinate_model, coordinate_params)
    metrics = coordinate_diagnostics(
        coordinate_model,
        coordinate_params,
        geometry["T"],
        geometry["X"],
        geometry["Xi_of_x"],
    )
    return coordinate_params, coordinate_fn, metrics, seconds


def build_minibatch_functions(
    geometry: Mapping[str, Any],
    coordinate_fn: Callable[[Array], Array],
) -> dict[str, Callable[..., Any]]:
    return {
        "vanilla": vanilla_utils.create_pde_minibatch_fn(
            vanilla_utils.IC_Burgers,
            vanilla_utils.BC_Burgers,
        ),
        "adaptive": adaptive_utils.create_pde_minibatch_fn(
            adaptive_utils.IC_Burgers,
            adaptive_utils.BC_Burgers,
            geometry["map_xi_to_x"],
        ),
        "tal": tal_utils.create_pde_minibatch_fn(
            tal_utils.IC_Burgers,
            tal_utils.BC_Burgers,
            geometry["map_xi_to_x"],
            coordinate_fn,
        ),
    }


def paired_initial_output_error(
    models: Mapping[str, Any],
    initial_params: Mapping[str, PyTree],
    coordinate_fn: Callable[[Array], Array],
    key: Array,
) -> float:
    points_2d = jax.random.uniform(key, (32, 2), dtype=TRAIN_DTYPE)
    xi = jax.vmap(coordinate_fn)(points_2d)
    points_3d = jnp.column_stack((points_2d, xi))
    prediction_2d = models["solution_2d"].apply(
        initial_params["vanilla_and_adaptive"], points_2d
    )
    prediction_3d = models["solution_3d"].apply(initial_params["tal"], points_3d)
    return float(jax.device_get(jnp.max(jnp.abs(prediction_2d - prediction_3d))))


def compute_initial_ntks(
    config: ExperimentConfig,
    models: Mapping[str, Any],
    initial_params: Mapping[str, PyTree],
    minibatch_functions: Mapping[str, Callable[..., Any]],
    key: Array,
) -> dict[str, dict[str, Any]]:
    batch_kwargs = {
        "pts_pde": config.ntk_pde_points,
        "pts_ics": config.ntk_ic_points,
        "pts_bcs": config.ntk_bc_points,
    }
    vanilla_batch = minibatch_functions["vanilla"](key, **batch_kwargs)
    adaptive_batch = minibatch_functions["adaptive"](key, **batch_kwargs)
    tal_batch = minibatch_functions["tal"](key, **batch_kwargs)
    for batch in (vanilla_batch, adaptive_batch, tal_batch):
        for leaf in jax.tree_util.tree_leaves(batch):
            leaf.block_until_ready()

    methods = {
        "vanilla": {
            "model": models["solution_2d"],
            "params": initial_params["vanilla"],
            "residual_points": vanilla_batch[0],
            "ic_points": vanilla_batch[1],
            "ic_targets": vanilla_batch[2],
            "bc_points": vanilla_batch[3],
            "bc_targets": vanilla_batch[4],
            "lifted": False,
            "geometry": None,
            "residual_weight": None,
            "residual_weight_mode": "mul",
        },
        "adaptive": {
            "model": models["solution_2d"],
            "params": initial_params["adaptive"],
            "residual_points": adaptive_batch[0],
            "ic_points": adaptive_batch[2],
            "ic_targets": adaptive_batch[3],
            "bc_points": adaptive_batch[4],
            "bc_targets": adaptive_batch[5],
            "lifted": False,
            "geometry": None,
            "residual_weight": adaptive_batch[1],
            "residual_weight_mode": "mul",
        },
        "tal": {
            "model": models["solution_3d"],
            "params": initial_params["tal"],
            "residual_points": tal_batch[0][:, :3],
            "ic_points": tal_batch[1],
            "ic_targets": tal_batch[2],
            "bc_points": tal_batch[3],
            "bc_targets": tal_batch[4],
            "lifted": True,
            "geometry": tal_batch[0][:, 3:6],
            "residual_weight": jnp.abs(tal_batch[0][:, 4]),
            "residual_weight_mode": "div",
        },
    }

    results: dict[str, dict[str, Any]] = {}
    for name, method in methods.items():
        print(f"Computing {name} initial NTK spectra ...")
        start = time.perf_counter()
        result = assemble_dirichlet_ntk(
            apply_fn=lambda params, point, model=method["model"]: model.apply(
                params, point
            ),
            params=method["params"],
            residual_points=method["residual_points"],
            nu=config.nu,
            ic_points=method["ic_points"],
            ic_targets=method["ic_targets"],
            bc_points=method["bc_points"],
            bc_targets=method["bc_targets"],
            lifted=method["lifted"],
            geometry=method["geometry"],
            residual_weight=method["residual_weight"],
            residual_weight_mode=method["residual_weight_mode"],
            w_ic=config.w_ic,
            w_bc=config.w_bc,
            chunk_size=config.ntk_chunk_size,
        )
        result["ntk_seconds"] = time.perf_counter() - start
        results[name] = result
        summary = spectral_summary(result["eigen_total"])
        print(
            f"[{name}] NTK={result['ntk_seconds']:.2f}s, "
            f"trace={summary['trace']:.3e}, "
            f"effective rank={summary['effective_rank']:.2f}"
        )
    return results


def prepare_test_problem(
    models: Mapping[str, Any],
    geometry: Mapping[str, Any],
    coordinate_fn: Callable[[Array], Array],
) -> dict[str, Any]:
    t_grid, x_grid, u_reference = geometry["T"], geometry["X"], geometry["U"]
    nt, nx = u_reference.shape
    t_flat, x_flat = t_grid.reshape(-1), x_grid.reshape(-1)
    labels = u_reference.reshape(-1, 1)
    inputs_2d = jnp.stack((t_flat, x_flat), axis=-1)
    inputs_3d = tal_utils.create_test_inputs(t_flat, x_flat, coordinate_fn)
    x_values = np.asarray(jax.device_get(x_grid[0]))
    interface_index = int(np.argmin(np.abs(x_values - 0.5)))
    return {
        "nt": int(nt),
        "nx": int(nx),
        "interface_index": interface_index,
        "interface_x": float(x_values[interface_index]),
        "labels": labels,
        "inputs_2d": inputs_2d,
        "inputs_3d": inputs_3d,
        "eval_2d": vanilla_utils.create_eval_fn(
            models["solution_2d"],
            nx_test=nx,
            interface_idx=interface_index,
            interface_mode="drop",
        ),
        "eval_3d": tal_utils.create_eval_fn(
            models["solution_3d"],
            nx_test=nx,
            interface_idx=interface_index,
            interface_mode="drop",
        ),
    }


def train_three_methods(
    config: ExperimentConfig,
    models: Mapping[str, Any],
    initial_params: Mapping[str, PyTree],
    minibatch_functions: Mapping[str, Callable[..., Any]],
    test_problem: Mapping[str, Any],
    key: Array,
) -> tuple[dict[str, PyTree], dict[str, Mapping[str, np.ndarray]]]:
    optimizer_2d = make_optimizer(config)
    optimizer_3d = make_optimizer(config)
    check_optimizer_dtype(
        optimizer_2d,
        initial_params["vanilla_and_adaptive"],
        "2D solution parameters",
    )
    check_optimizer_dtype(optimizer_3d, initial_params["tal"], "TAL parameters")

    common = {
        "key": key,
        "nu": config.nu,
        "max_iters": config.train_iters,
        "pts_pde": config.train_pde_points,
        "pts_ics": config.train_ic_points,
        "pts_bcs": config.train_bc_points,
        "max_runtime": config.max_runtime,
        "eval_every": config.train_eval_every,
    }

    print("Training Vanilla PINN ...")
    vanilla_params, _, vanilla_history = vanilla_utils.train_pde_stage(
        stage_name="Vanilla PINN",
        pde_params=initial_params["vanilla_and_adaptive"],
        pde_optimizer=optimizer_2d,
        pde_loss_grad_fn=vanilla_utils.create_pde_loss_grad_fn(
            models["solution_2d"], w_ic=config.w_ic, w_bc=config.w_bc
        ),
        pde_minibatch_fn=minibatch_functions["vanilla"],
        eval_fn=test_problem["eval_2d"],
        test_inputs=test_problem["inputs_2d"],
        test_labels=test_problem["labels"],
        **common,
    )

    print("Training adaptive-sampling PINN ...")
    adaptive_params, _, adaptive_history = adaptive_utils.train_pde_stage(
        stage_name="Adaptive-sampling PINN",
        pde_params=initial_params["vanilla_and_adaptive"],
        pde_optimizer=optimizer_2d,
        pde_loss_grad_fn=adaptive_utils.create_pde_loss_grad_fn(
            models["solution_2d"], w_ic=config.w_ic, w_bc=config.w_bc
        ),
        pde_minibatch_fn=minibatch_functions["adaptive"],
        eval_fn=test_problem["eval_2d"],
        test_inputs=test_problem["inputs_2d"],
        test_labels=test_problem["labels"],
        **common,
    )

    print("Training TAL-PINN ...")
    tal_params, _, tal_history = tal_utils.train_pde_stage(
        stage_name="TAL-PINN",
        pde_params=initial_params["tal"],
        pde_optimizer=optimizer_3d,
        pde_loss_grad_fn=tal_utils.create_pde_loss_grad_fn(
            models["solution_3d"], w_ic=config.w_ic, w_bc=config.w_bc
        ),
        pde_minibatch_fn=minibatch_functions["tal"],
        eval_fn=test_problem["eval_3d"],
        test_inputs=test_problem["inputs_3d"],
        test_labels=test_problem["labels"],
        **common,
    )
    final_params = {
        "vanilla": vanilla_params,
        "adaptive": adaptive_params,
        "tal": tal_params,
    }
    histories = {
        "vanilla": vanilla_history,
        "adaptive": adaptive_history,
        "tal": tal_history,
    }
    return final_params, histories


# ---------------------------------------------------------------------------
# Post-training physical residuals and saved analysis arrays
# ---------------------------------------------------------------------------


def make_final_residual_functions(
    models: Mapping[str, Any],
    coordinate_params: PyTree,
    nu: float,
) -> tuple[Callable[[PyTree, Array], Array], Callable[[PyTree, Array], Array]]:
    model_2d = models["solution_2d"]
    model_3d = models["solution_3d"]
    coordinate_model = models["coordinate"]
    nu_array = jnp.asarray(nu, dtype=TRAIN_DTYPE)

    def solution_2d(params: PyTree, tx: Array) -> Array:
        return jnp.squeeze(model_2d.apply(params, tx))

    solution_2d_grad = jax.grad(solution_2d, argnums=1)
    solution_2d_hess = jax.hessian(solution_2d, argnums=1)

    def residual_2d(params: PyTree, tx: Array) -> Array:
        value = solution_2d(params, tx)
        gradient = solution_2d_grad(params, tx)
        hessian = solution_2d_hess(params, tx)
        return gradient[0] + value * gradient[1] - nu_array * hessian[1, 1]

    def coordinate_value(tx: Array) -> Array:
        return jnp.squeeze(coordinate_model.apply(coordinate_params, tx))

    coordinate_grad = jax.grad(coordinate_value)
    coordinate_hess = jax.hessian(coordinate_value)

    def lifted_solution(params: PyTree, txi: Array) -> Array:
        return jnp.squeeze(model_3d.apply(params, txi))

    lifted_grad = jax.grad(lifted_solution, argnums=1)
    lifted_hess = jax.hessian(lifted_solution, argnums=1)

    def residual_tal(params: PyTree, tx: Array) -> Array:
        xi = coordinate_value(tx)
        grad_xi = coordinate_grad(tx)
        hess_xi = coordinate_hess(tx)
        txi = jnp.stack((tx[0], tx[1], xi))
        value = lifted_solution(params, txi)
        gradient = lifted_grad(params, txi)
        hessian = lifted_hess(params, txi)

        xi_t, xi_x = grad_xi
        xi_xx = hess_xi[1, 1]
        u_t = gradient[0] + gradient[2] * xi_t
        u_x = gradient[1] + gradient[2] * xi_x
        u_xx = (
            hessian[1, 1]
            + 2.0 * hessian[1, 2] * xi_x
            + hessian[2, 2] * xi_x**2
            + gradient[2] * xi_xx
        )
        return u_t + value * u_x - nu_array * u_xx

    return residual_2d, residual_tal


def _evaluate_points_in_chunks(
    point_fn: Callable[[PyTree, Array], Array],
    params: PyTree,
    points: np.ndarray | Array,
    chunk_size: int,
) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError(f"Residual points must have shape (N, 2); got {points_array.shape}.")
    if chunk_size <= 0:
        raise ValueError("Residual chunk size must be positive.")
    evaluator = jax.jit(jax.vmap(lambda point: point_fn(params, point)))
    pieces: list[np.ndarray] = []
    for start in range(0, points_array.shape[0], chunk_size):
        chunk = jnp.asarray(points_array[start : start + chunk_size], dtype=TRAIN_DTYPE)
        pieces.append(np.asarray(jax.device_get(evaluator(chunk))))
    return np.concatenate(pieces)


def evaluate_final_residuals(
    config: ExperimentConfig,
    models: Mapping[str, Any],
    coordinate_params: PyTree,
    final_params: Mapping[str, PyTree],
    geometry: Mapping[str, Any],
    key: Array,
) -> dict[str, np.ndarray]:
    residual_2d, residual_tal = make_final_residual_functions(
        models, coordinate_params, config.nu
    )
    t_reference = np.asarray(jax.device_get(geometry["T"]), dtype=np.float32)
    x_reference = np.asarray(jax.device_get(geometry["X"]), dtype=np.float32)
    time_index = _stride_indices(t_reference.shape[0], config.residual_time_stride)
    space_index = _stride_indices(x_reference.shape[1], config.residual_space_stride)
    grid_index = np.ix_(time_index, space_index)
    t_residual = t_reference[grid_index]
    x_residual = x_reference[grid_index]
    points = np.column_stack((t_residual.reshape(-1), x_residual.reshape(-1)))

    print(f"Evaluating final residuals on {t_residual.shape} physical grid ...")
    fields = {
        "vanilla": _evaluate_points_in_chunks(
            residual_2d,
            final_params["vanilla"],
            points,
            config.residual_chunk_size,
        ).reshape(t_residual.shape),
        "adaptive": _evaluate_points_in_chunks(
            residual_2d,
            final_params["adaptive"],
            points,
            config.residual_chunk_size,
        ).reshape(t_residual.shape),
        "tal": _evaluate_points_in_chunks(
            residual_tal,
            final_params["tal"],
            points,
            config.residual_chunk_size,
        ).reshape(t_residual.shape),
    }

    sample = jax.random.uniform(
        key,
        (config.plot_sample_points, 2),
        dtype=TRAIN_DTYPE,
    )
    sample_t = sample[:, 0]
    sample_xi = sample[:, 1]
    sample_x = geometry["map_xi_to_x"](sample_t, sample_xi)
    sample_points = jnp.stack((sample_t, sample_x), axis=-1)
    sample_residuals = {
        "vanilla": _evaluate_points_in_chunks(
            residual_2d,
            final_params["vanilla"],
            sample_points,
            config.residual_chunk_size,
        ),
        "adaptive": _evaluate_points_in_chunks(
            residual_2d,
            final_params["adaptive"],
            sample_points,
            config.residual_chunk_size,
        ),
        "tal": _evaluate_points_in_chunks(
            residual_tal,
            final_params["tal"],
            sample_points,
            config.residual_chunk_size,
        ),
    }

    result: dict[str, np.ndarray] = {
        "residual_T": t_residual,
        "residual_X": x_residual,
        "sample_t": np.asarray(jax.device_get(sample_t)),
        "sample_xi": np.asarray(jax.device_get(sample_xi)),
        "sample_x": np.asarray(jax.device_get(sample_x)),
    }
    for name, field in fields.items():
        result[f"{name}_final_residual"] = np.asarray(field)
        result[f"{name}_sample_final_residual"] = np.asarray(sample_residuals[name])
    return result


def flatten_analysis_data(
    config: ExperimentConfig,
    geometry: Mapping[str, Any],
    ntk_results: Mapping[str, Mapping[str, Any]],
    histories: Mapping[str, Mapping[str, np.ndarray]],
    residual_data: Mapping[str, np.ndarray],
    coordinate_metrics: Mapping[str, float],
    *,
    coordinate_seconds: float,
) -> dict[str, np.ndarray]:
    export: dict[str, np.ndarray] = {
        "reference_T": np.asarray(jax.device_get(geometry["T"])),
        "reference_X": np.asarray(jax.device_get(geometry["X"])),
        "reference_U": np.asarray(jax.device_get(geometry["U"])),
        "xi_grid": np.asarray(jax.device_get(geometry["xi_grid"])),
        "Xi_of_x": np.asarray(jax.device_get(geometry["Xi_of_x"])),
        "X_adapt": np.asarray(jax.device_get(geometry["X_adapt"])),
        "mesh_seconds": np.asarray([[geometry["mesh_seconds"]]], dtype=np.float64),
        "coordinate_training_seconds": np.asarray([[coordinate_seconds]], dtype=np.float64),
    }
    export.update({key: np.asarray(value) for key, value in residual_data.items()})

    scalar_ntk_keys = (
        "initial_residual_loss",
        "initial_total_loss",
        "initial_decay_quadratic",
        "n_residual_rows",
        "n_ic_rows",
        "n_bc_rows",
        "ntk_seconds",
    )
    for name, result in ntk_results.items():
        for key in ("eigen_residual_unscaled", "eigen_residual", "eigen_total"):
            values = np.asarray(result[key], dtype=np.float64)
            export[f"{name}_{key}"] = values
            export[f"{name}_{key}_index"] = np.arange(1, values.size + 1)
        for key in scalar_ntk_keys:
            export[f"{name}_{key}"] = np.asarray([[result[key]]])
        for block in ("residual", "total"):
            summary = spectral_summary(np.asarray(result[f"eigen_{block}"]))
            for metric, value in summary.items():
                export[f"{name}_{block}_{metric}"] = np.asarray([[value]])

    history_keys = (
        "iter",
        "loss",
        "train_time",
        "eval_iter",
        "eval_train_time",
        "mse",
        "rl2",
    )
    for name, history in histories.items():
        for key in history_keys:
            export[f"{name}_{key}"] = np.asarray(history[key])

    for key, value in coordinate_metrics.items():
        export[f"coordinate_{key}"] = np.asarray([[value]])
    for key, value in asdict(config).items():
        export[f"config_{key}"] = np.asarray([[value]])

    # Compatibility aliases for earlier plotting code.
    for number, name in enumerate(("vanilla", "adaptive", "tal"), start=1):
        export[f"idx_res{number}"] = export[f"{name}_eigen_residual_unscaled_index"]
        export[f"v_res{number}"] = export[f"{name}_eigen_residual_unscaled"]
        export[f"idx_total{number}"] = export[f"{name}_eigen_total_index"]
        export[f"v_total{number}"] = export[f"{name}_eigen_total"]
    return export


def save_analysis_data(
    prefix: Path,
    export: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "npz": Path(f"{prefix}_analysis.npz"),
        "mat": Path(f"{prefix}_analysis.mat"),
        "metadata": Path(f"{prefix}_metadata.json"),
        "checkpoint": Path(f"{prefix}_models.msgpack"),
    }

    npz_tmp = Path(f"{paths['npz']}.tmp")
    with npz_tmp.open("wb") as stream:
        np.savez_compressed(stream, **export)
    os.replace(npz_tmp, paths["npz"])

    mat_tmp = Path(f"{paths['mat']}.tmp")
    scipy.io.savemat(mat_tmp, dict(export), appendmat=False, do_compression=True)
    os.replace(mat_tmp, paths["mat"])

    json_tmp = Path(f"{paths['metadata']}.tmp")
    json_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(json_tmp, paths["metadata"])

    checkpoint_tmp = Path(f"{paths['checkpoint']}.tmp")
    checkpoint_tmp.write_bytes(serialization.to_bytes(checkpoint))
    os.replace(checkpoint_tmp, paths["checkpoint"])
    return paths


def load_analysis_data(prefix: Path) -> dict[str, np.ndarray]:
    path = Path(f"{prefix}_analysis.npz")
    if not path.is_file():
        raise FileNotFoundError(f"Saved analysis file not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------


METHODS = ("vanilla", "adaptive", "tal")
LABELS = {
    "vanilla": "Vanilla PINN",
    "adaptive": "Adaptive sampling PINN",
    "tal": "TAL-PINN",
}
COLORS = {
    "vanilla": "#4C78A8",
    "adaptive": "#F58518",
    "tal": "#54A24B",
}


def _save_figure(fig: Any, prefix: Path, suffix: str) -> tuple[Path, Path]:
    pdf_path = Path(f"{prefix}_{suffix}.pdf")
    png_path = Path(f"{prefix}_{suffix}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def plot_sampling_distribution(
    data: Mapping[str, np.ndarray],
    prefix: Path,
) -> tuple[Path, Path]:
    t = np.asarray(data["sample_t"]).reshape(-1)
    xi = np.asarray(data["sample_xi"]).reshape(-1)
    x = np.asarray(data["sample_x"]).reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True, sharey=True)
    axes[0].scatter(
        xi,
        t,
        s=3,
        alpha=0.35,
        color=COLORS["vanilla"],
        rasterized=True,
    )
    axes[0].set_title("Uniform physical-space sample")
    scatter = axes[1].scatter(
        x,
        t,
        c=xi,
        cmap="viridis",
        s=3,
        alpha=0.45,
        rasterized=True,
    )
    axes[1].set_title("Reference-generated adaptive sample")
    for axis in axes:
        axis.set_xlabel(r"physical coordinate $x$")
        axis.set_ylabel(r"time $t$")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
    fig.colorbar(scatter, ax=axes[1], label=r"computational coordinate $\xi$")
    fig.suptitle("Collocation-point density induced by the oracle map", y=1.02)
    fig.tight_layout()
    return _save_figure(fig, prefix, "sampling_distribution")


def plot_final_residual_fields(
    data: Mapping[str, np.ndarray],
    prefix: Path,
) -> tuple[Path, Path]:
    t_grid = np.asarray(data["residual_T"])
    x_grid = np.asarray(data["residual_X"])
    floor = 1.0e-14
    logged = {
        name: np.log10(np.abs(np.asarray(data[f"{name}_final_residual"])) + floor)
        for name in METHODS
    }
    all_values = np.concatenate([logged[name].reshape(-1) for name in METHODS])
    color_min, color_max = np.percentile(all_values, (1.0, 99.5))
    if color_max <= color_min:
        color_max = color_min + 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.2), sharex=True, sharey=True)
    mappable = None
    for axis, name in zip(axes, METHODS):
        mappable = axis.pcolormesh(
            x_grid,
            t_grid,
            logged[name],
            shading="auto",
            cmap="magma",
            vmin=color_min,
            vmax=color_max,
            rasterized=True,
        )
        axis.set_title(LABELS[name])
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$t$")
    assert mappable is not None
    fig.colorbar(
        mappable,
        ax=axes,
        label=r"$\log_{10}(|R|+10^{-14})$",
        shrink=0.92,
    )
    final_iteration = max(
        int(np.max(data[f"{name}_iter"]))
        if np.asarray(data[f"{name}_iter"]).size
        else 0
        for name in METHODS
    )
    fig.suptitle(
        f"Physical PDE residual from final checkpoints (iteration {final_iteration})",
        y=1.02,
    )
    fig.subplots_adjust(left=0.06, right=0.91, bottom=0.14, top=0.84, wspace=0.18)
    return _save_figure(fig, prefix, "final_residual_fields")


def plot_sampling_residual_correspondence(
    data: Mapping[str, np.ndarray],
    prefix: Path,
) -> tuple[Path, Path]:
    t = np.asarray(data["sample_t"]).reshape(-1)
    xi = np.asarray(data["sample_xi"]).reshape(-1)
    x = np.asarray(data["sample_x"]).reshape(-1)
    t_grid = np.asarray(data["residual_T"])
    x_grid = np.asarray(data["residual_X"])
    tal_field = np.abs(np.asarray(data["tal_final_residual"]))
    tal_sample = np.abs(np.asarray(data["tal_sample_final_residual"]).reshape(-1))
    floor = 1.0e-14
    log_sample = np.log10(tal_sample + floor)
    color_min, color_max = np.percentile(log_sample, (1.0, 99.5))
    if color_max <= color_min:
        color_max = color_min + 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.3), sharex=True, sharey=True)
    point_map = axes[0].scatter(
        x,
        t,
        c=xi,
        cmap="viridis",
        s=3,
        alpha=0.45,
        rasterized=True,
    )
    axes[0].set_title(r"Adaptive points (color: $\xi$)")
    fig.colorbar(point_map, ax=axes[0], shrink=0.85, label=r"$\xi$")

    field_map = axes[1].pcolormesh(
        x_grid,
        t_grid,
        np.log10(tal_field + floor),
        shading="auto",
        cmap="magma",
        vmin=color_min,
        vmax=color_max,
        rasterized=True,
    )
    axes[1].scatter(x, t, s=1.0, color="white", alpha=0.12, rasterized=True)
    axes[1].set_title("Final TAL residual with adaptive points")
    fig.colorbar(
        field_map,
        ax=axes[1],
        shrink=0.85,
        label=r"$\log_{10}(|R|+10^{-14})$",
    )

    residual_map = axes[2].scatter(
        x,
        t,
        c=log_sample,
        cmap="magma",
        vmin=color_min,
        vmax=color_max,
        s=4,
        alpha=0.60,
        rasterized=True,
    )
    axes[2].set_title("Adaptive points colored by final TAL residual")
    fig.colorbar(
        residual_map,
        ax=axes[2],
        shrink=0.85,
        label=r"$\log_{10}(|R|+10^{-14})$",
    )
    for axis in axes:
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$t$")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
    fig.suptitle("Adaptive point density versus post-training residual", y=1.02)
    fig.tight_layout()
    return _save_figure(fig, prefix, "sampling_residual_correspondence")


def plot_ntk_spectra(
    data: Mapping[str, np.ndarray],
    prefix: Path,
    *,
    trace_normalize: bool = False,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    for name in METHODS:
        for axis, key, title in (
            (
                axes[0],
                f"{name}_eigen_residual_unscaled",
                r"Residual block $K_{r,r}$ (before mean reduction)",
            ),
            (
                axes[1],
                f"{name}_eigen_total",
                r"Loss-consistent total NTK $K$",
            ),
        ):
            values = np.maximum(np.asarray(data[key], dtype=np.float64).reshape(-1), 0.0)
            if trace_normalize:
                values = values / max(np.sum(values), np.finfo(np.float64).tiny)
            floor = max(float(np.max(values)) * 1.0e-16, np.finfo(np.float64).tiny)
            axis.loglog(
                np.arange(1, values.size + 1),
                np.maximum(values, floor),
                color=COLORS[name],
                linewidth=1.7,
                label=LABELS[name],
            )
            axis.set_title(title)
    for axis in axes:
        axis.set_xlabel("Eigenvalue index")
        axis.set_ylabel(
            "Trace-normalized eigenvalue" if trace_normalize else r"Eigenvalue $\lambda$"
        )
        axis.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.8)
        axis.legend(frameon=False)
    fig.suptitle("NTK spectral decay at the paired initial state", y=1.02)
    fig.tight_layout()
    return _save_figure(fig, prefix, "ntk_spectra")


def plot_loss_histories(
    data: Mapping[str, np.ndarray],
    prefix: Path,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    for name in METHODS:
        iterations = np.asarray(data[f"{name}_iter"], dtype=np.int64).reshape(-1)
        losses = np.asarray(data[f"{name}_loss"], dtype=np.float64).reshape(-1)
        safe = np.maximum(losses, np.finfo(np.float64).tiny)
        axes[0].semilogy(
            iterations,
            safe,
            color=COLORS[name],
            linewidth=1.6,
            label=LABELS[name],
        )
        if safe.size:
            axes[1].semilogy(
                iterations,
                safe / safe[0],
                color=COLORS[name],
                linewidth=1.6,
                label=LABELS[name],
            )
    axes[0].set_title("Absolute training objective")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Loss normalized by first logged value")
    axes[1].set_ylabel(r"$L_k/L_0$")
    for axis in axes:
        axis.set_xlabel("Iteration")
        axis.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.8)
        axis.legend(frameon=False)
    fig.suptitle("Training-loss histories", y=1.02)
    fig.tight_layout()
    return _save_figure(fig, prefix, "loss_history")


def render_all_figures(
    data: Mapping[str, np.ndarray],
    prefix: Path,
) -> list[Path]:
    figure_paths: list[Path] = []
    for pair in (
        plot_sampling_distribution(data, prefix),
        plot_final_residual_fields(data, prefix),
        plot_sampling_residual_correspondence(data, prefix),
        plot_ntk_spectra(data, prefix),
        plot_loss_histories(data, prefix),
    ):
        figure_paths.extend(pair)
    return figure_paths


def print_final_summary(data: Mapping[str, np.ndarray]) -> None:
    header = (
        f"{'method':24s} {'final loss':>14s} {'final RL2':>14s} "
        f"{'residual RMS':>14s} {'max |R|':>14s}"
    )
    print(header)
    print("-" * len(header))
    for name in METHODS:
        losses = np.asarray(data[f"{name}_loss"], dtype=np.float64).reshape(-1)
        rl2 = np.asarray(data[f"{name}_rl2"], dtype=np.float64).reshape(-1)
        residual = np.abs(np.asarray(data[f"{name}_final_residual"], dtype=np.float64))
        final_loss = losses[-1] if losses.size else np.nan
        final_rl2 = rl2[-1] if rl2.size else np.nan
        rms = np.sqrt(np.mean(residual**2))
        print(
            f"{LABELS[name]:24s} {final_loss:14.6e} {final_rl2:14.6e} "
            f"{rms:14.6e} {np.max(residual):14.6e}"
        )


# ---------------------------------------------------------------------------
# Small notebook orchestration helpers
# ---------------------------------------------------------------------------


def validate_config(config: ExperimentConfig) -> None:
    if config.nu <= 0.0:
        raise ValueError("nu must be positive.")
    if config.learning_rate <= 0.0 or config.max_runtime <= 0.0:
        raise ValueError("learning_rate and max_runtime must be positive.")
    if config.w_ic < 0.0 or config.w_bc < 0.0:
        raise ValueError("loss weights must be non-negative.")


def create_experiment_keys(seed: int = 42) -> dict[str, Array]:
    """Create named, reproducible PRNG keys for separate notebook cells."""

    names = (
        "vanilla_init",
        "adaptive_init",
        "tal_init",
        "coordinate_init",
        "coordinate_train",
        "ntk",
        "train_vanilla",
        "train_adaptive",
        "train_tal",
        "residual",
    )
    values = jax.random.split(jax.random.PRNGKey(seed), len(names))
    return dict(zip(names, values))


def build_analysis_metadata(
    config: ExperimentConfig,
    geometry: Mapping[str, Any],
    ntk_results: Mapping[str, Mapping[str, Any]],
    coordinate_metrics: Mapping[str, float],
    *,
    coordinate_seconds: float,
) -> dict[str, Any]:
    """Build the JSON metadata after all numerical notebook cells finish."""

    nt, nx = geometry["U"].shape
    return {
        "experiment": "reference-coordinate static-shock Burgers mechanism diagnostic",
        "coordinate_source": "reference solution (oracle diagnostic only)",
        "training_dtype": "float32",
        "ntk_eigensolver_dtype": "numpy.float64",
        "data_path": str(geometry["data_path"]),
        "reference_shape": [int(nt), int(nx)],
        "config": asdict(config),
        "mesh_seconds": geometry["mesh_seconds"],
        "coordinate_training_seconds": coordinate_seconds,
        "coordinate_diagnostics": coordinate_metrics,
        "versions": {
            "jax": _package_version("jax"),
            "flax": _package_version("flax"),
            "optax": _package_version("optax"),
            "soap-jax": _package_version("soap-jax"),
        },
        "utilities": {
            "vanilla": VANILLA_UTILITY_PATH.name,
            "adaptive": ADAPTIVE_UTILITY_PATH.name,
            "tal": TAL_UTILITY_PATH.name,
        },
        "ntk_summary": {
            name: {
                "residual": spectral_summary(result["eigen_residual"]),
                "total": spectral_summary(result["eigen_total"]),
                "seconds": result["ntk_seconds"],
                "initial_residual_loss": result["initial_residual_loss"],
                "initial_total_loss": result["initial_total_loss"],
                "initial_decay_quadratic": result["initial_decay_quadratic"],
            }
            for name, result in ntk_results.items()
        },
    }


def build_checkpoint(
    coordinate_params,
    initial_params,
    final_params,
):
    return {
        "format_version": 3,
        "coordinate_source": "reference_solution",
        "coordinate_params": jax.device_get(coordinate_params),
        "initial_params": jax.device_get(
            {
                "vanilla": initial_params["vanilla"],
                "adaptive": initial_params["adaptive"],
                "tal": initial_params["tal"],
            }
        ),
        "final_params": jax.device_get(final_params),
    }


def main() -> None:
    """Intentionally empty: execute the analysis step by step in a notebook."""

    pass


if __name__ == "__main__":
    main()
