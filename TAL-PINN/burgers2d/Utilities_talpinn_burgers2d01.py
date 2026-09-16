import os
import numpy as np

import jax
import jax.numpy as jnp
from jax import jacfwd, vmap
from functools import partial
from typing import NamedTuple, Sequence, Callable

import flax.linen as nn
import optax
from soap_jax import soap

import time
import scipy.io
import matplotlib.pyplot as plt

from flax import serialization

class FNN(nn.Module):
    layer_sizes: Sequence[int]
    activation: Callable = nn.silu
    out_bias: bool = True

    @nn.compact
    def __call__(self, x):
        kinit = jax.nn.initializers.he_uniform()
        for feat in self.layer_sizes[:-1]:
            x = nn.Dense(feat, kernel_init=kinit)(x)
            x = self.activation(x)
        x = nn.Dense(self.layer_sizes[-1], kernel_init=kinit, use_bias=self.out_bias)(x)
        return x

class FNN_fourier(nn.Module):
    layer_sizes: Sequence[int]
    activation: Callable = nn.silu
    out_bias: bool = True
    fourier_dim: int = None

    @nn.compact
    def __call__(self, x):
        # Fourier feature layer
        fourier_dim = (self.layer_sizes[0] if self.fourier_dim is None else self.fourier_dim)

        x = nn.Dense(fourier_dim, kernel_init=jax.nn.initializers.normal(stddev=1.0))(x)
        x = jnp.sin(2.0 * jnp.pi * x)

        # Standard FNN
        kinit = jax.nn.initializers.he_uniform()

        for feat in self.layer_sizes[:-1]:
            x = nn.Dense(feat, kernel_init=kinit)(x)
            x = self.activation(x)

        x = nn.Dense(self.layer_sizes[-1], kernel_init=kinit, use_bias=self.out_bias)(x)

        return x


def _nodal_dudxy_periodic(U, x_src, y_src):
    """
    在周期张量积物理网格上计算节点处的 u_x 和 u_y。

    Parameters
    ----------
    U : (Nt, Ny, Nx)
        两个空间方向均包含重复的周期端点。
    x_src : (Nt, Nx)
    y_src : (Nt, Ny)

    Notes
    -----
    末端节点与首端节点表示同一个周期位置，因此先在
    ``Nx - 1`` / ``Ny - 1`` 个唯一节点上作周期中心差分，
    再把首端导数复制到末端。这样不会在周期边界引入单边差分误差。
    """
    eps = jnp.asarray(jnp.finfo(U.dtype).eps, dtype=U.dtype)

    # x derivative on the Nx - 1 unique periodic nodes.
    Ux_core = U[:, :, :-1]
    x_core = x_src[:, :-1]
    length_x = x_src[:, -1] - x_src[:, 0]

    x_prev = jnp.roll(x_core, shift=1, axis=1)
    x_next = jnp.roll(x_core, shift=-1, axis=1)
    x_prev = x_prev.at[:, 0].add(-length_x)
    x_next = x_next.at[:, -1].add(length_x)

    denom_x = x_next - x_prev
    denom_x = jnp.where(jnp.abs(denom_x) > eps, denom_x, eps)
    dUdx_core = (jnp.roll(Ux_core, shift=-1, axis=2) - jnp.roll(Ux_core, shift=1, axis=2)) / denom_x[:, None, :]
    dUdx = jnp.concatenate([dUdx_core, dUdx_core[:, :, :1]], axis=2)

    # y derivative on the Ny - 1 unique periodic nodes.
    Uy_core = U[:, :-1, :]
    y_core = y_src[:, :-1]
    length_y = y_src[:, -1] - y_src[:, 0]

    y_prev = jnp.roll(y_core, shift=1, axis=1)
    y_next = jnp.roll(y_core, shift=-1, axis=1)
    y_prev = y_prev.at[:, 0].add(-length_y)
    y_next = y_next.at[:, -1].add(length_y)

    denom_y = y_next - y_prev
    denom_y = jnp.where(jnp.abs(denom_y) > eps, denom_y, eps)
    dUdy_core = (jnp.roll(Uy_core, shift=-1, axis=1) - jnp.roll(Uy_core, shift=1, axis=1)) / denom_y[:, :, None]
    dUdy = jnp.concatenate([dUdy_core, dUdy_core[:, :1, :]], axis=1)

    return dUdx, dUdy

def _smooth_along_axis(values, axis):
    """
    沿指定轴进行 [1, 2, 1] 平滑。

    内部:
        (left + 2 * center + right) / 4

    两端:
        使用截断并重新归一化的 [1, 2, 1] 核，
        与原二维平滑中的 12、9 分母逻辑一致。
    """
    values_last = jnp.moveaxis(values, axis, -1)

    if values_last.shape[-1] == 1:
        return values

    smoothed = values_last
    smoothed = smoothed.at[..., 1:-1].set(0.25 * (values_last[..., :-2] + 2.0 * values_last[..., 1:-1] + values_last[..., 2:]))
    smoothed = smoothed.at[..., 0].set((2.0 * values_last[..., 0] + values_last[..., 1]) / 3.0)
    smoothed = smoothed.at[..., -1].set((values_last[..., -2] + 2.0 * values_last[..., -1]) / 3.0)

    return jnp.moveaxis(smoothed, -1, axis)


def _smooth_periodic_axis(values, axis):
    """沿含重复周期端点的轴作一次 [1, 2, 1] 周期平滑。"""
    values_last = jnp.moveaxis(values, axis, -1)
    core = values_last[..., :-1]
    core_smoothed = 0.25 * (jnp.roll(core, 1, axis=-1) + 2.0 * core + jnp.roll(core, -1, axis=-1))
    smoothed = jnp.concatenate([core_smoothed, core_smoothed[..., :1]], axis=-1)
    return jnp.moveaxis(smoothed, -1, axis)


def _smooth_3d_monitor(M, smooth_steps):
    """
    对 M(t, y, x) 同时在时间和两个空间方向平滑。

    M: (Nt, Ny, Nx)

    时间坐标 T 不移动，但 M[t] 会受到 M[t-1] 和 M[t+1]
    的影响。两个空间方向采用周期平滑，并保持重复端点一致。
    """
    def smooth_step(_, current_M):
        current_M = _smooth_along_axis(current_M, axis=0)
        current_M = _smooth_periodic_axis(current_M, axis=1)
        current_M = _smooth_periodic_axis(current_M, axis=2)

        return current_M

    return jax.lax.fori_loop(0, smooth_steps, smooth_step, M)


def _build_monitor_field(U, x_src, y_src, *, beta, alpha_floor, direction_x, direction_y, smooth_steps):
    """在固定物理网格上构造并平滑 M(t, y, x)。"""
    dUdx, dUdy = _nodal_dudxy_periodic(U, x_src, y_src)
    directional_derivative = direction_x * dUdx + direction_y * dUdy
    div = jnp.abs(directional_derivative) - directional_derivative

    dx = x_src[:, 1:] - x_src[:, :-1]
    dy = y_src[:, 1:] - y_src[:, :-1]

    # 每个物理单元上作二维梯形积分，再除以区域面积。
    div_cell = 0.25 * (div[:, :-1, :-1] + div[:, 1:, :-1] + div[:, :-1, 1:] + div[:, 1:, 1:])
    integral = jnp.sum(div_cell * dy[:, :, None] * dx[:, None, :], axis=(1, 2))

    eps = jnp.asarray(jnp.finfo(U.dtype).eps, dtype=U.dtype)
    area = jnp.maximum((x_src[:, -1] - x_src[:, 0]) * (y_src[:, -1] - y_src[:, 0]), eps)
    alpha = jnp.maximum(integral / area, jnp.asarray(alpha_floor, dtype=U.dtype))

    monitor = 1.0 + beta * div / alpha[:, None, None]

    # 明确恢复重复周期端点，随后沿 t、y、x 三方向平滑。
    monitor = monitor.at[:, :, -1].set(monitor[:, :, 0])
    monitor = monitor.at[:, -1, :].set(monitor[:, 0, :])
    return _smooth_3d_monitor(monitor, smooth_steps)


def _interp_rectilinear_2d(x_src, y_src, field, x_query, y_query):
    """在一个二维张量积物理网格上作双线性插值。"""
    eps = jnp.asarray(jnp.finfo(field.dtype).eps, dtype=field.dtype)

    x_query = jnp.clip(x_query, x_src[0], x_src[-1])
    y_query = jnp.clip(y_query, y_src[0], y_src[-1])

    ix = jnp.searchsorted(x_src, x_query, side="right") - 1
    iy = jnp.searchsorted(y_src, y_query, side="right") - 1
    ix = jnp.clip(ix, 0, x_src.shape[0] - 2)
    iy = jnp.clip(iy, 0, y_src.shape[0] - 2)

    x0 = x_src[ix]
    x1 = x_src[ix + 1]
    y0 = y_src[iy]
    y1 = y_src[iy + 1]

    wx = (x_query - x0) / jnp.maximum(x1 - x0, eps)
    wy = (y_query - y0) / jnp.maximum(y1 - y0, eps)

    f00 = field[iy, ix]
    f01 = field[iy, ix + 1]
    f10 = field[iy + 1, ix]
    f11 = field[iy + 1, ix + 1]

    f0 = (1.0 - wx) * f00 + wx * f01
    f1 = (1.0 - wx) * f10 + wx * f11
    return (1.0 - wy) * f0 + wy * f1


def _enforce_periodic_rectangle_boundaries(X, Y, x_left, x_right, y_left, y_right):
    """
    固定矩形边界的法向坐标，并配对周期边界的切向坐标。

    X 在 xi=0/1 上固定，在 eta=0/1 上周期；
    Y 在 eta=0/1 上固定，在 xi=0/1 上周期。
    """
    X = X.at[:, 0].set(x_left)
    X = X.at[:, -1].set(x_right)
    X = X.at[-1, :].set(X[0, :])

    Y = Y.at[0, :].set(y_left)
    Y = Y.at[-1, :].set(y_right)
    Y = Y.at[:, -1].set(Y[:, 0])
    return X, Y


def _mesh_min_jacobian(X, Y):
    """返回每个网格单元中心处 det(d(x,y)/d(xi,eta)) 的最小值。"""
    ny, nx = X.shape
    hxi = jnp.asarray(1.0 / (nx - 1), dtype=X.dtype)
    heta = jnp.asarray(1.0 / (ny - 1), dtype=X.dtype)

    X_xi = 0.5 * ((X[:-1, 1:] - X[:-1, :-1]) + (X[1:, 1:] - X[1:, :-1])) / hxi
    Y_xi = 0.5 * ((Y[:-1, 1:] - Y[:-1, :-1]) + (Y[1:, 1:] - Y[1:, :-1])) / hxi
    X_eta = 0.5 * ((X[1:, :-1] - X[:-1, :-1]) + (X[1:, 1:] - X[:-1, 1:])) / heta
    Y_eta = 0.5 * ((Y[1:, :-1] - Y[:-1, :-1]) + (Y[1:, 1:] - Y[:-1, 1:])) / heta

    return jnp.min(X_xi * Y_eta - X_eta * Y_xi)


def _backtrack_to_positive_mesh(X_old, Y_old, X_candidate, Y_candidate, jacobian_floor, max_backtracks):
    """必要时缩短一次网格更新，防止单元翻转。"""
    one = jnp.asarray(1.0, dtype=X_old.dtype)

    def condition(state):
        iteration, scale = state
        X_trial = X_old + scale * (X_candidate - X_old)
        Y_trial = Y_old + scale * (Y_candidate - Y_old)
        return jnp.logical_and(iteration < max_backtracks, _mesh_min_jacobian(X_trial, Y_trial) <= jacobian_floor)

    def body(state):
        iteration, scale = state
        return iteration + 1, 0.5 * scale

    _, scale = jax.lax.while_loop(condition, body, (jnp.asarray(0, dtype=jnp.int32), one))
    X_result = X_old + scale * (X_candidate - X_old)
    Y_result = Y_old + scale * (Y_candidate - Y_old)

    # 极端情况下回溯次数用尽，宁可保留旧网格，也不输出翻折网格。
    is_valid = _mesh_min_jacobian(X_result, Y_result) > jacobian_floor
    X_result = jnp.where(is_valid, X_result, X_old)
    Y_result = jnp.where(is_valid, Y_result, Y_old)
    return X_result, Y_result


def _mmpde5_relaxation_sweep(X, Y, X_previous_time, Y_previous_time, monitor, mass_coefficient, relaxation):
    """对隐式 MMPDE5 离散系统作一次红黑 SOR 扫描。"""
    ny, nx = X.shape
    inv_hxi2 = jnp.asarray((nx - 1) ** 2, dtype=X.dtype)
    inv_heta2 = jnp.asarray((ny - 1) ** 2, dtype=X.dtype)

    # X: xi 方向为固定法向边界，eta 方向为周期边界。
    X_prev_unique = X_previous_time[:-1, :]
    M_x = monitor[:-1, :]

    M_center_x = M_x[:, 1:-1]
    c_e_x = 0.5 * (M_center_x + M_x[:, 2:]) * inv_hxi2
    c_w_x = 0.5 * (M_center_x + M_x[:, :-2]) * inv_hxi2
    c_n_x = 0.5 * (M_center_x + jnp.roll(M_x, -1, axis=0)[:, 1:-1]) * inv_heta2
    c_s_x = 0.5 * (M_center_x + jnp.roll(M_x, 1, axis=0)[:, 1:-1]) * inv_heta2

    diagonal_x = mass_coefficient + c_e_x + c_w_x + c_n_x + c_s_x
    row_x, col_x = jnp.indices((ny - 1, nx - 2))
    red_x = ((row_x + col_x + 1) % 2) == 0

    # Y: eta 方向为固定法向边界，xi 方向为周期边界。
    Y_prev_unique = Y_previous_time[:, :-1]
    M_y = monitor[:, :-1]

    M_center_y = M_y[1:-1, :]
    c_e_y = 0.5 * (M_center_y + jnp.roll(M_y, -1, axis=1)[1:-1, :]) * inv_hxi2
    c_w_y = 0.5 * (M_center_y + jnp.roll(M_y, 1, axis=1)[1:-1, :]) * inv_hxi2
    c_n_y = 0.5 * (M_center_y + M_y[2:, :]) * inv_heta2
    c_s_y = 0.5 * (M_center_y + M_y[:-2, :]) * inv_heta2

    diagonal_y = mass_coefficient + c_e_y + c_w_y + c_n_y + c_s_y
    row_y, col_y = jnp.indices((ny - 2, nx - 1))
    red_y = ((row_y + 1 + col_y) % 2) == 0

    def update_one_color(X_current, Y_current, use_red):
        X_unique_current = X_current[:-1, :]
        candidate_x = (
            mass_coefficient * X_prev_unique[:, 1:-1]
            + c_e_x * X_unique_current[:, 2:]
            + c_w_x * X_unique_current[:, :-2]
            + c_n_x * jnp.roll(X_unique_current, -1, axis=0)[:, 1:-1]
            + c_s_x * jnp.roll(X_unique_current, 1, axis=0)[:, 1:-1]
        ) / diagonal_x
        X_center = X_unique_current[:, 1:-1]
        X_relaxed = (1.0 - relaxation) * X_center + relaxation * candidate_x
        X_center = jnp.where(red_x == use_red, X_relaxed, X_center)
        X_unique_current = X_unique_current.at[:, 1:-1].set(X_center)
        X_current = jnp.concatenate([X_unique_current, X_unique_current[:1, :]], axis=0)

        Y_unique_current = Y_current[:, :-1]
        candidate_y = (
            mass_coefficient * Y_prev_unique[1:-1, :]
            + c_e_y * jnp.roll(Y_unique_current, -1, axis=1)[1:-1, :]
            + c_w_y * jnp.roll(Y_unique_current, 1, axis=1)[1:-1, :]
            + c_n_y * Y_unique_current[2:, :]
            + c_s_y * Y_unique_current[:-2, :]
        ) / diagonal_y
        Y_center = Y_unique_current[1:-1, :]
        Y_relaxed = (1.0 - relaxation) * Y_center + relaxation * candidate_y
        Y_center = jnp.where(red_y == use_red, Y_relaxed, Y_center)
        Y_unique_current = Y_unique_current.at[1:-1, :].set(Y_center)
        Y_current = jnp.concatenate([Y_unique_current, Y_unique_current[:, :1]], axis=1)
        return X_current, Y_current

    X_new, Y_new = update_one_color(X, Y, True)
    X_new, Y_new = update_one_color(X_new, Y_new, False)
    return X_new, Y_new


def _solve_mmpde5_state(
    monitor_field,
    x_src,
    y_src,
    X_initial,
    Y_initial,
    X_previous_time,
    Y_previous_time,
    mass_coefficient,
    *,
    max_iter,
    tol,
    relaxation,
    jacobian_floor,
    max_backtracks,
):
    """求解一个稳态 Winslow 或一个隐式 MMPDE5 时间步。"""
    x_left, x_right = x_src[0], x_src[-1]
    y_left, y_right = y_src[0], y_src[-1]
    length_scale = jnp.maximum(x_right - x_left, y_right - y_left)
    eps = jnp.asarray(jnp.finfo(X_initial.dtype).eps, dtype=X_initial.dtype)

    X_initial, Y_initial = _enforce_periodic_rectangle_boundaries(
        X_initial, Y_initial, x_left, x_right, y_left, y_right
    )

    def condition(state):
        iteration, _, _, residual = state
        return jnp.logical_and(iteration < max_iter, residual > tol)

    def body(state):
        iteration, X_current, Y_current, _ = state

        # monitor 是物理空间中的场；网格移动后必须在新位置重新采样。
        monitor_nodes = _interp_rectilinear_2d(
            x_src, y_src, monitor_field, X_current, Y_current
        )

        X_candidate, Y_candidate = _mmpde5_relaxation_sweep(
            X_current,
            Y_current,
            X_previous_time,
            Y_previous_time,
            monitor_nodes,
            mass_coefficient,
            relaxation,
        )
        X_candidate, Y_candidate = _enforce_periodic_rectangle_boundaries(
            X_candidate, Y_candidate, x_left, x_right, y_left, y_right
        )
        X_new, Y_new = _backtrack_to_positive_mesh(
            X_current,
            Y_current,
            X_candidate,
            Y_candidate,
            jacobian_floor,
            max_backtracks,
        )

        displacement = jnp.maximum(jnp.max(jnp.abs(X_new - X_current)), jnp.max(jnp.abs(Y_new - Y_current)))
        residual = displacement / jnp.maximum(length_scale, eps)
        return iteration + 1, X_new, Y_new, residual

    initial_state = (
        jnp.asarray(0, dtype=jnp.int32),
        X_initial,
        Y_initial,
        jnp.asarray(jnp.inf, dtype=X_initial.dtype),
    )
    iterations, X_result, Y_result, residual = jax.lax.while_loop(condition, body, initial_state)
    return X_result, Y_result, iterations, residual


def _limit_mesh_motion(X_old, Y_old, X_candidate, Y_candidate, x_src, y_src, max_move_fraction):
    """限制一个物理时间步内的最大节点位移，并保持网格非翻转。"""
    eps = jnp.asarray(jnp.finfo(X_old.dtype).eps, dtype=X_old.dtype)
    max_dx = jnp.max(jnp.abs(X_candidate - X_old))
    max_dy = jnp.max(jnp.abs(Y_candidate - Y_old))
    allowed_x = max_move_fraction * jnp.min(x_src[1:] - x_src[:-1])
    allowed_y = max_move_fraction * jnp.min(y_src[1:] - y_src[:-1])

    scale_x = jnp.minimum(1.0, allowed_x / jnp.maximum(max_dx, eps))
    scale_y = jnp.minimum(1.0, allowed_y / jnp.maximum(max_dy, eps))
    scale = jnp.minimum(scale_x, scale_y)
    return X_old + scale * (X_candidate - X_old), Y_old + scale * (Y_candidate - Y_old)


@partial(
    jax.jit,
    static_argnames=("smooth_steps", "steady_max_iter", "time_max_iter", "max_backtracks"),
)
def _compute_adaptive_mesh_2d_mmpde5_impl(
    T,
    X,
    Y,
    U,
    *,
    beta,
    alpha_floor,
    direction_x,
    direction_y,
    smooth_steps,
    tau_mesh,
    steady_tol,
    steady_max_iter,
    time_tol,
    time_max_iter,
    relaxation,
    jacobian_floor,
    max_move_fraction,
    max_backtracks,
):
    nt, ny, nx = U.shape
    t_grid = T[:, 0, 0]
    x_src = X[:, 0, :]
    y_src = Y[:, :, 0]

    monitor = _build_monitor_field(
        U,
        x_src,
        y_src,
        beta=beta,
        alpha_floor=alpha_floor,
        direction_x=direction_x,
        direction_y=direction_y,
        smooth_steps=smooth_steps,
    )

    xi_grid = jnp.linspace(0.0, 1.0, nx, dtype=X.dtype)
    eta_grid = jnp.linspace(0.0, 1.0, ny, dtype=Y.dtype)

    # t=t0: 先求当前 monitor 对应的稳态二维 Winslow 网格。
    X0, Y0, steady_iterations, steady_residual = _solve_mmpde5_state(
        monitor[0],
        x_src[0],
        y_src[0],
        X[0],
        Y[0],
        X[0],
        Y[0],
        jnp.asarray(0.0, dtype=X.dtype),
        max_iter=steady_max_iter,
        tol=steady_tol,
        relaxation=relaxation,
        jacobian_floor=jacobian_floor,
        max_backtracks=max_backtracks,
    )

    def advance_one_time(carry, data):
        X_previous, Y_previous = carry
        dt, x_line, y_line, monitor_field = data
        eps = jnp.asarray(jnp.finfo(X.dtype).eps, dtype=X.dtype)
        mass_coefficient = tau_mesh / jnp.maximum(dt, eps)

        X_candidate, Y_candidate, iterations, residual = _solve_mmpde5_state(
            monitor_field,
            x_line,
            y_line,
            X_previous,
            Y_previous,
            X_previous,
            Y_previous,
            mass_coefficient,
            max_iter=time_max_iter,
            tol=time_tol,
            relaxation=relaxation,
            jacobian_floor=jacobian_floor,
            max_backtracks=max_backtracks,
        )

        # MMPDE 本身给出时间松弛；额外的位移限制只用于避免离散时间过粗时跨格。
        X_next, Y_next = _limit_mesh_motion(
            X_previous,
            Y_previous,
            X_candidate,
            Y_candidate,
            x_line,
            y_line,
            max_move_fraction,
        )
        X_next, Y_next = _backtrack_to_positive_mesh(
            X_previous,
            Y_previous,
            X_next,
            Y_next,
            jacobian_floor,
            max_backtracks,
        )

        return (X_next, Y_next), (X_next, Y_next, iterations, residual)

    (_, _), (X_tail, Y_tail, time_iterations, time_residuals) = jax.lax.scan(
        advance_one_time,
        (X0, Y0),
        (t_grid[1:] - t_grid[:-1], x_src[1:], y_src[1:], monitor[1:]),
    )

    X_adapt = jnp.concatenate([X0[None, :, :], X_tail], axis=0)
    Y_adapt = jnp.concatenate([Y0[None, :, :], Y_tail], axis=0)

    # 这些是自适应节点 (T, X_adapt, Y_adapt) 对应的精确计算坐标标签，
    # 直接用于训练逆坐标网络 (t,x,y) -> (xi,eta)。
    Xi_mesh = jnp.broadcast_to(xi_grid[None, None, :], (nt, ny, nx))
    Eta_mesh = jnp.broadcast_to(eta_grid[None, :, None], (nt, ny, nx))

    min_jacobian = jnp.min(jax.vmap(_mesh_min_jacobian)(X_adapt, Y_adapt))
    return (
        xi_grid,
        eta_grid,
        Xi_mesh,
        Eta_mesh,
        X_adapt,
        Y_adapt,
        steady_iterations,
        steady_residual,
        time_iterations,
        time_residuals,
        min_jacobian,
    )


def compute_adaptive_mesh_2d_mmpde5(
    T,
    X,
    Y,
    U,
    *,
    beta=0.5,
    alpha_floor=1.0,
    direction_x=1.0,
    direction_y=1.0,
    smooth_steps=4,
    tau_mesh=2.0e-2,
    steady_tol=1.0e-6,
    steady_max_iter=10000,
    time_tol=1.0e-6,
    time_max_iter=2000,
    relaxation=1.4,
    jacobian_floor=1.0e-6,
    max_move_fraction=0.5,
    max_backtracks=12,
):
    """
    构造二维空间、物理时间连续的 MMPDE5 型自适应网格。

    在 t=t0 上先求稳态 Winslow 方程，随后沿物理时间求解

        tau_mesh * Z_t = div_{xi,eta}(M grad_{xi,eta} Z),
        Z = (X, Y),

    的后向 Euler 离散。每个时间步只求二维问题，不组装三维矩阵；
    上一时间层网格直接进入下一层方程，因此不同时间切片并非独立。

    Parameters
    ----------
    T, X, Y, U : (Nt, Ny, Nx)
        原始周期张量积物理网格与解。x、y 数组均应包含重复周期端点。
    tau_mesh : float
        MMPDE5 网格响应时间。越小越接近逐层稳态 Winslow，越大越平滑但
        对移动激波的响应越慢。
    steady_tol, steady_max_iter : float, int
        初始时刻稳态 Winslow 迭代参数。
    time_tol, time_max_iter : float, int
        每个隐式 MMPDE5 时间步的非线性松弛参数。
    relaxation : float
        红黑 SOR 松弛系数，建议位于 (0, 2)。
    jacobian_floor : float
        单元正向 Jacobian 下界。更新会自动回溯以避免翻折。
    max_move_fraction : float
        一个物理时间步内允许的最大节点位移，以原网格最小间距为单位。

    Returns
    -------
    xi_grid, eta_grid : (Nx,), (Ny,)
        均匀计算坐标。
    Xi_mesh, Eta_mesh : (Nt, Ny, Nx)
        自适应物理节点对应的精确计算坐标标签。
    X_adapt, Y_adapt : (Nt, Ny, Nx)
        正向映射 (t,xi,eta) -> (x,y) 的节点值。
    """
    T = jnp.asarray(T)
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)
    U = jnp.asarray(U)

    if not (T.ndim == X.ndim == Y.ndim == U.ndim == 3):
        raise ValueError("T, X, Y and U must have shape (Nt, Ny, Nx).")
    if not (T.shape == X.shape == Y.shape == U.shape):
        raise ValueError("T, X, Y and U must have the same shape.")
    if T.shape[0] < 2 or T.shape[1] < 4 or T.shape[2] < 4:
        raise ValueError("MMPDE5 requires Nt >= 2 and at least three unique periodic nodes in x and y.")
    if beta < 0.0 or alpha_floor <= 0.0:
        raise ValueError("beta must be nonnegative and alpha_floor must be positive.")
    if smooth_steps < 0:
        raise ValueError("smooth_steps must be nonnegative.")
    if tau_mesh <= 0.0:
        raise ValueError("tau_mesh must be positive.")
    if steady_tol <= 0.0 or time_tol <= 0.0:
        raise ValueError("steady_tol and time_tol must be positive.")
    if steady_max_iter <= 0 or time_max_iter <= 0:
        raise ValueError("steady_max_iter and time_max_iter must be positive.")
    if not (0.0 < relaxation < 2.0):
        raise ValueError("relaxation must lie in (0, 2).")
    if jacobian_floor <= 0.0 or max_backtracks < 0:
        raise ValueError("jacobian_floor must be positive and max_backtracks must be nonnegative.")
    if max_move_fraction <= 0.0:
        raise ValueError("max_move_fraction must be positive.")

    outputs = _compute_adaptive_mesh_2d_mmpde5_impl(
        T,
        X,
        Y,
        U,
        beta=beta,
        alpha_floor=alpha_floor,
        direction_x=direction_x,
        direction_y=direction_y,
        smooth_steps=smooth_steps,
        tau_mesh=tau_mesh,
        steady_tol=steady_tol,
        steady_max_iter=steady_max_iter,
        time_tol=time_tol,
        time_max_iter=time_max_iter,
        relaxation=relaxation,
        jacobian_floor=jacobian_floor,
        max_move_fraction=max_move_fraction,
        max_backtracks=max_backtracks,
    )

    (
        xi_grid,
        eta_grid,
        Xi_mesh,
        Eta_mesh,
        X_adapt,
        Y_adapt,
        steady_iterations,
        steady_residual,
        time_iterations,
        time_residuals,
        min_jacobian,
    ) = outputs

    # 同步一次即可取得诊断量；大数组仍保留为 JAX arrays。
    diagnostics = jax.device_get(
        (
            steady_iterations,
            steady_residual,
            jnp.max(time_iterations),
            jnp.max(time_residuals),
            jnp.sum(time_residuals > time_tol),
            min_jacobian,
        )
    )
    print(
        "[MMPDE5] "
        f"steady_iter={int(diagnostics[0])}, steady_residual={float(diagnostics[1]):.3e}, "
        f"max_time_iter={int(diagnostics[2])}, max_time_residual={float(diagnostics[3]):.3e}, "
        f"unconverged_steps={int(diagnostics[4])}, min_jacobian={float(diagnostics[5]):.3e}"
    )

    return xi_grid, eta_grid, Xi_mesh, Eta_mesh, X_adapt, Y_adapt




def create_map_xieta_to_xy(T, xi_grid, eta_grid, X_adapt, Y_adapt):
    """
    构造正向坐标映射

        (t, xi, eta) -> (x, y).

    Parameters
    ----------
    T : (Nt, Ny, Nx), (Nt, Nx), or (Nt,)
        时间坐标。时间方向不移动。

    xi_grid : (Nx,)
        均匀计算坐标 xi。

    eta_grid : (Ny,)
        均匀计算坐标 eta。

    X_adapt, Y_adapt : (Nt, Ny, Nx)
        自适应物理坐标，数组轴顺序必须为
        (time, eta, xi).

    Returns
    -------
    map_xieta_to_xy
        接受可广播的 (t, xi, eta)，返回 (x, y)。
    """
    T = jnp.asarray(T)
    xi_grid = jnp.asarray(xi_grid)
    eta_grid = jnp.asarray(eta_grid)
    X_adapt = jnp.asarray(X_adapt)
    Y_adapt = jnp.asarray(Y_adapt)

    if T.ndim == 3:
        t_grid = T[:, 0, 0]
    elif T.ndim == 2:
        t_grid = T[:, 0]
    elif T.ndim == 1:
        t_grid = T
    else:
        raise ValueError("T must have shape (Nt,), (Nt, Nx), or (Nt, Ny, Nx).")

    expected_shape = (t_grid.shape[0], eta_grid.shape[0], xi_grid.shape[0])

    if X_adapt.shape != expected_shape:
        raise ValueError(f"X_adapt must have shape {expected_shape}, but got {X_adapt.shape}.")

    if Y_adapt.shape != expected_shape:
        raise ValueError(f"Y_adapt must have shape {expected_shape}, but got {Y_adapt.shape}.")

    @jax.jit
    def map_xieta_to_xy(t, xi, eta):
        """
        三线性插值计算

            x = x(t, xi, eta),
            y = y(t, xi, eta).

        t、xi、eta 可以是标量或任意可广播数组。
        """
        t, xi, eta = jnp.broadcast_arrays(t, xi, eta)

        t = jnp.clip(t, t_grid[0], t_grid[-1])
        xi = jnp.clip(xi, xi_grid[0], xi_grid[-1])
        eta = jnp.clip(eta, eta_grid[0], eta_grid[-1])

        # 定位 (t, eta, xi) 所在单元。
        it = jnp.searchsorted(t_grid, t, side="right") - 1
        ie = jnp.searchsorted(eta_grid, eta, side="right") - 1
        ix = jnp.searchsorted(xi_grid, xi, side="right") - 1

        it = jnp.clip(it, 0, t_grid.shape[0] - 2)
        ie = jnp.clip(ie, 0, eta_grid.shape[0] - 2)
        ix = jnp.clip(ix, 0, xi_grid.shape[0] - 2)

        t0 = t_grid[it]
        t1 = t_grid[it + 1]

        eta0 = eta_grid[ie]
        eta1 = eta_grid[ie + 1]

        xi0 = xi_grid[ix]
        xi1 = xi_grid[ix + 1]

        wt = (t - t0) / (t1 - t0)
        we = (eta - eta0) / (eta1 - eta0)
        wx = (xi - xi0) / (xi1 - xi0)

        def interpolate_field(field):
            # t = t0 平面
            f000 = field[it, ie, ix]
            f001 = field[it, ie, ix + 1]
            f010 = field[it, ie + 1, ix]
            f011 = field[it, ie + 1, ix + 1]

            # t = t1 平面
            f100 = field[it + 1, ie, ix]
            f101 = field[it + 1, ie, ix + 1]
            f110 = field[it + 1, ie + 1, ix]
            f111 = field[it + 1, ie + 1, ix + 1]

            # 先沿 xi 插值。
            f00 = (1.0 - wx) * f000 + wx * f001
            f01 = (1.0 - wx) * f010 + wx * f011
            f10 = (1.0 - wx) * f100 + wx * f101
            f11 = (1.0 - wx) * f110 + wx * f111

            # 再沿 eta 插值。
            f0 = (1.0 - we) * f00 + we * f01
            f1 = (1.0 - we) * f10 + we * f11

            # 最后沿时间插值。
            return (1.0 - wt) * f0 + wt * f1

        x = interpolate_field(X_adapt)
        y = interpolate_field(Y_adapt)

        return x, y

    return map_xieta_to_xy


def create_coor_loss_grad_fn(model, lambda_pos=0.1, jac_det_min=1.0e-3):
    """
    为一般二维逆坐标网络构造损失及梯度函数。

    网络映射：
        (t, x, y) -> (xi, eta)

    inputs : (N, 3)
    targets: (N, 2)
    """
    apply_fn = model.apply

    def coor_single(params, txy):
        output = apply_fn(params, txy)
        return jnp.asarray(output).reshape(2)

    # 单点 Jacobian:
    #
    # [[xi_t,  xi_x,  xi_y ],
    #  [eta_t, eta_x, eta_y]]
    coor_jacobian = jax.jacrev(coor_single, argnums=1)
    batch_jacobian = jax.vmap(coor_jacobian, in_axes=(None, 0), out_axes=0)

    def loss_fn(params, inputs, targets):
        coor_pred = jnp.asarray(apply_fn(params, inputs)).reshape(-1, 2)
        coor_true = jnp.asarray(targets).reshape(-1, 2)

        # 坐标拟合损失。
        loss_fit = jnp.mean((coor_pred - coor_true) ** 2)
        jac = batch_jacobian(params, inputs,)  # (N, 2, 3)

        xi_x = jac[:, 0, 1]
        xi_y = jac[:, 0, 2]

        eta_x = jac[:, 1, 1]
        eta_y = jac[:, 1, 2]

        jac_det = xi_x * eta_y - xi_y * eta_x

        # 一般 Winslow 映射允许四个空间偏导均为非零；
        # 这里只约束完整空间 Jacobian 的方向，不施加三角映射限制。
        violation_det = jax.nn.relu(jac_det_min - jac_det)
        loss_pos = jnp.mean(violation_det**2)
        loss = loss_fit + lambda_pos * loss_pos

        return loss

    return jax.jit(jax.value_and_grad(loss_fn))


@partial(jax.jit, static_argnames=['pts'])
def coor_minibatch(key, inputs, targets, pts):
    idx = jax.random.randint(key, shape=(pts,), minval=0, maxval=inputs.shape[0])
    return inputs[idx], targets[idx]


def create_coor_net_update_fn(loss_grad_fn, optimizer):
    @jax.jit
    def update_step(params, opt_state, inputs, targets):
        loss, grads = loss_grad_fn(params, inputs, targets)
        updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)

        return (new_params, new_opt_state, loss)

    return update_step


def IC_Burgers_2D(x, y):
    return jnp.sin(0.5 * jnp.pi * (x + y))


def create_pde_loss_grad_fn(model, w_ic=10.0, w_bc=10.0, jac_floor=1.0e-6, normalize_importance=True):
    """
    PDE 网络:
        U = U(t, x, y, xi, eta)

    对应物理解:
        u(t,x,y) = U(t,x,y,xi(t,x,y),eta(t,x,y))

    边界条件采用周期配对，不使用参考解边界值。
    """
    apply_fn = model.apply

    def get_U(params, inputs):
        return jnp.asarray(apply_fn(params, inputs)).reshape(())

    grad_U = jax.grad(get_U, argnums=1)
    hess_U = jax.hessian(get_U, argnums=1)

    def residual_single(params, data, nu):
        lifted_inputs = data[:5]

        xi_t, xi_x, xi_y = data[5:8]
        xi_xx = data[8]
        xi_yy = data[10]

        eta_t, eta_x, eta_y = data[11:14]
        eta_xx = data[14]
        eta_yy = data[16]

        U = get_U(params, lifted_inputs)
        grad = grad_U(params, lifted_inputs)       # (5,)
        H = hess_U(params, lifted_inputs)          # (5, 5)

        zero = jnp.zeros_like(xi_t)
        one = jnp.ones_like(xi_t)

        # q = (t,x,y,xi,eta)
        q_t = jnp.stack([one, zero, zero, xi_t, eta_t])
        q_x = jnp.stack([zero, one, zero, xi_x, eta_x])
        q_y = jnp.stack([zero, zero, one, xi_y, eta_y])

         # 一阶物理导数
        u = U
        u_t = jnp.dot(grad, q_t)
        u_x = jnp.dot(grad, q_x)
        u_y = jnp.dot(grad, q_y)

        # 二阶物理导数
        u_xx = q_x @ H @ q_x + grad[3] * xi_xx + grad[4] * eta_xx
        u_yy = q_y @ H @ q_y + grad[3] * xi_yy + grad[4] * eta_yy

        # 标准 Laplace 黏性项
        residual = u_t + u * u_x + u * u_y - nu * (u_xx + u_yy)

        return residual

    residual_batch = jax.vmap(residual_single, in_axes=(None, 0, None))

    def loss_fn(params, inputs_pde, inputs_ics, targets_ics, inputs_bcs_lo, inputs_bcs_hi, nu=1.0e-3):
        # -----------------------------------------------------
        # PDE residual
        # -----------------------------------------------------
        residual = residual_batch(params, inputs_pde, nu)

        xi_x = inputs_pde[:, 6]
        xi_y = inputs_pde[:, 7]
        eta_x = inputs_pde[:, 12]
        eta_y = inputs_pde[:, 13]

        # 完整逆映射空间 Jacobian
        jac_det = xi_x * eta_y - xi_y * eta_x

        # 在 (xi,eta) 上均匀采样时，对物理空间积分的修正
        importance = 1.0 / jnp.maximum(jnp.abs(jac_det), jnp.asarray(jac_floor, dtype=inputs_pde.dtype))

        # 归一化只改变 PDE 损失的整体尺度，不改变点之间的相对权重
        if normalize_importance:
            importance = importance / jnp.mean(importance)

        loss_pde = jnp.mean(importance * residual**2)

        # -----------------------------------------------------
        # Initial condition
        # -----------------------------------------------------
        pred_ics = jnp.asarray(apply_fn(params, inputs_ics)).reshape(-1)
        true_ics = jnp.asarray(targets_ics).reshape(-1)
        loss_ics = jnp.mean((pred_ics - true_ics) ** 2)

        # -----------------------------------------------------
        # Periodic boundary condition
        #
        # inputs_bcs_lo:
        #   x=x_min 以及 y=y_min
        #
        # inputs_bcs_hi:
        #   对应的 x=x_max 以及 y=y_max
        # -----------------------------------------------------
        pred_bcs_lo = jnp.asarray(apply_fn(params, inputs_bcs_lo)).reshape(-1)
        pred_bcs_hi = jnp.asarray(apply_fn(params, inputs_bcs_hi)).reshape(-1)

        loss_bcs = jnp.mean((pred_bcs_lo - pred_bcs_hi) ** 2)

        return loss_pde + w_ic * loss_ics + w_bc * loss_bcs

    return jax.jit(jax.value_and_grad(loss_fn))


def create_pde_minibatch_fn(IC_map, map_xieta_to_xy, coor_net, *, t_bounds, x_bounds=(0.0, 4.0), y_bounds=(0.0, 4.0), dtype=jnp.float32):
    """
    Parameters
    ----------
    IC_map:
        (x, y) -> u(t_min, x, y)

    map_xieta_to_xy:
        (t, xi, eta) -> (x, y)

    coor_net:
        单点函数:
            [t, x, y] -> [xi, eta]

        例如:
            coor_net = lambda z: coor_model.apply(coor_params, z)

    pts_bcs:
        每个周期方向的配对点数。
        最终每一侧数组包含 2 * pts_bcs 个点：
            pts_bcs 个 x 周期配对
            pts_bcs 个 y 周期配对
    """
    t_min = jnp.asarray(t_bounds[0], dtype=dtype)
    t_max = jnp.asarray(t_bounds[1], dtype=dtype)

    x_min = jnp.asarray(x_bounds[0], dtype=dtype)
    x_max = jnp.asarray(x_bounds[1], dtype=dtype)

    y_min = jnp.asarray(y_bounds[0], dtype=dtype)
    y_max = jnp.asarray(y_bounds[1], dtype=dtype)

    def coor_single(txy):
        return jnp.asarray(coor_net(txy)).reshape((2,))

    coor_jac_single = jax.jacrev(coor_single)
    coor_hess_single = jax.hessian(coor_single)

    batch_coor = jax.vmap(coor_single)
    batch_coor_jac = jax.vmap(coor_jac_single)
    batch_coor_hess = jax.vmap(coor_hess_single)

    def lift_points(txy):
        """
        [t,x,y] -> [t,x,y,xi,eta]
        """
        xieta = batch_coor(txy)
        return jnp.concatenate([txy, xieta], axis=-1)

    def make_pde_data(txy):
        """
        构造完整的 17 列 PDE 输入。
        """
        xieta = batch_coor(txy)             # (N, 2)
        jac = batch_coor_jac(txy)           # (N, 2, 3)
        hess = batch_coor_hess(txy)         # (N, 2, 3, 3)

        xi_first = jac[:, 0, :]             # xi_t, xi_x, xi_y
        eta_first = jac[:, 1, :]            # eta_t, eta_x, eta_y

        xi_second = jnp.stack([hess[:, 0, 1, 1], hess[:, 0, 1, 2], hess[:, 0, 2, 2]], axis=-1)
        eta_second = jnp.stack([hess[:, 1, 1, 1], hess[:, 1, 1, 2], hess[:, 1, 2, 2]], axis=-1)

        return jnp.concatenate([txy, xieta, xi_first, xi_second, eta_first, eta_second], axis=-1)

    @partial(jax.jit, static_argnames=("pts_pde", "pts_ics", "pts_bcs"))
    def pde_minibatch(key, pts_pde, pts_ics, pts_bcs):
        key_pde, key_ics, key_bcx, key_bcy = jax.random.split(key, 4)

        
        # 1. PDE points
        #    时间在真实时间区间采样；
        #    空间在计算坐标 (xi,eta) 中均匀采样。
        
        sample_pde = jax.random.uniform(key_pde, shape=(pts_pde, 3), dtype=dtype)

        t_pde = t_min + (t_max - t_min) * sample_pde[:, 0]
        xi_seed = sample_pde[:, 1]
        eta_seed = sample_pde[:, 2]

        x_pde, y_pde = map_xieta_to_xy(t_pde, xi_seed, eta_seed)
        txy_pde = jnp.stack([t_pde, x_pde, y_pde], axis=-1)
        inputs_pde = make_pde_data(txy_pde)

        
        # 2. Initial-condition points
        #    初值点在物理空间中均匀采样。
        
        sample_ics = jax.random.uniform(key_ics, shape=(pts_ics, 2), dtype=dtype)

        x_ics = x_min + (x_max - x_min) * sample_ics[:, 0]
        y_ics = y_min + (y_max - y_min) * sample_ics[:, 1]
        t_ics = jnp.full_like(x_ics, t_min)
        txy_ics = jnp.stack([t_ics, x_ics, y_ics], axis=-1)

        inputs_ics = lift_points(txy_ics)
        targets_ics = IC_map(x_ics, y_ics).reshape(-1, 1)

        
        # 3. x 方向周期边界
        #    (t, x_min, y) <-> (t, x_max, y)
        
        sample_bcx = jax.random.uniform(key_bcx, shape=(pts_bcs, 2), dtype=dtype)

        t_bcx = t_min + (t_max - t_min) * sample_bcx[:, 0]
        y_bcx = y_min + (y_max - y_min) * sample_bcx[:, 1]

        x_lo = jnp.full_like(t_bcx, x_min)
        x_hi = jnp.full_like(t_bcx, x_max)

        txy_x_lo = jnp.stack([t_bcx, x_lo, y_bcx], axis=-1)
        txy_x_hi = jnp.stack([t_bcx, x_hi, y_bcx], axis=-1)

        
        # 4. y 方向周期边界
        #    (t, x, y_min) <-> (t, x, y_max)
        
        sample_bcy = jax.random.uniform(key_bcy, shape=(pts_bcs, 2), dtype=dtype)

        t_bcy = t_min + (t_max - t_min) * sample_bcy[:, 0]
        x_bcy = x_min + (x_max - x_min) * sample_bcy[:, 1]

        y_lo = jnp.full_like(t_bcy, y_min)
        y_hi = jnp.full_like(t_bcy, y_max)

        txy_y_lo = jnp.stack([t_bcy, x_bcy, y_lo], axis=-1)
        txy_y_hi = jnp.stack([t_bcy, x_bcy, y_hi], axis=-1)

        # 配对顺序必须保持一致
        inputs_bcs_lo = jnp.concatenate([lift_points(txy_x_lo), lift_points(txy_y_lo)], axis=0)
        inputs_bcs_hi = jnp.concatenate([lift_points(txy_x_hi), lift_points(txy_y_hi)], axis=0)

        return (inputs_pde, inputs_ics, targets_ics, inputs_bcs_lo, inputs_bcs_hi)

    return pde_minibatch


def create_pde_net_update_fn(loss_grad_fn, optimizer, nu):

    @jax.jit
    def update_step(params, opt_state, inputs_pde, inputs_ics, targets_ics, inputs_bcs_lo, inputs_bcs_hi):

        # 1. Compute loss and gradients
        loss, grads = loss_grad_fn(params, inputs_pde, inputs_ics, targets_ics, inputs_bcs_lo, inputs_bcs_hi, nu)

        # 2. Optimizer update
        updates, new_opt_state = optimizer.update(grads, opt_state, params=params)

        # 3. Apply updates
        new_params = optax.apply_updates(params, updates)

        return (new_params, new_opt_state, loss)

    return update_step


def create_identity_maps(xL, xR, yL, yR):
    """
    初始非自适应坐标映射。

    Forward:
        (t, xi, eta) -> (x, y)

    Inverse:
        (t, x, y) -> (xi, eta)

    时间坐标保持不变。
    """
    xL = jnp.asarray(xL)
    xR = jnp.asarray(xR)
    yL = jnp.asarray(yL)
    yR = jnp.asarray(yR)

    length_x = xR - xL
    length_y = yR - yL

    @jax.jit
    def identity_map_xieta_to_xy(t, xi, eta):
        # 支持标量、向量及可广播形状
        _, xi, eta = jnp.broadcast_arrays(t, xi, eta)

        x = xL + length_x * xi
        y = yL + length_y * eta

        return x, y

    @jax.jit
    def identity_coor_net(txy):
        """
        txy[..., :] = [t, x, y]
        return [..., 2] = [xi, eta]
        """
        txy = jnp.asarray(txy)

        x = txy[..., 1]
        y = txy[..., 2]

        xi = (x - xL) / length_x
        eta = (y - yL) / length_y

        return jnp.stack([xi, eta], axis=-1)

    return identity_map_xieta_to_xy, identity_coor_net


def _discontinuity_sum_mask_jax(x, y, discontinuity_sums, atol):
    """标记满足 x + y = c 的点，其中 c 取自 discontinuity_sums。"""
    xy_sum = jnp.asarray(x) + jnp.asarray(y)
    levels = jnp.asarray(discontinuity_sums, dtype=xy_sum.dtype)
    return jnp.any(
        jnp.isclose(
            xy_sum[..., None],
            levels,
            rtol=0.0,
            atol=jnp.asarray(atol, dtype=xy_sum.dtype),
        ),
        axis=-1,
    )


def _discontinuity_sum_mask_numpy(x, y, discontinuity_sums, atol):
    """NumPy 版本的 x + y = c 间断面掩码。"""
    xy_sum = np.asarray(x) + np.asarray(y)
    levels = np.asarray(discontinuity_sums, dtype=xy_sum.dtype)
    return np.any(
        np.isclose(xy_sum[..., None], levels, rtol=0.0, atol=atol),
        axis=-1,
    )


def _replace_masked_error_with_four_neighbor_average(error, mask):
    """
    仅为绘图替换被屏蔽点的误差。

    每个被屏蔽点使用上、下、左、右四个相邻点的原始误差平均值。
    当前问题在 x、y 方向均为周期边界，且网格包含重复周期端点；
    因此边界点的邻点取自另一端最靠近边界的非重复节点。
    """
    error = np.asarray(error)
    mask = np.asarray(mask, dtype=bool)

    if error.ndim != 2 or mask.shape != error.shape:
        raise ValueError("error and mask must be two-dimensional arrays with the same shape.")

    ny, nx = error.shape
    if ny < 3 or nx < 3:
        raise ValueError("At least three grid points in each direction are required for four-neighbor averaging.")

    row_prev = np.arange(ny) - 1
    row_next = np.arange(ny) + 1
    col_prev = np.arange(nx) - 1
    col_next = np.arange(nx) + 1

    # 第 0 个和最后一个节点是周期配对点，跨边界时跳过重复端点。
    row_prev[0] = ny - 2
    row_next[-1] = 1
    col_prev[0] = nx - 2
    col_next[-1] = 1

    neighbor_average = 0.25 * (
        error[row_prev, :]
        + error[row_next, :]
        + error[:, col_prev]
        + error[:, col_next]
    )

    error_for_plot = error.copy()
    error_for_plot[mask] = neighbor_average[mask]
    return error_for_plot


def create_eval_fn(model, discontinuity_sums=(2.0, 6.0), discontinuity_atol=1.0e-6):
    """
    model input:
        [t, x, y, xi, eta]

    model output:
        scalar u

    MSE 和相对 L2 误差均排除满足
        x + y = discontinuity_sums[k]
    的间断面网格点。
    """
    discontinuity_sums = tuple(float(value) for value in discontinuity_sums)
    if discontinuity_atol < 0.0:
        raise ValueError("discontinuity_atol must be non-negative.")

    apply_fn = model.apply

    def predict_single(params, inputs):
        return jnp.asarray(apply_fn(params, inputs)).reshape(())

    predict_batch = jax.vmap(predict_single, in_axes=(None, 0))

    def eval_error(params, inputs, labels):
        predictions = predict_batch(params, inputs).reshape(-1)
        labels = jnp.asarray(labels).reshape(-1)

        diff = predictions - labels

        on_discontinuity = _discontinuity_sum_mask_jax(
            inputs[:, 1],
            inputs[:, 2],
            discontinuity_sums,
            discontinuity_atol,
        )
        valid = jnp.logical_not(on_discontinuity)

        squared_error = jnp.where(valid, diff**2, 0.0)
        squared_label = jnp.where(valid, labels**2, 0.0)
        valid_count = jnp.sum(valid)

        mse = jnp.sum(squared_error) / jnp.maximum(valid_count, 1)

        numerator = jnp.sum(squared_error)
        denominator = jnp.sum(squared_label)

        rl2 = jnp.sqrt(numerator / jnp.maximum(denominator, 1.0e-12))

        return mse, rl2

    return jax.jit(eval_error)


def train_pde_stage(stage_name, pde_params, pde_optimizer, pde_loss_grad_fn, pde_minibatch_fn, eval_fn, test_inputs, test_labels,
    key, nu, max_iters=10000, pts_pde=10000, pts_ics=1000, pts_bcs=1000, max_runtime=10000.0, eval_every=500):
    """
    pts_bcs:
        每个周期方向的配对点数。

        minibatch 最终产生：
            pts_bcs 个 x 周期配对；
            pts_bcs 个 y 周期配对。
    """
    if eval_every <= 0:
        raise ValueError("eval_every must be positive.")

    opt_state = pde_optimizer.init(pde_params)
    update_fn = create_pde_net_update_fn(pde_loss_grad_fn, pde_optimizer, nu=nu)

    # JIT warm-up
    warmup_key = jax.random.PRNGKey(0)
    warmup_dataset = pde_minibatch_fn(warmup_key, pts_pde=pts_pde, pts_ics=pts_ics, pts_bcs=pts_bcs)
    jax.tree_util.tree_map(lambda value: value.block_until_ready(), warmup_dataset)

    _, _, warmup_loss = update_fn(pde_params, opt_state, *warmup_dataset)
    warmup_loss.block_until_ready()
    print(f"[{stage_name}] JIT warm-up finished.")

    runtime = 0.0
    history = {
        "iter": [],
        "loss": [],
        "train_time": [],
        "eval_iter": [],
        "eval_train_time": [],
        "mse": [],
        "rl2": [],
    }

    def record_evaluation(iteration, train_time):
        mse, rl2 = eval_fn(pde_params, test_inputs, test_labels)
        mse_value, rl2_value = jax.device_get((mse, rl2))
        mse_value = float(np.asarray(mse_value))
        rl2_value = float(np.asarray(rl2_value))

        history["eval_iter"].append(iteration)
        history["eval_train_time"].append(train_time)
        history["mse"].append(mse_value)
        history["rl2"].append(rl2_value)
        return mse_value, rl2_value

    mse_value, rl2_value = record_evaluation(
        iteration=0,
        train_time=0.0,
    )
    print(
        f"[{stage_name}] iter=00000, time=0.00s | "
        f"MSE={mse_value:.2e} | RL2={rl2_value:.2e}"
    )

    # Training
    for it in range(1, max_iters + 1):
        if runtime >= max_runtime:
            break

        start = time.perf_counter()

        key, key_batch = jax.random.split(key)
        dataset = pde_minibatch_fn(key_batch, pts_pde=pts_pde, pts_ics=pts_ics, pts_bcs=pts_bcs)
        pde_params, opt_state, loss = update_fn(pde_params, opt_state, *dataset)
        loss_value = float(loss.block_until_ready())

        runtime += time.perf_counter() - start

        history["iter"].append(it)
        history["loss"].append(loss_value)
        history["train_time"].append(runtime)

        # Evaluation
        if it % eval_every == 0:
            mse_value, rl2_value = record_evaluation(
                iteration=it,
                train_time=runtime,
            )

            print(f"[{stage_name}] iter={it:05d}, time={runtime:.2f}s, loss={loss_value:.2e} | MSE={mse_value:.2e} | RL2={rl2_value:.2e}")

    if history["iter"]:
        final_iter = history["iter"][-1]
        if history["eval_iter"][-1] != final_iter:
            mse_value, rl2_value = record_evaluation(
                iteration=final_iter,
                train_time=runtime,
            )
            print(
                f"[{stage_name}] final iter={final_iter:05d}, "
                f"time={runtime:.2f}s | MSE={mse_value:.2e} | "
                f"RL2={rl2_value:.2e}"
            )

    # Convert history to NumPy
    history["iter"] = np.asarray(history["iter"], dtype=np.int32)
    history["loss"] = np.asarray(history["loss"], dtype=np.float64)
    history["train_time"] = np.asarray(history["train_time"], dtype=np.float64)
    history["eval_iter"] = np.asarray(history["eval_iter"], dtype=np.int32)
    history["eval_train_time"] = np.asarray(history["eval_train_time"], dtype=np.float64)
    history["mse"] = np.asarray(history["mse"], dtype=np.float64)
    history["rl2"] = np.asarray(history["rl2"], dtype=np.float64)
    history["runtime"] = float(runtime)

    return pde_params, key, history


def plot_training_history(history, metric="rl2", title="Training history"):
    iters = np.asarray(history["iter"])
    loss = np.asarray(history["loss"])
    eval_iters = np.asarray(history["eval_iter"])

    if metric == "rl2":
        metric_values = np.asarray(history["rl2"])
        metric_label = r"Relative $L^2$ error"
        metric_title = r"Relative $L^2$ error"

    elif metric == "mse":
        metric_values = np.asarray(history["mse"])
        metric_label = "MSE"
        metric_title = "Mean squared error"

    else:
        raise ValueError("metric must be 'rl2' or 'mse'.")

    loss_safe = np.maximum(loss, 1.0e-30)
    metric_safe = np.maximum(metric_values, 1.0e-30)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    axes[0].semilogy(iters, loss_safe, linewidth=1.5, color="tab:red")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.6)

    # Evaluation error
    axes[1].semilogy(eval_iters, metric_safe, marker="o", markersize=4, linewidth=1.5, color="tab:blue", label=metric_label)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(metric_label)
    axes[1].set_title(metric_title)
    axes[1].grid(True, which="both", linestyle=":", alpha=0.6)
    axes[1].legend()

    fig.suptitle(title, fontsize=14)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.show()

    return fig, axes



def generate_adaptive_mesh(
    pde_net,
    pde_params,
    T_grid,
    X_grid,
    Y_grid,
    coor_fn,
    *,
    beta=0.5,
    alpha_floor=1.0,
    direction_x=1.0,
    direction_y=1.0,
    smooth_steps=4,
    tau_mesh=2.0e-2,
    steady_tol=1.0e-6,
    steady_max_iter=10000,
    time_tol=1.0e-6,
    time_max_iter=2000,
    relaxation=1.4,
    jacobian_floor=1.0e-6,
    max_move_fraction=0.5,
    prediction_batch_size=131072,
):
    """
    根据当前二维 Burgers 近似解生成 Winslow/MMPDE5 自适应网格。

    Parameters
    ----------
    T_grid, X_grid, Y_grid : (Nt, Ny, Nx)
        均匀物理网格。要求为空间张量积网格。

    coor_fn :
        单点逆坐标映射
            [t, x, y] -> [xi, eta]

    Returns
    -------
    xi_grid : (Nx,)
    eta_grid : (Ny,)
    Xi_mesh : (Nt, Ny, Nx)
    Eta_mesh : (Nt, Ny, Nx)
        自适应节点对应的计算坐标标签。
    X_adapt : (Nt, Ny, Nx)
    Y_adapt : (Nt, Ny, Nx)
    map_xieta_to_xy :
        连续正向映射 (t,xi,eta) -> (x,y)
    solution_coarse : (Nt, Ny, Nx)
        当前 PDE 网络在均匀物理网格上的预测。
    """
    start = time.perf_counter()

    T_grid = jnp.asarray(T_grid)
    X_grid = jnp.asarray(X_grid)
    Y_grid = jnp.asarray(Y_grid)

    if T_grid.ndim != 3:
        raise ValueError("T_grid, X_grid and Y_grid must have shape (Nt, Ny, Nx).")

    if not (T_grid.shape == X_grid.shape == Y_grid.shape):
        raise ValueError("T_grid, X_grid and Y_grid must have the same shape.")

    if prediction_batch_size <= 0:
        raise ValueError("prediction_batch_size must be positive.")

    # 1. 在均匀物理网格上计算当前 Burgers 解
    t_flat = T_grid.reshape(-1)
    x_flat = X_grid.reshape(-1)
    y_flat = Y_grid.reshape(-1)

    num_points = t_flat.shape[0]
    batch_size = min(prediction_batch_size, num_points)

    batch_coor = jax.vmap(coor_fn)

    @jax.jit
    def predict_chunk(params, t_chunk, x_chunk, y_chunk):
        txy_chunk = jnp.stack([t_chunk, x_chunk, y_chunk], axis=-1)

        xieta_chunk = batch_coor(txy_chunk)
        xieta_chunk = jnp.asarray(xieta_chunk).reshape(-1, 2)

        inputs_chunk = jnp.concatenate([txy_chunk, xieta_chunk], axis=-1)
        u_chunk = pde_net.apply(params, inputs_chunk)

        return jnp.asarray(u_chunk).reshape(-1)

    # 2. Chunked prediction on every grid point
    # =========================================================
    u_chunks = []
    for begin in range(0, num_points, batch_size):
        end = min(begin + batch_size, num_points)
        valid_size = end - begin

        t_chunk = t_flat[begin:end]
        x_chunk = x_flat[begin:end]
        y_chunk = y_flat[begin:end]

        # 最后一块补齐到相同形状，避免 JAX 因形状变化再次编译。
        if valid_size < batch_size:
            pad_size = batch_size - valid_size

            t_chunk = jnp.pad(t_chunk, (0, pad_size), mode="edge")
            x_chunk = jnp.pad(x_chunk, (0, pad_size), mode="edge")
            y_chunk = jnp.pad(y_chunk, (0, pad_size), mode="edge")

        u_chunk = predict_chunk(pde_params, t_chunk, x_chunk, y_chunk)
        u_chunk.block_until_ready()

        u_chunks.append(u_chunk[:valid_size])

    u_flat = jnp.concatenate(u_chunks, axis=0)

    solution_coarse = u_flat.reshape(T_grid.shape)
    solution_coarse.block_until_ready()

    prediction_time = time.perf_counter() - start

    print(f"[Adaptive Mesh] predicted {num_points} points in batches of {batch_size}, time={prediction_time:.2f}s")

    # 2. 初始稳态 Winslow + 物理时间 MMPDE5 网格推进

    (xi_grid, eta_grid, Xi_mesh, Eta_mesh, X_adapt, Y_adapt) = compute_adaptive_mesh_2d_mmpde5(
        T_grid,
        X_grid,
        Y_grid,
        solution_coarse,
        beta=beta,
        alpha_floor=alpha_floor,
        direction_x=direction_x,
        direction_y=direction_y,
        smooth_steps=smooth_steps,
        tau_mesh=tau_mesh,
        steady_tol=steady_tol,
        steady_max_iter=steady_max_iter,
        time_tol=time_tol,
        time_max_iter=time_max_iter,
        relaxation=relaxation,
        jacobian_floor=jacobian_floor,
        max_move_fraction=max_move_fraction,
    )

    X_adapt.block_until_ready()
    Y_adapt.block_until_ready()

    # 3. 连续映射 (t,xi,eta) -> (x,y)
    map_xieta_to_xy = create_map_xieta_to_xy(T_grid, xi_grid, eta_grid, X_adapt, Y_adapt)
    runtime_s = time.perf_counter() - start

    print(f"[Adaptive Mesh] method=mmpde5, tau_mesh={tau_mesh:.3e}, time={runtime_s:.2f}s")

    return (xi_grid, eta_grid, Xi_mesh, Eta_mesh, X_adapt, Y_adapt, map_xieta_to_xy, solution_coarse)


def train_coordinate_net(coor_net, coor_params, coor_optimizer, coor_loss_grad_fn, T_grid, X_grid, Y_grid, Xi_of_xy, Eta_of_xy,
    key, max_iters=5000, batch_size=4096, eval_size=50000, eval_every=500):
    """
    训练二维逆坐标网络：
        [t,x,y] -> [xi,eta]

    ``X_grid, Y_grid`` 可以是自适应物理节点；此时 ``Xi_of_xy,
    Eta_of_xy`` 应为这些节点对应的计算坐标标签。
    """
    T_grid = jnp.asarray(T_grid)
    X_grid = jnp.asarray(X_grid)
    Y_grid = jnp.asarray(Y_grid)
    Xi_of_xy = jnp.asarray(Xi_of_xy)
    Eta_of_xy = jnp.asarray(Eta_of_xy)

    if not (T_grid.shape == X_grid.shape == Y_grid.shape == Xi_of_xy.shape == Eta_of_xy.shape):
        raise ValueError("T_grid, X_grid, Y_grid, Xi_of_xy and Eta_of_xy must have the same shape.")

    # Training dataset
    inputs = jnp.stack([T_grid.reshape(-1), X_grid.reshape(-1), Y_grid.reshape(-1)], axis=-1)
    targets = jnp.stack([Xi_of_xy.reshape(-1), Eta_of_xy.reshape(-1)], axis=-1)

    num_data = inputs.shape[0]

    # Optimizer
    opt_state = coor_optimizer.init(coor_params)
    update_fn = create_coor_net_update_fn(coor_loss_grad_fn, coor_optimizer)

    if eval_size > 0:
        eval_size = min(eval_size, num_data)
        key, key_eval = jax.random.split(key)

        eval_idx = jax.random.randint(key_eval, shape=(eval_size,), minval=0, maxval=num_data)

        inputs_eval = inputs[eval_idx]
        targets_eval = targets[eval_idx]

        @jax.jit
        def evaluate_sample(params):
            predictions = jnp.asarray(coor_net.apply(params, inputs_eval)).reshape(-1, 2)
            squared_error = (predictions - targets_eval) ** 2

            mse = jnp.mean(squared_error)
            component_mse = jnp.mean(squared_error, axis=0)

            return mse, component_mse

    # JIT warm-up
    warmup_key = jax.random.PRNGKey(0)
    inputs_warmup, targets_warmup = coor_minibatch(warmup_key, inputs, targets, pts=batch_size)
    inputs_warmup.block_until_ready()
    targets_warmup.block_until_ready()
    _, _, warmup_loss = update_fn(coor_params, opt_state, inputs_warmup, targets_warmup)
    warmup_loss.block_until_ready()

    print("[Coordinate] JIT warm-up finished.")

    runtime = 0.0
    # Training
    for it in range(1, max_iters + 1):
        start = time.perf_counter()

        key, key_batch = jax.random.split(key)
        inputs_batch, targets_batch = coor_minibatch(key_batch, inputs, targets, pts=batch_size)
        coor_params, opt_state, loss = update_fn(coor_params, opt_state, inputs_batch, targets_batch)
        loss_value = float(loss.block_until_ready())
        runtime += time.perf_counter() - start

        if it % eval_every == 0:
            if eval_size > 0:
                mse, component_mse = evaluate_sample(coor_params)
                mse_value, component_mse_value = jax.device_get((mse, component_mse))
                mse_value = float(mse_value)
                component_mse_value = np.asarray(component_mse_value, dtype=np.float64)

                print(f"[Coordinate] iter={it:05d}, time={runtime:.2f}s, batch loss={loss_value:.2e}, full mse={mse_value:.2e}, mse(xi,eta)=({component_mse_value[0]:.2e}, {component_mse_value[1]:.2e})")

            else:
                print(f"[Coordinate] iter={it:05d}, time={runtime:.2f}s, batch loss={loss_value:.2e}")

    return coor_params, key


def create_coor_fn(coor_net, coor_params):
    """
    返回固定参数的单点坐标函数：
        [t,x,y] -> [xi,eta]
    """
    def coor_fn(txy):
        return jnp.asarray(coor_net.apply(coor_params, txy)).reshape((2,))

    return coor_fn


def create_test_inputs(t_test, x_test, y_test, coor_fn):
    """
    构造 PDE 网络测试输入：
        [t,x,y,xi,eta]
    """
    t_test = jnp.asarray(t_test)
    x_test = jnp.asarray(x_test)
    y_test = jnp.asarray(y_test)

    if not (t_test.shape == x_test.shape == y_test.shape):
        raise ValueError("t_test, x_test and y_test must have the same shape.")

    t_flat = t_test.reshape(-1)
    x_flat = x_test.reshape(-1)
    y_flat = y_test.reshape(-1)

    # (N_test, 3)
    txy = jnp.stack([t_flat, x_flat, y_flat], axis=-1)

    # (N_test, 2)
    xieta_test = jax.vmap(coor_fn)(txy)
    xieta_test = jnp.asarray(xieta_test).reshape(-1, 2)

    # (N_test, 5): [t,x,y,xi,eta]
    return jnp.concatenate([txy, xieta_test], axis=-1)


def plot_evaluation(
    params,
    model,
    t_test,
    x_test,
    y_test,
    labels_test,
    coor_fn,
    cmap="jet",
    error_cmap="jet",
    discontinuity_sums=(2.0, 6.0),
    discontinuity_atol=1.0e-6,
):
    """
    评估二维 Burgers 标量解，并绘制最终时刻的：
        1. 参考解
        2. 预测解
        3. 绝对误差
    Parameters
    ----------
    t_test, x_test, y_test:
        完整张量积测试网格上的坐标，可以是展开数组，
        也可以是相同形状的 (Nt, Ny, Nx) 数组。

    labels_test:
        标量参考解，可以为 (Nt, Ny, Nx)、(N,1) 或 (N,)。

    coor_fn:
        单点坐标映射
            [t,x,y] -> [xi,eta]

    Notes
    -----
    误差指标排除 x+y=2 和 x+y=6 上的点。参考解和预测解保持原值；
    仅在误差图中用上、下、左、右四邻点的平均误差替换间断面点。
    """
    discontinuity_sums = tuple(float(value) for value in discontinuity_sums)
    if discontinuity_atol < 0.0:
        raise ValueError("discontinuity_atol must be non-negative.")

    apply_fn = model.apply

    # 1. Prepare test inputs
    t_test_jax = jnp.asarray(t_test).reshape(-1)
    x_test_jax = jnp.asarray(x_test).reshape(-1)
    y_test_jax = jnp.asarray(y_test).reshape(-1)

    if not (t_test_jax.shape == x_test_jax.shape == y_test_jax.shape):
        raise ValueError("t_test, x_test and y_test must contain the same number of points.")

    txy_test = jnp.stack([t_test_jax, x_test_jax, y_test_jax], axis=-1)
    batch_coor = jax.vmap(coor_fn)

    @jax.jit
    def predict_fn(params, txy):
        # (N,2): [xi,eta]
        xieta = batch_coor(txy)
        xieta = jnp.asarray(xieta).reshape(-1, 2)

        # (N,5): [t,x,y,xi,eta]
        inputs = jnp.concatenate([txy, xieta], axis=-1)
        predictions = apply_fn(params, inputs)

        return jnp.asarray(predictions).reshape(-1)

    predictions = predict_fn(params, txy_test)
    predictions.block_until_ready()


    # 2. Convert to NumPy

    predictions_np = np.asarray(predictions).reshape(-1)
    labels_np = np.asarray(labels_test).reshape(-1)

    if predictions_np.shape != labels_np.shape:
        raise ValueError("Predictions and labels_test must contain the same number of scalar values.")

    t_np = np.asarray(t_test_jax).reshape(-1)
    x_np = np.asarray(x_test_jax).reshape(-1)
    y_np = np.asarray(y_test_jax).reshape(-1)


    # 3. Reconstruct structured (t,y,x) grid
    #
    # np.lexsort 的最后一个键是主排序键，因此排序顺序为：
    #     t -> y -> x

    sort_idx = np.lexsort((x_np, y_np, t_np))

    t_sorted = t_np[sort_idx]
    x_sorted = x_np[sort_idx]
    y_sorted = y_np[sort_idx]

    pred_sorted = predictions_np[sort_idx]
    true_sorted = labels_np[sort_idx]

    t_unique = np.unique(t_sorted)
    x_unique = np.unique(x_sorted)
    y_unique = np.unique(y_sorted)

    nt_test = len(t_unique)
    nx_test = len(x_unique)
    ny_test = len(y_unique)

    expected_size = nt_test * ny_test * nx_test

    if expected_size != len(t_sorted):
        raise ValueError("The supplied test points do not form a complete rectangular (t,y,x) tensor-product grid.")

    # 进一步检查是否真的包含每个张量积节点，
    # 防止存在重复点和缺失点但总数量恰好相同。
    T_expected, Y_expected, X_expected = np.meshgrid(t_unique, y_unique, x_unique, indexing="ij")

    if not (np.allclose(t_sorted, T_expected.reshape(-1)) and np.allclose(y_sorted, Y_expected.reshape(-1)) and np.allclose(x_sorted, X_expected.reshape(-1))):
        raise ValueError("The supplied test coordinates contain missing or duplicated tensor-product grid points.")

    print(f"Plotting test grid: nt={nt_test}, ny={ny_test}, nx={nx_test}")

    # (Nt,Ny,Nx)
    pred_grid = pred_sorted.reshape(nt_test, ny_test, nx_test)
    true_grid = true_sorted.reshape(nt_test, ny_test, nx_test)

    # 4. Global space-time errors. 间断面点不参与任何误差指标。

    on_discontinuity = _discontinuity_sum_mask_numpy(
        X_expected,
        Y_expected,
        discontinuity_sums,
        discontinuity_atol,
    )
    valid = np.logical_not(on_discontinuity)
    if not np.any(valid):
        raise ValueError("No off-discontinuity test points remain for error evaluation.")

    diff = pred_grid - true_grid
    squared_error = diff**2
    mse = np.mean(squared_error[valid])
    numerator = np.sum(squared_error[valid])
    denominator = np.sum((true_grid**2)[valid])
    rl2 = np.sqrt(numerator / max(denominator, 1.0e-12))

    # 5. Final-time errors
    tend = t_unique[-1]
    true_final = true_grid[-1]
    pred_final = pred_grid[-1]
    raw_error_final = np.abs(pred_final - true_final)
    final_on_discontinuity = on_discontinuity[-1]
    final_valid = np.logical_not(final_on_discontinuity)
    if not np.any(final_valid):
        raise ValueError("No off-discontinuity final-time points remain for error evaluation.")

    final_squared_error = (pred_final - true_final) ** 2
    final_mse = np.mean(final_squared_error[final_valid])
    final_numerator = np.sum(final_squared_error[final_valid])
    final_denominator = np.sum((true_final**2)[final_valid])
    final_rl2 = np.sqrt(final_numerator / max(final_denominator, 1.0e-12))

    # 只修改误差图使用的数据；true_final 和 pred_final 始终保持原值。
    error_final = _replace_masked_error_with_four_neighbor_average(
        raw_error_final,
        final_on_discontinuity,
    )

    print(
        "Evaluation errors (excluding x+y=2 and x+y=6):\n"
        f"  Full space-time: "
        f"MSE={mse:.6e}, RL2={rl2:.6e}\n"
        f"  Final time     : "
        f"MSE={final_mse:.6e}, RL2={final_rl2:.6e}\n"
        f"  Excluded points: full={np.count_nonzero(on_discontinuity)}, "
        f"final={np.count_nonzero(final_on_discontinuity)}"
    )


    # 6. Shared levels for reference and prediction

    solution_min = min(np.nanmin(true_final), np.nanmin(pred_final))
    solution_max = max(np.nanmax(true_final), np.nanmax(pred_final))

    if solution_max <= solution_min:
        solution_max = solution_min + 1.0e-12

    solution_levels = np.linspace(solution_min, solution_max, 101)
    error_max = np.nanmax(error_final)

    if error_max <= 0.0:
        error_max = 1.0e-12

    error_levels = np.linspace(0.0, error_max, 101)

    # 7. Plot final-time fields

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)

    contour_true = axes[0].contourf(x_unique, y_unique, true_final, levels=solution_levels, cmap=cmap, extend="both")
    fig.colorbar(contour_true, ax=axes[0], label=r"$u$")
    axes[0].set_title(rf"Reference solution at $t={tend:.4g}$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")
    axes[0].set_aspect("equal", adjustable="box")

    contour_pred = axes[1].contourf(x_unique, y_unique, pred_final, levels=solution_levels, cmap=cmap, extend="both")
    fig.colorbar(contour_pred, ax=axes[1], label=r"$u$")
    axes[1].set_title(rf"Predicted solution at $t={tend:.4g}$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$y$")
    axes[1].set_aspect("equal", adjustable="box")

    contour_error = axes[2].contourf(x_unique, y_unique, error_final, levels=error_levels, cmap=error_cmap, extend="max")
    fig.colorbar(contour_error, ax=axes[2], label=r"$|u_{\mathrm{pred}}-u_{\mathrm{ref}}|$")
    axes[2].set_title(rf"Absolute error at $t={tend:.4g}$, $\mathrm{{RL2}}_{{\rm excl.}}={final_rl2:.2e}$")
    axes[2].set_xlabel(r"$x$")
    axes[2].set_ylabel(r"$y$")
    axes[2].set_aspect("equal", adjustable="box")

    plt.show()
    return mse, rl2


def plot_adaptive_mesh(T_grid, X_adapt, Y_adapt, time_index=-1, stride=4):
    """绘制一个物理时间层上的二维自适应网格。"""
    T_grid = np.asarray(T_grid)
    X_adapt = np.asarray(X_adapt)
    Y_adapt = np.asarray(Y_adapt)

    if not (T_grid.shape == X_adapt.shape == Y_adapt.shape):
        raise ValueError("T_grid, X_adapt and Y_adapt must have the same shape.")
    if T_grid.ndim != 3:
        raise ValueError("T_grid, X_adapt and Y_adapt must have shape (Nt, Ny, Nx).")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    nt, ny, nx = X_adapt.shape
    time_index = time_index % nt
    x_mesh = X_adapt[time_index]
    y_mesh = Y_adapt[time_index]
    t_value = T_grid[time_index, 0, 0]

    row_indices = list(range(0, ny, stride))
    col_indices = list(range(0, nx, stride))
    if row_indices[-1] != ny - 1:
        row_indices.append(ny - 1)
    if col_indices[-1] != nx - 1:
        col_indices.append(nx - 1)

    fig, ax = plt.subplots(figsize=(6.5, 6.2), constrained_layout=True)
    for row in row_indices:
        ax.plot(x_mesh[row, :], y_mesh[row, :], color="tab:blue", linewidth=0.55)
    for col in col_indices:
        ax.plot(x_mesh[:, col], y_mesh[:, col], color="tab:blue", linewidth=0.55)

    ax.set_title(rf"MMPDE5 adaptive mesh at $t={t_value:.4g}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    plt.show()
    return fig, ax



def plot_coordinate_mapping(T_grid, X_grid, Y_grid, coor_fn, cmap="jet"):
    """
    绘制最终时刻坐标网络预测的：

        xi(T,x,y), eta(T,x,y)

    coor_fn:
        单点映射 [t,x,y] -> [xi,eta]
    """
    T_grid = jnp.asarray(T_grid)
    X_grid = jnp.asarray(X_grid)
    Y_grid = jnp.asarray(Y_grid)

    if T_grid.ndim != 3:
        raise ValueError("T_grid, X_grid and Y_grid must have shape (Nt, Ny, Nx).")

    if not (T_grid.shape == X_grid.shape == Y_grid.shape):
        raise ValueError("T_grid, X_grid and Y_grid must have the same shape.")

    # 1. Final-time physical grid
    T_final = T_grid[-1]
    X_final = X_grid[-1]
    Y_final = Y_grid[-1]

    ny, nx = X_final.shape
    t_end = float(np.asarray(T_final[0, 0]))

    txy_final = jnp.stack([T_final.reshape(-1), X_final.reshape(-1), Y_final.reshape(-1)], axis=-1)

    # 2. Evaluate coordinate network
    batch_coor = jax.jit(jax.vmap(coor_fn))

    xieta_pred = batch_coor(txy_final)
    xieta_pred.block_until_ready()

    xieta_pred = np.asarray(xieta_pred).reshape(ny, nx, 2)

    xi_final = xieta_pred[:, :, 0]
    eta_final = xieta_pred[:, :, 1]

    X_final_np = np.asarray(X_final)
    Y_final_np = np.asarray(Y_final)

    print(
        f"Coordinate ranges at t={t_end:.6g}:\n"
        f"  xi : [{np.min(xi_final):.6e}, {np.max(xi_final):.6e}]\n"
        f"  eta: [{np.min(eta_final):.6e}, {np.max(eta_final):.6e}]"
    )

    # 坐标理论范围为 [0,1]
    filled_levels = np.linspace(0.0, 1.0, 101)
    line_levels = np.linspace(0.0, 1.0, 11)

    # 3. Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

    # xi(T,x,y)
    contour_xi = axes[0].contourf(X_final_np, Y_final_np, xi_final, levels=filled_levels, cmap=cmap, extend="both")
    axes[0].contour(X_final_np, Y_final_np, xi_final, levels=line_levels, colors="black", linewidths=0.45, alpha=0.65)
    fig.colorbar(contour_xi, ax=axes[0], label=r"$\xi_\theta(T,x,y)$")
    axes[0].set_title(rf"$\xi_\theta(T,x,y)$ at $T={t_end:.4g}$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")
    axes[0].set_aspect("equal", adjustable="box")

    # eta(T,x,y)
    contour_eta = axes[1].contourf(X_final_np, Y_final_np, eta_final, levels=filled_levels, cmap=cmap, extend="both")
    axes[1].contour(X_final_np, Y_final_np, eta_final, levels=line_levels, colors="black", linewidths=0.45, alpha=0.65)
    fig.colorbar(contour_eta, ax=axes[1], label=r"$\eta_\theta(T,x,y)$")
    axes[1].set_title(rf"$\eta_\theta(T,x,y)$ at $T={t_end:.4g}$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$y$")
    axes[1].set_aspect("equal", adjustable="box")

    plt.show()

    return fig, axes

def save_training_results(
    pde_params_by_stage,
    coor_params_by_stage,
    pde_histories_by_stage,
    checkpoint_path="talpinn_burgers2d01.msgpack",
    history_path=None,
    metadata=None,
):
    """Save both PDE stages, the coordinate network, and PDE histories."""
    pde_stage_names = ("stage1", "stage2")
    coor_stage_names = ("stage1",)
    history_keys = (
        "iter",
        "loss",
        "train_time",
        "eval_iter",
        "eval_train_time",
        "mse",
        "rl2",
        "runtime",
    )

    if set(pde_params_by_stage) != set(pde_stage_names):
        raise ValueError(f"pde_params_by_stage keys must be {pde_stage_names}")
    if set(coor_params_by_stage) != set(coor_stage_names):
        raise ValueError(f"coor_params_by_stage keys must be {coor_stage_names}")
    if set(pde_histories_by_stage) != set(pde_stage_names):
        raise ValueError(
            f"pde_histories_by_stage keys must be {pde_stage_names}"
        )

    saved_histories = {}
    for stage_name in pde_stage_names:
        history = pde_histories_by_stage[stage_name]
        missing_keys = [name for name in history_keys if name not in history]
        if missing_keys:
            raise ValueError(
                f"{stage_name} history is missing keys: {missing_keys}"
            )
        saved_histories[stage_name] = {
            name: np.asarray(history[name]) for name in history_keys
        }

    checkpoint = {
        "format_version": 1,
        "method": "talpinn",
        "pde_params": jax.device_get({
            name: pde_params_by_stage[name] for name in pde_stage_names
        }),
        "coor_params": jax.device_get({
            name: coor_params_by_stage[name] for name in coor_stage_names
        }),
        "pde_histories": saved_histories,
        "metadata": {} if metadata is None else dict(metadata),
    }

    checkpoint_path = os.path.abspath(checkpoint_path)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint_tmp = checkpoint_path + ".tmp"
    with open(checkpoint_tmp, "wb") as file:
        file.write(serialization.to_bytes(checkpoint))
    os.replace(checkpoint_tmp, checkpoint_path)

    if history_path is None:
        history_path = os.path.splitext(checkpoint_path)[0] + "_history.npz"
    history_path = os.path.abspath(history_path)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    history_arrays = {
        f"{stage_name}_{name}": value
        for stage_name, history in saved_histories.items()
        for name, value in history.items()
    }
    history_tmp = history_path + ".tmp"
    with open(history_tmp, "wb") as file:
        np.savez_compressed(file, **history_arrays)
    os.replace(history_tmp, history_path)

    print(f"Training checkpoint saved to: {checkpoint_path}")
    print(f"Training histories saved to: {history_path}")


def load_training_results(checkpoint_path):
    """Load a complete two-stage TAL-PINN checkpoint."""
    with open(checkpoint_path, "rb") as file:
        checkpoint = serialization.msgpack_restore(file.read())

    if checkpoint.get("format_version") != 1:
        raise ValueError(
            f"Unsupported checkpoint format: "
            f"{checkpoint.get('format_version')}"
        )

    checkpoint["pde_params"] = jax.tree_util.tree_map(
        jnp.asarray,
        checkpoint["pde_params"],
    )
    checkpoint["coor_params"] = jax.tree_util.tree_map(
        jnp.asarray,
        checkpoint["coor_params"],
    )
    print(f"Training checkpoint loaded from: {checkpoint_path}")
    return checkpoint


def load_training_histories(history_path):
    """Load both scalar Burgers PDE histories from the NumPy archive."""
    stage_names = ("stage1", "stage2")
    history_keys = (
        "iter",
        "loss",
        "train_time",
        "eval_iter",
        "eval_train_time",
        "mse",
        "rl2",
        "runtime",
    )
    with np.load(history_path, allow_pickle=False) as data:
        histories = {
            stage_name: {
                name: np.array(data[f"{stage_name}_{name}"])
                for name in history_keys
            }
            for stage_name in stage_names
        }

    print(f"Training histories loaded from: {history_path}")
    return histories


def main():

    # 1. Data
    data_path = "Burgers_2D_01.mat"
    data = scipy.io.loadmat(data_path)

    # U 已经按照 (Nt, Ny, Nx) 保存
    U_data = jnp.asarray(data["U"])

    Nt, Ny, Nx = U_data.shape

    # MATLAB 中是一行向量，reshape(-1) 转成一维
    t = jnp.asarray(data["t"]).reshape(-1)
    x = jnp.asarray(data["x"]).reshape(-1)
    y = jnp.asarray(data["y"]).reshape(-1)

    assert t.shape[0] == Nt
    assert x.shape[0] == Nx
    assert y.shape[0] == Ny

    # 构造 (Nt, Ny, Nx) 坐标张量
    # broadcast_to 在表达层面不需要先存储三份完整坐标数据
    T_data = jnp.broadcast_to(t[:, None, None], (Nt, Ny, Nx))
    X_data = jnp.broadcast_to(x[None, None, :], (Nt, Ny, Nx))
    Y_data = jnp.broadcast_to(y[None, :, None], (Nt, Ny, Nx))

    t0 = float(T_data[0, 0, 0])
    tend = float(T_data[-1, 0, 0])

    xL = float(X_data[0, 0, 0])
    xR = float(X_data[0, 0, -1])

    yL = float(Y_data[0, 0, 0])
    yR = float(Y_data[0, -1, 0])

    trange = (t0, tend)
    xrange = (xL, xR)
    yrange = (yL, yR)

    T_grid = T_data
    X_grid = X_data
    Y_grid = Y_data

    T_test = T_data[-1]
    X_test = X_data[-1]
    Y_test = Y_data[-1]
    test_labels = U_data[-1].reshape(-1)

    print(f"Data grid: Nt={Nt}, Ny={Ny}, Nx={Nx}")
    print(f"Time domain: {trange}")
    print(f"x domain: {xrange}")
    print(f"y domain: {yrange}")
    print(f"Test snapshot: t={tend:.6g}")

    # 2. Random keys and models
    seed=42
    key = jax.random.PRNGKey(seed)
    key_pde_init, key_coor_init, key_train = jax.random.split(key, 3)

    pde_net = FNN_fourier(layer_sizes=[40, 40, 1], activation=nn.silu)
    pde_params = pde_net.init(key_pde_init, jnp.ones((5,), dtype=T_grid.dtype))

    coor_net = FNN_fourier(layer_sizes=[40, 40, 2], activation=nn.silu)
    coor_params = coor_net.init(key_coor_init, jnp.ones((3,), dtype=T_grid.dtype))

    # 3. Optimizers and losses
    pde_optimizer = soap(learning_rate=3.0e-3, b1=0.95, b2=0.95, weight_decay=0.01, precondition_frequency=10, precondition_1d=False)
    coor_optimizer = soap(learning_rate=3.0e-3, b1=0.95, b2=0.95, weight_decay=0.01, precondition_frequency=10, precondition_1d=False)

    pde_loss_grad_fn = create_pde_loss_grad_fn( pde_net, w_ic=10.0, w_bc=10.0, jac_floor=1.0e-6, normalize_importance=True)
    coor_loss_grad_fn = create_coor_loss_grad_fn(
        coor_net,
        lambda_pos=0.1,
        jac_det_min=1.0e-3,
    )
    eval_fn = create_eval_fn(pde_net)

    identity_map_xieta_to_xy, identity_coor_fn = create_identity_maps(xL, xR, yL, yR)

    # 4. Shared helpers
    def make_minibatch(map_xieta_to_xy, coor_fn):
        return create_pde_minibatch_fn(IC_map=IC_Burgers_2D, map_xieta_to_xy=map_xieta_to_xy, coor_net=coor_fn, t_bounds=trange, x_bounds=xrange, y_bounds=yrange, dtype=T_grid.dtype)


    #stage1
    test_inputs_stage1 = create_test_inputs(T_test, X_test, Y_test, identity_coor_fn)
    minibatch_stage1 = make_minibatch(identity_map_xieta_to_xy, identity_coor_fn)

    pde_params, key_train, history_stage1 = train_pde_stage(stage_name="Stage 1", 
        pde_params=pde_params, pde_optimizer=pde_optimizer, pde_loss_grad_fn=pde_loss_grad_fn, pde_minibatch_fn=minibatch_stage1,
        eval_fn=eval_fn, test_inputs=test_inputs_stage1, test_labels=test_labels,
        key=key_train, nu=1.0e-2, max_iters=4000, pts_pde=50000, pts_ics=5000, pts_bcs=5000, max_runtime=10000.0, eval_every=100)
    pde_params_stage1 = pde_params

    plot_training_history(history_stage1, metric="rl2", title=f"Stage 1 training history")
    plot_evaluation(params=pde_params, model=pde_net, t_test=T_test, x_test=X_test, y_test=Y_test, labels_test=test_labels, coor_fn=identity_coor_fn)

    # Generate first adaptive mesh
    xi_grid, eta_grid, Xi_mesh, Eta_mesh, X_adapt, Y_adapt, map_xieta_to_xy, solution_stage1 = generate_adaptive_mesh(
        pde_net,
        pde_params,
        T_grid,
        X_grid,
        Y_grid,
        identity_coor_fn,
        beta=0.5,
        alpha_floor=1.0,
        direction_x=1.0,
        direction_y=1.0,
        smooth_steps=4,
        tau_mesh=2.0e-2,
        steady_tol=1.0e-6,
        steady_max_iter=10000,
        time_tol=1.0e-6,
        time_max_iter=2000,
        relaxation=1.4,
        jacobian_floor=1.0e-6,
        max_move_fraction=0.5,
    )
    plot_adaptive_mesh(T_grid, X_adapt, Y_adapt, time_index=-1, stride=4)

    # Train first coordinate network
    coor_params, key_train = train_coordinate_net( coor_net=coor_net, coor_params=coor_params, coor_optimizer=coor_optimizer, coor_loss_grad_fn=coor_loss_grad_fn,
        T_grid=T_grid, X_grid=X_adapt, Y_grid=Y_adapt, Xi_of_xy=Xi_mesh, Eta_of_xy=Eta_mesh,
        key=key_train, max_iters=10000, batch_size=10000, eval_every=100)
    coor_params_stage1 = coor_params

    trained_coor_fn = create_coor_fn(coor_net, coor_params)
    plot_coordinate_mapping(T_grid, X_grid, Y_grid, trained_coor_fn)

    #stage2
    test_inputs_stage2 = create_test_inputs(T_test, X_test, Y_test, trained_coor_fn)
    minibatch_stage2 = make_minibatch(map_xieta_to_xy, trained_coor_fn)

    pde_params, key_train, history_stage2 = train_pde_stage(stage_name="Stage 2", 
        pde_params=pde_params, pde_optimizer=pde_optimizer, pde_loss_grad_fn=pde_loss_grad_fn, pde_minibatch_fn=minibatch_stage2,
        eval_fn=eval_fn, test_inputs=test_inputs_stage2, test_labels=test_labels,
        key=key_train, nu=1.0e-3, max_iters=10000, pts_pde=50000, pts_ics=5000, pts_bcs=5000, max_runtime=10000.0, eval_every=100)
    pde_params_stage2 = pde_params

    save_training_results(
        pde_params_by_stage={
            "stage1": pde_params_stage1,
            "stage2": pde_params_stage2,
        },
        coor_params_by_stage={
            "stage1": coor_params_stage1,
        },
        pde_histories_by_stage={
            "stage1": history_stage1,
            "stage2": history_stage2,
        },
        checkpoint_path="talpinn_burgers2d01.msgpack",
        metadata={
            "data_path": data_path,
            "seed": seed,
            "test_time": tend,
        },
    )

    plot_training_history(history_stage2, metric="rl2", title=f"Stage 2 training history")
    plot_evaluation(params=pde_params, model=pde_net, t_test=T_test, x_test=X_test, y_test=Y_test, labels_test=test_labels, coor_fn=trained_coor_fn)


if __name__ == "__main__":
    main()
