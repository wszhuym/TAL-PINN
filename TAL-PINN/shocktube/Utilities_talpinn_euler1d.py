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

class Euler_net(nn.Module):
    n_nodes: int = 40
    fourier_dim: int = 40
    activation: Callable = nn.silu

    def setup(self):
        # Fourier feature layer
        kinit_fourier = jax.nn.initializers.normal(stddev=1.0)
        self.feature_map = nn.Dense(self.fourier_dim, kernel_init=kinit_fourier)

        # Shared trunk
        self.trunk = FNN(layer_sizes=[self.n_nodes, self.n_nodes], activation=self.activation)

        # Three output branches
        branch_sizes = [self.n_nodes, self.n_nodes, 1]
        self.branch_rho = FNN(layer_sizes=branch_sizes, activation=self.activation, out_bias=False)
        self.branch_u = FNN(layer_sizes=branch_sizes, activation=self.activation, out_bias=False)
        self.branch_p = FNN(layer_sizes=branch_sizes, activation=self.activation, out_bias=False)

    def __call__(self, inputs):
        # Fourier features
        features = self.feature_map(inputs)
        features = jnp.sin(2.0 * jnp.pi * features)

        # Shared representation
        hidden = self.trunk(features)

        # Three branches
        rho_raw = self.branch_rho(hidden)[..., 0]
        u_raw = self.branch_u(hidden)[..., 0]
        p_raw = self.branch_p(hidden)[..., 0]

        # Physical constraints
        rho = jax.nn.softplus(rho_raw) + 1e-5
        u = u_raw
        p = jax.nn.softplus(p_raw) + 1e-5

        return rho, u, p




@jax.jit
def compute_adaptive_mesh_1d_integral(T, X, U):
    # Gradient on the original physical mesh
    dX = X[:, 1:] - X[:, :-1]
    dU = (U[:, 1:] - U[:, :-1]) / dX
    div   = jnp.abs(dU) - dU

    alpha = (0.5 * (div[:, :-1] + div[:, 1:]) * (X[:, 1:-1] - X[:, :-2]))
    alpha = jnp.sum(alpha, axis=1)
    alpha = jnp.maximum(alpha, 1.0)
    alpha = alpha.reshape(-1, 1)

    # Monitor function
    M = (1.0+0.5 * div / alpha)

    # Smooth monitor function
    def smooth_step(_, M):
        return M.at[:, 1:-1].set(0.25 * (M[:, :-2] + 2.0 * M[:, 1:-1] + M[:, 2:]))
    M = jax.lax.fori_loop(0, 4, smooth_step, M)

    # Construct xi(t_i, x_j)
    increments = M * dX
    Xi_of_x = jnp.concatenate([jnp.zeros((X.shape[0], 1), dtype=X.dtype), jnp.cumsum(increments, axis=1)], axis=1)

    # Normalize to [0, 1]
    Xi_of_x = (Xi_of_x / Xi_of_x[:, -1:])

    # Uniform computational grid
    xi_grid = jnp.linspace(0.0, 1.0, X.shape[1],  dtype=X.dtype)

    # Invert xi(t_i, x) -> x(t_i, xi)
    def invert_one_slice(xi_of_x, x):
        return jnp.interp(xi_grid, xi_of_x, x)

    X_adapt = jax.vmap(invert_one_slice, in_axes=(0, 0), out_axes=0)(Xi_of_x, X)

    return xi_grid, Xi_of_x, X_adapt



def _nodal_dudx(U, X):
    """在原始物理网格上计算节点处的 u_x。"""
    dUdx = jnp.zeros_like(U)
    dUdx = dUdx.at[:, 1:-1].set((U[:, 2:] - U[:, :-2]) / (X[:, 2:] - X[:, :-2]))
    dUdx = dUdx.at[:, 0].set((U[:, 1] - U[:, 0]) / (X[:, 1] - X[:, 0]))
    dUdx = dUdx.at[:, -1].set((U[:, -1] - U[:, -2])/ (X[:, -1] - X[:, -2]))

    return dUdx

def _interp_rows(X_src, F_src, X_query):
    """
    对每一个时间层分别进行一维插值。
    X_src, F_src, X_query: (Nt, Nx)
    """
    return jax.vmap(lambda xs, fs, xq: jnp.interp(xq, xs, fs), in_axes=(0, 0, 0),out_axes=0)(X_src, F_src, X_query)

def _smooth_1d_monitor(omega, smooth_steps):
    """对应原代码底边和上边的四次一维平滑。"""

    def smooth_step(_, w):
        return w.at[1:-1].set(0.25 * (w[:-2] + 2.0 * w[1:-1] + w[2:]))

    return jax.lax.fori_loop(0, smooth_steps, smooth_step, omega)

def _smooth_2d_monitor(omega, smooth_steps):
    """对应原代码内部单元监视器的二维平滑。"""

    def smooth_step(_, w):
        ws = w

        # 内部
        ws = ws.at[1:-1, 1:-1].set(4.0 / 16.0 * w[1:-1, 1:-1] 
            + 2.0 / 16.0 * (w[:-2, 1:-1] + w[2:, 1:-1] + w[1:-1, :-2] + w[1:-1, 2:])
            + 1.0 / 16.0 * (w[:-2, :-2] + w[:-2, 2:] + w[2:, :-2] + w[2:, 2:]))
        # 下边
        ws = ws.at[0, 1:-1].set(4.0 / 12.0 * w[0, 1:-1]
            + 2.0 / 12.0 * (w[0, :-2] + w[0, 2:] + w[1, 1:-1])
            + 1.0 / 12.0 * (w[1, :-2] + w[1, 2:]))
        # 上边
        ws = ws.at[-1, 1:-1].set(4.0 / 12.0 * w[-1, 1:-1]
            + 2.0 / 12.0 * (w[-1, :-2] + w[-1, 2:] + w[-2, 1:-1])
            + 1.0 / 12.0 * (w[-2, :-2] + w[-2, 2:]))
        # 左边
        ws = ws.at[1:-1, 0].set(4.0 / 12.0 * w[1:-1, 0]
            + 2.0 / 12.0 * (w[:-2, 0] + w[2:, 0] + w[1:-1, 1])
            + 1.0 / 12.0 * (w[:-2, 1] + w[2:, 1]))
        # 右边
        ws = ws.at[1:-1, -1].set(4.0 / 12.0 * w[1:-1, -1]
            + 2.0 / 12.0 * (w[:-2, -1] + w[2:, -1] + w[1:-1, -2])
            + 1.0 / 12.0 * ( w[:-2, -2] + w[2:, -2]))
        # 四个角
        ws = ws.at[0, 0].set(4.0 / 9.0 * w[0, 0]
            + 2.0 / 9.0 * (w[0, 1] + w[1, 0])
            + 1.0 / 9.0 * w[1, 1])
        ws = ws.at[0, -1].set(4.0 / 9.0 * w[0, -1]
            + 2.0 / 9.0 * (w[0, -2] + w[1, -1])
            + 1.0 / 9.0 * w[1, -2])
        ws = ws.at[-1, 0].set(4.0 / 9.0 * w[-1, 0]
            + 2.0 / 9.0 * (w[-1, 1] + w[-2, 0])
            + 1.0 / 9.0 * w[-2, 1])
        ws = ws.at[-1, -1].set(4.0 / 9.0 * w[-1, -1]
            + 2.0 / 9.0 * (w[-1, -2] + w[-2, -1])
            + 1.0 / 9.0 * w[-2, -2])

        return ws

    return jax.lax.fori_loop(0, smooth_steps, smooth_step, omega)


@partial(jax.jit, static_argnames=("max_iter", "smooth_steps"))
def compute_adaptive_mesh_1d_winslow(T, X, U, *, tol=1.0e-5, max_iter=100_000, smooth_steps=4):
    """
    Winslow 型时空网格迭代，但只移动空间坐标 X。

    Parameters
    ----------
    T, X, U : (Nt, Nx)
        原始时间坐标、空间坐标和数值解。
    tol : float
        对应原代码中的 sumd 容差。
    max_iter : int
        最大迭代次数。
    smooth_steps : int
        监视器平滑次数。

    Returns
    -------
    xi_grid : (Nx,)
        均匀计算坐标。
    Xi_of_x : (Nt, Nx)
        原始物理节点 X 对应的计算坐标 xi(t, x)。
    X_adapt : (Nt, Nx)
        Winslow 迭代得到的 x(t, xi)。
    """
    if T.ndim != 2 or X.ndim != 2 or U.ndim != 2:
        raise ValueError("T, X and U must all have shape (Nt, Nx).")

    if T.shape != X.shape or X.shape != U.shape:
        raise ValueError("T, X and U must have the same shape.")

    nt, nx = U.shape

    if nt < 3 or nx < 3:
        raise ValueError("The legacy 2-D Winslow stencil requires Nt >= 3 and Nx >= 3.")

    # 原始物理网格和原始导数场保持不变。
    X_src = X
    dUdx_src = _nodal_dudx(U, X_src)

    xi_grid = jnp.linspace(0.0, 1.0, nx, dtype=X.dtype)

    tol = jnp.asarray(tol, dtype=X.dtype)

    def winslow_sweep(X_old):
        # 关键：网格移动以后，在新的 X_old 上重新取 dU/dx。
        # 因此监视器并不是固定不变的。
        dUdx_cur = _interp_rows(X_src, dUdx_src, X_old)

        div = jnp.abs(dUdx_cur) - dUdx_cur

        # alpha 同样使用当前网格间距重新计算。
        alpha = (0.5 * (div[:, :-1] + div[:, 1:]) * (X_old[:, 1:] - X_old[:, :-1]))
        alpha = jnp.sum(alpha, axis=1, keepdims=True)
        alpha = jnp.maximum(alpha, 1.0)

        rho = 1.0 + 0.5 * div / alpha

        # 在每一轮开始时 xx = x。
        X_new = X_old

        # =====================================================
        # 底部时间边界
        # =====================================================
        omega_bottom = 0.5 * (rho[0, 1:] + rho[0, :-1])
        omega_bottom = _smooth_1d_monitor(omega_bottom, smooth_steps)

        # 按原 NumPy 代码的次序执行 Gauss-Seidel 更新。
        def update_bottom(k, x_new):
            value = (omega_bottom[k - 1] * x_new[0, k - 1] + omega_bottom[k] * x_new[0, k + 1]) / (omega_bottom[k - 1] + omega_bottom[k])

            return x_new.at[0, k].set(value)

        X_new = jax.lax.fori_loop(1, nx - 1, update_bottom, X_new)

        # =====================================================
        # 内部网格
        # =====================================================
        omega = 0.25 * (rho[1:, 1:] + rho[1:, :-1] + rho[:-1, 1:] + rho[:-1, :-1])
        omega = _smooth_2d_monitor(omega, smooth_steps)

        nxi = nx - 2

        def update_interior(flat_index, x_new):
            # 按 j 外层、k 内层的顺序展开，
            # 从而保留原代码的 Gauss-Seidel 更新顺序。
            j = flat_index // nxi + 1
            k = flat_index % nxi + 1

            w00 = omega[j - 1, k - 1]
            w01 = omega[j - 1, k]
            w10 = omega[j, k - 1]
            w11 = omega[j, k]

            numerator = ((w00 + w01) * x_new[j - 1, k] + (w00 + w10) * x_new[j, k - 1] + (w10 + w11) * X_old[j + 1, k] + (w01 + w11) * X_old[j, k + 1])
            value = (0.5 * numerator / (w00 + w01 + w10 + w11))

            return x_new.at[j, k].set(value)

        X_new = jax.lax.fori_loop(0, (nt - 2) * (nx - 2), update_interior, X_new)

        # =====================================================
        # 顶部时间边界
        # =====================================================
        omega_top = 0.5 * (rho[-1, 1:] + rho[-1, :-1])

        omega_top = _smooth_1d_monitor(omega_top, smooth_steps)

        def update_top(k, x_new):
            value = (omega_top[k - 1] * x_new[-1, k - 1] + omega_top[k] * x_new[-1, k + 1]) / (omega_top[k - 1] + omega_top[k])

            return x_new.at[-1, k].set(value)

        X_new = jax.lax.fori_loop(1, nx - 1, update_top, X_new)

        return X_new


    # Winslow 外层迭代

    def cond_fun(state):
        iteration, _, residual = state

        return ((iteration < max_iter) & (residual > tol))

    def body_fun(state):
        iteration, X_old, _ = state

        X_new = winslow_sweep(X_old)

        # 与原代码完全一致，使用平方差的总和。
        residual = jnp.sum((X_new - X_old) ** 2)

        return (iteration + 1, X_new, residual,)

    initial_state = (jnp.asarray(0, dtype=jnp.int32), X_src, jnp.asarray(1.0, dtype=X.dtype))

    _, X_adapt, _ = jax.lax.while_loop(cond_fun, body_fun, initial_state)

    # X_adapt 表示 x(t, xi)。
    # 在原始 X 节点上反插值，得到 xi(t, x)。
    Xi_of_x = jax.vmap(lambda x_adapt, x_src: jnp.interp(x_src, x_adapt, xi_grid), in_axes=(0, 0), out_axes=0)(X_adapt, X_src)

    return xi_grid, Xi_of_x, X_adapt





def create_map_xi_to_x(T, xi_grid, X_adapt):
    T = jnp.asarray(T)
    xi_grid = jnp.asarray(xi_grid)
    X_adapt = jnp.asarray(X_adapt)

    # Extract the 1D time coordinate
    if T.ndim == 2:
        t_grid = T[:, 0]
    else:
        t_grid = T

    @jax.jit
    def map_xi_to_x(t, xi):
        # Bilinear interpolation of x(t, xi).
        # Support scalar / vector / arbitrary compatible shapes
        t, xi = jnp.broadcast_arrays(t, xi)

        # Keep queries inside the interpolation domain
        t = jnp.clip(t, t_grid[0], t_grid[-1])
        xi = jnp.clip(xi, xi_grid[0], xi_grid[-1])

        # Locate the interpolation cell
        i = jnp.searchsorted(t_grid, t, side="right") - 1
        j = jnp.searchsorted(xi_grid, xi, side="right") - 1

        i = jnp.clip(i, 0, t_grid.shape[0] - 2)
        j = jnp.clip(j, 0, xi_grid.shape[0] - 2)

        # Coordinates of the four surrounding nodes
        t0 = t_grid[i]
        t1 = t_grid[i + 1]

        xi0 = xi_grid[j]
        xi1 = xi_grid[j + 1]

        # Local interpolation coordinates
        alpha = (t - t0) / (t1 - t0)
        beta = (xi - xi0) / (xi1 - xi0)

        # Values at four corners
        x00 = X_adapt[i,     j]
        x01 = X_adapt[i,     j + 1]
        x10 = X_adapt[i + 1, j]
        x11 = X_adapt[i + 1, j + 1]

        # Bilinear interpolation
        x = ((1.0 - alpha) * (1.0 - beta) * x00 
            + (1.0 - alpha) * beta * x01
            + alpha * (1.0 - beta) * x10
            + alpha * beta * x11)

        return x

    return map_xi_to_x


def create_coor_loss_grad_fn(model, lambda_pos=0.1, xi_x_min=0.05):
    apply_fn = model.apply

    def coor_single(params, tx):
        return jnp.squeeze(apply_fn(params, tx))

    coor_grad = jax.grad(coor_single, argnums=1)
    batch_grad = jax.vmap(coor_grad, in_axes=(None, 0))

    def loss_fn(params, inputs, targets):
        xi_pred = jnp.asarray(apply_fn(params, inputs)).reshape(-1)
        xi_true = jnp.asarray(targets).reshape(-1)
        loss_fit = jnp.mean((xi_pred - xi_true) ** 2)

        grad_xi = batch_grad(params, inputs)
        xi_x = grad_xi[:, 1]

        violation = jax.nn.relu(xi_x_min - xi_x)
        loss_pos = jnp.mean(violation**2)

        loss = (loss_fit + lambda_pos * loss_pos)
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



def create_pde_loss_grad_fn(model, w_ic=10.0, gamma=1.4):
    apply_fn = model.apply
    # Single-point primitive variables
    # inputs = [t, x, xi]
    # output = [rho, u, p]

    def get_primitive(params, inputs):
        rho, velocity, pressure = apply_fn(params, inputs)
        rho = jnp.squeeze(rho)
        velocity = jnp.squeeze(velocity)
        pressure = jnp.squeeze(pressure)

        return jnp.stack([rho, velocity, pressure])

    # Conservative variables
    # Q = [rho, rho*u, E]
    def get_Q(params, inputs):
        rho, velocity, pressure = get_primitive(params, inputs)
        momentum = rho * velocity
        energy = (0.5 * rho * velocity**2 + pressure / (gamma - 1.0))

        return jnp.stack([rho, momentum, energy])

    # Euler flux
    # F = [rho*u, rho*u^2+p, u(E+p)]
    def get_F(params, inputs):
        rho, velocity, pressure = get_primitive(params, inputs)
        energy = (0.5 * rho * velocity**2 + pressure / (gamma - 1.0))

        return jnp.stack([rho * velocity, rho * velocity**2 + pressure, velocity * (energy + pressure)])

    jac_Q = jax.jacfwd(get_Q, argnums=1)
    jac_F = jax.jacfwd(get_F, argnums=1)
    hess_Q = jax.jacfwd(jac_Q, argnums=1)

    # Single-point Euler residual
    #
    # data = [
    #     t, x, xi,
    #     xi_t, xi_x, xi_xx
    # ]
    def residual_single(params, data, nu):
        inputs = data[:3]

        xi_t = data[3]
        xi_x = data[4]
        xi_xx = data[5]

        # JQ.shape = (3, 3)
        # 第一维：守恒方程分量
        # 第二维：[t, x, xi]
        JQ = jac_Q(params, inputs)
        JF = jac_F(params, inputs)

        # HQ.shape = (3, 3, 3)
        HQ = hess_Q(params, inputs)

        # Physical time derivative
        # q(t,x) = Q(t,x,xi(t,x))
        # q_t = Q_t + Q_xi * xi_t
        q_t = (JQ[:, 0] + JQ[:, 2] * xi_t)

        # Physical flux derivative
        # f_x = F_x + F_xi * xi_x
        f_x = (JF[:, 1] + JF[:, 2] * xi_x)

        # Physical second derivative
        # q_xx =
        #     Q_xx
        #   + 2 Q_xxi xi_x
        #   + Q_xixi xi_x^2
        #   + Q_xi xi_xx
        q_xx = (HQ[:, 1, 1] + 2.0 * HQ[:, 1, 2] * xi_x + HQ[:, 2, 2] * xi_x**2 + JQ[:, 2] * xi_xx)

        # Viscous/artificial-viscosity Euler residual
        residual = (q_t + f_x - nu * q_xx)

        return residual

    residual_batch = jax.vmap(residual_single, in_axes=(None, 0, None))
    primitive_batch = jax.vmap(get_primitive, in_axes=(None, 0))

    # Total loss
    def loss_fn(params, inputs_pde, inputs_ics, targets_ics, nu=1.0e-3):
        # 1. PDE loss
        # residual.shape = (N_pde, 3)
        residual = residual_batch(params, inputs_pde, nu)

        # importance correction:
        # rho_sampling(t,x) ~ |xi_x|
        xi_x = inputs_pde[:, 4]

        weight = jnp.maximum(jnp.abs(xi_x), 1.0e-8)
        loss_pde = jnp.mean(jnp.sum(residual**2, axis=-1) / weight)

        # 2. Initial-condition loss
        # targets_ics = [rho, u, p]
        pred_ics = primitive_batch(params, inputs_ics)
        loss_ics = jnp.mean(jnp.sum((pred_ics - targets_ics) ** 2, axis=-1))

        # Total loss
        loss = (loss_pde + w_ic * loss_ics)

        return loss

    return jax.jit(jax.value_and_grad(loss_fn))


def IC_Riemann_1D_single(x, x_jump, crhoL, cuL, cpL, crhoR, cuR, cpR):
    rho = jnp.where(x < x_jump, crhoL, jnp.where(x > x_jump, crhoR,  0.5 * (crhoL + crhoR)))
    velocity = jnp.where(x < x_jump, cuL, jnp.where(x > x_jump, cuR, 0.5 * (cuL + cuR)))
    pressure = jnp.where(x < x_jump, cpL, jnp.where(x > x_jump, cpR, 0.5 * (cpL + cpR)))

    return rho, velocity, pressure

IC_vmap = jax.vmap(IC_Riemann_1D_single, in_axes=(0, None, None, None, None, None, None, None))


def create_pde_minibatch_fn(IC_map, map_xi_to_x, coor_net, xL, xR, t0, tend, left_state, right_state, x_jump=None):
    crhoL, cuL, cpL = left_state
    crhoR, cuR, cpR = right_state

    if x_jump is None:
        x_jump = 0.5 * (xL + xR)

    # 保证 coor_net 的单点输出严格为标量，
    # 否则 jax.grad 不接受 shape=(1,) 的输出。
    def coor_scalar(tx):
        return jnp.squeeze(coor_net(tx))

    coor_grad = jax.grad(coor_scalar)
    coor_hess = jax.hessian(coor_scalar)

    batch_coor = jax.vmap(coor_scalar)
    batch_grad = jax.vmap(coor_grad)
    batch_hess = jax.vmap(coor_hess)

    @partial(jax.jit, static_argnames=("pts_pde", "pts_ics"))
    def pde_minibatch(key, pts_pde, pts_ics):
        key_pde, key_ics = jax.random.split(key, 2)

        sample_pde = jax.random.uniform(key_pde, shape=(pts_pde, 2), minval=0.0, maxval=1.0)
        t_pde = (t0 + (tend - t0) * sample_pde[:, 0])
        xi_sample = sample_pde[:, 1]

        x_pde = jnp.ravel(map_xi_to_x(t_pde, xi_sample))
        tx_pde = jnp.stack([t_pde, x_pde], axis=-1,)

        xi_pde = batch_coor(tx_pde)

        grad_xi = batch_grad(tx_pde)

        xi_t = grad_xi[:, 0]
        xi_x = grad_xi[:, 1]

        hess_xi = batch_hess(tx_pde)
        xi_xx = hess_xi[:, 1, 1]

        # [t, x, xi, xi_t, xi_x, xi_xx]
        inputs_pde = jnp.stack([t_pde, x_pde, xi_pde, xi_t, xi_x, xi_xx], axis=-1)

        # 2. Initial-condition points
        sample_ics = jax.random.uniform(key_ics, shape=(pts_ics,), minval=0.0, maxval=1.0)

        x_ics = (xL + (xR - xL) * sample_ics)
        t_ics = jnp.full_like(x_ics, t0)
        tx_ics = jnp.stack([t_ics, x_ics], axis=-1)
        xi_ics = batch_coor(tx_ics)

        # [t, x, xi]
        inputs_ics = jnp.stack([t_ics, x_ics, xi_ics], axis=-1)

        # Riemann 初值
        rho_ics, velocity_ics, pressure_ics = IC_map(x_ics, x_jump, crhoL, cuL, cpL, crhoR, cuR, cpR)

        # targets_ics.shape = (pts_ics, 3)
        # [rho, u, p]
        targets_ics = jnp.stack([rho_ics, velocity_ics, pressure_ics], axis=-1)

        return (inputs_pde, inputs_ics, targets_ics)

    return pde_minibatch



def create_pde_net_update_fn(loss_grad_fn, optimizer, nu):

    @jax.jit
    def update_step(params, opt_state, inputs_pde, inputs_ics, targets_ics):

        # 1. Compute loss and gradients
        loss, grads = loss_grad_fn(params, inputs_pde, inputs_ics, targets_ics, nu)

        # 2. Optimizer update
        updates, new_opt_state = optimizer.update(grads, opt_state, params=params)

        # 3. Apply updates
        new_params = optax.apply_updates(params, updates)

        return (new_params, new_opt_state, loss)

    return update_step


def create_identity_maps(xL, xR):

    @jax.jit
    def identity_map_xi_to_x(t, xi):
        return xL + (xR - xL) * xi

    @jax.jit
    def identity_coor_net(tx):
        t, x = tx
        return (x - xL) / (xR - xL)

    return identity_map_xi_to_x, identity_coor_net


def create_eval_fn(model):
    apply_fn = model.apply

    def predict_single(params, inputs):
        rho, velocity, pressure = apply_fn(params, inputs)

        return jnp.stack([jnp.squeeze(rho), jnp.squeeze(velocity), jnp.squeeze(pressure)])

    predict_batch = jax.vmap(predict_single, in_axes=(None, 0))

    def eval_error(params, inputs, labels):
        # shapes: (N_test, 3)
        predictions = predict_batch(params, inputs).reshape(-1, 3)
        labels = jnp.asarray(labels).reshape(-1, 3)

        diff = predictions - labels

        # 分别计算 rho、u、p 的 MSE
        mse = jnp.mean(diff**2, axis=0)

        # 分别计算 rho、u、p 的 relative L2
        numerator = jnp.sum(diff**2, axis=0)
        denominator = jnp.sum(labels**2, axis=0,)

        rl2 = jnp.sqrt(numerator / jnp.maximum(denominator, 1.0e-12))
        # mse.shape = (3,)
        # rl2.shape = (3,)
        return mse, rl2

    return jax.jit(eval_error)

def train_pde_stage(stage_name, pde_params, pde_optimizer, pde_loss_grad_fn, pde_minibatch_fn, eval_fn, test_inputs, test_labels, 
    key, nu, max_iters=10000, pts_pde=10000, pts_ics=1000, max_runtime=10000.0, eval_every=500):

    opt_state = pde_optimizer.init(pde_params)
    update_fn = create_pde_net_update_fn(pde_loss_grad_fn, pde_optimizer, nu=nu)

    # JIT warm-up
    warmup_key = jax.random.PRNGKey(0)
    warmup_dataset = pde_minibatch_fn(warmup_key, pts_pde=pts_pde, pts_ics=pts_ics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), warmup_dataset)

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
        mse_value = np.asarray(mse_value, dtype=np.float64)
        rl2_value = np.asarray(rl2_value, dtype=np.float64)

        history["eval_iter"].append(iteration)
        history["eval_train_time"].append(train_time)
        history["mse"].append(mse_value.copy())
        history["rl2"].append(rl2_value.copy())

        return mse_value, rl2_value

    mse_value, rl2_value = record_evaluation(iteration=0, train_time=0.0)
    print(
        f"[{stage_name}] iter=00000, time=0.00s | "
        f"MSE(rho,u,p)=({mse_value[0]:.2e}, {mse_value[1]:.2e}, {mse_value[2]:.2e}) | "
        f"RL2(rho,u,p)=({rl2_value[0]:.2e}, {rl2_value[1]:.2e}, {rl2_value[2]:.2e})"
    )

    # Training
    for it in range(1, max_iters + 1):
        if runtime >= max_runtime:
            break

        start = time.perf_counter()

        key, key_batch = jax.random.split(key)
        dataset = pde_minibatch_fn(key_batch, pts_pde=pts_pde, pts_ics=pts_ics)
        pde_params, opt_state, loss = update_fn(pde_params, opt_state, *dataset)
        loss_value = float(loss.block_until_ready())

        runtime += time.perf_counter() - start

        history["iter"].append(it)
        history["loss"].append(loss_value)
        history["train_time"].append(runtime)

        # Evaluation
        if it % eval_every == 0:
            mse_value, rl2_value = record_evaluation(iteration=it, train_time=runtime)

            print(f"[{stage_name}] iter={it:05d}, time={runtime:.2f}s, loss={loss_value:.2e} | "
                f"MSE(rho,u,p)=({mse_value[0]:.2e}, {mse_value[1]:.2e}, {mse_value[2]:.2e}) | "
                f"RL2(rho,u,p)=({rl2_value[0]:.2e}, {rl2_value[1]:.2e}, {rl2_value[2]:.2e})")

    if history["iter"]:
        final_iter = history["iter"][-1]
        if history["eval_iter"][-1] != final_iter:
            mse_value, rl2_value = record_evaluation(iteration=final_iter, train_time=runtime)
            print(
                f"[{stage_name}] final iter={final_iter:05d}, time={runtime:.2f}s | "
                f"MSE(rho,u,p)=({mse_value[0]:.2e}, {mse_value[1]:.2e}, {mse_value[2]:.2e}) | "
                f"RL2(rho,u,p)=({rl2_value[0]:.2e}, {rl2_value[1]:.2e}, {rl2_value[2]:.2e})"
            )

    # Convert history to NumPy arrays
    history["iter"] = np.asarray(history["iter"], dtype=np.int32)
    history["loss"] = np.asarray(history["loss"], dtype=np.float64)
    history["train_time"] = np.asarray(history["train_time"], dtype=np.float64)
    history["eval_iter"] = np.asarray(history["eval_iter"], dtype=np.int32)
    history["eval_train_time"] = np.asarray(history["eval_train_time"], dtype=np.float64)

    history["mse"] = np.asarray(history["mse"], dtype=np.float64).reshape(-1, 3)
    history["rl2"] = np.asarray(history["rl2"], dtype=np.float64).reshape(-1, 3)

    return pde_params, key, history


def plot_training_history(history, metric="rl2",  title="Training history"):
    iters = np.asarray(history["iter"])
    loss = np.asarray(history["loss"])
    eval_iters = np.asarray(history["eval_iter"])

    if metric == "rl2":
        metric_values = np.asarray(history["rl2"])
        metric_label = r"Relative $L^2$ error"
        metric_title = r"Component-wise relative $L^2$ error"

    elif metric == "mse":
        metric_values = np.asarray(history["mse"])
        metric_label = "MSE"
        metric_title = "Component-wise mean squared error"

    else:
        raise ValueError("metric must be 'rl2' or 'mse'." )

    if metric_values.ndim != 2 or metric_values.shape[1] != 3:
        raise ValueError("history metric must have shape (num_evaluations, 3), with columns [rho, u, p].")

    loss_safe = np.maximum(loss, 1.0e-30)
    metric_safe = np.maximum(metric_values, 1.0e-30)
    component_labels = [r"Density $\rho$",  r"Velocity $u$",  r"Pressure $p$"]

    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    axes[0].semilogy(iters, loss_safe, linewidth=1.5, color="tab:red",)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss")
    axes[0].grid(True,  which="both", linestyle=":", alpha=0.6)

    # Component-wise evaluation errors
    for component_idx in range(3):
        axes[1].semilogy(eval_iters, metric_safe[:, component_idx], marker="o", markersize=4, linewidth=1.5, color=colors[component_idx], label=component_labels[component_idx])

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(metric_label)
    axes[1].set_title(metric_title)
    axes[1].grid(True, which="both", linestyle=":", alpha=0.6,)
    axes[1].legend()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.show()

    return fig, axes



def generate_adaptive_mesh(pde_net, pde_params, T_grid, X_grid, coor_fn,
    mesh_method="winslow", winslow_tol=1.0e-5, winslow_max_iter=100_000, smooth_steps=4,):

    start = time.perf_counter()

    T_grid = jnp.asarray(T_grid)
    X_grid = jnp.asarray(X_grid)

    if T_grid.shape != X_grid.shape:
        raise ValueError("T_grid and X_grid must have the same shape.")

    # 1. 在均匀物理网格上计算当前Euler解
    t_flat = T_grid.reshape(-1)
    x_flat = X_grid.reshape(-1)
    tx = jnp.stack([t_flat, x_flat], axis=-1,)

    xi_flat = jax.vmap(coor_fn)(tx)
    xi_flat = jnp.asarray(xi_flat).reshape(-1)

    inputs = jnp.stack([t_flat, x_flat, xi_flat], axis=-1)

    rho_flat, velocity_flat, pressure_flat = pde_net.apply(pde_params, inputs)

    rho_grid = jnp.asarray(rho_flat).reshape(T_grid.shape)
    velocity_grid = jnp.asarray(velocity_flat).reshape(T_grid.shape)
    pressure_grid = jnp.asarray(pressure_flat).reshape(T_grid.shape)

    # shape=(Nt,Nx,3)
    solution_coarse = jnp.stack([rho_grid, velocity_grid, pressure_grid], axis=-1)
    solution_coarse.block_until_ready()

    U_monitor = velocity_grid
    # 3. 选择网格生成方法
    mesh_method = mesh_method.lower()

    if mesh_method == "integral":
        xi_grid, Xi_of_x, X_adapt = (compute_adaptive_mesh_1d_integral(T_grid, X_grid, U_monitor))

    elif mesh_method == "winslow":
        xi_grid, Xi_of_x, X_adapt = (compute_adaptive_mesh_1d_winslow(T_grid, X_grid, U_monitor,
            tol=winslow_tol, max_iter=winslow_max_iter, smooth_steps=smooth_steps))

    else:
        raise ValueError("mesh_method must be 'integral' or 'winslow'.")

    X_adapt.block_until_ready()

    # 4. 构造连续映射 (t,xi) -> x
    map_xi_to_x = create_map_xi_to_x(T_grid, xi_grid, X_adapt)
    runtime_s = time.perf_counter() - start

    print(f"[Adaptive Mesh] method={mesh_method}, time={runtime_s:.2f}s")

    return (xi_grid, Xi_of_x, X_adapt, map_xi_to_x, solution_coarse)


def train_coordinate_net(coor_net, coor_params, coor_optimizer, coor_loss_grad_fn, T_grid, X_grid, Xi_of_x,
    key, max_iters=5000, batch_size=4096, eval_every=500):

    # Training dataset
    inputs = jnp.stack([T_grid.reshape(-1), X_grid.reshape(-1)], axis=-1)
    targets = Xi_of_x.reshape(-1, 1)

    # Optimizer
    opt_state = coor_optimizer.init(coor_params)
    update_fn = create_coor_net_update_fn(coor_loss_grad_fn, coor_optimizer)

    @jax.jit
    def evaluate_full_grid(params, inputs_eval, targets_eval):
        predictions = jnp.asarray(coor_net.apply(params, inputs_eval)).reshape(-1)
        targets_eval = jnp.asarray(targets_eval).reshape(-1)

        return jnp.mean((predictions - targets_eval) ** 2)




    warmup_key = jax.random.PRNGKey(0)
    inputs_warmup, targets_warmup = coor_minibatch(warmup_key, inputs, targets, pts=batch_size)

    # Make sure minibatch compilation is finished
    inputs_warmup.block_until_ready()
    targets_warmup.block_until_ready()

    # Compile update step
    _, _, warmup_loss = update_fn(coor_params, opt_state, inputs_warmup, targets_warmup)
    warmup_loss.block_until_ready()
    print("[Coordinate] JIT warm-up finished.")

    runtime = 0.0

    for it in range(1, max_iters+1):
        start = time.perf_counter()

        key, key_batch = jax.random.split(key)
        inputs_batch, targets_batch = coor_minibatch(key_batch, inputs, targets, pts=batch_size)
        coor_params, opt_state, loss = update_fn(coor_params, opt_state, inputs_batch, targets_batch)
        loss_value = float(loss.block_until_ready())

        runtime += time.perf_counter() - start

        # Full-grid evaluation
        if it % eval_every == 0:
            mse = evaluate_full_grid(coor_params, inputs, targets)
            mse_value = float(mse.block_until_ready())

            print(f"[Coordinate] iter={it:05d}, time={runtime:.2f}s, batch loss={loss_value:.2e}, full mse={mse_value:.2e}")

    return coor_params, key


def create_coor_fn(coor_net, coor_params):
    def coor_fn(tx):
        return jnp.squeeze(coor_net.apply(coor_params, tx))

    return coor_fn



def create_test_inputs(t_test, x_test, coor_fn):
    t_test = jnp.asarray(t_test)
    x_test = jnp.asarray(x_test)

    if t_test.shape != x_test.shape:
        raise ValueError("t_test and x_test must have the same shape.")

    t_flat = t_test.reshape(-1)
    x_flat = x_test.reshape(-1)

    tx = jnp.stack([t_flat, x_flat], axis=-1)

    xi_test = jax.vmap(coor_fn)(tx)
    xi_test = jnp.asarray(xi_test).reshape(-1)

    # shape=(N_test,3)
    return jnp.stack([t_flat, x_flat, xi_test], axis=-1)


def plot_evaluation(params, model, t_test, x_test, labels_test, coor_fn, cmap="jet"):

    apply_fn = model.apply

    # 1. Prepare test inputs
    t_test_jax = jnp.asarray(t_test).reshape(-1)
    x_test_jax = jnp.asarray(x_test).reshape(-1)

    if t_test_jax.shape != x_test_jax.shape:
        raise ValueError("t_test and x_test must contain the same number of points.")

    tx_test = jnp.stack([t_test_jax, x_test_jax], axis=-1)
    batch_coor = jax.vmap(coor_fn)

    @jax.jit
    def predict_fn(params, tx):
        xi = batch_coor(tx)
        xi = jnp.asarray(xi).reshape(-1)

        inputs = jnp.stack([tx[:, 0], tx[:, 1], xi], axis=-1)
        rho, velocity, pressure = apply_fn(params, inputs)

        return jnp.stack([jnp.asarray(rho).reshape(-1), jnp.asarray(velocity).reshape(-1), jnp.asarray(pressure).reshape(-1)], axis=-1)

    predictions = predict_fn(params, tx_test)
    predictions.block_until_ready()

    # 2. Convert to NumPy
    predictions_np = np.asarray(predictions).reshape(-1, 3)
    labels_np = np.asarray(labels_test).reshape(-1, 3)

    if predictions_np.shape != labels_np.shape:
        raise ValueError("Predictions and labels must have the same shape. labels_test should have the last dimension [rho,u,p].")

    t_np = np.asarray(t_test_jax).reshape(-1)
    x_np = np.asarray(x_test_jax).reshape(-1)

    # 3. Reconstruct structured (t,x) grid
    sort_idx = np.lexsort((x_np, t_np))

    t_sorted = t_np[sort_idx]
    x_sorted = x_np[sort_idx]

    pred_sorted = predictions_np[sort_idx]
    true_sorted = labels_np[sort_idx]

    t_unique = np.unique(t_sorted)
    x_unique = np.unique(x_sorted)

    nt_test = len(t_unique)
    nx_test = len(x_unique)

    if nt_test * nx_test != len(t_sorted):
        raise ValueError("The supplied test points do not form a complete rectangular (t,x) grid.")

    print(f"Plotting test grid: nt={nt_test}, nx={nx_test}")

    # shapes=(Nt,Nx,3)
    pred_grid = pred_sorted.reshape(nt_test, nx_test, 3)
    true_grid = true_sorted.reshape(nt_test, nx_test, 3)

    # 4. Component-wise errors
    diff = pred_grid - true_grid
    mse = np.mean(diff**2, axis=(0, 1))
    numerator = np.sum(diff**2, axis=(0, 1))
    denominator = np.sum(true_grid**2, axis=(0, 1))
    rl2 = np.sqrt(numerator / np.maximum(denominator, 1.0e-12))

    print(
        "Evaluation errors:\n"
        f"  rho: MSE={mse[0]:.6e}, RL2={rl2[0]:.6e}\n"
        f"  u  : MSE={mse[1]:.6e}, RL2={rl2[1]:.6e}\n"
        f"  p  : MSE={mse[2]:.6e}, RL2={rl2[2]:.6e}"
    )

    # 5. Density contour data
    rho_true = true_grid[:, :, 0]
    rho_pred = pred_grid[:, :, 0]

    rho_min = min(np.nanmin(rho_true), np.nanmin(rho_pred))
    rho_max = max(np.nanmax(rho_true), np.nanmax(rho_pred))

    if rho_max <= rho_min:
        rho_max = rho_min + 1.0e-12

    rho_levels = np.linspace(rho_min, rho_max, 101)

    # 6. Final-time profiles
    tend = t_unique[-1]

    true_final = true_grid[-1, :, :]
    pred_final = pred_grid[-1, :, :]

    component_names = [r"$\rho$", r"$u$", r"$p$"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    # 7. Plot
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    # Reference density
    contour_true = axes[0].contourf(x_unique, t_unique, rho_true, levels=rho_levels, cmap=cmap)
    fig.colorbar(contour_true, ax=axes[0], label=r"$\rho$")
    axes[0].set_title(r"Reference density $\rho(t,x)$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$t$")

    # Predicted density
    contour_pred = axes[1].contourf(x_unique, t_unique, rho_pred, levels=rho_levels, cmap=cmap)
    fig.colorbar(contour_pred, ax=axes[1], label=r"$\rho$")
    axes[1].set_title(r"Predicted density $\rho(t,x)$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$t$")

    # Final-time profiles of rho, u and p
    for component_idx in range(3):
        axes[2].plot(x_unique, true_final[:, component_idx], color=colors[component_idx], linestyle="-", linewidth=2.0,
            label=(f"Reference {component_names[component_idx]}"))
        axes[2].plot(x_unique, pred_final[:, component_idx], color=colors[component_idx], linestyle="--", linewidth=2.0,
            label=(f"Prediction {component_names[component_idx]}"))

    axes[2].set_title(rf"Profiles at $t={tend:.4g}$")
    axes[2].set_xlabel(r"$x$")
    axes[2].set_ylabel("Primitive variables")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="best", ncol=2, fontsize=9)

    plt.tight_layout()
    plt.show()

    return mse, rl2




def plot_coordinate_mapping(T_grid, X_grid, xi_grid, Xi_of_x, X_adapt, coor_fn, n_times=5):

    # Convert to NumPy
    T_np = np.asarray(T_grid)
    X_np = np.asarray(X_grid)
    Xi_target = np.asarray(Xi_of_x)

    if T_np.ndim == 2:
        t_grid = T_np[:, 0]
    else:
        t_grid = T_np

    Nt = len(t_grid)

    # Select several representative times
    time_indices = np.linspace(0, Nt - 1, n_times, dtype=int)
    time_indices = np.unique(time_indices)

    # Predict xi(t,x) on the whole physical grid
    tx_all = jnp.stack([jnp.asarray(T_grid).reshape(-1), jnp.asarray(X_grid).reshape(-1)], axis=-1)
    xi_pred = jax.vmap(coor_fn)(tx_all)
    xi_pred = np.asarray(xi_pred).reshape(X_np.shape)
    xi_error = np.abs(xi_pred - Xi_target)

    # Diagnostics
    mse = np.mean((xi_pred - Xi_target) ** 2)
    max_error = np.max(xi_error)
    print(f"Coordinate MSE = {mse:.6e}")
    print(f"Coordinate max error = {max_error:.6e}")

    coor_grad = jax.grad(coor_fn)
    grad_all = jax.vmap(coor_grad)(tx_all)
    grad_all = np.asarray(grad_all)
    xi_x = grad_all[:, 1].reshape(X_np.shape)

    print(f"xi_x min = {np.min(xi_x):.6e}")
    print(f"xi_x max = {np.max(xi_x):.6e}")
    print(f"negative xi_x points = {np.sum(xi_x <= 0)}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 2D coordinate field:
    ax = axes[0]
    contour = ax.contourf(X_np, T_np, xi_pred, levels=101, cmap="jet")
    fig.colorbar(contour, ax=ax, label=r"$\xi_\theta(t,x)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.set_title(r"Learned coordinate field $\xi_\theta(t,x)$")

    # 2. x -> xi
    ax = axes[1]
    for i in time_indices:
        x_i = X_np[i, :]
        ax.plot(x_i, Xi_target[i, :], linewidth=2, label=rf"Target $t={t_grid[i]:.2f}$")
        ax.plot(x_i, xi_pred[i, :], linestyle="--", linewidth=1.5, label=rf"NN $t={t_grid[i]:.2f}$")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\xi(t,x)$")
    ax.set_title(r"$x \rightarrow \xi$")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, ncol=2)

    # 3. x -> xi error
    ax = axes[2]
    for i in time_indices:
        ax.plot(X_np[i, :], xi_error[i, :], linewidth=1.5, label=rf"$t={t_grid[i]:.2f}$")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$|\xi_{\mathrm{NN}}-\xi_{\mathrm{target}}|$")
    ax.set_title(r"Coordinate fitting error")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

def save_training_results(pde_params_by_stage, coor_params_by_stage, pde_histories_by_stage,
    checkpoint_path="talpinn_euler1d.msgpack", history_path=None):
    """Save both Euler PDE stages, the coordinate network, and histories."""
    pde_stage_names = ("stage1", "stage2")
    coor_stage_names = ("stage1",)
    history_keys = (
        "iter", "loss", "train_time",
        "eval_iter", "eval_train_time", "mse", "rl2",
    )

    if set(pde_params_by_stage) != set(pde_stage_names):
        raise ValueError(f"pde_params_by_stage keys must be {pde_stage_names}")
    if set(coor_params_by_stage) != set(coor_stage_names):
        raise ValueError(f"coor_params_by_stage keys must be {coor_stage_names}")
    if set(pde_histories_by_stage) != set(pde_stage_names):
        raise ValueError(f"pde_histories_by_stage keys must be {pde_stage_names}")

    histories = {}
    for stage_name in pde_stage_names:
        history = pde_histories_by_stage[stage_name]
        missing_keys = [name for name in history_keys if name not in history]
        if missing_keys:
            raise ValueError(f"{stage_name} history is missing keys: {missing_keys}")
        histories[stage_name] = {
            name: np.asarray(history[name]) for name in history_keys
        }

    checkpoint = {
        "format_version": 1,
        "pde_params": jax.device_get({
            name: pde_params_by_stage[name] for name in pde_stage_names
        }),
        "coor_params": jax.device_get({
            name: coor_params_by_stage[name] for name in coor_stage_names
        }),
        "pde_histories": histories,
    }

    checkpoint_path = os.path.abspath(checkpoint_path)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint_tmp = checkpoint_path + ".tmp"
    with open(checkpoint_tmp, "wb") as f:
        f.write(serialization.to_bytes(checkpoint))
    os.replace(checkpoint_tmp, checkpoint_path)

    if history_path is None:
        history_path = os.path.splitext(checkpoint_path)[0] + "_history.npz"
    history_path = os.path.abspath(history_path)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    history_arrays = {
        f"{stage_name}_{name}": values
        for stage_name, history in histories.items()
        for name, values in history.items()
    }
    history_tmp = history_path + ".tmp"
    with open(history_tmp, "wb") as f:
        np.savez_compressed(f, **history_arrays)
    os.replace(history_tmp, history_path)

    print(f"Training checkpoint saved to: {checkpoint_path}")
    print(f"Training histories saved to: {history_path}")


def load_training_results(checkpoint_path):
    """Load the complete Euler checkpoint and return JAX parameter arrays."""
    with open(checkpoint_path, "rb") as f:
        checkpoint = serialization.msgpack_restore(f.read())

    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format_version')}")

    checkpoint["pde_params"] = jax.tree_util.tree_map(
        jnp.asarray, checkpoint["pde_params"]
    )
    checkpoint["coor_params"] = jax.tree_util.tree_map(
        jnp.asarray, checkpoint["coor_params"]
    )

    print(f"Training checkpoint loaded from: {checkpoint_path}")
    return checkpoint


def load_training_histories(history_path):
    """Load both component-wise Euler histories from the NumPy archive."""
    stage_names = ("stage1", "stage2")
    history_keys = (
        "iter", "loss", "train_time",
        "eval_iter", "eval_train_time", "mse", "rl2",
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



def main(mesh_method="winslow"):

    # 1. Load Sod shock-tube data
    path = "SodShocktube_data.mat"
    #path = "LaxShocktube_data.mat"
    data = scipy.io.loadmat(path)

    sss = 8
    Rho0 = jnp.asarray(data["Rho"])[:, ::sss]
    U0 = jnp.asarray(data["U"])[:, ::sss]
    P0 = jnp.asarray(data["P"])[:, ::sss]
    X0 = jnp.asarray(data["X"])[:, ::sss]
    T0 = jnp.asarray(data["T"])[:, ::sss]

    Nt_test, Nx_test = Rho0.shape
    xl = float(X0[0, 0])
    xr = float(X0[0, -1])
    t0 = float(T0[0, 0])
    tend = float(T0[-1, 0])

    x_jump = 0.5 * (xl + xr)
    ic_params = (float(Rho0[0, 0]), float(U0[0, 0]), float(P0[0, 0]), float(Rho0[0, -1]), float(U0[0, -1]), float(P0[0, -1]))
    left_state = ic_params[:3]
    right_state = ic_params[3:]

    trange = (t0, tend)
    xrange = (xl, xr)

    T_grid = T0
    X_grid = X0

    t_test = T0.reshape(-1)
    x_test = X0.reshape(-1)

    test_labels = jnp.stack([Rho0.reshape(-1), U0.reshape(-1),P0.reshape(-1)], axis=-1)

    print(f"Test grid: Nt={Nt_test}, Nx={Nx_test}")
    print(f"Time domain: {trange}")
    print(f"Space domain: {xrange}")
    print(f"Interface position: {x_jump}")
    print(f"Left state  [rho,u,p]: {left_state}")
    print(f"Right state [rho,u,p]: {right_state}")


    # 2. Random keys
    seed = 42
    key = jax.random.PRNGKey(seed)
    key_pde_init, key_coor_init, key_train = jax.random.split(key, 3)

    # 3. Models
    pde_net = Euler_net(n_nodes=40, fourier_dim=20, activation=nn.silu)
    pde_params = pde_net.init(key_pde_init, jnp.ones((3,), dtype=T0.dtype))
    coor_net = FNN_fourier(layer_sizes=[40, 40, 1], activation=nn.silu, fourier_dim=20)
    coor_params = coor_net.init(key_coor_init, jnp.ones((2,), dtype=T0.dtype))

    # 4. Optimizers
    pde_optimizer = soap(learning_rate=3.0e-3, b1=0.95, b2=0.95, weight_decay=0.01, precondition_frequency=10, precondition_1d=False)
    coor_optimizer = soap(learning_rate=3.0e-3, b1=0.95, b2=0.95, weight_decay=0.01, precondition_frequency=10, precondition_1d=False)

    # 5. Common functions
    pde_loss_grad_fn = create_pde_loss_grad_fn(pde_net, w_ic=100.0, gamma=1.4)
    coor_loss_grad_fn = create_coor_loss_grad_fn(coor_net, lambda_pos=0.1, xi_x_min=0.05)
    eval_fn = create_eval_fn(pde_net)

    # Initial affine coordinate mapping
    identity_map_xi_to_x, identity_coor_fn = create_identity_maps(xl, xr)

    def make_minibatch(map_xi_to_x, coor_fn):
        return create_pde_minibatch_fn(IC_map=IC_vmap, map_xi_to_x=map_xi_to_x, coor_net=coor_fn,
            xL=xl, xR=xr, t0=t0, tend=tend, left_state=left_state, right_state=right_state, x_jump=x_jump)


    # Stage 1: affine coordinate, nu=1e-2
    print("\n========== Stage 1 ==========")
    test_inputs_stage1 = create_test_inputs(t_test, x_test, identity_coor_fn)
    minibatch_stage1 = make_minibatch(identity_map_xi_to_x, identity_coor_fn)

    pde_params, key_train, history_stage1 = train_pde_stage(stage_name="Stage 1", 
    pde_params=pde_params, pde_optimizer=pde_optimizer, pde_loss_grad_fn=pde_loss_grad_fn, pde_minibatch_fn=minibatch_stage1,
    eval_fn=eval_fn, test_inputs=test_inputs_stage1,  test_labels=test_labels,
    key=key_train, nu=1.0e-3, max_iters=10000, pts_pde=10000, pts_ics=1000, max_runtime=10000.0, eval_every=100)
    pde_params_stage1 = pde_params

    plot_training_history(history_stage1, metric="rl2", title="Stage 1 training history")
    plot_evaluation(params=pde_params, model=pde_net, t_test=t_test, x_test=x_test, labels_test=test_labels, coor_fn=identity_coor_fn)

    # Generate first adaptive mesh
    xi_grid, Xi_of_x, X_adapt, map_xi_to_x, solution_stage1 = generate_adaptive_mesh(pde_net=pde_net, pde_params=pde_params, T_grid=T_grid, X_grid=X_grid, coor_fn=identity_coor_fn,
        mesh_method='integral', winslow_tol=1.0e-5, winslow_max_iter=100000, smooth_steps=4)

    # Train first coordinate network
    coor_params, key_train = train_coordinate_net(coor_net=coor_net, coor_params=coor_params, coor_optimizer=coor_optimizer, coor_loss_grad_fn=coor_loss_grad_fn, T_grid=T_grid, X_grid=X_grid, Xi_of_x=Xi_of_x,
        key=key_train, max_iters=10000, batch_size=10000, eval_every=1000)
    coor_params_stage1 = coor_params

    trained_coor_fn = create_coor_fn(coor_net, coor_params)

    plot_coordinate_mapping(T_grid=T_grid, X_grid=X_grid, xi_grid=xi_grid, Xi_of_x=Xi_of_x, X_adapt=X_adapt, coor_fn=trained_coor_fn, n_times=6)


    # Stage 2: first adaptive coordinate, nu=1e-3

    print("\n========== Stage 2 ==========")
    # Stage 2: first adaptive coordinate, nu=1e-3
    test_inputs_stage2 = create_test_inputs(t_test, x_test, trained_coor_fn)
    minibatch_stage2 = make_minibatch(map_xi_to_x, trained_coor_fn)

    pde_params, key_train, history_stage2 = train_pde_stage(stage_name="Stage 2", 
        pde_params=pde_params, pde_optimizer=pde_optimizer, pde_loss_grad_fn=pde_loss_grad_fn, pde_minibatch_fn=minibatch_stage2,
        eval_fn=eval_fn, test_inputs=test_inputs_stage2,  test_labels=test_labels,
        key=key_train, nu=1.0e-4, max_iters=10000, pts_pde=10000, pts_ics=1000, max_runtime=10000.0, eval_every=100)
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
        checkpoint_path="talpinn_euler1d.msgpack",
    )

    plot_training_history(history_stage2, metric="rl2", title="Stage 2 training history",)

    plot_evaluation(params=pde_params, model=pde_net, t_test=t_test, x_test=x_test, labels_test=test_labels, coor_fn=trained_coor_fn)
