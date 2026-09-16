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


@jax.jit
def compute_adaptive_mesh_1d(T, X, U):
    # Gradient on the original physical mesh
    dX = X[:, 1:] - X[:, :-1]
    dU = (U[:, 1:] - U[:, :-1]) / dX

    # Monitor function
    M = jnp.sqrt(1.0 + dU**2)

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
        xi_pred = apply_fn(params, inputs)
        loss_fit = jnp.mean((xi_pred - targets) ** 2)

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


def create_pde_loss_grad_fn(model, w_ic=10.0, w_bc=10.0):
    apply_fn = model.apply

    
    # Single-point lifted solution
    # inputs = [t, x, xi]
    def get_U(params, inputs):
        U = apply_fn(params, inputs)
        return jnp.squeeze(U)

    grad_U = jax.grad(get_U, argnums=1)
    hess_U = jax.hessian(get_U, argnums=1)

    # Single-point PDE residual
    # data = [t, x, xi, xi_t, xi_x, xi_xx]
    
    def residual_single(params, data, nu):
        inputs = data[:3]

        xi_t = data[3]
        xi_x = data[4]
        xi_xx = data[5]

        U = get_U(params, inputs)

        grad = grad_U(params, inputs)

        U_t = grad[0]
        U_x = grad[1]
        U_xi = grad[2]

        H = hess_U(params, inputs)

        U_xx = H[1, 1]
        U_xxi = H[1, 2]
        U_xixi = H[2, 2]

        # Chain rule
        # u(t,x) = U(t,x,xi(t,x))

        u = U
        u_t = U_t + U_xi * xi_t
        u_x = U_x + U_xi * xi_x
        u_xx = U_xx + 2.0 * U_xxi * xi_x + U_xixi * xi_x**2 + U_xi * xi_xx

        # viscous Burgers
        residual = u_t + u * u_x - nu * u_xx

        return residual

    residual_batch = jax.vmap(residual_single, in_axes=(None, 0, None))
    # Total loss
    
    def loss_fn(params, inputs_pde, inputs_ics, targets_ics, inputs_bcs, targets_bcs, nu=1e-3):

        # PDE loss
        residual = residual_batch(params, inputs_pde, nu)

        # importance correction:
        # rho(t,x) ~ |xi_x|
        xi_x = inputs_pde[:, 4]
        weight = jnp.maximum(jnp.abs(xi_x), 1e-8)

        loss_pde = jnp.mean(residual**2 / weight)

        # Initial-condition loss
        # inputs_ics = [t, x, xi]
        pred_ics = apply_fn(params, inputs_ics)
        loss_ics = jnp.mean((pred_ics - targets_ics) ** 2)

        # Dirichlet boundary-condition loss
        # inputs_bcs = [t, x_boundary, xi]
        pred_bcs = apply_fn(params, inputs_bcs)
        loss_bcs = jnp.mean((pred_bcs - targets_bcs) ** 2)

        # Total
        loss = loss_pde + w_ic * loss_ics + w_bc * loss_bcs

        return loss

    return jax.jit(jax.value_and_grad(loss_fn))


def IC_Burgers(x):
    return jnp.sin(2 * jnp.pi * x) + 0.5 * jnp.sin(jnp.pi * x)

def BC_Burgers(t, x):
    return jnp.zeros_like(x)


def create_pde_minibatch_fn(IC_map, BC_map, map_xi_to_x, coor_net):
    """
    IC_map: x -> u(0, x)
    BC_map: (t, x) -> g_bc(t, x)
    map_xi_to_x: (t, xi) -> x(t, xi)
    coor_net: [t, x] -> xi(t, x), scalar output
    """

    # Coordinate-network derivatives
    coor_grad = jax.grad(coor_net)
    coor_hess = jax.hessian(coor_net)

    batch_coor = jax.vmap(coor_net)
    batch_grad = jax.vmap(coor_grad)
    batch_hess = jax.vmap(coor_hess)

    @partial(jax.jit, static_argnames=["pts_pde", "pts_ics", "pts_bcs"])
    def pde_minibatch(key, pts_pde, pts_ics, pts_bcs):
        key_pde, key_ics, key_bcs = jax.random.split(key, 3)

        # 1. PDE collocation points
        # Uniform in computational coordinates (t, xi)
        sample_pde = jax.random.uniform(key_pde, shape=(pts_pde, 2), minval=0.0, maxval=1.0)

        t_pde = sample_pde[:, 0]
        xi_sample = sample_pde[:, 1]

        # (t, xi) -> physical x
        x_pde = map_xi_to_x(t_pde, xi_sample)
        tx_pde = jnp.stack([t_pde, x_pde], axis=-1)

        # xi(t,x)
        xi_pde = batch_coor(tx_pde)

        # [xi_t, xi_x]
        grad_xi = batch_grad(tx_pde)
        xi_t = grad_xi[:, 0]
        xi_x = grad_xi[:, 1]

        # Hessian
        hess_xi = batch_hess(tx_pde)
        xi_xx = hess_xi[:, 1, 1]

        inputs_pde = jnp.stack([t_pde, x_pde, xi_pde, xi_t, xi_x, xi_xx], axis=-1)
        
        # 2. Initial-condition points
        # Uniform in physical x
        x_ics = jax.random.uniform(key_ics, shape=(pts_ics,), minval=0.0, maxval=1.0)
        t_ics = jnp.zeros_like(x_ics)
        tx_ics = jnp.stack([t_ics, x_ics], axis=-1)
        xi_ics = batch_coor(tx_ics)

        inputs_ics = jnp.stack([t_ics, x_ics, xi_ics], axis=-1)
        targets_ics = IC_map(x_ics).reshape(-1, 1)
        
        # 3. Dirichlet boundary-condition points
        # x = 0 and x = 1
        t_bcs = jax.random.uniform(key_bcs, shape=(pts_bcs,), minval=0.0, maxval=1.0)

        # left boundary
        x_left = jnp.zeros_like(t_bcs)
        tx_left = jnp.stack([t_bcs, x_left], axis=-1)
        xi_left = batch_coor(tx_left)

        inputs_left = jnp.stack([t_bcs, x_left, xi_left], axis=-1)
        targets_left = BC_map(t_bcs, x_left).reshape(-1, 1)

        # right boundary
        x_right = jnp.ones_like(t_bcs)
        tx_right = jnp.stack([t_bcs, x_right], axis=-1)
        xi_right = batch_coor(tx_right)

        inputs_right = jnp.stack([t_bcs, x_right, xi_right], axis=-1)
        targets_right = BC_map(t_bcs, x_right).reshape(-1, 1)

        # concatenate two boundaries
        inputs_bcs = jnp.concatenate([inputs_left, inputs_right], axis=0)
        targets_bcs = jnp.concatenate([targets_left, targets_right], axis=0)

        return (inputs_pde, inputs_ics, targets_ics, inputs_bcs, targets_bcs)

    return pde_minibatch



def create_pde_net_update_fn(loss_grad_fn, optimizer, nu):

    @jax.jit
    def update_step(params, opt_state, inputs_pde, inputs_ics, targets_ics, inputs_bcs, targets_bcs):

        # 1. Compute loss and gradients
        loss, grads = loss_grad_fn(params, inputs_pde, inputs_ics, targets_ics, inputs_bcs, targets_bcs, nu)

        # 2. Optimizer update
        updates, new_opt_state = optimizer.update(grads, opt_state, params=params)

        # 3. Apply updates
        new_params = optax.apply_updates(params, updates)

        return (new_params, new_opt_state, loss)

    return update_step


@jax.jit
def identity_map_xi_to_x(t, xi):
    """
    Identity map: (t, xi) -> x = xi
    """
    return xi


@jax.jit
def identity_coor_net(tx):
    """
    Identity coordinate map: (t, x) -> xi = x
    tx = [t, x]
    """
    return tx[1]


def create_eval_fn(model, nx_test=None, interface_idx=None, interface_mode="none"):
    """
    interface_mode:
        "none"    : no special treatment
        "drop"    : exclude the interface point from MSE / RL2
        "average" : replace the interface value by the average
                    of its two neighboring points
    """
    apply_fn = model.apply

    if interface_mode not in ["none", "drop", "average"]:
        raise ValueError("interface_mode must be 'none', 'drop', or 'average'")

    if interface_mode != "none":
        if nx_test is None or interface_idx is None:
            raise ValueError("nx_test and interface_idx must be provided when interface_mode is not 'none'.")

    def eval_error(params, inputs, labels):
        # prediction
        u_pred = apply_fn(params, inputs)

        # flatten
        u_pred = jnp.asarray(u_pred).reshape(-1)
        u_true = jnp.asarray(labels).reshape(-1)

        if interface_mode == "none":
            diff = u_pred - u_true
            mse = jnp.mean(diff**2)
            rl2 = jnp.sqrt(jnp.sum(diff**2) / jnp.maximum(jnp.sum(u_true**2), 1e-12))

        elif interface_mode == "drop":

            u_pred_grid = u_pred.reshape(-1, nx_test)
            u_true_grid = u_true.reshape(-1, nx_test)
            diff = (u_pred_grid - u_true_grid)

            # 1 everywhere, 0 at interface
            mask = jnp.ones((nx_test), dtype=diff.dtype)
            mask = mask.at[interface_idx].set(0.0)

            # broadcast over time
            diff2 = diff**2 * mask[None, :]
            mse = (jnp.sum(diff2) / jnp.maximum(jnp.sum(mask) * diff.shape[0], 1.0))
            numerator = jnp.sum(diff2)
            denominator = jnp.sum(u_true_grid**2 * mask[None, :])
            rl2 = jnp.sqrt(numerator / jnp.maximum(denominator, 1e-12))

        else:
            u_pred_grid = u_pred.reshape(-1, nx_test)
            u_true_grid = u_true.reshape(-1, nx_test)

            pred_avg = 0.5 * (u_pred_grid[:, interface_idx - 1] + u_pred_grid[:, interface_idx + 1])
            true_avg = 0.5 * (u_true_grid[:, interface_idx - 1] + u_true_grid[:, interface_idx + 1])

            u_pred_grid = u_pred_grid.at[:, interface_idx].set(pred_avg)
            u_true_grid = u_true_grid.at[:, interface_idx].set(true_avg)

            u_pred = u_pred_grid.reshape(-1)
            u_true = u_true_grid.reshape(-1)

            diff = u_pred - u_true
            mse = jnp.mean(diff**2)
            rl2 = jnp.sqrt(jnp.sum(diff**2) / jnp.maximum(jnp.sum(u_true**2), 1e-12))

        return mse, rl2

    return jax.jit(eval_error)




def train_pde_stage(stage_name, pde_params, pde_optimizer, pde_loss_grad_fn, pde_minibatch_fn, eval_fn, test_inputs, test_labels, 
    key, nu, max_iters=10000, pts_pde=10000, pts_ics=1000, pts_bcs=1000, max_runtime=10000.0, eval_every=500):
    """
    Train one PDE stage.
    The PDE parameters are inherited from the previous stage, while the optimizer state is reset for each new stage.
    """

    # Reset optimizer state for this stage
    opt_state = pde_optimizer.init(pde_params)
    update_fn = create_pde_net_update_fn(pde_loss_grad_fn, pde_optimizer, nu=nu)

    warmup_key = jax.random.PRNGKey(0)
    warmup_dataset = pde_minibatch_fn(warmup_key, pts_pde=pts_pde, pts_ics=pts_ics, pts_bcs=pts_bcs)

    # Make sure minibatch compilation/execution is finished
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), warmup_dataset)
    
    # Compile update step
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
        mse_value, rl2_value = map(float, jax.device_get((mse, rl2)))

        history["eval_iter"].append(iteration)
        history["eval_train_time"].append(train_time)
        history["mse"].append(mse_value)
        history["rl2"].append(rl2_value)

        return mse_value, rl2_value

    # Record the error before this stage starts. This is especially useful for
    # showing the change at a coordinate-update/stage boundary.
    mse_value, rl2_value = record_evaluation(iteration=0, train_time=0.0)
    print(f"[{stage_name}] iter = 00000, time = 0.00s | mse = {mse_value:.2e}, rl2 = {rl2_value:.2e}")

    for it in range(1, max_iters + 1):
        if runtime >= max_runtime:
            break
        start = time.time()

        key, key_batch = jax.random.split(key)
        dataset = pde_minibatch_fn(key_batch, pts_pde=pts_pde, pts_ics=pts_ics, pts_bcs=pts_bcs)
        (pde_params, opt_state, loss) = update_fn(pde_params, opt_state, *dataset)
        loss.block_until_ready()

        runtime += time.time() - start

        history["iter"].append(it)
        history["loss"].append(float(loss))
        history["train_time"].append(runtime)

        if it % eval_every == 0:
            mse_value, rl2_value = record_evaluation(iteration=it, train_time=runtime)

            print(f"[{stage_name}] iter = {it:05d}, time = {runtime:.2f}s, loss = {float(loss):.2e} | mse = {mse_value:.2e}, rl2 = {rl2_value:.2e}")

    # If max_runtime stops a stage between two scheduled evaluations, retain
    # the error corresponding to the actual final parameters as well.
    if history["iter"]:
        final_iter = history["iter"][-1]
        if history["eval_iter"][-1] != final_iter:
            mse_value, rl2_value = record_evaluation(iteration=final_iter, train_time=runtime)
            print(f"[{stage_name}] final iter = {final_iter:05d}, time = {runtime:.2f}s | mse = {mse_value:.2e}, rl2 = {rl2_value:.2e}")

    history["iter"] = np.asarray(history["iter"])
    history["loss"] = np.asarray(history["loss"])
    history["train_time"] = np.asarray(history["train_time"])
    history["eval_iter"] = np.asarray(history["eval_iter"])
    history["eval_train_time"] = np.asarray(history["eval_train_time"])
    history["mse"] = np.asarray(history["mse"])
    history["rl2"] = np.asarray(history["rl2"])

    return pde_params, key, history


def plot_training_history(history, metric="rl2", title="Training history"):
    # Loss history
    iters = np.asarray(history["iter"])
    loss = np.asarray(history["loss"])
    loss_safe = np.maximum(loss, 1e-30)
    ln_loss = np.log(loss_safe)

    # Evaluation history
    eval_iters = np.asarray(history["eval_iter"])

    if metric == "rl2":
        metric_values = np.asarray(history["rl2"])
        metric_safe = np.maximum(metric_values, 1e-30)
        ln_metric = np.log(metric_safe)
        metric_label = r"Relative $L^2$ error"
        metric_title = r"Relative $L^2$ error"

    elif metric == "mse":
        metric_values = np.asarray(history["mse"])
        metric_safe = np.maximum(metric_values, 1e-30)
        ln_metric = np.log(metric_safe)
        metric_label = "MSE"
        metric_title = "Mean squared error"

    else:
        raise ValueError("metric must be 'rl2' or 'mse'")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),)

    # ln(loss)
    axes[0].plot(iters, ln_loss, linewidth=1.5)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$\ln(\mathcal{L})$")
    axes[0].set_title(r"Training loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # RL2 / MSE
    axes[1].plot(eval_iters, ln_metric, marker="o", linewidth=1.5)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(metric_label)
    axes[1].set_title(metric_title)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()



def generate_adaptive_mesh(pde_net, pde_params, T_grid, X_grid, coor_fn):

    start = time.perf_counter()
    # Evaluate current solution on uniform physical grid
    t_flat = T_grid.reshape(-1)
    x_flat = X_grid.reshape(-1)
    tx = jnp.stack([t_flat, x_flat], axis=-1)

    # Current lifting coordinate
    xi_flat = jax.vmap(coor_fn)(tx)
    inputs = jnp.stack([t_flat, x_flat, xi_flat], axis=-1)
    U_coarse = pde_net.apply(pde_params, inputs).reshape(T_grid.shape)
    U_coarse.block_until_ready()


    # Adaptive mesh
    (xi_grid, Xi_of_x,  X_adapt) = compute_adaptive_mesh_1d(T_grid, X_grid, U_coarse)
    X_adapt.block_until_ready()

    # Continuous map:
    map_xi_to_x = create_map_xi_to_x(T_grid, xi_grid, X_adapt)

    runtime_ms = (time.perf_counter() - start)
    print(f"[Adaptive Mesh] time = {runtime_ms:.2f} s")

    return (xi_grid, Xi_of_x, X_adapt, map_xi_to_x, U_coarse)


def train_coordinate_net(coor_net, coor_params, coor_optimizer, coor_loss_grad_fn, T_grid, X_grid, Xi_of_x,
    key, max_iters=5000, batch_size=4096, eval_every=500):

    # Training dataset
    inputs = jnp.stack([T_grid.reshape(-1), X_grid.reshape(-1)], axis=-1)
    targets = Xi_of_x.reshape(-1, 1)

    # Optimizer
    opt_state = coor_optimizer.init(coor_params)
    update_fn = create_coor_net_update_fn(coor_loss_grad_fn, coor_optimizer)

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
        start = time.time()

        key, key_batch = jax.random.split(key)
        inputs_batch, targets_batch = coor_minibatch(key_batch, inputs, targets, pts=batch_size)
        (coor_params, opt_state, loss) = update_fn(coor_params, opt_state, inputs_batch, targets_batch)
        loss.block_until_ready()

        runtime += time.time() - start

        # Full-grid evaluation
        if it % eval_every == 0:
            pred = coor_net.apply(coor_params, inputs)
            mse = jnp.mean((pred - targets) ** 2)
            mse.block_until_ready()
            print(f"[Coordinate] iter = {it:05d}, time = {runtime:.2f}s, batch loss = {float(loss):.2e}, full mse = {float(mse):.2e}")

    return coor_params, key


def create_coor_fn(coor_net, coor_params):
    def coor_fn(tx):
        return jnp.squeeze(coor_net.apply(coor_params, tx))

    return coor_fn



def create_test_inputs(t_test, x_test, coor_fn):

    tx = jnp.stack([t_test, x_test], axis=-1)
    xi_test = jax.vmap(coor_fn)(tx)

    return jnp.stack([t_test, x_test, xi_test], axis=-1)



def plot_evaluation(params, model, t_test, x_test, labels_test, coor_fn, interface_x=None, interface_tol=1e-12, 
    metric_interface_mode="none", plot_interface_mode="none", cmap="jet"):
    """
    metric_interface_mode:
        "none"
        "drop"
        "average"

    plot_interface_mode:
        "none"
        "zero"
        "average"
    """

    # 1. Prepare test inputs
    t_test_jax = jnp.asarray(t_test).reshape(-1)
    x_test_jax = jnp.asarray(x_test).reshape(-1)
    tx_test = jnp.stack([t_test_jax, x_test_jax], axis=-1)

    batch_coor = jax.vmap(coor_fn)

    @jax.jit
    def predict_fn(params, tx):
        xi = batch_coor(tx)
        inputs = jnp.stack([tx[:, 0], tx[:, 1], xi], axis=-1)

        return model.apply(params, inputs)

    u_pred = predict_fn(params, tx_test)
    u_pred.block_until_ready()

    # 2. Convert to NumPy
    u_pred_np = np.asarray(u_pred).reshape(-1)
    t_np = np.asarray(t_test_jax).reshape(-1)
    x_np = np.asarray(x_test_jax).reshape(-1)
    u_true_np = np.asarray(labels_test).reshape(-1)

    # 3. Resolve structured (t, x) grid
    sort_idx = np.lexsort((x_np, t_np))
    
    t_sorted = t_np[sort_idx]
    x_sorted = x_np[sort_idx]

    u_pred_sorted = u_pred_np[sort_idx]
    u_true_sorted = u_true_np[sort_idx]

    t_unique = np.unique(t_sorted)
    x_unique = np.unique(x_sorted)

    nt_test = len(t_unique)
    nx_test = len(x_unique)

    if nt_test * nx_test != len(t_sorted):
        raise ValueError("The supplied test points do not form a complete rectangular (t, x) grid.")

    print(f"Plotting... Test grid shape: nt={nt_test}, nx={nx_test}")

    gt_u_raw = u_true_sorted.reshape(nt_test, nx_test)
    u_raw = u_pred_sorted.reshape(nt_test, nx_test)

    # 4. Detect interface
    interface_idx = None

    if interface_x is not None:
        interface_mask = np.isclose(x_unique, interface_x, atol=interface_tol, rtol=0.0)
        if np.any(interface_mask):
            interface_idx = int(np.where(interface_mask)[0][0])
            print(f"Interface detected at x = {x_unique[interface_idx]:.8f}, index = {interface_idx}")

        else:
            print(f"No grid point detected at x={interface_x}.")

    # 5. Error metrics
    u_metric = u_raw.copy()
    gt_metric = gt_u_raw.copy()

    if (interface_idx is not None and metric_interface_mode == "average"):
        u_metric[:, interface_idx] = 0.5 * (u_metric[:, interface_idx - 1] + u_metric[:, interface_idx + 1])
        gt_metric[:, interface_idx] = 0.5 * (gt_metric[:, interface_idx - 1] + gt_metric[:, interface_idx + 1])

    diff_metric = (u_metric - gt_metric)

    if (interface_idx is not None and metric_interface_mode == "drop"):
        mask = np.ones(nx_test, dtype=bool)
        mask[interface_idx] = False
        diff_eval = diff_metric[:, mask]
        gt_eval = gt_metric[:, mask]

    else:
        diff_eval = diff_metric
        gt_eval = gt_metric

    mse = np.mean(diff_eval**2)
    denominator = max(np.sum(gt_eval**2), 1e-12)
    rl2 = np.sqrt(np.sum(diff_eval**2) / denominator)

    print(f"MSE = {mse:.6e}\n RL2 = {rl2:.6e}")

    # 6. Prepare plotting data
    u = u_raw.copy()
    # Reference remains untouched by default
    gt_u = gt_u_raw.copy()

    if (interface_idx is not None and plot_interface_mode == "zero"):
        # Same convention as your old plotting code
        u[:, interface_idx] = 0.0

    elif (interface_idx is not None and plot_interface_mode == "average"):
        u[:, interface_idx] = 0.5 * (u[:, interface_idx - 1] + u[:, interface_idx + 1])
        gt_u[:, interface_idx] = 0.5 * (gt_u[:, interface_idx - 1] + gt_u[:, interface_idx + 1])

    abs_err = np.abs(u - gt_u)

    # Old plotting convention:
    # do not display interface-point error
    if (interface_idx is not None and plot_interface_mode == "zero"):
        abs_err[:, interface_idx] = 0.0

    # 7. Final-time profiles
    gt_u_1d = gt_u[-1, :]
    u_1d = u[-1, :]
    abs_err_1d = abs_err[-1, :]
    tend = t_unique[-1]

    # 8. Common contour ranges
    con_lv = 101

    sol_min = min(np.nanmin(gt_u), np.nanmin(u))
    sol_max = max(np.nanmax(gt_u), np.nanmax(u))

    if sol_max <= sol_min:
        sol_max = sol_min + 1e-12

    sol_levels = np.linspace(sol_min, sol_max, con_lv)

    err_vmax = float(np.nanmax(abs_err))

    if err_vmax <= 0.0:
        err_vmax = 1e-12

    err_levels = np.linspace(0.0, err_vmax, con_lv)

    # 9. Plot
    fig = plt.figure(figsize=(16, 9))

    # Reference solution
    ax1 = fig.add_subplot(2, 3, 1)
    clv1 = ax1.contourf(x_unique, t_unique, gt_u, levels=sol_levels, cmap=cmap)
    fig.colorbar(clv1, ax=ax1)
    ax1.set_title("Reference solution")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$t$")

    # Predicted solution
    ax2 = fig.add_subplot(2, 3, 2)
    clv2 = ax2.contourf(x_unique, t_unique, u, levels=sol_levels, cmap=cmap)
    fig.colorbar(clv2, ax=ax2)
    ax2.set_title("Predicted solution")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$t$")

    # Absolute error
    ax3 = fig.add_subplot(2, 3, 3)
    clv_err = ax3.contourf(x_unique, t_unique, abs_err, levels=err_levels, cmap=cmap)
    err_ticks = [0.0, 0.5 * err_vmax, err_vmax]
    cbar_err = fig.colorbar(clv_err, ax=ax3, ticks=err_ticks)
    cbar_err.set_label(r"$|u_{\mathrm{pred}}-u_{\mathrm{ref}}|$")
    cbar_err.ax.set_yticklabels([r"$0$", f"{0.5 * err_vmax:.2e}", f"{err_vmax:.2e}"])
    ax3.set_title("Absolute error")
    ax3.set_xlabel(r"$x$")
    ax3.set_ylabel(r"$t$")

    # Reference profile
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(x_unique, gt_u_1d, linewidth=2, label="Reference")
    ax4.set_title(rf"Reference profile at $t={tend:.2f}$")
    ax4.set_xlabel(r"$x$")
    ax4.set_ylabel(r"$u$")
    ax4.legend(loc="best")
    ax4.grid(True, linestyle=":", alpha=0.6)

    # Predicted profile
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(x_unique, u_1d, linewidth=2, linestyle="--", label="Prediction")
    ax5.set_title(rf"Predicted profile at $t={tend:.2f}$")
    ax5.set_xlabel(r"$x$")
    ax5.set_ylabel(r"$u$")
    ax5.legend(loc="best")
    ax5.grid(True, linestyle=":", alpha=0.6)

    # Absolute-error profile
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(x_unique, abs_err_1d, linewidth=1.5, label="Absolute error")
    ax6.set_title(rf"Absolute error at $t={tend:.2f}$")
    ax6.set_xlabel(r"$x$")
    ax6.set_ylabel(r"$|u_{\mathrm{pred}}-u_{\mathrm{ref}}|$")
    ax6.legend(loc="best")
    ax6.grid(True, linestyle=":", alpha=0.6)

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
    checkpoint_path="checkpoint.msgpack", history_path=None):
    """
    Save all TAL-PINN stage parameters and PDE training histories.

    The msgpack checkpoint contains everything needed to recover the three PDE
    parameter sets, the two coordinate-network parameter sets, and all three
    histories. A second compressed NPZ file stores the histories in a format
    that can be loaded directly by NumPy for plotting and post-processing.

    ``coor_params["stage1"]`` is the coordinate network learned after PDE
    stage 1 and used by PDE stage 2; ``coor_params["stage2"]`` is learned after
    PDE stage 2 and used by PDE stage 3.
    """
    expected_pde_stages = ("stage1", "stage2", "stage3")
    expected_coor_stages = ("stage1", "stage2")
    history_keys = (
        "iter", "loss", "train_time",
        "eval_iter", "eval_train_time", "mse", "rl2",
    )

    if set(pde_params_by_stage) != set(expected_pde_stages):
        raise ValueError(f"pde_params_by_stage keys must be {expected_pde_stages}")
    if set(coor_params_by_stage) != set(expected_coor_stages):
        raise ValueError(f"coor_params_by_stage keys must be {expected_coor_stages}")
    if set(pde_histories_by_stage) != set(expected_pde_stages):
        raise ValueError(f"pde_histories_by_stage keys must be {expected_pde_stages}")

    histories = {}
    for stage_name in expected_pde_stages:
        history = pde_histories_by_stage[stage_name]
        missing_keys = [name for name in history_keys if name not in history]
        if missing_keys:
            raise ValueError(f"{stage_name} history is missing keys: {missing_keys}")
        histories[stage_name] = {
            name: np.asarray(history[name]) for name in history_keys
        }

    # Move device arrays to the host before serialization and wait for any
    # outstanding JAX computation to finish.
    checkpoint = {
        "format_version": 1,
        "pde_params": jax.device_get({
            name: pde_params_by_stage[name] for name in expected_pde_stages
        }),
        "coor_params": jax.device_get({
            name: coor_params_by_stage[name] for name in expected_coor_stages
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
    """Load the complete checkpoint saved by ``save_training_results``."""
    with open(checkpoint_path, "rb") as f:
        checkpoint = serialization.msgpack_restore(f.read())

    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format_version')}")

    # msgpack_restore returns NumPy arrays; convert parameter leaves back to
    # JAX arrays so the returned PyTrees are immediately ready for model.apply.
    checkpoint["pde_params"] = jax.tree_util.tree_map(
        jnp.asarray, checkpoint["pde_params"]
    )
    checkpoint["coor_params"] = jax.tree_util.tree_map(
        jnp.asarray, checkpoint["coor_params"]
    )

    print(f"Training checkpoint loaded from: {checkpoint_path}")
    return checkpoint


def load_training_histories(history_path):
    """Load the three PDE histories from the compressed NumPy archive."""
    stage_names = ("stage1", "stage2", "stage3")
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



def main():
    # 1. Data
    data = scipy.io.loadmat("burgers_godunov.mat")

    U0 = jnp.asarray(data["U"])
    X0 = jnp.asarray(data["X"])
    T0 = jnp.asarray(data["T"])

    Nt_test, Nx_test = U0.shape

    T_grid = T0
    X_grid = X0

    t_test = T0.reshape(-1)
    x_test = X0.reshape(-1)
    test_labels = U0.reshape(-1, 1)

    # 2. Random keys
    seed = 42
    key = jax.random.PRNGKey(seed)
    (key_pde_init, key_coor_init, key_train) = jax.random.split(key, 3)

    # 3. Models
    layer_sizes = [40, 40, 1]

    pde_net = FNN_fourier(layer_sizes, fourier_dim=20)
    pde_params = pde_net.init(key_pde_init, jnp.ones((3,)))

    coor_net = FNN_fourier(layer_sizes, fourier_dim=20)
    coor_params = coor_net.init(key_coor_init, jnp.ones((2,)))

    # 4. Optimizers
    pde_optimizer = soap(learning_rate=3e-3, b1=0.95, b2=0.95, weight_decay=0.01, precondition_frequency=10, precondition_1d=False)
    coor_optimizer = soap(learning_rate=3e-3, b1=0.95, b2=0.95, weight_decay=0.01, precondition_frequency=10, precondition_1d=False)

    # 5. Common functions
    pde_loss_grad_fn = create_pde_loss_grad_fn(pde_net)
    coor_loss_grad_fn = create_coor_loss_grad_fn(coor_net, lambda_pos=0.01, xi_x_min=0.05)
    eval_fn = create_eval_fn(pde_net, nx_test=Nx_test)

    
    # Stage 1
    test_inputs_stage1 = create_test_inputs(t_test, x_test, identity_coor_net)
    minibatch_stage1 = create_pde_minibatch_fn(IC_Burgers, BC_Burgers, identity_map_xi_to_x, identity_coor_net)

    pde_params, key_train, history_stage1 = train_pde_stage(stage_name="Stage 1", 
            pde_params=pde_params, pde_optimizer=pde_optimizer, pde_loss_grad_fn=pde_loss_grad_fn, pde_minibatch_fn=minibatch_stage1,
            eval_fn=eval_fn, test_inputs=test_inputs_stage1, test_labels=test_labels, 
            key=key_train, nu=1e-2, max_iters=2000, pts_pde=10000, pts_ics=1000, pts_bcs=1000, max_runtime=10000.0, eval_every=100)

    # JAX parameter PyTrees are immutable, so retaining the reference here is a
    # valid snapshot even though ``pde_params`` is rebound in later stages.
    pde_params_stage1 = pde_params

    plot_training_history(history_stage1, metric="rl2", title="Stage 1 training history")
    plot_evaluation(params=pde_params, model=pde_net, t_test=t_test, x_test=x_test, labels_test=test_labels, coor_fn=identity_coor_net)

    # Adaptive mesh
    (xi_grid, Xi_of_x, X_adapt, map_xi_to_x, U_stage1) = generate_adaptive_mesh(pde_net, pde_params, T_grid, X_grid, identity_coor_net)

    # Coordinate network
    coor_params, key_train = train_coordinate_net(coor_net=coor_net, coor_params=coor_params, coor_optimizer=coor_optimizer, coor_loss_grad_fn=coor_loss_grad_fn,
        T_grid=T_grid, X_grid=X_grid, Xi_of_x=Xi_of_x, key=key_train, max_iters=10000, batch_size=10000, eval_every=1000)

    coor_params_stage1 = coor_params

    trained_coor_net = create_coor_fn(coor_net,coor_params)

    plot_coordinate_mapping(T_grid=T_grid, X_grid=X_grid, xi_grid=xi_grid, Xi_of_x=Xi_of_x, X_adapt=X_adapt, coor_fn=trained_coor_net, n_times=6)

    # Stage 2
    test_inputs_stage2 = create_test_inputs(t_test, x_test, trained_coor_net)
    minibatch_stage2 = create_pde_minibatch_fn(IC_Burgers, BC_Burgers, map_xi_to_x, trained_coor_net)

    pde_params, key_train, history_stage2 = train_pde_stage(stage_name="Stage 2",
        pde_params=pde_params, pde_optimizer=pde_optimizer, pde_loss_grad_fn=pde_loss_grad_fn, pde_minibatch_fn=minibatch_stage2,
        eval_fn=eval_fn, test_inputs=test_inputs_stage2, test_labels=test_labels,
        key=key_train, nu=1e-3, max_iters=2000, pts_pde=10000, pts_ics=1000, pts_bcs=1000, max_runtime=10000.0, eval_every=100)

    pde_params_stage2 = pde_params

    plot_training_history(history_stage2, metric="rl2", title="Stage 2 training history")
    plot_evaluation(params=pde_params, model=pde_net, t_test=t_test, x_test=x_test, labels_test=test_labels, coor_fn=trained_coor_net)


    # Adaptive mesh
    (xi_grid, Xi_of_x, X_adapt, map_xi_to_x, U_stage2) = generate_adaptive_mesh(pde_net, pde_params, T_grid, X_grid, trained_coor_net)

    # Coordinate network
    coor_params, key_train = train_coordinate_net(coor_net=coor_net, coor_params=coor_params, coor_optimizer=coor_optimizer, coor_loss_grad_fn=coor_loss_grad_fn,
        T_grid=T_grid, X_grid=X_grid, Xi_of_x=Xi_of_x, key=key_train, max_iters=10000, batch_size=10000, eval_every=1000)

    coor_params_stage2 = coor_params

    trained_coor_net = create_coor_fn(coor_net,coor_params)

    plot_coordinate_mapping(T_grid=T_grid, X_grid=X_grid, xi_grid=xi_grid, Xi_of_x=Xi_of_x, X_adapt=X_adapt, coor_fn=trained_coor_net, n_times=6)

    # Stage 3
    test_inputs_stage3 = create_test_inputs(t_test, x_test, trained_coor_net)
    minibatch_stage3 = create_pde_minibatch_fn(IC_Burgers, BC_Burgers, map_xi_to_x, trained_coor_net)

    pde_params, key_train, history_stage3 = train_pde_stage(stage_name="Stage 3",
        pde_params=pde_params, pde_optimizer=pde_optimizer, pde_loss_grad_fn=pde_loss_grad_fn, pde_minibatch_fn=minibatch_stage3,
        eval_fn=eval_fn, test_inputs=test_inputs_stage3, test_labels=test_labels,
        key=key_train, nu=1e-4, max_iters=4000, pts_pde=10000, pts_ics=1000, pts_bcs=1000, max_runtime=10000.0, eval_every=100)

    pde_params_stage3 = pde_params

    save_training_results(
        pde_params_by_stage={
            "stage1": pde_params_stage1,
            "stage2": pde_params_stage2,
            "stage3": pde_params_stage3,
        },
        coor_params_by_stage={
            "stage1": coor_params_stage1,
            "stage2": coor_params_stage2,
        },
        pde_histories_by_stage={
            "stage1": history_stage1,
            "stage2": history_stage2,
            "stage3": history_stage3,
        },
        checkpoint_path="talpinn_burgers_moving.msgpack",
    )

    plot_training_history(history_stage3, metric="rl2", title="Stage 3 training history")
    plot_evaluation(params=pde_params, model=pde_net, t_test=t_test, x_test=x_test, labels_test=test_labels, coor_fn=trained_coor_net)
