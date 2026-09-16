import os
import time
from functools import partial
from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import scipy.io
from flax import serialization
from soap_jax import soap


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

        return nn.Dense(
            self.layer_sizes[-1],
            kernel_init=kinit,
            use_bias=self.out_bias,
        )(x)


class Euler_net(nn.Module):
    """Fourier-feature network for primitive variables ``(rho, u, p)``."""

    n_nodes: int = 40
    fourier_dim: int = 40
    activation: Callable = nn.silu

    def setup(self):
        self.feature_map = nn.Dense(
            self.fourier_dim,
            kernel_init=jax.nn.initializers.normal(stddev=1.0),
        )

        self.trunk = FNN(
            layer_sizes=[self.n_nodes, self.n_nodes],
            activation=self.activation,
        )

        branch_sizes = [self.n_nodes, self.n_nodes, 1]
        self.branch_rho = FNN(
            layer_sizes=branch_sizes,
            activation=self.activation,
            out_bias=False,
        )
        self.branch_u = FNN(
            layer_sizes=branch_sizes,
            activation=self.activation,
            out_bias=False,
        )
        self.branch_p = FNN(
            layer_sizes=branch_sizes,
            activation=self.activation,
            out_bias=False,
        )

    def __call__(self, inputs):
        # ``inputs`` contains only the physical coordinates [t, x].
        features = self.feature_map(inputs)
        features = jnp.sin(2.0 * jnp.pi * features)
        hidden = self.trunk(features)

        rho_raw = self.branch_rho(hidden)[..., 0]
        velocity = self.branch_u(hidden)[..., 0]
        p_raw = self.branch_p(hidden)[..., 0]

        # Preserve positivity of density and pressure.
        rho = jax.nn.softplus(rho_raw) + 1.0e-5
        pressure = jax.nn.softplus(p_raw) + 1.0e-5

        return rho, velocity, pressure


def create_pde_loss_grad_fn(model, w_ic=10.0, gamma=1.4):
    """
    构造一维 Euler 方程的 vanilla PINN 损失。

    网络只接收原始物理坐标：

        (t, x) -> (rho, u, p).

    守恒变量和通量分别为

        Q = [rho, rho*u, E],
        F = [rho*u, rho*u**2 + p, u*(E+p)],
        E = 0.5*rho*u**2 + p/(gamma-1).

    沿用原程序中的人工黏性形式：

        Q_t + F_x - nu Q_xx = 0.

    原文件没有单独的边界损失，因此这里保留相同的 PDE + IC 结构。
    """
    apply_fn = model.apply

    def get_primitive(params, tx):
        rho, velocity, pressure = apply_fn(params, tx)
        return jnp.stack(
            [
                jnp.squeeze(rho),
                jnp.squeeze(velocity),
                jnp.squeeze(pressure),
            ]
        )

    def get_Q(params, tx):
        rho, velocity, pressure = get_primitive(params, tx)
        momentum = rho * velocity
        energy = 0.5 * rho * velocity**2 + pressure / (gamma - 1.0)
        return jnp.stack([rho, momentum, energy])

    def get_F(params, tx):
        rho, velocity, pressure = get_primitive(params, tx)
        energy = 0.5 * rho * velocity**2 + pressure / (gamma - 1.0)
        return jnp.stack(
            [
                rho * velocity,
                rho * velocity**2 + pressure,
                velocity * (energy + pressure),
            ]
        )

    jac_Q = jax.jacfwd(get_Q, argnums=1)
    jac_F = jax.jacfwd(get_F, argnums=1)
    hess_Q = jax.jacfwd(jac_Q, argnums=1)

    def residual_single(params, tx, nu):
        # Shapes:
        #   JQ, JF: (3, 2), with coordinate order [t, x]
        #   HQ:     (3, 2, 2)
        JQ = jac_Q(params, tx)
        JF = jac_F(params, tx)
        HQ = hess_Q(params, tx)

        q_t = JQ[:, 0]
        f_x = JF[:, 1]
        q_xx = HQ[:, 1, 1]

        return q_t + f_x - nu * q_xx

    residual_batch = jax.vmap(
        residual_single,
        in_axes=(None, 0, None),
    )
    primitive_batch = jax.vmap(
        get_primitive,
        in_axes=(None, 0),
    )

    def loss_fn(
        params,
        inputs_pde,
        inputs_ics,
        targets_ics,
        nu=1.0e-4,
    ):
        residual = residual_batch(params, inputs_pde, nu)
        loss_pde = jnp.mean(jnp.sum(residual**2, axis=-1))

        pred_ics = primitive_batch(params, inputs_ics)
        loss_ics = jnp.mean(
            jnp.sum((pred_ics - targets_ics) ** 2, axis=-1)
        )

        return loss_pde + w_ic * loss_ics

    return jax.jit(jax.value_and_grad(loss_fn))


def IC_Riemann_1D_single(
    x,
    x_jump,
    crhoL,
    cuL,
    cpL,
    crhoR,
    cuR,
    cpR,
):
    rho = jnp.where(
        x < x_jump,
        crhoL,
        jnp.where(x > x_jump, crhoR, 0.5 * (crhoL + crhoR)),
    )
    velocity = jnp.where(
        x < x_jump,
        cuL,
        jnp.where(x > x_jump, cuR, 0.5 * (cuL + cuR)),
    )
    pressure = jnp.where(
        x < x_jump,
        cpL,
        jnp.where(x > x_jump, cpR, 0.5 * (cpL + cpR)),
    )
    return rho, velocity, pressure


IC_vmap = jax.vmap(
    IC_Riemann_1D_single,
    in_axes=(0, None, None, None, None, None, None, None),
)


def create_pde_minibatch_fn(
    IC_map,
    *,
    x_bounds,
    t_bounds,
    left_state,
    right_state,
    x_jump=None,
    dtype=jnp.float32,
):
    """在原始物理时空区域 ``(t,x)`` 中均匀采样 PDE 和初值点。"""
    xL = jnp.asarray(x_bounds[0], dtype=dtype)
    xR = jnp.asarray(x_bounds[1], dtype=dtype)
    t0 = jnp.asarray(t_bounds[0], dtype=dtype)
    tend = jnp.asarray(t_bounds[1], dtype=dtype)

    crhoL, cuL, cpL = (
        jnp.asarray(value, dtype=dtype) for value in left_state
    )
    crhoR, cuR, cpR = (
        jnp.asarray(value, dtype=dtype) for value in right_state
    )

    if x_jump is None:
        x_jump = 0.5 * (xL + xR)
    else:
        x_jump = jnp.asarray(x_jump, dtype=dtype)

    @partial(jax.jit, static_argnames=("pts_pde", "pts_ics"))
    def pde_minibatch(key, pts_pde, pts_ics):
        key_pde, key_ics = jax.random.split(key, 2)

        # PDE points sampled uniformly in the physical (t, x) domain.
        sample_pde = jax.random.uniform(
            key_pde,
            shape=(pts_pde, 2),
            dtype=dtype,
        )
        t_pde = t0 + (tend - t0) * sample_pde[:, 0]
        x_pde = xL + (xR - xL) * sample_pde[:, 1]
        inputs_pde = jnp.stack([t_pde, x_pde], axis=-1)

        # Initial-condition points sampled uniformly in x.
        sample_ics = jax.random.uniform(
            key_ics,
            shape=(pts_ics,),
            dtype=dtype,
        )
        x_ics = xL + (xR - xL) * sample_ics
        t_ics = jnp.full_like(x_ics, t0)
        inputs_ics = jnp.stack([t_ics, x_ics], axis=-1)

        rho_ics, velocity_ics, pressure_ics = IC_map(
            x_ics,
            x_jump,
            crhoL,
            cuL,
            cpL,
            crhoR,
            cuR,
            cpR,
        )
        targets_ics = jnp.stack(
            [rho_ics, velocity_ics, pressure_ics],
            axis=-1,
        )

        return inputs_pde, inputs_ics, targets_ics

    return pde_minibatch


def create_pde_net_update_fn(loss_grad_fn, optimizer, nu):
    @jax.jit
    def update_step(
        params,
        opt_state,
        inputs_pde,
        inputs_ics,
        targets_ics,
    ):
        loss, grads = loss_grad_fn(
            params,
            inputs_pde,
            inputs_ics,
            targets_ics,
            nu,
        )
        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            params=params,
        )
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    return update_step


def create_eval_fn(model):
    """计算 ``rho``、``u`` 和 ``p`` 各自的 MSE 与相对 L2 误差。"""
    apply_fn = model.apply

    def eval_error(params, inputs, labels):
        rho, velocity, pressure = apply_fn(params, inputs)
        predictions = jnp.stack(
            [
                jnp.asarray(rho).reshape(-1),
                jnp.asarray(velocity).reshape(-1),
                jnp.asarray(pressure).reshape(-1),
            ],
            axis=-1,
        )
        labels = jnp.asarray(labels).reshape(-1, 3)
        diff = predictions - labels

        mse = jnp.mean(diff**2, axis=0)
        numerator = jnp.sum(diff**2, axis=0)
        denominator = jnp.sum(labels**2, axis=0)
        rl2 = jnp.sqrt(
            numerator / jnp.maximum(denominator, 1.0e-12)
        )
        return mse, rl2

    return jax.jit(eval_error)


def train_pinn(
    pde_params,
    pde_optimizer,
    pde_loss_grad_fn,
    pde_minibatch_fn,
    eval_fn,
    test_inputs,
    test_labels,
    key,
    nu,
    max_iters=10000,
    pts_pde=10000,
    pts_ics=1000,
    max_runtime=10000.0,
    eval_every=500,
):
    """单阶段训练 vanilla Euler PINN；预热和评估不计入训练时间。"""
    if eval_every <= 0:
        raise ValueError("eval_every must be positive.")

    opt_state = pde_optimizer.init(pde_params)
    update_fn = create_pde_net_update_fn(
        pde_loss_grad_fn,
        pde_optimizer,
        nu=nu,
    )

    # JIT warm-up is excluded from training time.
    warmup_key = jax.random.PRNGKey(0)
    warmup_dataset = pde_minibatch_fn(
        warmup_key,
        pts_pde=pts_pde,
        pts_ics=pts_ics,
    )
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready(),
        warmup_dataset,
    )

    _, _, warmup_loss = update_fn(
        pde_params,
        opt_state,
        *warmup_dataset,
    )
    warmup_loss.block_until_ready()
    print("[Vanilla Euler PINN] JIT warm-up finished.")

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

    mse_value, rl2_value = record_evaluation(
        iteration=0,
        train_time=0.0,
    )
    print(
        "[Vanilla Euler PINN] iter=00000, time=0.00s | "
        f"MSE(rho,u,p)=({mse_value[0]:.2e}, "
        f"{mse_value[1]:.2e}, {mse_value[2]:.2e}) | "
        f"RL2(rho,u,p)=({rl2_value[0]:.2e}, "
        f"{rl2_value[1]:.2e}, {rl2_value[2]:.2e})"
    )

    for it in range(1, max_iters + 1):
        if runtime >= max_runtime:
            break

        start = time.perf_counter()

        key, key_batch = jax.random.split(key)
        dataset = pde_minibatch_fn(
            key_batch,
            pts_pde=pts_pde,
            pts_ics=pts_ics,
        )
        pde_params, opt_state, loss = update_fn(
            pde_params,
            opt_state,
            *dataset,
        )
        loss_value = float(loss.block_until_ready())

        # Sampling and optimization are counted; evaluation below is not.
        runtime += time.perf_counter() - start

        history["iter"].append(it)
        history["loss"].append(loss_value)
        history["train_time"].append(runtime)

        if it % eval_every == 0:
            mse_value, rl2_value = record_evaluation(
                iteration=it,
                train_time=runtime,
            )

            print(
                f"[Vanilla Euler PINN] iter={it:05d}, "
                f"time={runtime:.2f}s, loss={loss_value:.2e} | "
                f"MSE(rho,u,p)=({mse_value[0]:.2e}, "
                f"{mse_value[1]:.2e}, {mse_value[2]:.2e}) | "
                f"RL2(rho,u,p)=({rl2_value[0]:.2e}, "
                f"{rl2_value[1]:.2e}, {rl2_value[2]:.2e})"
            )

    if history["iter"]:
        final_iter = history["iter"][-1]
        if history["eval_iter"][-1] != final_iter:
            mse_value, rl2_value = record_evaluation(
                iteration=final_iter,
                train_time=runtime,
            )
            print(
                f"[Vanilla Euler PINN] final iter={final_iter:05d}, "
                f"time={runtime:.2f}s | "
                f"MSE(rho,u,p)=({mse_value[0]:.2e}, "
                f"{mse_value[1]:.2e}, {mse_value[2]:.2e}) | "
                f"RL2(rho,u,p)=({rl2_value[0]:.2e}, "
                f"{rl2_value[1]:.2e}, {rl2_value[2]:.2e})"
            )

    history["iter"] = np.asarray(history["iter"], dtype=np.int32)
    history["loss"] = np.asarray(history["loss"], dtype=np.float64)
    history["train_time"] = np.asarray(
        history["train_time"],
        dtype=np.float64,
    )
    history["eval_iter"] = np.asarray(
        history["eval_iter"],
        dtype=np.int32,
    )
    history["eval_train_time"] = np.asarray(
        history["eval_train_time"],
        dtype=np.float64,
    )
    history["mse"] = np.asarray(
        history["mse"],
        dtype=np.float64,
    ).reshape(-1, 3)
    history["rl2"] = np.asarray(
        history["rl2"],
        dtype=np.float64,
    ).reshape(-1, 3)
    history["runtime"] = float(runtime)

    return pde_params, key, history


def plot_training_history(
    history,
    metric="rl2",
    title="Vanilla Euler PINN training history",
):
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
        raise ValueError("metric must be 'rl2' or 'mse'.")

    if metric_values.ndim != 2 or metric_values.shape[1] != 3:
        raise ValueError(
            "history metric must have shape (num_evaluations, 3), "
            "with columns [rho, u, p]."
        )

    loss_safe = np.maximum(loss, 1.0e-30)
    metric_safe = np.maximum(metric_values, 1.0e-30)
    component_labels = [
        r"Density $\rho$",
        r"Velocity $u$",
        r"Pressure $p$",
    ]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(
        iters,
        loss_safe,
        linewidth=1.5,
        color="tab:red",
    )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.6)

    for component_idx in range(3):
        axes[1].semilogy(
            eval_iters,
            metric_safe[:, component_idx],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=colors[component_idx],
            label=component_labels[component_idx],
        )

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(metric_label)
    axes[1].set_title(metric_title)
    axes[1].grid(True, which="both", linestyle=":", alpha=0.6)
    axes[1].legend()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.show()
    return fig, axes


def create_test_inputs(t_test, x_test):
    """构造只含原始物理坐标 ``[t, x]`` 的测试输入。"""
    t_test = jnp.asarray(t_test)
    x_test = jnp.asarray(x_test)

    if t_test.shape != x_test.shape:
        raise ValueError("t_test and x_test must have the same shape.")

    return jnp.stack(
        [t_test.reshape(-1), x_test.reshape(-1)],
        axis=-1,
    )


def plot_evaluation(
    params,
    model,
    t_test,
    x_test,
    labels_test,
    cmap="jet",
):
    """评估并绘制一维 Euler 方程的 ``rho``、``u`` 和 ``p``。"""
    t_test_jax = jnp.asarray(t_test).reshape(-1)
    x_test_jax = jnp.asarray(x_test).reshape(-1)

    if t_test_jax.shape != x_test_jax.shape:
        raise ValueError(
            "t_test and x_test must contain the same number of points."
        )

    test_inputs = jnp.stack([t_test_jax, x_test_jax], axis=-1)

    @jax.jit
    def predict_fn(current_params, inputs):
        rho, velocity, pressure = model.apply(current_params, inputs)
        return jnp.stack(
            [
                jnp.asarray(rho).reshape(-1),
                jnp.asarray(velocity).reshape(-1),
                jnp.asarray(pressure).reshape(-1),
            ],
            axis=-1,
        )

    predictions = predict_fn(params, test_inputs)
    predictions.block_until_ready()

    predictions_np = np.asarray(predictions).reshape(-1, 3)
    labels_np = np.asarray(labels_test).reshape(-1, 3)
    if predictions_np.shape != labels_np.shape:
        raise ValueError(
            "Predictions and labels must have the same shape; "
            "the last dimension must be [rho, u, p]."
        )

    t_np = np.asarray(t_test_jax).reshape(-1)
    x_np = np.asarray(x_test_jax).reshape(-1)

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
        raise ValueError(
            "The supplied test points do not form a complete rectangular "
            "(t,x) tensor-product grid."
        )

    T_expected, X_expected = np.meshgrid(
        t_unique,
        x_unique,
        indexing="ij",
    )
    if not (
        np.allclose(t_sorted, T_expected.reshape(-1))
        and np.allclose(x_sorted, X_expected.reshape(-1))
    ):
        raise ValueError(
            "The supplied test coordinates contain missing or duplicated "
            "tensor-product grid points."
        )

    print(f"Plotting test grid: nt={nt_test}, nx={nx_test}")

    pred_grid = pred_sorted.reshape(nt_test, nx_test, 3)
    true_grid = true_sorted.reshape(nt_test, nx_test, 3)

    diff = pred_grid - true_grid
    mse = np.mean(diff**2, axis=(0, 1))
    numerator = np.sum(diff**2, axis=(0, 1))
    denominator = np.sum(true_grid**2, axis=(0, 1))
    rl2 = np.sqrt(
        numerator / np.maximum(denominator, 1.0e-12)
    )

    print(
        "Evaluation errors:\n"
        f"  rho: MSE={mse[0]:.6e}, RL2={rl2[0]:.6e}\n"
        f"  u  : MSE={mse[1]:.6e}, RL2={rl2[1]:.6e}\n"
        f"  p  : MSE={mse[2]:.6e}, RL2={rl2[2]:.6e}"
    )

    rho_true = true_grid[:, :, 0]
    rho_pred = pred_grid[:, :, 0]

    rho_min = min(np.nanmin(rho_true), np.nanmin(rho_pred))
    rho_max = max(np.nanmax(rho_true), np.nanmax(rho_pred))
    if rho_max <= rho_min:
        rho_max = rho_min + 1.0e-12
    rho_levels = np.linspace(rho_min, rho_max, 101)

    tend = t_unique[-1]
    true_final = true_grid[-1]
    pred_final = pred_grid[-1]
    component_names = [r"$\rho$", r"$u$", r"$p$"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    contour_true = axes[0].contourf(
        x_unique,
        t_unique,
        rho_true,
        levels=rho_levels,
        cmap=cmap,
    )
    fig.colorbar(contour_true, ax=axes[0], label=r"$\rho$")
    axes[0].set_title(r"Reference density $\rho(t,x)$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$t$")

    contour_pred = axes[1].contourf(
        x_unique,
        t_unique,
        rho_pred,
        levels=rho_levels,
        cmap=cmap,
    )
    fig.colorbar(contour_pred, ax=axes[1], label=r"$\rho$")
    axes[1].set_title(r"Predicted density $\rho(t,x)$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$t$")

    for component_idx in range(3):
        axes[2].plot(
            x_unique,
            true_final[:, component_idx],
            color=colors[component_idx],
            linestyle="-",
            linewidth=2.0,
            label=f"Reference {component_names[component_idx]}",
        )
        axes[2].plot(
            x_unique,
            pred_final[:, component_idx],
            color=colors[component_idx],
            linestyle="--",
            linewidth=2.0,
            label=f"Prediction {component_names[component_idx]}",
        )

    axes[2].set_title(rf"Profiles at $t={tend:.4g}$")
    axes[2].set_xlabel(r"$x$")
    axes[2].set_ylabel("Primitive variables")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="best", ncol=2, fontsize=9)

    plt.tight_layout()
    plt.show()
    return mse, rl2


def save_training_results(
    pde_params,
    history,
    checkpoint_path="pinn_euler1d.msgpack",
    history_path=None,
    metadata=None,
):
    """Save the vanilla-PINN parameters and its complete training history."""
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
    missing_keys = [name for name in history_keys if name not in history]
    if missing_keys:
        raise ValueError(f"history is missing keys: {missing_keys}")

    saved_history = {
        name: np.asarray(history[name]) for name in history_keys
    }
    checkpoint = {
        "format_version": 1,
        "method": "vanilla_pinn",
        "pde_params": jax.device_get({"stage1": pde_params}),
        "pde_histories": {"stage1": saved_history},
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
    history_tmp = history_path + ".tmp"
    with open(history_tmp, "wb") as file:
        np.savez_compressed(
            file,
            **{
                f"stage1_{name}": value
                for name, value in saved_history.items()
            },
        )
    os.replace(history_tmp, history_path)

    print(f"Training checkpoint saved to: {checkpoint_path}")
    print(f"Training history saved to: {history_path}")


def load_training_results(checkpoint_path):
    """Load a complete vanilla Euler PINN checkpoint."""
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
    print(f"Training checkpoint loaded from: {checkpoint_path}")
    return checkpoint


def load_training_histories(history_path):
    """Load the component-wise vanilla-PINN history from its NumPy archive."""
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
        history = {
            name: np.array(data[f"stage1_{name}"])
            for name in history_keys
        }

    print(f"Training history loaded from: {history_path}")
    return {"stage1": history}


def main():
    # 1. Load Sod shock-tube data.
    data_path = "SodShocktube_data.mat"
    data = scipy.io.loadmat(data_path)

    # Match the TAL-PINN Euler comparison grid exactly.
    sss = 8
    Rho0 = jnp.asarray(data["Rho"])[:, ::sss]
    U0 = jnp.asarray(data["U"])[:, ::sss]
    P0 = jnp.asarray(data["P"])[:, ::sss]
    X0 = jnp.asarray(data["X"])[:, ::sss]
    T0 = jnp.asarray(data["T"])[:, ::sss]

    if not (Rho0.shape == U0.shape == P0.shape == X0.shape == T0.shape):
        raise ValueError("Rho, U, P, X and T must have the same shape.")

    Nt_test, Nx_test = Rho0.shape
    xl = float(X0[0, 0])
    xr = float(X0[0, -1])
    t0 = float(T0[0, 0])
    tend = float(T0[-1, 0])

    x_jump = 0.5 * (xl + xr)
    left_state = (
        float(Rho0[0, 0]),
        float(U0[0, 0]),
        float(P0[0, 0]),
    )
    right_state = (
        float(Rho0[0, -1]),
        float(U0[0, -1]),
        float(P0[0, -1]),
    )

    trange = (t0, tend)
    xrange = (xl, xr)

    t_test = T0.reshape(-1)
    x_test = X0.reshape(-1)
    test_inputs = create_test_inputs(t_test, x_test)
    test_labels = jnp.stack(
        [
            Rho0.reshape(-1),
            U0.reshape(-1),
            P0.reshape(-1),
        ],
        axis=-1,
    )

    print(f"Test grid: Nt={Nt_test}, Nx={Nx_test}")
    print(f"Time domain: {trange}")
    print(f"Space domain: {xrange}")
    print(f"Interface position: {x_jump}")
    print(f"Left state  [rho,u,p]: {left_state}")
    print(f"Right state [rho,u,p]: {right_state}")

    # 2. Vanilla Euler PINN: the network input is only [t, x].
    seed = 42
    key = jax.random.PRNGKey(seed)
    key_pde_init, key_train = jax.random.split(key, 2)

    pde_net = Euler_net(
        n_nodes=40,
        fourier_dim=40,
        activation=nn.silu,
    )
    pde_params = pde_net.init(
        key_pde_init,
        jnp.ones((2,), dtype=T0.dtype),
    )

    # Directly train at the intended final artificial-viscosity value.
    nu = 1.0e-4

    pde_optimizer = soap(
        learning_rate=3.0e-3,
        b1=0.95,
        b2=0.95,
        weight_decay=0.01,
        precondition_frequency=10,
        precondition_1d=False,
    )
    pde_loss_grad_fn = create_pde_loss_grad_fn(
        pde_net,
        w_ic=10.0,
        gamma=1.4,
    )
    pde_minibatch_fn = create_pde_minibatch_fn(
        IC_vmap,
        x_bounds=xrange,
        t_bounds=trange,
        left_state=left_state,
        right_state=right_state,
        x_jump=x_jump,
        dtype=T0.dtype,
    )
    eval_fn = create_eval_fn(pde_net)

    pde_params, key_train, history = train_pinn(
        pde_params=pde_params,
        pde_optimizer=pde_optimizer,
        pde_loss_grad_fn=pde_loss_grad_fn,
        pde_minibatch_fn=pde_minibatch_fn,
        eval_fn=eval_fn,
        test_inputs=test_inputs,
        test_labels=test_labels,
        key=key_train,
        nu=nu,
        max_iters=1000,
        pts_pde=10000,
        pts_ics=1000,
        max_runtime=10000.0,
        eval_every=100,
    )

    save_training_results(
        pde_params=pde_params,
        history=history,
        checkpoint_path="pinn_euler1d.msgpack",
        metadata={
            "seed": seed,
            "nu": nu,
            "max_iters": 1000,
            "pts_pde": 10000,
            "pts_ics": 1000,
            "eval_every": 100,
            "spatial_stride": sss,
        },
    )

    plot_training_history(
        history,
        metric="rl2",
        title="Vanilla Euler PINN training history",
    )
    plot_evaluation(
        params=pde_params,
        model=pde_net,
        t_test=t_test,
        x_test=x_test,
        labels_test=test_labels,
    )

    return pde_params, history


if __name__ == "__main__":
    main()
