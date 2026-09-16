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


class FNN_fourier(nn.Module):
    layer_sizes: Sequence[int]
    activation: Callable = nn.silu
    out_bias: bool = True
    fourier_dim: int = None

    @nn.compact
    def __call__(self, x):
        """Fourier feature layer followed by a standard fully connected net."""
        fourier_dim = self.layer_sizes[0] if self.fourier_dim is None else self.fourier_dim

        x = nn.Dense(
            fourier_dim,
            kernel_init=jax.nn.initializers.normal(stddev=1.0),
        )(x)
        x = jnp.sin(2.0 * jnp.pi * x)

        kinit = jax.nn.initializers.he_uniform()
        for feat in self.layer_sizes[:-1]:
            x = nn.Dense(feat, kernel_init=kinit)(x)
            x = self.activation(x)

        return nn.Dense(
            self.layer_sizes[-1],
            kernel_init=kinit,
            use_bias=self.out_bias,
        )(x)


def IC_Burgers_2D(x, y):
    return jnp.sin(0.5 * jnp.pi * (x + y))


def create_pde_loss_grad_fn(model, w_ic=10.0, w_bc=10.0):
    """
    构造二维黏性 Burgers 方程的 vanilla PINN 损失：

        u_t + u u_x + u u_y - nu (u_xx + u_yy) = 0.

    网络只接收原始物理坐标：

        (t, x, y) -> u(t, x, y).

    边界条件沿用原程序的周期值配对形式。
    """
    apply_fn = model.apply

    def get_u(params, txy):
        return jnp.asarray(apply_fn(params, txy)).reshape(())

    grad_u = jax.grad(get_u, argnums=1)
    hess_u = jax.hessian(get_u, argnums=1)

    def residual_single(params, txy, nu):
        u = get_u(params, txy)
        grad = grad_u(params, txy)
        hess = hess_u(params, txy)

        u_t = grad[0]
        u_x = grad[1]
        u_y = grad[2]
        u_xx = hess[1, 1]
        u_yy = hess[2, 2]

        return u_t + u * u_x + u * u_y - nu * (u_xx + u_yy)

    residual_batch = jax.vmap(residual_single, in_axes=(None, 0, None))

    def loss_fn(
        params,
        inputs_pde,
        inputs_ics,
        targets_ics,
        inputs_bcs_lo,
        inputs_bcs_hi,
        nu=1.0e-3,
    ):
        # PDE residual on points sampled uniformly in the physical domain.
        residual = residual_batch(params, inputs_pde, nu)
        loss_pde = jnp.mean(residual**2)

        # Initial condition.
        pred_ics = jnp.asarray(apply_fn(params, inputs_ics)).reshape(-1)
        true_ics = jnp.asarray(targets_ics).reshape(-1)
        loss_ics = jnp.mean((pred_ics - true_ics) ** 2)

        # Periodic value matching in the x and y directions.
        pred_bcs_lo = jnp.asarray(apply_fn(params, inputs_bcs_lo)).reshape(-1)
        pred_bcs_hi = jnp.asarray(apply_fn(params, inputs_bcs_hi)).reshape(-1)
        loss_bcs = jnp.mean((pred_bcs_lo - pred_bcs_hi) ** 2)

        return loss_pde + w_ic * loss_ics + w_bc * loss_bcs

    return jax.jit(jax.value_and_grad(loss_fn))


def create_pde_minibatch_fn(
    IC_map,
    *,
    t_bounds,
    x_bounds=(0.0, 4.0),
    y_bounds=(0.0, 4.0),
    dtype=jnp.float32,
):
    """
    在原始物理时空区域中均匀采样 vanilla PINN 训练点。

    ``pts_bcs`` 是每个周期方向的配对点数。返回的每一侧边界
    数组包含 ``pts_bcs`` 个 x 周期配对点和 ``pts_bcs`` 个 y
    周期配对点。
    """
    t_min = jnp.asarray(t_bounds[0], dtype=dtype)
    t_max = jnp.asarray(t_bounds[1], dtype=dtype)
    x_min = jnp.asarray(x_bounds[0], dtype=dtype)
    x_max = jnp.asarray(x_bounds[1], dtype=dtype)
    y_min = jnp.asarray(y_bounds[0], dtype=dtype)
    y_max = jnp.asarray(y_bounds[1], dtype=dtype)

    @partial(jax.jit, static_argnames=("pts_pde", "pts_ics", "pts_bcs"))
    def pde_minibatch(key, pts_pde, pts_ics, pts_bcs):
        key_pde, key_ics, key_bcx, key_bcy = jax.random.split(key, 4)

        # 1. PDE points: uniform sampling in (t, x, y).
        sample_pde = jax.random.uniform(
            key_pde,
            shape=(pts_pde, 3),
            dtype=dtype,
        )
        t_pde = t_min + (t_max - t_min) * sample_pde[:, 0]
        x_pde = x_min + (x_max - x_min) * sample_pde[:, 1]
        y_pde = y_min + (y_max - y_min) * sample_pde[:, 2]
        inputs_pde = jnp.stack([t_pde, x_pde, y_pde], axis=-1)

        # 2. Initial-condition points: uniform sampling in (x, y).
        sample_ics = jax.random.uniform(
            key_ics,
            shape=(pts_ics, 2),
            dtype=dtype,
        )
        x_ics = x_min + (x_max - x_min) * sample_ics[:, 0]
        y_ics = y_min + (y_max - y_min) * sample_ics[:, 1]
        t_ics = jnp.full_like(x_ics, t_min)

        inputs_ics = jnp.stack([t_ics, x_ics, y_ics], axis=-1)
        targets_ics = IC_map(x_ics, y_ics).reshape(-1, 1)

        # 3. x-periodic boundary pairs:
        #    (t, x_min, y) <-> (t, x_max, y).
        sample_bcx = jax.random.uniform(
            key_bcx,
            shape=(pts_bcs, 2),
            dtype=dtype,
        )
        t_bcx = t_min + (t_max - t_min) * sample_bcx[:, 0]
        y_bcx = y_min + (y_max - y_min) * sample_bcx[:, 1]

        txy_x_lo = jnp.stack(
            [t_bcx, jnp.full_like(t_bcx, x_min), y_bcx],
            axis=-1,
        )
        txy_x_hi = jnp.stack(
            [t_bcx, jnp.full_like(t_bcx, x_max), y_bcx],
            axis=-1,
        )

        # 4. y-periodic boundary pairs:
        #    (t, x, y_min) <-> (t, x, y_max).
        sample_bcy = jax.random.uniform(
            key_bcy,
            shape=(pts_bcs, 2),
            dtype=dtype,
        )
        t_bcy = t_min + (t_max - t_min) * sample_bcy[:, 0]
        x_bcy = x_min + (x_max - x_min) * sample_bcy[:, 1]

        txy_y_lo = jnp.stack(
            [t_bcy, x_bcy, jnp.full_like(t_bcy, y_min)],
            axis=-1,
        )
        txy_y_hi = jnp.stack(
            [t_bcy, x_bcy, jnp.full_like(t_bcy, y_max)],
            axis=-1,
        )

        inputs_bcs_lo = jnp.concatenate([txy_x_lo, txy_y_lo], axis=0)
        inputs_bcs_hi = jnp.concatenate([txy_x_hi, txy_y_hi], axis=0)

        return (
            inputs_pde,
            inputs_ics,
            targets_ics,
            inputs_bcs_lo,
            inputs_bcs_hi,
        )

    return pde_minibatch


def create_pde_net_update_fn(loss_grad_fn, optimizer, nu):
    @jax.jit
    def update_step(
        params,
        opt_state,
        inputs_pde,
        inputs_ics,
        targets_ics,
        inputs_bcs_lo,
        inputs_bcs_hi,
    ):
        loss, grads = loss_grad_fn(
            params,
            inputs_pde,
            inputs_ics,
            targets_ics,
            inputs_bcs_lo,
            inputs_bcs_hi,
            nu,
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    return update_step


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
        raise ValueError(
            "error and mask must be two-dimensional arrays with the same shape."
        )

    ny, nx = error.shape
    if ny < 3 or nx < 3:
        raise ValueError(
            "At least three grid points in each direction are required "
            "for four-neighbor averaging."
        )

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


def create_eval_fn(
    model,
    discontinuity_sums=(2.0, 6.0),
    discontinuity_atol=1.0e-6,
):
    """
    使用物理坐标输入 ``[t, x, y]`` 计算 MSE 和相对 L2 误差。

    两个指标都排除满足 ``x + y = discontinuity_sums[k]`` 的点。
    """
    discontinuity_sums = tuple(float(value) for value in discontinuity_sums)
    if discontinuity_atol < 0.0:
        raise ValueError("discontinuity_atol must be non-negative.")

    apply_fn = model.apply

    def eval_error(params, inputs, labels):
        predictions = jnp.asarray(apply_fn(params, inputs)).reshape(-1)
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
        rl2 = jnp.sqrt(
            jnp.sum(squared_error)
            / jnp.maximum(jnp.sum(squared_label), 1.0e-12)
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
    pts_bcs=1000,
    max_runtime=10000.0,
    eval_every=500,
):
    """单阶段训练 vanilla PINN；计时不包括 JIT 预热和误差评估。"""
    if eval_every <= 0:
        raise ValueError("eval_every must be positive.")

    opt_state = pde_optimizer.init(pde_params)
    update_fn = create_pde_net_update_fn(
        pde_loss_grad_fn,
        pde_optimizer,
        nu=nu,
    )

    # JIT warm-up is deliberately excluded from training time.
    warmup_key = jax.random.PRNGKey(0)
    warmup_dataset = pde_minibatch_fn(
        warmup_key,
        pts_pde=pts_pde,
        pts_ics=pts_ics,
        pts_bcs=pts_bcs,
    )
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready(),
        warmup_dataset,
    )
    _, _, warmup_loss = update_fn(pde_params, opt_state, *warmup_dataset)
    warmup_loss.block_until_ready()
    print("[Vanilla PINN] JIT warm-up finished.")

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
        "[Vanilla PINN] iter=00000, time=0.00s | "
        f"MSE={mse_value:.2e} | RL2={rl2_value:.2e}"
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
            pts_bcs=pts_bcs,
        )
        pde_params, opt_state, loss = update_fn(
            pde_params,
            opt_state,
            *dataset,
        )
        loss_value = float(loss.block_until_ready())

        # Sampling plus the optimization update are counted as training time.
        # The evaluation block below is not counted.
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
                f"[Vanilla PINN] iter={it:05d}, "
                f"time={runtime:.2f}s, loss={loss_value:.2e} | "
                f"MSE={mse_value:.2e} | RL2={rl2_value:.2e}"
            )

    if history["iter"]:
        final_iter = history["iter"][-1]
        if history["eval_iter"][-1] != final_iter:
            mse_value, rl2_value = record_evaluation(
                iteration=final_iter,
                train_time=runtime,
            )
            print(
                f"[Vanilla PINN] final iter={final_iter:05d}, "
                f"time={runtime:.2f}s | MSE={mse_value:.2e} | "
                f"RL2={rl2_value:.2e}"
            )

    history["iter"] = np.asarray(history["iter"], dtype=np.int32)
    history["loss"] = np.asarray(history["loss"], dtype=np.float64)
    history["train_time"] = np.asarray(history["train_time"], dtype=np.float64)
    history["eval_iter"] = np.asarray(history["eval_iter"], dtype=np.int32)
    history["eval_train_time"] = np.asarray(history["eval_train_time"], dtype=np.float64)
    history["mse"] = np.asarray(history["mse"], dtype=np.float64)
    history["rl2"] = np.asarray(history["rl2"], dtype=np.float64)
    history["runtime"] = float(runtime)

    return pde_params, key, history


def plot_training_history(
    history,
    metric="rl2",
    title="Vanilla PINN training history",
):
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

    axes[1].semilogy(
        eval_iters,
        metric_safe,
        marker="o",
        markersize=4,
        linewidth=1.5,
        color="tab:blue",
        label=metric_label,
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


def create_test_inputs(t_test, x_test, y_test):
    """构造只含原始物理坐标 ``[t, x, y]`` 的测试输入。"""
    t_test = jnp.asarray(t_test)
    x_test = jnp.asarray(x_test)
    y_test = jnp.asarray(y_test)

    if not (t_test.shape == x_test.shape == y_test.shape):
        raise ValueError("t_test, x_test and y_test must have the same shape.")

    return jnp.stack(
        [t_test.reshape(-1), x_test.reshape(-1), y_test.reshape(-1)],
        axis=-1,
    )


def plot_evaluation(
    params,
    model,
    t_test,
    x_test,
    y_test,
    labels_test,
    cmap="jet",
    error_cmap="jet",
    discontinuity_sums=(2.0, 6.0),
    discontinuity_atol=1.0e-6,
):
    """
    评估二维 Burgers 标量解，并绘制最后一个给定时刻的参考解、
    预测解和绝对误差。

    误差指标排除 ``x+y=2`` 和 ``x+y=6`` 上的点。参考解和预测解
    保持原值；仅在误差图中用上、下、左、右四邻点的平均误差替换
    间断面点。
    """
    discontinuity_sums = tuple(float(value) for value in discontinuity_sums)
    if discontinuity_atol < 0.0:
        raise ValueError("discontinuity_atol must be non-negative.")

    t_test_jax = jnp.asarray(t_test).reshape(-1)
    x_test_jax = jnp.asarray(x_test).reshape(-1)
    y_test_jax = jnp.asarray(y_test).reshape(-1)

    if not (t_test_jax.shape == x_test_jax.shape == y_test_jax.shape):
        raise ValueError(
            "t_test, x_test and y_test must contain the same number of points."
        )

    test_inputs = jnp.stack(
        [t_test_jax, x_test_jax, y_test_jax],
        axis=-1,
    )

    @jax.jit
    def predict_fn(current_params, inputs):
        return jnp.asarray(model.apply(current_params, inputs)).reshape(-1)

    predictions = predict_fn(params, test_inputs)
    predictions.block_until_ready()

    predictions_np = np.asarray(predictions).reshape(-1)
    labels_np = np.asarray(labels_test).reshape(-1)
    if predictions_np.shape != labels_np.shape:
        raise ValueError(
            "Predictions and labels_test must contain the same number "
            "of scalar values."
        )

    t_np = np.asarray(t_test_jax).reshape(-1)
    x_np = np.asarray(x_test_jax).reshape(-1)
    y_np = np.asarray(y_test_jax).reshape(-1)

    # Sort as t -> y -> x and reconstruct the tensor-product grid.
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
        raise ValueError(
            "The supplied test points do not form a complete rectangular "
            "(t,y,x) tensor-product grid."
        )

    T_expected, Y_expected, X_expected = np.meshgrid(
        t_unique,
        y_unique,
        x_unique,
        indexing="ij",
    )
    if not (
        np.allclose(t_sorted, T_expected.reshape(-1))
        and np.allclose(y_sorted, Y_expected.reshape(-1))
        and np.allclose(x_sorted, X_expected.reshape(-1))
    ):
        raise ValueError(
            "The supplied test coordinates contain missing or duplicated "
            "tensor-product grid points."
        )

    print(
        f"Plotting test grid: nt={nt_test}, ny={ny_test}, nx={nx_test}"
    )

    pred_grid = pred_sorted.reshape(nt_test, ny_test, nx_test)
    true_grid = true_sorted.reshape(nt_test, ny_test, nx_test)

    # Error metrics exclude all points on the prescribed discontinuities.
    on_discontinuity = _discontinuity_sum_mask_numpy(
        X_expected,
        Y_expected,
        discontinuity_sums,
        discontinuity_atol,
    )
    valid = np.logical_not(on_discontinuity)
    if not np.any(valid):
        raise ValueError(
            "No off-discontinuity test points remain for error evaluation."
        )

    squared_error = (pred_grid - true_grid) ** 2
    mse = np.mean(squared_error[valid])
    rl2 = np.sqrt(
        np.sum(squared_error[valid])
        / max(np.sum((true_grid**2)[valid]), 1.0e-12)
    )

    tend = t_unique[-1]
    true_final = true_grid[-1]
    pred_final = pred_grid[-1]
    raw_error_final = np.abs(pred_final - true_final)

    final_on_discontinuity = on_discontinuity[-1]
    final_valid = np.logical_not(final_on_discontinuity)
    if not np.any(final_valid):
        raise ValueError(
            "No off-discontinuity final-time points remain for error evaluation."
        )

    final_squared_error = (pred_final - true_final) ** 2
    final_mse = np.mean(final_squared_error[final_valid])
    final_rl2 = np.sqrt(
        np.sum(final_squared_error[final_valid])
        / max(np.sum((true_final**2)[final_valid]), 1.0e-12)
    )

    error_final = _replace_masked_error_with_four_neighbor_average(
        raw_error_final,
        final_on_discontinuity,
    )

    print(
        "Evaluation errors (excluding x+y=2 and x+y=6):\n"
        f"  Supplied test set: MSE={mse:.6e}, RL2={rl2:.6e}\n"
        f"  Final time       : MSE={final_mse:.6e}, "
        f"RL2={final_rl2:.6e}\n"
        f"  Excluded points  : full={np.count_nonzero(on_discontinuity)}, "
        f"final={np.count_nonzero(final_on_discontinuity)}"
    )

    solution_min = min(np.nanmin(true_final), np.nanmin(pred_final))
    solution_max = max(np.nanmax(true_final), np.nanmax(pred_final))
    if solution_max <= solution_min:
        solution_max = solution_min + 1.0e-12

    solution_levels = np.linspace(solution_min, solution_max, 101)
    error_max = np.nanmax(error_final)
    if error_max <= 0.0:
        error_max = 1.0e-12
    error_levels = np.linspace(0.0, error_max, 101)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.2),
        constrained_layout=True,
    )

    contour_true = axes[0].contourf(
        x_unique,
        y_unique,
        true_final,
        levels=solution_levels,
        cmap=cmap,
        extend="both",
    )
    fig.colorbar(contour_true, ax=axes[0], label=r"$u$")
    axes[0].set_title(rf"Reference solution at $t={tend:.4g}$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")
    axes[0].set_aspect("equal", adjustable="box")

    contour_pred = axes[1].contourf(
        x_unique,
        y_unique,
        pred_final,
        levels=solution_levels,
        cmap=cmap,
        extend="both",
    )
    fig.colorbar(contour_pred, ax=axes[1], label=r"$u$")
    axes[1].set_title(rf"Predicted solution at $t={tend:.4g}$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$y$")
    axes[1].set_aspect("equal", adjustable="box")

    contour_error = axes[2].contourf(
        x_unique,
        y_unique,
        error_final,
        levels=error_levels,
        cmap=error_cmap,
        extend="max",
    )
    fig.colorbar(
        contour_error,
        ax=axes[2],
        label=r"$|u_{\mathrm{pred}}-u_{\mathrm{ref}}|$",
    )
    axes[2].set_title(
        rf"Absolute error at $t={tend:.4g}$, "
        rf"$\mathrm{{RL2}}_{{\rm excl.}}={final_rl2:.2e}$"
    )
    axes[2].set_xlabel(r"$x$")
    axes[2].set_ylabel(r"$y$")
    axes[2].set_aspect("equal", adjustable="box")

    plt.show()
    return mse, rl2


def save_training_results(
    pde_params,
    history,
    checkpoint_path="pinn_burgers2d01.msgpack",
    history_path=None,
    metadata=None,
):
    """Save the vanilla-PINN parameters and complete scalar-error history."""
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
    """Load the vanilla 2D Burgers PINN checkpoint."""
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
    """Load the scalar MSE, relative-L2, loss and timing histories."""
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
    # 1. Use exactly the same reference data layout as TAL-PINN.
    data_path = "Burgers_2D_01.mat"
    data = scipy.io.loadmat(data_path)

    U_data = jnp.asarray(data["U"])
    Nt, Ny, Nx = U_data.shape

    t = jnp.asarray(data["t"]).reshape(-1)
    x = jnp.asarray(data["x"]).reshape(-1)
    y = jnp.asarray(data["y"]).reshape(-1)
    if not (len(t) == Nt and len(x) == Nx and len(y) == Ny):
        raise ValueError("The lengths of t, x and y must match U.shape.")

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

    # The current comparison evaluates the final available snapshot.
    T_test = T_data[-1]
    X_test = X_data[-1]
    Y_test = Y_data[-1]
    test_labels = U_data[-1].reshape(-1)
    test_inputs = create_test_inputs(T_test, X_test, Y_test)

    print(f"Data grid: Nt={Nt}, Ny={Ny}, Nx={Nx}")
    print(f"Time domain: {trange}")
    print(f"x domain: {xrange}")
    print(f"y domain: {yrange}")
    print(f"Test snapshot: t={tend:.6g}")

    # 2. Vanilla PINN: FNN_fourier receives only (t, x, y).
    seed = 42
    key = jax.random.PRNGKey(seed)
    key_pde_init, key_train = jax.random.split(key, 2)

    pde_net = FNN_fourier(
        layer_sizes=[40, 40, 1],
        activation=nn.silu,
    )
    pde_params = pde_net.init(
        key_pde_init,
        jnp.ones((3,), dtype=T_data.dtype),
    )

    # 3. One direct training stage at the target viscosity.
    # Change this value if the reference data use another viscosity.
    nu = 1.0e-3

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
        w_bc=10.0,
    )
    pde_minibatch_fn = create_pde_minibatch_fn(
        IC_Burgers_2D,
        t_bounds=trange,
        x_bounds=xrange,
        y_bounds=yrange,
        dtype=T_data.dtype,
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
        max_iters=10000,
        pts_pde=10000,
        pts_ics=1000,
        pts_bcs=1000,
        max_runtime=10000.0,
        eval_every=100,
    )

    save_training_results(
        pde_params=pde_params,
        history=history,
        checkpoint_path="pinn_burgers2d01.msgpack",
        metadata={
            "data_path": data_path,
            "seed": seed,
            "nu": nu,
            "test_time": tend,
            "max_iters": 10000,
            "pts_pde": 10000,
            "pts_ics": 1000,
            "pts_bcs": 1000,
            "eval_every": 100,
        },
    )

    plot_training_history(
        history,
        metric="rl2",
        title="Vanilla PINN training history",
    )
    plot_evaluation(
        params=pde_params,
        model=pde_net,
        t_test=T_test,
        x_test=X_test,
        y_test=Y_test,
        labels_test=test_labels,
    )

    return pde_params, history


if __name__ == "__main__":
    main()
