import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.precision", 6)

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


def pairwise_distances(Y):
    diff = Y[:, None, :] - Y[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def check_distance_matrix(D, tol=1e-10, sample_triples=None, seed=42):
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("Distance matrix must be square")
    diag_abs_max = float(np.max(np.abs(np.diag(D))))
    symmetry_abs_max = float(np.max(np.abs(D - D.T)))
    negative_count = int(np.sum(D < -tol))
    if sample_triples is None:
        I, J, K = np.indices((n, n, n))
        triangle_violations = D[I, J] > D[I, K] + D[K, J] + tol
        total_triples = n**3
    else:
        local_rng = np.random.default_rng(seed)
        I = local_rng.integers(0, n, size=sample_triples)
        J = local_rng.integers(0, n, size=sample_triples)
        K = local_rng.integers(0, n, size=sample_triples)
        triangle_violations = D[I, J] > D[I, K] + D[K, J] + tol
        total_triples = sample_triples
    triangle_violation_count = int(np.sum(triangle_violations))
    return {
        "n": n,
        "diag_abs_max": diag_abs_max,
        "diag_ok": diag_abs_max <= tol,
        "symmetry_abs_max": symmetry_abs_max,
        "symmetry_ok": symmetry_abs_max <= tol,
        "negative_count": negative_count,
        "nonnegative_ok": negative_count == 0,
        "triangle_triples_checked": int(total_triples),
        "triangle_violation_count": triangle_violation_count,
        "triangle_violation_fraction": triangle_violation_count / total_triples,
    }


def centering_matrix(n):
    one = np.ones((n, 1))
    return np.eye(n) - (one @ one.T) / n


def gram_from_distances(D):
    D2 = D**2
    J = centering_matrix(D.shape[0])
    B = -0.5 * J @ D2 @ J
    return D2, J, 0.5 * (B + B.T)


def gram_spectrum(B, tol=1e-9):
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    counts = {
        "positive": int(np.sum(evals > tol)),
        "zero": int(np.sum(np.abs(evals) <= tol)),
        "negative": int(np.sum(evals < -tol)),
        "rank": int(np.sum(evals > tol)),
        "min_embedding_dim": int(np.sum(evals > tol)),
    }
    return evals, evecs, counts


def classical_mds(D, m=None, tol=1e-9):
    D2, J, B = gram_from_distances(D)
    evals, evecs, counts = gram_spectrum(B, tol=tol)
    if m is None:
        use = np.where(evals > tol)[0]
    else:
        use = np.arange(min(m, len(evals)))
    lambdas = np.maximum(evals[use], 0)
    Y = evecs[:, use] * np.sqrt(lambdas)
    return {"Y": Y, "B": B, "D2": D2, "J": J, "evals": evals, "evecs": evecs, "counts": counts}


def distance_errors(D_ref, Y_hat):
    D_hat = pairwise_distances(Y_hat)
    diff = D_ref - D_hat
    return {
        "E_max": float(np.max(np.abs(diff))),
        "E_F": float(np.linalg.norm(diff, ord="fro")),
        "E_rel": float(np.linalg.norm(diff, ord="fro") / np.linalg.norm(D_ref, ord="fro")),
    }


def procrustes_align(Y_hat, Y_ref):
    A = Y_hat - Y_hat.mean(axis=0)
    B_ref = Y_ref - Y_ref.mean(axis=0)
    U, _, Vt = np.linalg.svd(A.T @ B_ref, full_matrices=False)
    Q = U @ Vt
    return A @ Q + Y_ref.mean(axis=0)


def save_case3_outputs():
    rng = np.random.default_rng(SEED)
    n = 45
    t = np.linspace(0, 4 * np.pi, n)
    Y_true = np.column_stack([
        np.cos(t) + 0.06 * rng.normal(size=n),
        np.sin(t) + 0.06 * rng.normal(size=n),
        0.22 * t + 0.04 * rng.normal(size=n),
    ])
    D = pairwise_distances(Y_true)
    pd.DataFrame([check_distance_matrix(D)]).to_csv(RESULTS / "case3_basic_checks.csv", index=False)

    D2, J, B = gram_from_distances(D)
    evals, evecs, spectrum_counts = gram_spectrum(B)
    pd.DataFrame({"index": np.arange(1, len(evals) + 1), "eigenvalue": evals}).to_csv(
        RESULTS / "case3_gram_spectrum.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(evals) + 1), evals, marker="o")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Кейс 3: спектр матрицы Грама")
    ax.set_xlabel("Номер собственного значения")
    ax.set_ylabel("Собственное значение")
    fig.tight_layout()
    fig.savefig(FIGURES / "case3_gram_spectrum.png", dpi=180)
    plt.close(fig)

    mds_full = classical_mds(D)
    Y_full = mds_full["Y"]
    Y_full_aligned = procrustes_align(Y_full, Y_true)
    coord_rmse = float(np.sqrt(np.mean((Y_full_aligned - Y_true) ** 2)))
    exact_errors = distance_errors(D, Y_full)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(Y_true[:, 0], Y_true[:, 1], Y_true[:, 2], c=t, cmap="viridis", s=35)
    ax1.set_title("Исходная 3D-конфигурация")
    ax1.set_xlabel("y1")
    ax1.set_ylabel("y2")
    ax1.set_zlabel("y3")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(Y_full_aligned[:, 0], Y_full_aligned[:, 1], Y_full_aligned[:, 2], c=t, cmap="viridis", s=35)
    ax2.set_title("Восстановление после выравнивания")
    ax2.set_xlabel("y1")
    ax2.set_ylabel("y2")
    ax2.set_zlabel("y3")
    fig.tight_layout()
    fig.savefig(FIGURES / "case3_true_vs_recovered_3d.png", dpi=180)
    plt.close(fig)

    rows = []
    for m in range(1, 9):
        Y_m = classical_mds(D, m=m)["Y"]
        row = distance_errors(D, Y_m)
        row["m"] = m
        rows.append(row)
    dim_errors = pd.DataFrame(rows)[["m", "E_max", "E_F", "E_rel"]]
    dim_errors.to_csv(RESULTS / "case3_dimension_errors.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dim_errors["m"], dim_errors["E_F"], marker="o", label="E_F(m)")
    ax.plot(dim_errors["m"], dim_errors["E_rel"], marker="s", label="E_rel(m)")
    ax.axvline(spectrum_counts["min_embedding_dim"], color="black", linestyle="--", label="минимальная точная размерность")
    ax.set_title("Кейс 3: ошибка восстановления расстояний от размерности")
    ax.set_xlabel("m")
    ax.set_ylabel("Ошибка")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "case3_dimension_error_curve.png", dpi=180)
    plt.close(fig)

    Y2 = classical_mds(D, m=2)["Y"]
    D_hat_2 = pairwise_distances(Y2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(Y2[:, 0], Y2[:, 1], c=t, cmap="viridis", s=45)
    axes[0].set_title("2D-вложение")
    axes[0].set_xlabel("coord 1")
    axes[0].set_ylabel("coord 2")
    iu = np.triu_indices_from(D, k=1)
    axes[1].scatter(D[iu], D_hat_2[iu], alpha=0.75)
    lo = min(D[iu].min(), D_hat_2[iu].min())
    hi = max(D[iu].max(), D_hat_2[iu].max())
    axes[1].plot([lo, hi], [lo, hi], color="black", linestyle="--")
    axes[1].set_title("Исходные и восстановленные расстояния в 2D")
    axes[1].set_xlabel("d_ij")
    axes[1].set_ylabel("d_hat_ij")
    fig.tight_layout()
    fig.savefig(FIGURES / "case3_2d_embedding_and_distances.png", dpi=180)
    plt.close(fig)

    R = rng.uniform(-1.0, 1.0, size=(n, n))
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 0.0)
    median_distance = np.median(D[np.triu_indices_from(D, k=1)])
    noise_rows = []
    for eps in [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]:
        D_eps = D + eps * median_distance * R
        np.fill_diagonal(D_eps, 0.0)
        checks = check_distance_matrix(D_eps, tol=1e-10)
        mds_eps = classical_mds(D_eps)
        err_against_exact = distance_errors(D, mds_eps["Y"])
        noise_rows.append({
            "epsilon": eps,
            "triangle_violation_fraction": checks["triangle_violation_fraction"],
            "negative_distance_count": checks["negative_count"],
            "positive_eigenvalues": mds_eps["counts"]["positive"],
            "negative_eigenvalues": mds_eps["counts"]["negative"],
            "E_F_against_exact_D": err_against_exact["E_F"],
            "E_rel_against_exact_D": err_against_exact["E_rel"],
        })
    noise_summary = pd.DataFrame(noise_rows)
    noise_summary.to_csv(RESULTS / "case3_noise_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(noise_summary["epsilon"], noise_summary["triangle_violation_fraction"], marker="o")
    axes[0].set_title("Доля нарушений треугольника")
    axes[0].set_xlabel("epsilon")
    axes[0].set_ylabel("fraction")
    axes[1].plot(noise_summary["epsilon"], noise_summary["negative_eigenvalues"], marker="o", color="tab:red")
    axes[1].set_title("Число отрицательных собственных значений")
    axes[1].set_xlabel("epsilon")
    axes[1].set_ylabel("count")
    axes[2].plot(noise_summary["epsilon"], noise_summary["E_rel_against_exact_D"], marker="o", color="tab:green")
    axes[2].set_title("Относительная ошибка к точным расстояниям")
    axes[2].set_xlabel("epsilon")
    axes[2].set_ylabel("E_rel")
    fig.tight_layout()
    fig.savefig(FIGURES / "case3_noise_stability.png", dpi=180)
    plt.close(fig)

    return {
        "min_embedding_dim": spectrum_counts["min_embedding_dim"],
        "coord_rmse": coord_rmse,
        "distance_E_rel": exact_errors["E_rel"],
    }


def true_additive_components(X):
    return [
        np.sin(2 * np.pi * X[:, 0]),
        0.5 * (X[:, 1] - 0.5) ** 2,
        np.exp(-3 * X[:, 2]),
    ]


def make_additive_data(n_samples=1400, sigma=0.08, seed=42):
    local_rng = np.random.default_rng(seed)
    X = local_rng.uniform(0, 1, size=(n_samples, 3))
    components = true_additive_components(X)
    y_clean = np.sum(components, axis=0)
    y = y_clean + local_rng.normal(0, sigma, size=n_samples)
    return X, y, y_clean


def polynomial_basis_1d(x, M):
    x = np.asarray(x)
    return np.column_stack([x**m for m in range(1, M + 1)])


def make_additive_design(X, M):
    blocks = [polynomial_basis_1d(X[:, j], M) for j in range(X.shape[1])]
    return np.column_stack([np.ones(X.shape[0]), *blocks])


def fit_ridge_closed_form(Z, y, alpha=0.0):
    penalty = np.eye(Z.shape[1])
    penalty[0, 0] = 0.0
    A = Z.T @ Z + alpha * penalty
    b = Z.T @ y
    if alpha == 0:
        return np.linalg.lstsq(Z, y, rcond=None)[0]
    return np.linalg.solve(A, b)


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MSE": float(mse), "RMSE": float(np.sqrt(mse)), "R2": float(r2_score(y_true, y_pred))}


def component_values_from_coefs(X, coefs, M):
    comps = []
    for j in range(X.shape[1]):
        Phi = polynomial_basis_1d(X[:, j], M)
        comps.append(Phi @ coefs[j])
    return np.column_stack(comps)


def fit_additive_model(X, y, M, alpha=1e-5):
    Z = make_additive_design(X, M)
    theta_raw = fit_ridge_closed_form(Z, y, alpha)
    beta0_raw = float(theta_raw[0])
    coefs = theta_raw[1:].reshape(X.shape[1], M)
    raw_components = component_values_from_coefs(X, coefs, M)
    component_means = raw_components.mean(axis=0)
    beta0 = beta0_raw + float(np.sum(component_means))
    return {"M": M, "coefs": coefs, "component_means": component_means, "beta0": beta0, "n_features": X.shape[1]}


def component_values(model, X):
    raw_components = component_values_from_coefs(X, model["coefs"], model["M"])
    return raw_components - model["component_means"]


def predict_additive_model(model, X):
    return model["beta0"] + component_values(model, X).sum(axis=1)


def additive_n_params(model):
    return 1 + model["n_features"] * model["M"]


def make_nonadditive_data(n_samples=1400, sigma=0.08, seed=7):
    local_rng = np.random.default_rng(seed)
    X = local_rng.uniform(0, 1, size=(n_samples, 3))
    y_clean = np.sin(2 * np.pi * (X[:, 0] + X[:, 1])) + 0.5 * X[:, 2]
    y = y_clean + local_rng.normal(0, sigma, size=n_samples)
    return X, y, y_clean


def add_metrics_row(rows, name, yhat_train, yhat_test, y_train, y_test, n_params):
    row = {"model": name, "n_params": n_params}
    row.update({f"train_{k}": v for k, v in regression_metrics(y_train, yhat_train).items()})
    row.update({f"test_{k}": v for k, v in regression_metrics(y_test, yhat_test).items()})
    rows.append(row)


def save_case7_outputs():
    X_add, y_add, y_add_clean = make_additive_data()
    X_train, X_test, y_train, y_test, y_clean_train, y_clean_test = train_test_split(
        X_add,
        y_add,
        y_add_clean,
        test_size=0.25,
        random_state=SEED,
    )
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    M_chosen = 10
    additive_model = fit_additive_model(X_train, y_train, M=M_chosen, alpha=1e-5)
    pred_train_add = predict_additive_model(additive_model, X_train)
    pred_test_add = predict_additive_model(additive_model, X_test)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    pred_train_lin = linear_model.predict(X_train)
    pred_test_lin = linear_model.predict(X_test)

    metrics_rows = []
    add_metrics_row(metrics_rows, "linear", pred_train_lin, pred_test_lin, y_train, y_test, X_train.shape[1] + 1)
    add_metrics_row(
        metrics_rows,
        f"additive_poly_M{M_chosen}",
        pred_train_add,
        pred_test_add,
        y_train,
        y_test,
        additive_n_params(additive_model),
    )
    pd.DataFrame(metrics_rows).to_csv(RESULTS / "case7_additive_vs_linear_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test, pred_test_lin, alpha=0.7, label="linear")
    axes[0].scatter(y_test, pred_test_add, alpha=0.7, label="additive", marker="x")
    lo = min(y_test.min(), pred_test_lin.min(), pred_test_add.min())
    hi = max(y_test.max(), pred_test_lin.max(), pred_test_add.max())
    axes[0].plot([lo, hi], [lo, hi], color="black", linestyle="--")
    axes[0].set_title("Тест: истинные и предсказанные значения")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("y_hat")
    axes[0].legend()
    residuals = y_test - pred_test_add
    axes[1].hist(residuals, bins=25, edgecolor="black")
    axes[1].set_title("Остатки аддитивной модели на тесте")
    axes[1].set_xlabel("y - y_hat")
    axes[1].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIGURES / "case7_predictions_and_residuals.png", dpi=180)
    plt.close(fig)

    sweep_rows = []
    for M in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]:
        model = fit_additive_model(X_train, y_train, M=M, alpha=1e-5)
        yhat_train = predict_additive_model(model, X_train)
        yhat_test = predict_additive_model(model, X_test)
        row = {"M": M, "n_params": additive_n_params(model)}
        row.update({f"train_{k}": v for k, v in regression_metrics(y_train, yhat_train).items()})
        row.update({f"test_{k}": v for k, v in regression_metrics(y_test, yhat_test).items()})
        sweep_rows.append(row)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(RESULTS / "case7_M_sweep.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(sweep["M"], sweep["train_RMSE"], marker="o", label="train RMSE")
    axes[0].plot(sweep["M"], sweep["test_RMSE"], marker="s", label="test RMSE")
    axes[0].set_title("Ошибка от числа базисных функций")
    axes[0].set_xlabel("M")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    axes[1].plot(sweep["M"], sweep["train_R2"], marker="o", label="train R2")
    axes[1].plot(sweep["M"], sweep["test_R2"], marker="s", label="test R2")
    axes[1].set_title("R2 от числа базисных функций")
    axes[1].set_xlabel("M")
    axes[1].set_ylabel("R2")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "case7_M_sweep.png", dpi=180)
    plt.close(fig)

    grid = np.linspace(0, 1, 300)
    X_grid_template = np.zeros((len(grid), X_train.shape[1]))
    estimated_components = []
    true_components_centered = []
    train_true_comps = true_additive_components(X_train)
    train_true_means = [c.mean() for c in train_true_comps]
    for j in range(X_train.shape[1]):
        Xg = X_grid_template.copy()
        Xg[:, j] = grid
        estimated_components.append(component_values(additive_model, Xg)[:, j])
        if j == 0:
            true_g = np.sin(2 * np.pi * grid)
        elif j == 1:
            true_g = 0.5 * (grid - 0.5) ** 2
        else:
            true_g = np.exp(-3 * grid)
        true_components_centered.append(true_g - train_true_means[j])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for j, ax in enumerate(axes):
        ax.plot(grid, true_components_centered[j], label="true centered", linewidth=3)
        ax.plot(grid, estimated_components[j], label="estimated", linestyle="--")
        ax.set_title(f"g_{j + 1}(x_{j + 1})")
        ax.set_xlabel(f"x_{j + 1}")
        ax.set_ylabel("centered contribution")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "case7_recovered_components.png", dpi=180)
    plt.close(fig)

    full_poly_degree = 5
    full_model = make_pipeline(PolynomialFeatures(degree=full_poly_degree, include_bias=False), Ridge(alpha=1e-5))
    full_model.fit(X_train, y_train)
    pred_train_full = full_model.predict(X_train)
    pred_test_full = full_model.predict(X_test)
    n_full_params = full_model.named_steps["polynomialfeatures"].fit_transform(X_train[:1]).shape[1] + 1

    full_rows = []
    add_metrics_row(
        full_rows,
        f"additive_poly_M{M_chosen}",
        pred_train_add,
        pred_test_add,
        y_train,
        y_test,
        additive_n_params(additive_model),
    )
    add_metrics_row(
        full_rows,
        f"full_poly_degree{full_poly_degree}",
        pred_train_full,
        pred_test_full,
        y_train,
        y_test,
        n_full_params,
    )
    pd.DataFrame(full_rows).to_csv(RESULTS / "case7_additive_vs_full_on_additive_data.csv", index=False)

    X_non, y_non, y_non_clean = make_nonadditive_data()
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_non, y_non, test_size=0.25, random_state=SEED)
    scaler_non = MinMaxScaler()
    Xn_train = scaler_non.fit_transform(Xn_train)
    Xn_test = scaler_non.transform(Xn_test)

    additive_non = fit_additive_model(Xn_train, yn_train, M=M_chosen, alpha=1e-5)
    pred_train_add_non = predict_additive_model(additive_non, Xn_train)
    pred_test_add_non = predict_additive_model(additive_non, Xn_test)

    full_non = make_pipeline(PolynomialFeatures(degree=full_poly_degree, include_bias=False), Ridge(alpha=1e-5))
    full_non.fit(Xn_train, yn_train)
    pred_train_full_non = full_non.predict(Xn_train)
    pred_test_full_non = full_non.predict(Xn_test)

    linear_non = LinearRegression().fit(Xn_train, yn_train)
    pred_train_lin_non = linear_non.predict(Xn_train)
    pred_test_lin_non = linear_non.predict(Xn_test)

    non_rows = []
    add_metrics_row(non_rows, "linear", pred_train_lin_non, pred_test_lin_non, yn_train, yn_test, Xn_train.shape[1] + 1)
    add_metrics_row(
        non_rows,
        f"additive_poly_M{M_chosen}",
        pred_train_add_non,
        pred_test_add_non,
        yn_train,
        yn_test,
        additive_n_params(additive_non),
    )
    add_metrics_row(
        non_rows,
        f"full_poly_degree{full_poly_degree}",
        pred_train_full_non,
        pred_test_full_non,
        yn_train,
        yn_test,
        n_full_params,
    )
    nonadditive_metrics = pd.DataFrame(non_rows)
    nonadditive_metrics.to_csv(RESULTS / "case7_nonadditive_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(yn_test, pred_test_add_non, alpha=0.7, label="additive")
    axes[0].scatter(yn_test, pred_test_full_non, alpha=0.7, label="full polynomial", marker="x")
    lo = min(yn_test.min(), pred_test_add_non.min(), pred_test_full_non.min())
    hi = max(yn_test.max(), pred_test_add_non.max(), pred_test_full_non.max())
    axes[0].plot([lo, hi], [lo, hi], color="black", linestyle="--")
    axes[0].set_title("Неаддитивные данные: сравнение предсказаний")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("y_hat")
    axes[0].legend()
    axes[1].bar(nonadditive_metrics["model"], nonadditive_metrics["test_RMSE"])
    axes[1].set_title("Test RMSE на неаддитивных данных")
    axes[1].set_ylabel("RMSE")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "case7_nonadditive_comparison.png", dpi=180)
    plt.close(fig)

    best = sweep.loc[sweep["test_RMSE"].idxmin()]
    return {"best_M": int(best["M"]), "best_test_RMSE": float(best["test_RMSE"])}


def main():
    case3 = save_case3_outputs()
    case7 = save_case7_outputs()
    print("case3_min_embedding_dim", case3["min_embedding_dim"])
    print("case3_coord_rmse", f"{case3['coord_rmse']:.6e}")
    print("case3_distance_E_rel", f"{case3['distance_E_rel']:.6e}")
    print("case7_best_M", case7["best_M"])
    print("case7_best_test_RMSE", f"{case7['best_test_RMSE']:.6f}")


if __name__ == "__main__":
    main()
