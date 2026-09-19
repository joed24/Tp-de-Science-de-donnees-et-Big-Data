"""Point d'entrée de l'application Nuées dynamiques."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

from src.dynamic_clouds import DynamicClouds, load_numeric_csv, make_demo_dataset

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODE_LABELS = {
    "points": "Points représentatifs",
    "axes": "Axes factoriels",
    "distribution": "Distribution probabiliste",
    "structure": "Structure représentative",
    "point": "Point (K-Means)",
}


def _ellipse_from_covariance(ax, mean, covariance, n_std=2.0):
    """Trace une ellipse de covariance pour un problème 2D."""
    cov = np.asarray(covariance, dtype=float)[:2, :2]
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.maximum(vals, 0.0)
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(xy=np.asarray(mean)[:2], width=width, height=height, angle=angle,
                      fill=False, linewidth=1.8)
    ax.add_patch(ellipse)


def plot_result(X, model, title, filename):
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    labels = model.labels_

    for j in range(model.k):
        pts = X[labels == j]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=24, alpha=0.65, label=f"Classe {j + 1}")

    for proto in model.prototypes:
        if model.mode == "point":
            p = proto.center
            ax.scatter(p[0], p[1], marker="X", s=180, edgecolor="black", linewidth=1.2)

        elif model.mode == "points":
            reps = proto.representatives
            ax.scatter(reps[:, 0], reps[:, 1], marker="D", s=78,
                       edgecolor="black", linewidth=0.8)

        elif model.mode == "axes":
            c = proto.center
            ax.scatter(c[0], c[1], marker="X", s=150, edgecolor="black", linewidth=1.0)
            for r, v in enumerate(proto.axes.T):
                scale = 1.5 * np.sqrt(max(float(proto.eigenvalues[r]), 0.1))
                ax.plot([c[0] - scale * v[0], c[0] + scale * v[0]],
                        [c[1] - scale * v[1], c[1] + scale * v[1]], linewidth=2.0)

        elif model.mode == "distribution":
            ax.scatter(proto.mean[0], proto.mean[1], marker="X", s=150,
                       edgecolor="black", linewidth=1.0)
            if X.shape[1] >= 2:
                _ellipse_from_covariance(ax, proto.mean, proto.covariance)

        elif model.mode == "structure":
            ax.scatter(proto.center[0], proto.center[1], marker="X", s=150,
                       edgecolor="black", linewidth=1.0)
            reps = proto.representatives
            ax.scatter(reps[:, 0], reps[:, 1], marker="D", s=65,
                       edgecolor="black", linewidth=0.8)
            if X.shape[1] >= 2:
                _ellipse_from_covariance(ax, proto.center, proto.covariance)

    ax.set_title(title)
    ax.set_xlabel("Variable 1 (standardisée)")
    ax.set_ylabel("Variable 2 (standardisée)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=160)
    plt.close(fig)


def plot_history(model, filename):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(range(1, len(model.history_) + 1), model.history_, marker="o")
    ax.set_xlabel("Itération")
    ax.set_ylabel("Critère d'adéquation")
    ax.set_title(f"Convergence — {MODE_LABELS[model.mode]}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=160)
    plt.close(fig)


def run_mode(X, mode, k, q, seed):
    model = DynamicClouds(k=k, mode=mode, q=q, seed=seed)
    model.fit(X)
    plot_result(X, model, f"Nuées dynamiques — {MODE_LABELS[mode]} — K={k}",
                f"{mode}_partition.png")
    plot_history(model, f"{mode}_convergence.png")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Nuées dynamiques — implémentation from scratch")
    parser.add_argument("--mode", choices=[*MODE_LABELS.keys(), "all"], default=None)
    parser.add_argument("--k", type=int, default=4, help="Nombre de classes")
    parser.add_argument("--q", type=int, default=3,
                        help="Nombre d'étalons ou d'axes demandé")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default=None,
                        help="Chemin vers un CSV numérique optionnel")
    return parser.parse_args()


def interactive_choice():
    print("\n============================================================")
    print("              CLASSIFICATION PAR NUÉES DYNAMIQUES")
    print("============================================================")
    print("1 - Ensemble de points représentatifs")
    print("2 - Axes factoriels")
    print("3 - Distribution probabiliste")
    print("4 - Structure représentative")
    print("5 - Point (K-Means)")
    print("6 - Exécuter les cinq représentations")
    print("============================================================")
    choice = input("Votre choix [1-6] : ").strip()
    mapping = {
        "1": "points", "2": "axes", "3": "distribution",
        "4": "structure", "5": "point", "6": "all",
    }
    if choice not in mapping:
        raise ValueError("Choix invalide.")
    k = int(input("Nombre de classes K [4] : ") or "4")
    q = int(input("Nombre d'étalons/axes q [3] : ") or "3")
    return mapping[choice], k, q


def main():
    args = parse_args()
    if args.mode is None:
        mode, k, q = interactive_choice()
    else:
        mode, k, q = args.mode, args.k, args.q

    if args.data:
        X, df, ignored = load_numeric_csv(args.data)
        print(f"Dataset chargé : {args.data}")
        print(f"Observations : {len(df)} | Variables numériques : {X.shape[1]}")
        if ignored:
            print(f"Colonnes ignorées : {', '.join(ignored)}")
    else:
        X, _ = make_demo_dataset(seed=args.seed)
        pd.DataFrame(X, columns=["variable_1", "variable_2"]).to_csv(
            ROOT / "data" / "dataset_demo_standardized.csv", index=False
        )
        print(f"Dataset de démonstration : {len(X)} observations, {X.shape[1]} variables.")

    modes = list(MODE_LABELS) if mode == "all" else [mode]
    summary = []

    for current in modes:
        model = run_mode(X, current, k, q, args.seed)
        criterion = model.history_[-1]
        sizes = np.bincount(model.labels_, minlength=k).tolist()
        summary.append({
            "representation": current,
            "iterations": len(model.history_),
            "criterion_final": criterion,
            "cluster_sizes": ";".join(map(str, sizes)),
        })
        print(f"\n--- {MODE_LABELS[current]} ---")
        print(f"Itérations : {len(model.history_)}")
        print(f"Critère final : {criterion:.6f}")
        print(f"Taille des classes : {sizes}")

    pd.DataFrame(summary).to_csv(ROOT / "results" / "summary.csv", index=False)
    print("\n============================================================")
    print("Exécution terminée.")
    print("Figures : results/figures/")
    print("Résumé  : results/summary.csv")
    print("============================================================")


if __name__ == "__main__":
    main()
