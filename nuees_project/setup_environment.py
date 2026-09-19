"""Vérifie les dépendances et n'installe que celles qui manquent.

Usage :
    python setup_environment.py

Le script tente d'abord un import et une vérification de version. Ainsi,
si numpy/pandas/matplotlib sont déjà correctement installés, aucun téléchargement
n'est lancé.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

REQUIREMENTS = {
    "numpy": ("numpy", "1.26"),
    "pandas": ("pandas", "2.2"),
    "matplotlib": ("matplotlib", "3.8"),
}


def version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for token in value.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_package(distribution: str, minimum: str) -> bool:
    try:
        importlib.import_module(distribution)
        installed = version(distribution)
        ok = version_tuple(installed) >= version_tuple(minimum)
        status = "OK" if ok else f"version {installed} < {minimum}"
        print(f"[{'OK' if ok else 'À METTRE À JOUR'}] {distribution}: {installed} ({status})")
        return ok
    except (ImportError, PackageNotFoundError):
        print(f"[MANQUANT] {distribution}")
        return False


def main() -> None:
    missing = []
    for distribution, (_, minimum) in REQUIREMENTS.items():
        if not check_package(distribution, minimum):
            missing.append(f"{distribution}>={minimum}")

    if not missing:
        print("\nToutes les dépendances nécessaires sont déjà disponibles.")
        print("Aucun téléchargement n'a été effectué.")
        return

    print("\nDépendances à installer/mettre à jour :")
    print("  " + ", ".join(missing))
    answer = input("Autoriser pip à installer uniquement celles-ci ? [O/n] ").strip().lower()
    if answer not in {"", "o", "oui", "y", "yes"}:
        print("Installation annulée.")
        print("Le programme peut démarrer seulement si les dépendances manquantes sont installées.")
        return

    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    print("\nVérification finale :")
    for distribution, (_, minimum) in REQUIREMENTS.items():
        if not check_package(distribution, minimum):
            raise SystemExit(f"Échec de la vérification de {distribution}.")
    print("Environnement prêt.")


if __name__ == "__main__":
    main()
