import json
import sys
from pathlib import Path

PACKAGE = "insurance_pricing"

PYPROJECT = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "insurance-pricing"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
"""

REQUIREMENTS = """pandas
numpy
scipy
statsmodels
scikit-learn
xgboost
matplotlib
joblib
fastapi
uvicorn
pydantic
pytest
httpx
"""

GITIGNORE = """__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
venv/
*.egg-info/
.env
.DS_Store

data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
artifacts/*
!artifacts/.gitkeep
"""

DOCKERIGNORE = """.git
.venv
venv
data
notebooks
reports
tests
**/__pycache__
**/.ipynb_checkpoints
*.egg-info
"""

README = """# Motor insurance pricing

Claim frequency-severity modelling on motor insurance data: Poisson/NB/Gamma/Tweedie GLMs vs. XGBoost, served with FastAPI and Docker.
"""

NOTEBOOKS = [
    "01_exposure_and_leakage",
    "02_poisson_glm",
    "03_residuals_nb",
    "04_severity_pure_premium",
    "05_xgboost",
    "06_final_evaluation",
]

SRC_MODULES = [
    "__init__",
    "config",
    "data",
    "split",
    "preprocessing",
    "metrics",
    "models",
    "train",
    "predict",
]

GITKEEP_DIRS = [
    "data/raw",
    "data/processed",
    "artifacts",
    "reports/figures",
]


def empty_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1) + "\n"


def build_files():
    files = {
        "README.md": README,
        "pyproject.toml": PYPROJECT,
        "requirements.txt": REQUIREMENTS,
        ".gitignore": GITIGNORE,
        ".dockerignore": DOCKERIGNORE,
        "Dockerfile": "",
        "app/__init__.py": "",
        "app/main.py": "",
        "app/schemas.py": "",
        "tests/__init__.py": "",
        "tests/test_metrics.py": "",
        "tests/test_preprocessing.py": "",
        "tests/test_api.py": "",
    }
    for name in SRC_MODULES:
        files[f"src/{PACKAGE}/{name}.py"] = ""
    for name in NOTEBOOKS:
        files[f"notebooks/{name}.ipynb"] = empty_notebook()
    for d in GITKEEP_DIRS:
        files[f"{d}/.gitkeep"] = ""
    return files


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    root = root.resolve()
    created, skipped = [], []
    for rel, content in build_files().items():
        path = root / rel
        if path.exists():
            skipped.append(rel)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(rel)
    print(f"Root: {root}")
    print(f"Created {len(created)} files")
    for rel in created:
        print(f"  + {rel}")
    if skipped:
        print(f"Skipped {len(skipped)} existing files")
        for rel in skipped:
            print(f"  = {rel}")


if __name__ == "__main__":
    main()
