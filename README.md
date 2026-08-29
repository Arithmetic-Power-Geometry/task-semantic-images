# Task-Semantic Images (TSI) — Reproducibility Package

Reference implementation for **Beyond Complete Compilation: Task-Semantic Images and the Complexity of Computing Only What a Query Can See**.

## Reproduce everything

```bash
python -m pip install -e .[dev]
pytest -q
python -m tasksemantic.experiments
```

or simply:

```bash
make reproduce
```

The workflow regenerates all CSV tables and PDF/PNG figures under `results/` and `figures/`.

## Interactive app

```bash
python app.py
```

The app exposes four exact finite laboratories: cardinality images, proper-affine image verification, minimum-cover semantic-resolution states, and an exact fixed-order OBDD diagnostic. Parameters can be changed interactively. Exhaustive modes are deliberately capped at small `n` because they enumerate all assignments; polynomial matching-based cardinality experiments scale substantially farther.

## CLI examples

```bash
tasksemantic cardinality --n 40 --family cycle
tasksemantic affine --n 12 --k 4
tasksemantic barrier --n 12 --p 0.3 --seed 2026
tasksemantic bdd --n 18 --family cycle
tasksemantic reproduce
```

## Scientific scope

The package verifies finite instances of the paper's exact identities and reports measured runtimes/state counts. The exponential DNNF lower bound and NP-completeness boundary are literature theorems; this software does **not** fabricate empirical DNNF lower bounds or claim to solve P vs NP.

## Citation

Akhtar, M. A. K. (2026). *Beyond Complete Compilation: Task-Semantic Images and the Complexity of Computing Only What a Query Can See* (Version V1). Zenodo. https://doi.org/10.5281/zenodo.22160359


## License

Apache-2.0. Copyright 2026 Mohammad Amir Khusru Akhtar.
