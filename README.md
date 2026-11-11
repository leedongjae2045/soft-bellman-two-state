# Existence and Local Stability of Fixed Points of Projected Soft Bellman Iteration as the Discount Factor Varies: A Two-State Case Study

Code accompanying Section 9 ("Numerical Illustration") of the paper above (author: Dongjae Lee).

`reproduce_section9.py` regenerates everything reported in Section 9:

- the thresholds of Table 3,
- the fixed points on the 119-point grid of discount factors,
- the non-existence check for gamma in (gamma_*, 1),
- the rows of Table 4,
- the iteration trajectories (u_0 = 0, 400 iterations),
- Figure 5.

It then compares every number quoted in Section 9 with the recomputed value and prints PASS or FAIL for each of the 62 comparisons.

## Requirements

- Python 3
- NumPy and SciPy
- Matplotlib (only for Figure 5; it can be skipped with `--no-fig`)

The results in the paper were produced with Python 3.13.18, NumPy 2.4.6 and SciPy 1.16.3.

The version used in the paper is the release `v1.0`.

## Usage

```
pip install numpy scipy matplotlib
python reproduce_section9.py
```

Options:

- `--outdir DIR`  write `numerics.pdf` and `numerics.png` to `DIR` (default: current directory)
- `--no-fig`      skip Figure 5

The script is deterministic and runs in a few seconds. The last lines of the output are

```
62/62 checks passed
```

and the exit code is 0 if all comparisons pass, 1 otherwise.

The script can also be pasted into a single cell of Google Colab or a Jupyter notebook; the figure is then displayed below the cell.

## License

MIT License. See the `LICENSE` file.
