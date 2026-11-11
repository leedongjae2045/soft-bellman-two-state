#!/usr/bin/env python3
"""
Reproduction of Section 9 ("Numerical Illustration") of
"Existence and Local Stability of Fixed Points of Projected Soft Bellman Iteration
 as the Discount Factor Varies: A Two-State Case Study" (v5).

Usage
    python reproduce_section9.py [--outdir DIR] [--no-fig]
    Colab / Jupyter:  paste the whole file into one cell and run it (figure is shown below the
                      cell and saved to the working directory, /content in Colab), or upload it and
                      run  !python reproduce_section9.py

Requires  numpy, scipy; matplotlib for Figure 5 (skip it with --no-fig).  All three are
pre-installed in Colab.

What it does (in the order of the paper's Section 9)
    1. Table 3   thresholds gamma_1, gamma_2, gamma_*, gamma^dagger.
    2. Grid      119 values of gamma per kernel: existence, uniqueness of the scalar-scan
                 roots, monotonicity of the spectral radius.
    3. Table 4   fixed points at gamma = gamma_*(1 - eps).
    4. Non-existence check   60 values of gamma in (gamma_*, 1): the scalar scan finds no sign
                 change of f_gamma, f_gamma > 0 at every scanned point, and eta(gamma) > 0.
    5. Iteration trajectories from u0 = 0 at 0.97 gamma_* and 1.02 gamma_* (400 iterations;
                 errors recorded at k = 120 and k = 400).
    6. Check     every number quoted in Section 9 is compared with the computed value
                 and reported as PASS / FAIL.
    7. Figure 5  written to  numerics.pdf  and  numerics.png.

Notation follows the paper: u = w/tau, x = exp(u), M = Psi * Pmat, F(u) = gamma M V(u).
"""
import argparse
import os
import sys
import warnings

import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.special import expit

# ======================================================================================
# 1. Model (Sections 4 and 6.1)
# ======================================================================================
PHI = np.array([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=float)  # rows: (s1,a1),(s1,a2),(s2,a1),(s2,a2)
PSI = np.linalg.solve(PHI.T @ PHI, PHI.T)                      # (Phi^T Phi)^{-1} Phi^T, 2 x 4


def kernel(p, alpha):
    """Pmat in R^{4x2} for a kernel with P(s1|s1,a1)=p, beta2-beta4=alpha (Section 6.1).

    Everything depends on the kernel only through (p, alpha); we take beta4 = t, beta2 = t+alpha,
    beta3 = 2-p-alpha-2t with t the midpoint of its admissible interval."""
    lo = max(0.0, -alpha, (1 - p - alpha) / 2)
    hi = min(1.0, 1 - alpha, (2 - p - alpha) / 2)
    assert lo <= hi + 1e-12, "(p, alpha) not in Theta"
    t = 0.5 * (lo + hi)
    b4, b2 = t, t + alpha
    b3 = 2 - p - alpha - 2 * t
    P1 = np.array([p, b2, b3, b4])                   # P(s1 | s, a), rows ordered as in PHI
    assert np.all((P1 >= -1e-12) & (P1 <= 1 + 1e-12))
    return np.column_stack([P1, 1 - P1])


def setup(p, alpha):
    """M = Psi*Pmat (eq. (2)), Delta = 4p-3alpha-2 (eq. (4)), X = 2-p+alpha."""
    M = PSI @ kernel(p, alpha)
    return M, 4 * p - 3 * alpha - 2, 2 - p + alpha


def V(u):
    """Soft state values V_j(u) = log sum_a exp(phi(s_j,a)^T u)."""
    Q = PHI @ u
    return np.array([np.logaddexp(Q[0], Q[1]), np.logaddexp(Q[2], Q[3])])


def F(u, gamma, M):
    """The iteration map u_{k+1} = gamma M V(u_k) (eq. (3))."""
    return gamma * M @ V(u)


def policy(u):
    """(pi_1, pi_2) = (pi(a1|s1), pi(a1|s2)) = (x1/(x1+x2), 1/(1+x1)), eq. (5)."""
    Q = PHI @ u
    return float(expit(Q[0] - Q[1])), float(expit(Q[2] - Q[3]))


def Pi_phi(u):
    """Pi^phi(w) = [[pi1, 1-pi1], [2-pi2, 1]] (Proposition 5.1)."""
    pi1, pi2 = policy(u)
    return np.array([[pi1, 1 - pi1], [2 - pi2, 1.0]])


def rho(u, gamma, M):
    """Spectral radius of J_gamma(w*) = gamma M Pi^phi(w*)."""
    return float(np.max(np.abs(np.linalg.eigvals(gamma * M @ Pi_phi(u)))))


# ======================================================================================
# 2. Thresholds (eq. (8)) and the sign-change point gamma^dagger (Lemma 7.4)
# ======================================================================================
def thresholds(p, alpha):
    """Returns (gamma_1, gamma_2, gamma_star, Delta)."""
    _, D, X = setup(p, alpha)
    g1 = 3 / (X + np.sqrt(X ** 2 + 2 * D)) if X ** 2 + 2 * D >= 0 else np.inf
    g2 = 6 / ((2 + p) + np.sqrt((2 + p) ** 2 - 4 * D)) if (2 + p) ** 2 >= 4 * D else np.inf
    return g1, g2, min(g1, g2), D


def gamma_dagger(p, alpha):
    """gamma^dagger = 3(alpha-p)/|Delta| if p < alpha, else None."""
    D = 4 * p - 3 * alpha - 2
    return 3 * (alpha - p) / abs(D) if p < alpha else None


# ======================================================================================
# 3. Fixed points: scalar reduction (Lemma 7.2) and fsolve for |eta| < 0.02
# ======================================================================================
ETA_SWITCH = 0.02          # below this |eta| the exponents A..D are huge -> use fsolve instead


def exponents(gamma, M, D):
    """eta and the exponents A,B,C,D of Lemma 7.2."""
    detM = D / 9
    eta = M[0, 0] - gamma * detM
    a = gamma * detM
    b = gamma * detM + M[1, 0]
    c = -M[0, 1]
    d = 1 / gamma - M[0, 1] - M[1, 1]
    return eta, a / eta, b / eta, c / eta, d / eta


def _scaled_f(t, A, B, C, D):
    """f(x1) = x1 + (1+x1)^A x1^B - (1+x1)^C x1^D at x1 = exp(t), multiplied by exp(-m) > 0.

    The three summands are formed in log-space and divided by the largest one, so there is no
    overflow and the sign (which is all the root search uses) is unchanged."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    l1p = np.logaddexp(0.0, t)                               # log(1 + x1)
    T = np.vstack([t, A * l1p + B * t, C * l1p + D * t])     # logs of the three summands
    m = T.max(axis=0)
    return np.exp(T[0] - m) + np.exp(T[1] - m) - np.exp(T[2] - m)


def t_grids(tmax=1e4):
    """The two grids of the paper in t = log x1: 24,001 equally spaced points on [-60, 60] and
    4,000 geometrically spaced points in (60, tmax] (the point 60 itself belongs to the first)."""
    return [np.linspace(-60.0, 60.0, 24001), np.geomspace(60.0, tmax, 4001)[1:]]


def scalar_scan(gamma, M, D, tmax=1e4):
    """All sign changes of f in t=log x1 on the two grids of t_grids, joined into one array so that
    the interval between the last point of the first grid (t = 60) and the first point of the
    second is also tested; each sign change is refined by Brent's method;
    u = (t, log x2) with x2 = (1+x1)^A x1^B."""
    _, A, B, C, Dd = exponents(gamma, M, D)
    grids = [np.concatenate(t_grids(tmax))]      # one array: the sign change across t = 60 is tested too
    f1 = lambda t: float(_scaled_f(t, A, B, C, Dd)[0])
    roots = []
    for g in grids:
        v = _scaled_f(g, A, B, C, Dd)
        for i in np.nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]:
            roots.append(brentq(f1, g[i], g[i + 1], xtol=1e-14, maxiter=500))
    return [np.array([t, A * np.logaddexp(0.0, t) + B * t]) for t in roots]


def residual_ok(u, gamma, M):
    """Acceptance test of the paper: ||F(u)-u||_inf <= 1e-9 max{1, ||u||_inf}."""
    return bool(np.max(np.abs(F(u, gamma, M) - u)) <= 1e-9 * max(1.0, np.max(np.abs(u))))


def solve_at(gamma, p, alpha, prev=None, tmax=1e4):
    """Fixed points at one gamma.  |eta| >= 0.02: scalar scan.  |eta| < 0.02: fsolve started from
    the solution at the previous grid value (zeros if there is none).  Only accepted solutions are
    returned; `method` records which route was used."""
    M, D, _ = setup(p, alpha)
    eta = M[0, 0] - gamma * D / 9
    if abs(eta) < ETA_SWITCH:
        u0 = np.zeros(2) if prev is None else np.asarray(prev, dtype=float)
        # fsolve may stop with status 3 ("xtol too small") after reaching machine precision, so the
        # paper's residual test, not fsolve's exit status, decides whether the solution is accepted.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            u = fsolve(lambda v: F(v, gamma, M) - v, u0, xtol=1e-14)
        good = [u] if residual_ok(u, gamma, M) else []
        return dict(roots=good, method="fsolve", eta=eta)
    good = [u for u in scalar_scan(gamma, M, D, tmax) if residual_ok(u, gamma, M)]
    return dict(roots=good, method="scalar", eta=eta)


# ======================================================================================
# 4. The experiments
# ======================================================================================
KERNELS = [(0.9, 0.5), (0.6, 0.9), (5 / 6, 1.0)]
SUBTITLE = [r"$\gamma_1<\gamma_2$", r"$\gamma_2<\gamma_1$", r"$\gamma_1=\gamma_2$ (tie)"]
EPS_TABLE4 = (1e-2, 1e-4, 1e-6)


def make_grid(gs):
    """119 values: 60 equally spaced in [0.02, 0.9 gs] and gs(1-eps), eps geometric in [1e-1,1e-4]
    (eps=1e-1 is 0.9 gs, already in the first block).  Increasing in gamma."""
    first = np.linspace(0.02, 0.9 * gs, 60)
    second = gs * (1 - np.geomspace(1e-1, 1e-4, 60))[1:]
    return np.concatenate([first, second])


def run_grid(p, alpha):
    """Fixed point, ||u*||, rho along the grid; fsolve is warm-started from the previous point."""
    M, _, _ = setup(p, alpha)
    gam = make_grid(thresholds(p, alpha)[2])
    prev, rows = None, []
    for g in gam:
        r = solve_at(g, p, alpha, prev)
        if r["roots"]:
            prev = r["roots"][0]
            u = r["roots"][0]
            rows.append(dict(gamma=g, n=len(r["roots"]), method=r["method"],
                             norm=float(np.linalg.norm(u)), rho=rho(u, g, M)))
        else:
            rows.append(dict(gamma=g, n=0, method=r["method"], norm=np.nan, rho=np.nan))
    return rows


def table4_rows(p, alpha):
    """Rows of Table 4: gamma = gs(1-eps); eps=1e-6 uses the search range t <= 1e7."""
    M, _, _ = setup(p, alpha)
    gs = thresholds(p, alpha)[2]
    out = []
    for eps in EPS_TABLE4:
        g = gs * (1 - eps)
        r = solve_at(g, p, alpha, None, tmax=1e7 if eps == 1e-6 else 1e4)
        assert len(r["roots"]) == 1 and r["method"] == "scalar", (p, alpha, eps, len(r["roots"]))
        u = r["roots"][0]
        rh = rho(u, g, M)
        pi1, pi2 = policy(u)
        out.append(dict(eps=eps, gamma=g, norm=float(np.linalg.norm(u)), gap=1 - rh,
                        ratio=(1 - rh) / eps, pi1=pi1, pi2=pi2, u=u))
    return out


def nonexistence_scan(p, alpha, n=60):
    """Scalar scan for gamma = gs + (1-gs) s, s geometric in [1e-3, 0.99] (n values, all in
    (gs, 1)), on the same t-grids as above and with f evaluated in log form.
    Returns the gammas, the eta values, the number of sign changes of f and min f/scale over all
    scanned points (the sign of f is all that matters; scale > 0)."""
    M, D, _ = setup(p, alpha)
    gs = thresholds(p, alpha)[2]
    gam = gs + (1 - gs) * np.geomspace(1e-3, 0.99, n)
    etas, changes, fmin = [], 0, np.inf
    for g in gam:
        eta, A, B, C, Dd = exponents(g, M, D)
        etas.append(eta)
        for grid in [np.concatenate(t_grids())]:
            v = _scaled_f(grid, A, B, C, Dd)
            changes += int(np.count_nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0))
            fmin = min(fmin, float(v.min()))
    return gam, np.array(etas), changes, fmin


def trajectory(gamma, M, n, ustar=None):
    """||u_k|| (and ||u_k - u*||) for k = 1..n, u_0 = 0, u_{k+1} = gamma M V(u_k)."""
    u = np.zeros(2)
    norms, errs = np.empty(n), np.empty(n)
    for k in range(n):
        u = F(u, gamma, M)
        norms[k] = np.linalg.norm(u)
        errs[k] = np.linalg.norm(u - ustar) if ustar is not None else np.nan
    return norms, errs


# ======================================================================================
# 5. The numbers printed in Section 9 (copied from the paper) and the comparison
# ======================================================================================
# Table 3: (Delta, gamma_1, gamma_2, gamma_*, gamma^dagger) ; compared after rounding to 4 decimals
TABLE3 = {
    (0.9, 0.5):  (0.1,  0.9199, 1.0471, 0.9199, None),
    (0.6, 0.9):  (-2.3, 0.9583, 0.9098, 0.9098, 0.3913),
    (5 / 6, 1.0): (-5 / 3, 0.9000, 0.9000, 0.9000, 0.3),
}
# Table 4: (gamma, ||u*||, 1-rho, (1-rho)/eps, pi1, pi2) as printed (strings fix the precision)
TABLE4 = {
    (0.9, 0.5): [("0.91067", "5.54", "3.37e-2", "3.37", "0.0714", "8.18e-2"),
                 ("0.91978", "13.9", "7.18e-4", "7.18", "0.0016", "2.28e-3"),
                 ("0.91987", "23.2", "1.13e-5", "11.26", "2.18e-5", "4.05e-5")],
    (0.6, 0.9): [("0.90068", "5.33", "2.36e-2", "2.36", "0.6872", "1.56e-2"),
                 ("0.90969", "21.9", "5.05e-4", "5.05", "0.9827", "2.81e-8"),
                 ("0.90978", "43.2", "8.97e-6", "8.97", "0.9997", "1.31e-15")],
    (5 / 6, 1.0): [("0.89100", "10.8", "1.02e-2", "1.02", "0.4157", "5.70e-4"),
                   ("0.89991", "1.08e3", "1.00e-4", "1.00", "0.4148", "<1e-300"),
                   ("0.90000", "1.08e5", "1.00e-6", "1.00", "0.4147", "<1e-300")],
}


def matches(x, printed):
    """True if x rounded to the number of significant digits of `printed` equals `printed`."""
    if printed.startswith("<"):
        return x < float(printed[1:])
    mant = printed.lower().split("e")[0].lstrip("+-")
    nd = len(mant.replace(".", "").lstrip("0"))
    return float(f"{x:.{nd - 1}e}") == float(printed)


CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((bool(ok), name, detail))


# ======================================================================================
# 6. Figure 5
# ======================================================================================
def make_figure(results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "mathtext.fontset": "dejavusans",
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    # the dashed style for gamma > gamma_* is a second, colour-independent cue.
    INK, C_NORM, C_RHO, C_BELOW, C_ABOVE = "#222222", "#1b5fa5", "#c4501b", "#0f8a5f", "#a3287f"
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.5))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.075, hspace=0.42, wspace=0.30)

    for c, R in enumerate(results):
        gs, gd = R["gs"], R["gd"]
        gam = np.array([r["gamma"] for r in R["grid"]])
        nrm = np.array([r["norm"] for r in R["grid"]])
        rh = np.array([r["rho"] for r in R["grid"]])
        top, mid, bot = axes[0, c], axes[1, c], axes[2, c]

        top.set_title(rf"$(p,\alpha)=({R['plabel']},\,{R['alpha']:g})$" + "\n" + SUBTITLE[c],
                      fontsize=7, color=INK)
        top.semilogy(gam, nrm, color=C_NORM, lw=1.4)
        mid.plot(gam, rh, color=C_RHO, lw=1.4)
        mid.axhline(1.0, color=INK, ls=":", lw=0.8)
        mid.set_ylim(0, 1.05)
        mid.set_yticks(np.arange(0, 1.01, 0.2))
        for ax in (top, mid):
            ax.axvline(gs, color="k", ls="--", lw=0.8)
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.01, 0.2))
        top.text(gs + 0.012, 0.03, rf"$\gamma_*={gs:.4f}$", transform=top.get_xaxis_transform(),
                 rotation=90, va="bottom", ha="left", fontsize=5.5, color=INK)
        if gd is not None:
            top.axvline(gd, color="0.4", ls=":", lw=0.8)
        mid.set_xlabel(r"$\gamma$")

        ks = np.arange(1, len(R["below"]) + 1)
        bot.semilogy(ks, R["below"], color=C_BELOW, lw=1.2,
                     label=rf"$\gamma={R['g_below']:.3f}$ ($<\gamma_*$)")
        bot.semilogy(ks, R["above"], color=C_ABOVE, lw=1.2, ls=(0, (5, 2)),
                     label=rf"$\gamma={R['g_above']:.3f}$ ($>\gamma_*$)")
        bot.set_xlim(-6, len(ks) + 6)
        bot.set_xticks([0, 50, 100])
        bot.set_xlabel(r"iteration $k$")
        bot.legend(loc="upper left", fontsize=6, frameon=False, handlelength=1.8, borderaxespad=0.2)
        if c == 0:
            top.set_ylabel(r"$\|u^*\|\ \ (u^*=w^*/\tau)$")
            mid.set_ylabel(r"$\rho(J_\gamma(w^*))$")
            bot.set_ylabel(r"$\|u_k\|,\ u_0=0$")
        for ax in axes[:, c]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors=INK)

    paths = [os.path.join(outdir, "numerics.pdf"), os.path.join(outdir, "numerics.png")]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=200)
    plt.close(fig)
    return paths


# ======================================================================================
# 7. Main
# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".", help="where to write numerics.pdf / numerics.png")
    ap.add_argument("--no-fig", action="store_true", help="skip Figure 5")
    # parse_known_args: Jupyter/Colab put their own "-f kernel.json" into sys.argv
    args, _ = ap.parse_known_args()
    os.makedirs(args.outdir, exist_ok=True)

    results = []

    # ---------------- Table 3 ----------------
    print("=== Table 3: thresholds (eq. (8)) ===")
    print(f"{'(p,alpha)':<12}{'Delta':>9}{'gamma_1':>9}{'gamma_2':>9}{'gamma_*':>9}{'gamma_dag':>11}")
    for (p, a) in KERNELS:
        g1, g2, gs, D = thresholds(p, a)
        gd = gamma_dagger(p, a)
        print(f"({p:.3f},{a:.1f}){D:11.4f}{g1:9.4f}{g2:9.4f}{gs:9.4f}{'' if gd is None else f'{gd:11.4f}':>11}")
        eD, e1, e2, es, ed = TABLE3[(p, a)]
        ok = (0.90 - 1e-9 <= gs <= 0.92                       # Section 9: 0.90 <= gamma_* <= 0.92 (gs = 0.9 up to rounding on the tie curve)
              and abs(D - eD) < 1e-9 and round(g1, 4) == e1 and round(g2, 4) == e2 and round(gs, 4) == es
              and ((gd is None and ed is None) or (gd is not None and ed is not None and round(gd, 4) == ed)))
        check(f"Table 3, (p,alpha)=({p:.3f},{a})", ok)
        results.append(dict(p=p, alpha=a, plabel="5/6" if abs(p - 5 / 6) < 1e-12 else f"{p:g}",
                            gs=gs, gd=gd))

    # ---------------- 119-point grid ----------------
    print("\n=== Grid of 119 values of gamma (Theorem 6.1) ===")
    for R in results:
        rows = run_grid(R["p"], R["alpha"])
        R["grid"] = rows
        n = len(rows)
        found = sum(r["n"] >= 1 for r in rows)
        scal = [r for r in rows if r["method"] == "scalar"]
        n_fs = n - len(scal)
        uniq = sum(r["n"] == 1 for r in scal)
        rh = np.array([r["rho"] for r in rows])
        mono = bool(np.all(np.isfinite(rh)) and np.all(np.diff(rh) > -1e-12))
        print(f"({R['p']:.3f},{R['alpha']:.1f}): {n} grid values | fixed point found {found}/{n} | "
              f"scalar scan {len(scal)} pts, exactly one root at {uniq}/{len(scal)} | fsolve {n_fs} pts | "
              f"rho {rh[0]:.4f} -> {rh[-1]:.6f}, monotone={mono}, max<1={bool(np.nanmax(rh) < 1)}")
        tag = f"({R['p']:.3f},{R['alpha']:.1f})"
        check(f"(a) {tag}: 119 grid values", n == 119)
        check(f"(a) {tag}: fixed point at every grid value", found == n)
        check(f"(a) {tag}: scalar scan returned exactly one root", uniq == len(scal))
        check(f"(b) {tag}: rho monotone along grid", mono)
        check(f"(b) {tag}: rho(0.02) about 0.02", 0.015 < rh[0] < 0.025, f"{rh[0]:.4f}")
        check(f"(b) {tag}: rho at last grid value > 0.999 and < 1", 0.999 < rh[-1] < 1, f"{rh[-1]:.6f}")
        check(f"(b) {tag}: rho < 1 throughout", bool(np.nanmax(rh) < 1))

    # ---------------- Table 4 ----------------
    print("\n=== Table 4: gamma = gamma_*(1-eps) ===")
    print(f"{'(p,alpha)':<12}{'eps':>7}{'gamma':>10}{'||u*||':>11}{'1-rho':>11}{'(1-rho)/eps':>13}"
          f"{'pi1':>11}{'pi2':>11}  match")
    for R in results:
        rows = table4_rows(R["p"], R["alpha"])
        R["t4"] = rows
        for i, (r, exp) in enumerate(zip(rows, TABLE4[(R["p"], R["alpha"])])):
            vals = [r["gamma"], r["norm"], r["gap"], r["ratio"], r["pi1"], r["pi2"]]
            ok = all(matches(v, e) for v, e in zip(vals, exp))
            label = f"({R['p']:.3f},{R['alpha']:.1f})" if i == 0 else ""
            pi2_txt = "<1e-300" if r["pi2"] < 1e-300 else f"{r['pi2']:.3e}"
            print(f"{label:<12}{r['eps']:7.0e}{r['gamma']:10.5f}{r['norm']:11.4g}{r['gap']:11.3e}"
                  f"{r['ratio']:13.3f}{r['pi1']:11.4g}{pi2_txt:>11}"
                  f"  {'ok' if ok else 'MISMATCH'}")
            check(f"Table 4, ({R['p']:.3f},{R['alpha']:.1f}), eps={r['eps']:.0e}", ok)

    # ---------------- Non-existence for gamma in (gamma_*, 1) ----------------
    print("\n=== Non-existence check: 60 values of gamma in (gamma_*, 1) ===")
    for R in results:
        gam, etas, changes, fmin = nonexistence_scan(R["p"], R["alpha"])
        tag = f"({R['p']:.3f},{R['alpha']:.1f})"
        print(f"{tag}: gamma in [{gam[0]:.5f}, {gam[-1]:.5f}] | sign changes of f: {changes} | "
              f"min of scaled f: {fmin:.3e} | min eta: {etas.min():.4f}")
        check(f"(a) {tag}: non-existence scan uses 60 values of gamma in (gamma_*, 1)",
              len(gam) == 60 and bool(np.all(gam > R["gs"]) and np.all(gam < 1)))
        check(f"(a) {tag}: no sign change of f_gamma for gamma in (gamma_*, 1)", changes == 0)
        check(f"(a) {tag}: f_gamma > 0 at every scanned point", fmin > 0, f"{fmin:.3e}")
        check(f"(a) {tag}: eta(gamma) >= 0.11 on the scanned values", bool(etas.min() >= 0.11),
              f"{etas.min():.4f}")

    # ---------------- (c), (d): limits and rates ----------------
    t1, t2, t3 = (R["t4"] for R in results)
    check("(c) kernel 1: pi1 ~ 2e-5 at eps=1e-6", 1.5e-5 < t1[2]["pi1"] < 2.5e-5, f"{t1[2]['pi1']:.3g}")
    check("(c) kernel 1: policy -> (0,0)", t1[2]["pi1"] < 1e-3 and t1[2]["pi2"] < 1e-3)
    check("(c) kernel 2: policy -> (1,0)", t2[2]["pi1"] > 0.999 and t2[2]["pi2"] < 1e-10)
    check("(c) tie: pi2 -> 0", t3[2]["pi2"] < 1e-100)
    check("(c) tie: pi1 settles near 0.415", round(t3[2]["pi1"], 3) == 0.415, f"{t3[2]['pi1']:.5f}")
    d21 = float(t3[2]["u"][1] - t3[2]["u"][0])
    check("(c) tie: u2*-u1* -> 0.344 (as stated in the paper)", round(d21, 3) == 0.344,
          f"computed {d21:.5f} at eps=1e-6 (paper: 0.344)")
    check("(d) tie: (1-rho)/eps about 1", all(0.99 < r["ratio"] < 1.02 for r in t3))
    check("(d) tie: ||u*|| ~ 1/eps (x100 from eps=1e-4 to 1e-6)",
          90 < t3[2]["norm"] / t3[1]["norm"] < 110, f"{t3[2]['norm'] / t3[1]['norm']:.1f}")

    # ---------------- Trajectories ----------------
    print("\n=== Trajectories, u0 = 0 ===")
    for R in results:
        p, a, gs = R["p"], R["alpha"], R["gs"]
        M, _, _ = setup(p, a)
        gb, ga = 0.97 * gs, 1.02 * gs
        ustar = solve_at(gb, p, a)["roots"][0]
        nb, eb = trajectory(gb, M, 400, ustar)
        na, _ = trajectory(ga, M, 400)
        R.update(g_below=gb, g_above=ga, below=nb[:120], above=na[:120])
        tag = f"({p:.3f},{a:.1f})"
        print(f"{tag}: gamma={gb:.4f}: ||u_120-u*||={eb[119]:.2e}, ||u_400-u*||={eb[399]:.2e} | "
              f"gamma={ga:.4f}: ||u_120||={na[119]:.4g}, ||u_400||={na[399]:.4g}")
        check(f"(e) {tag}: ||u_120-u*|| < 1e-2 at 0.97 gamma_*", eb[119] < 1e-2, f"{eb[119]:.2e}")
        check(f"(e) {tag}: ||u_400-u*|| < 1e-7 at 0.97 gamma_*", eb[399] < 1e-7, f"{eb[399]:.2e}")
        check(f"(e) {tag}: ||u_400|| > 1e3 and still growing at 1.02 gamma_*",
              na[399] > 1e3 and na[399] > na[119], f"{na[399]:.4g}")

    # ---------------- Summary ----------------
    print("\n=== Comparison with the numbers quoted in Section 9 ===")
    for ok, name, detail in CHECKS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))
    n_fail = sum(not ok for ok, _, _ in CHECKS)
    print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed" + ("" if n_fail == 0 else f", {n_fail} FAILED"))

    # ---------------- Figure 5 ----------------
    if not args.no_fig:
        try:
            paths = make_figure(results, args.outdir)
            print("Figure 5 written to: " + ", ".join(os.path.abspath(q) for q in paths))
            if IN_NOTEBOOK:                      # show the figure under the cell
                from IPython.display import Image, display
                display(Image(filename=paths[1]))
        except ImportError:
            print("matplotlib not installed: Figure 5 skipped (pip install matplotlib)")
    return 1 if n_fail else 0


IN_NOTEBOOK = "ipykernel" in sys.modules or "google.colab" in sys.modules

if __name__ == "__main__":
    status = main()
    if not IN_NOTEBOOK:                          # in a notebook, sys.exit would only raise SystemExit
        sys.exit(status)
