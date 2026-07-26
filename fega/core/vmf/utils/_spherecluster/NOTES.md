# vMF installer notes

The patch installer copies `_spk.py`, `_vmfm.py`, `_vmf_numerics.py`,
`_vmfm_factor.py`, and `_vmfm_factor_em.py` into the installed `spherecluster`
package. The numerical and factor modules use sibling imports rather than
`fega` imports so the patched package remains usable outside FEGA.

The normalizer implements the paper formula with SciPy's scaled Bessel function.
When that value underflows, SciPy adaptive quadrature evaluates the exact NIST
DLMF 10.32.2 integral after mode centering and scaling. It fails explicitly when
the integral misses its error contract and contains no asymptotic or alternative
scientific estimator.

Fixed-M initialization selects the largest finite full mixture likelihood. This
is the reproducibility convention recorded in `todo.md`, not a new scientific
gate or a guarantee of the global optimum.

The factor backend is an exact row-span reformulation: explicit rows use the
complete economy-QR coordinates, hidden Gram input uses unmodified Cholesky only
after eligibility checks, and ambient dimension remains authoritative in the
concentration and normalizer. The module is offline until FEGA integration is
separately approved; every guarded ambiguity requests a full dense rerun.
Promotion is limited to float64 clouds with `2 <= n <= 64`,
`2 <= d <= 256000`, and the exact calibrated NumPy/SciPy/BLAS fingerprint.
Input, environment, or terminal numerical failures select a complete dense
restart from the original rows, seed, initialization count, and iteration budget.
