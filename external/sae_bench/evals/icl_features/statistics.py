from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def one_sample_mean_degradation_test(
    comparison_accuracies: Sequence[float],
    *,
    reference_accuracy: float,
) -> dict[str, Any] | None:
    """Test whether mean comparison accuracy is below a fixed reference accuracy."""
    if len(comparison_accuracies) < 2:
        return None
    import numpy as np
    from scipy.stats import ttest_1samp

    values = np.asarray(comparison_accuracies, dtype=np.float64)
    drops = float(reference_accuracy) - values
    if np.allclose(drops, 0.0, rtol=0.0, atol=1.0e-15):
        statistic = 0.0
        p_value = 1.0
    elif float(np.std(values, ddof=1)) <= 1.0e-15:
        statistic = float("inf") if float(np.mean(drops)) > 0.0 else float("-inf")
        p_value = 0.0 if statistic > 0.0 else 1.0
    else:
        result = ttest_1samp(
            values,
            popmean=float(reference_accuracy),
            alternative="less",
        )
        statistic = float(-result.statistic)
        p_value = float(result.pvalue)
    return {
        "test": "one_sided_one_sample_t_test_over_random_trials",
        "alternative": "mean random-ablation accuracy below baseline accuracy",
        "trials": int(values.size),
        "mean_accuracy": float(np.mean(values)),
        "sample_std_accuracy": float(np.std(values, ddof=1)),
        "mean_accuracy_drop": float(np.mean(drops)),
        "sample_std_accuracy_drop": float(np.std(drops, ddof=1)),
        "t_statistic_for_accuracy_drop": statistic,
        "p_value_one_sided": p_value,
    }


def exact_binomial_upper_tail(successes: int, trials: int) -> float:
    """Return P[X >= successes] for X ~ Binomial(trials, 0.5)."""
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if trials == 0:
        return 1.0
    from scipy.stats import binom

    return float(binom.sf(successes - 1, trials, 0.5))


def exact_binomial_log10_upper_tail(successes: int, trials: int) -> float:
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if trials == 0:
        return 0.0
    import numpy as np
    from scipy.special import logsumexp
    from scipy.stats import binom

    support = np.arange(successes, trials + 1, dtype=np.int64)
    log_p = float(logsumexp(binom.logpmf(support, trials, 0.5)))
    return log_p / float(np.log(10.0))


def exact_mcnemar(
    reference_correct: Sequence[bool],
    comparison_correct: Sequence[bool],
) -> dict[str, Any]:
    """Exact paired McNemar test for a one-sided accuracy degradation."""
    if len(reference_correct) != len(comparison_correct):
        raise ValueError("Paired outcomes must have the same length")
    reference_only = sum(
        bool(reference) and not bool(comparison)
        for reference, comparison in zip(
            reference_correct, comparison_correct, strict=True
        )
    )
    comparison_only = sum(
        not bool(reference) and bool(comparison)
        for reference, comparison in zip(
            reference_correct, comparison_correct, strict=True
        )
    )
    discordant = reference_only + comparison_only
    one_sided = exact_binomial_upper_tail(reference_only, discordant)
    two_sided = min(
        1.0,
        2.0
        * min(
            exact_binomial_upper_tail(reference_only, discordant),
            exact_binomial_upper_tail(comparison_only, discordant),
        ),
    )
    return {
        "test": "exact_paired_mcnemar",
        "alternative": "reference accuracy greater than comparison accuracy",
        "reference_correct_comparison_wrong": reference_only,
        "reference_wrong_comparison_correct": comparison_only,
        "discordant_pairs": discordant,
        "p_value_one_sided": one_sided,
        "log10_p_value_one_sided": exact_binomial_log10_upper_tail(
            reference_only, discordant
        ),
        "p_value_two_sided": two_sided,
    }


def empirical_upper_tail(observed: float, null_values: Sequence[float]) -> float:
    """Finite-sample corrected empirical upper-tail p-value."""
    if not null_values:
        raise ValueError("At least one null value is required")
    exceedances = sum(float(value) >= float(observed) for value in null_values)
    return (exceedances + 1.0) / (len(null_values) + 1.0)


def accuracy_summary(correct: Sequence[bool]) -> dict[str, Any]:
    total = len(correct)
    count = sum(bool(value) for value in correct)
    return {
        "correct": count,
        "total": total,
        "accuracy": count / total if total else 0.0,
    }
