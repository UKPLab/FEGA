### credit to: https://github.com/jasonlaska/spherecluster/pull/42
# ruff: noqa

import warnings

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.special import logsumexp

from ._vmf_numerics import (
    log_vmf_normalizer,
    log_vmf_normalizer_plus_kappa,
    vmf_mixture_log_likelihood,
)

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

# depecated k_means_ dependencies
#from sklearn.cluster.k_means_ import _init_centroids, _tolerance, _validate_center_shape
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from sklearn.utils import check_array, check_random_state, as_float_array
from sklearn.utils.extmath import squared_norm
from sklearn.utils.extmath import row_norms
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils.validation import check_is_fitted

from sklearn.utils.sparsefuncs import mean_variance_axis

MAX_CONTENTRATION = 1e10

def _tolerance(X, tol):
    """Return a tolerance which is independent of the dataset"""
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol


def _validate_center_shape(X, n_centers, centers):
    """Check if centers is compatible with X and n_centers"""
    if len(centers) != n_centers:
        raise ValueError('The shape of the initial centers (%s) '
                         'does not match the number of clusters %i'
                         % (centers.shape, n_centers))
    if centers.shape[1] != X.shape[1]:
        raise ValueError(
            "The number of features of the initial centers %s "
            "does not match the number of features of the data %s."
            % (centers.shape[1], X.shape[1]))


# PHU'S ENHANCE: validate fixed mixture weights before EM so alpha is a
# nonnegative probability vector and log-space posterior math stays finite.
def _validate_force_weights(force_weights, n_clusters):
    weights = np.asarray(force_weights, dtype=np.float64)
    if weights.shape != (n_clusters,):
        raise ValueError(
            "force_weights shape={} but must equal ({},).".format(
                weights.shape, n_clusters
            )
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("force_weights must be finite.")
    if np.any(weights < 0.0):
        raise ValueError("force_weights must be non-negative.")
    weight_sum = weights.sum()
    if weight_sum <= 0.0:
        raise ValueError("force_weights must contain positive mass.")
    return weights / weight_sum


# PHU'S ENHANCE: exact zero mixture weights must remain impossible in the E-step.
def _log_weights(weights):
    weights = np.asarray(weights, dtype=np.float64)
    logged = np.full(weights.shape, -np.inf, dtype=np.float64)
    positive = weights > 0.0
    logged[positive] = np.log(weights[positive])
    return logged


def _k_init(
    X,
    n_clusters,
    x_squared_norms,
    random_state,
    n_local_trials=None,
    trace=None,
):
    """Init n_clusters seeds according to k-means++

    Parameters
    ----------
    X : array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    n_clusters : integer
        The number of seeds to choose

    x_squared_norms : array, shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : int, RandomState instance
        The generator used to initialize the centers. Use an int to make the
        randomness deterministic.
        See :term:`Glossary <random_state>`.

    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and retain its seeded identity when tracing.
    center_id = random_state.randint(n_samples)
    if trace is not None:
        trace["first_center_index"] = int(center_id)
        trace["selections"] = []
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        sampling_potential = float(current_pot)
        rand_vals = random_state.random_sample(n_local_trials) * sampling_potential
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        if trace is not None:
            trace["selections"].append(
                {
                    "center_position": int(c),
                    "sampling_potential": sampling_potential,
                    "selected_potential": float(current_pot),
                    "random_values": [float(value) for value in rand_vals.tolist()],
                    "candidate_indices": [int(value) for value in candidate_ids.tolist()],
                    "candidate_potentials": [
                        float(value) for value in candidates_pot.tolist()
                    ],
                    "selected_index": int(best_candidate),
                }
            )

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]

    return centers
    
def _init_centroids(X, k, init, random_state=None, x_squared_norms=None,
                    init_size=None, trace=None):
    """Compute the initial centroids

    Parameters
    ----------

    X : array, shape (n_samples, n_features)

    k : int
        number of centroids

    init : {'k-means++', 'random' or ndarray or callable} optional
        Method for initialization

    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    x_squared_norms : array, shape (n_samples,), optional
        Squared euclidean norm of each data point. Pass it if you have it at
        hands already to avoid it being recomputed here. Default: None

    init_size : int, optional
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than k.

    Returns
    -------
    centers : array, shape(k, n_features)
    """
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)

    if init_size is not None and init_size < n_samples:
        if init_size < k:
            warnings.warn(
                "init_size=%d should be larger than k=%d. "
                "Setting it to 3*k" % (init_size, k),
                RuntimeWarning, stacklevel=2)
            init_size = 3 * k
        init_indices = random_state.randint(0, n_samples, init_size)
        X = X[init_indices]
        x_squared_norms = x_squared_norms[init_indices]
        n_samples = X.shape[0]
    elif n_samples < k:
        raise ValueError(
            "n_samples=%d should be larger than k=%d" % (n_samples, k))

    if isinstance(init, str) and init == 'k-means++':
        centers = _k_init(X, k, random_state=random_state,
                          x_squared_norms=x_squared_norms, trace=trace)
    elif isinstance(init, str) and init == 'random':
        seeds = random_state.permutation(n_samples)[:k]
        centers = X[seeds]
    elif hasattr(init, '__array__'):
        # ensure that the centers have the same dtype as X
        # this is a requirement of fused types of cython
        centers = np.array(init, dtype=X.dtype)
    elif callable(init):
        centers = init(X, k, random_state=random_state)
        centers = np.asarray(centers, dtype=X.dtype)
    else:
        raise ValueError("the init parameter for the k-means should "
                         "be 'k-means++' or 'random' or an ndarray, "
                         "'%s' (type '%s') was passed." % (init, type(init)))

    if sp.issparse(centers):
        centers = centers.toarray()

    _validate_center_shape(X, k, centers)
    return centers
    
def _inertia_from_labels(X, centers, labels):
    """Compute inertia with cosine distance using known labels.
    """
    n_examples, n_features = X.shape
    inertia = np.zeros((n_examples,))
    for ee in range(n_examples):
        inertia[ee] = 1 - X[ee, :].dot(centers[int(labels[ee]), :].T)

    return np.sum(inertia)


def _labels_inertia(X, centers):
    """Compute labels and inertia with cosine distance.
    """
    n_examples, n_features = X.shape
    n_clusters, n_features = centers.shape

    labels = np.zeros((n_examples,))
    inertia = np.zeros((n_examples,))

    for ee in range(n_examples):
        dists = np.zeros((n_clusters,))
        for cc in range(n_clusters):
            dists[cc] = 1 - X[ee, :].dot(centers[cc, :].T)

        labels[ee] = np.argmin(dists)
        inertia[ee] = dists[int(labels[ee])]

    return labels, np.sum(inertia)


def _vmf_log(X, kappa, mu):
    """Return per-row vMF log densities through the shared normalizer.

    The compatibility helper retains the vendored module's historical call
    shape while routing every dimension and concentration through the single
    order-aware normalizer authority.
    """
    # Combine the shifted normalizer with alignment residuals without cancellation.
    _, n_features = X.shape
    shifted_log_norm = log_vmf_normalizer_plus_kappa(n_features, float(kappa))
    return kappa * (X.dot(mu).T - 1.0) + shifted_log_norm


def _init_unit_centers(X, n_clusters, random_state, init, trace=None):
    """Initializes unit norm centers.

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)

    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    init:  (string) one of
        k-means++ : uses sklearn k-means++ initialization algorithm
        spherical-k-means : use centroids from one pass of spherical k-means
        random : random unit norm vectors
        random-orthonormal : random orthonormal vectors
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
    """
    # Route every stochastic initialization through its prederived local seed.
    n_examples, n_features = np.shape(X)
    if isinstance(init, np.ndarray):
        n_init_clusters, n_init_features = init.shape
        assert n_init_clusters == n_clusters
        assert n_init_features == n_features

        # ensure unit normed centers
        centers = init
        for cc in range(n_clusters):
            centers[cc, :] = centers[cc, :] / np.linalg.norm(centers[cc, :])

        return centers

    elif init == "spherical-k-means":
        raise NotImplementedError("This option from the original spherecluster implementation is deprecated")
        #labels, inertia, centers, iters = spherical_kmeans._spherical_kmeans_single_lloyd(
        #    X, n_clusters, x_squared_norms=np.ones((n_examples,)), init="k-means++"
        #)
        #return centers

    elif init == "random":
        centers = random_state.randn(n_clusters, n_features)
        for cc in range(n_clusters):
            centers[cc, :] = centers[cc, :] / np.linalg.norm(centers[cc, :])

        return centers

    elif init == "k-means++":
        centers = _init_centroids(
            X,
            n_clusters,
            "k-means++",
            random_state=random_state,
            x_squared_norms=np.ones((n_examples,)),
            trace=trace,
        )

        for cc in range(n_clusters):
            centers[cc, :] = centers[cc, :] / np.linalg.norm(centers[cc, :])

        return centers

    elif init == "random-orthonormal":
        centers = random_state.randn(n_clusters, n_features)
        q, r = np.linalg.qr(centers.T, mode="reduced")

        return q.T

    elif init == "random-class":
        centers = np.zeros((n_clusters, n_features))
        for cc in range(n_clusters):
            while np.linalg.norm(centers[cc, :]) == 0:
                labels = random_state.randint(0, n_clusters, n_examples)
                centers[cc, :] = X[labels == cc, :].sum(axis=0)

        for cc in range(n_clusters):
            centers[cc, :] = centers[cc, :] / np.linalg.norm(centers[cc, :])

        return centers


def _expectation(X, centers, weights, concentrations, posterior_type="soft"):
    """Compute component responsibilities with the shared vMF normalizer.

    Parameters
    ----------
    centers (mu) : array, [n_centers x n_features]
    weights (alpha) : array, [n_centers, ] (alpha)
    concentrations (kappa) : array, [n_centers, ]

    Returns
    ----------
    posterior : array, [n_centers, n_examples]
    """
    # Evaluate every component through the same density path.
    n_examples, n_features = np.shape(X)
    n_clusters, _ = centers.shape

    f_log = np.zeros((n_clusters, n_examples))
    for cc in range(n_clusters):
        f_log[cc, :] = _vmf_log(X, concentrations[cc], centers[cc, :])

    posterior = np.zeros((n_clusters, n_examples))
    if posterior_type == "soft":
        weights_log = _log_weights(weights)
        posterior = np.tile(weights_log.T, (n_examples, 1)).T + f_log
        for ee in range(n_examples):
            posterior[:, ee] = np.exp(posterior[:, ee] - logsumexp(posterior[:, ee]))

    elif posterior_type == "hard":
        weights_log = _log_weights(weights)
        weighted_f_log = np.tile(weights_log.T, (n_examples, 1)).T + f_log
        for ee in range(n_examples):
            posterior[np.argmax(weighted_f_log[:, ee]), ee] = 1.0

    return posterior


def _dense_weighted_component_sums(X, posterior):
    """Accumulate dense component resultants with inherited float ordering.

    NumPy's unoptimized contraction traverses examples in the historical
    ``sum(axis=0)`` order without materializing a scaled ``n x d`` array.
    Optimized einsum and BLAS matmul are deliberately excluded because their
    reduction order can change low bits in the corrected dense authority.
    """
    # Contract every component without allocating vocabulary-sized intermediates.
    return np.einsum("kn,nd->kd", posterior, X, optimize=False)


def _maximization(X, posterior, force_weights=None, trace=None):
    """Estimate new centers, weights, and concentrations from

    Parameters
    ----------
    posterior : array, [n_centers, n_examples]
        The posterior matrix from the expectation step.

    force_weights : None or array, [n_centers, ]
        If None is passed, will estimate weights.
        If an array is passed, will use instead of estimating.

    Returns
    ----------
    centers (mu) : array, [n_centers x n_features]
    weights (alpha) : array, [n_centers, ] (alpha)
    concentrations (kappa) : array, [n_centers, ]
    """
    n_examples, n_features = X.shape
    n_clusters, n_examples = posterior.shape
    concentrations = np.zeros((n_clusters,))
    centers = (
        np.zeros((n_clusters, n_features))
        if sp.issparse(X)
        else _dense_weighted_component_sums(X, posterior)
    )
    if force_weights is None:
        weights = np.zeros((n_clusters,))
    else:
        weights = _validate_force_weights(force_weights, n_clusters)

    component_trace = [] if trace is not None else None
    for cc in range(n_clusters):
        # update weights (alpha)
        if force_weights is None:
            weights[cc] = np.mean(posterior[cc, :])

        # update centers (mu)
        if sp.issparse(X):
            X_scaled = X.copy()
            X_scaled.data *= posterior[cc, :].repeat(np.diff(X_scaled.indptr))
            centers[cc, :] = X_scaled.sum(axis=0)

        # normalize centers
        center_norm = np.linalg.norm(centers[cc, :])
        if (
            not np.isfinite(weights[cc])
            or weights[cc] <= np.finfo(np.float64).eps
            or not np.isfinite(center_norm)
            or center_norm <= 1e-8
        ):
            # PHU'S ENHANCE: the vanilla update divides by component mass.
            # Empty components carry zero alpha; kappa=0 is the uniform vMF.
            weights[cc] = max(0.0, weights[cc]) if np.isfinite(weights[cc]) else 0.0
            centers[cc, :] = 0.0
            centers[cc, cc % n_features] = 1.0
            concentrations[cc] = 0.0
            if component_trace is not None:
                component_trace.append(
                    {
                        "component": int(cc),
                        "weight": float(weights[cc]),
                        "resultant_norm": float(center_norm),
                        "rbar": None,
                        "concentration": 0.0,
                        "fallback": True,
                    }
                )
            if sp.issparse(X):
                del X_scaled
            continue

        centers[cc, :] = centers[cc, :] / center_norm

        # update concentration (kappa) [TODO: add other kappa approximations]
        rbar = center_norm / (n_examples * weights[cc])
        if not np.isfinite(rbar):
            concentrations[cc] = 0.0
        elif rbar >= 1.0 - 1e-10:
            concentrations[cc] = MAX_CONTENTRATION
        else:
            rbar = max(0.0, rbar)
            concentrations[cc] = rbar * n_features - np.power(rbar, 3.0)
            concentrations[cc] /= 1.0 - np.power(rbar, 2.0)

        if component_trace is not None:
            component_trace.append(
                {
                    "component": int(cc),
                    "weight": float(weights[cc]),
                    "resultant_norm": float(center_norm),
                    "rbar": float(rbar) if np.isfinite(rbar) else None,
                    "concentration": float(concentrations[cc]),
                    "fallback": False,
                }
            )

        # let python know we can free this (good for large dense X)
        if sp.issparse(X):
            del X_scaled

    if trace is not None:
        trace["components"] = component_trace
    return centers, weights, concentrations


def _movMF(
    X,
    n_clusters,
    posterior_type="soft",
    force_weights=None,
    max_iter=300,
    verbose=False,
    init="random-class",
    random_state=None,
    tol=1e-6,
    trace=None,
):
    """Mixture of von Mises Fisher clustering.

    Implements the algorithms (i) and (ii) from

      "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions"
      by Banerjee, Dhillon, Ghosh, and Sra.

    TODO: Currently only supports Banerjee et al 2005 approximation of kappa,
          however, there are numerous other approximations see _update_params.

    Attribution
    ----------
    Approximation of log-vmf distribution function from movMF R-package.

    movMF: An R Package for Fitting Mixtures of von Mises-Fisher Distributions
    by Kurt Hornik, Bettina Grun, 2014

    Find more at:
      https://cran.r-project.org/web/packages/movMF/vignettes/movMF.pdf
      https://cran.r-project.org/web/packages/movMF/index.html

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    posterior_type: 'soft' or 'hard'
        Type of posterior computed in exepectation step.
        See note about attribute: self.posterior_

    force_weights : None or array [n_clusters, ]
        If None, the algorithm will estimate the weights.
        If an array of weights, algorithm will estimate concentrations and
        centers with given weights.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    init:  (string) one of
        random-class [default]: random class assignment & centroid computation
        k-means++ : uses sklearn k-means++ initialization algorithm
        spherical-k-means : use centroids from one pass of spherical k-means
        random : random unit norm vectors
        random-orthonormal : random orthonormal vectors
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    tol : float, default: 1e-6
        Relative tolerance with regards to inertia to declare convergence

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    verbose : int, default 0
        Verbosity mode.

    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.
    """
    random_state = check_random_state(random_state)
    n_examples, n_features = np.shape(X)
    if force_weights is not None:
        force_weights = _validate_force_weights(force_weights, n_clusters)

    # Initialize centers and capture exact branch values only when requested.
    initialization_trace = {} if trace is not None else None
    centers = _init_unit_centers(
        X, n_clusters, random_state, init, trace=initialization_trace
    )

    # init weights (alphas)
    if force_weights is None:
        weights = np.ones((n_clusters,))
        weights = weights / np.sum(weights)
    else:
        weights = force_weights

    # init concentrations (kappas)
    concentrations = np.ones((n_clusters,))

    if verbose:
        print("Initialization complete")

    iteration_trace = [] if trace is not None else None
    for iter in range(max_iter):
        centers_prev = centers.copy()

        # expectation step
        posterior = _expectation(
            X, centers, weights, concentrations, posterior_type=posterior_type
        )

        # maximization step
        maximization_trace = {} if trace is not None else None
        centers, weights, concentrations = _maximization(
            X,
            posterior,
            force_weights=force_weights,
            trace=maximization_trace,
        )

        # check convergence
        tolcheck = squared_norm(centers_prev - centers)
        if trace is not None:
            iteration_trace.append(
                {
                    "iteration": int(iter),
                    "posterior": np.asarray(posterior, dtype=np.float64).tolist(),
                    "center_shift": float(tolcheck),
                    "tolerance": float(tol),
                    "maximization": maximization_trace,
                }
            )
        if tolcheck <= tol:
            if verbose:
                print(
                    "Converged at iteration %d: "
                    "center shift %e within tolerance %e" % (iter, tolcheck, tol)
                )
            break

    # labels come for free via posterior
    labels = np.zeros((n_examples,))
    for ee in range(n_examples):
        labels[ee] = np.argmax(posterior[:, ee])

    inertia = _inertia_from_labels(X, centers, labels)

    if trace is not None:
        trace.update(
            {
                "initialization": initialization_trace,
                "iterations": iteration_trace,
                "iteration_count": len(iteration_trace),
                "converged": bool(
                    iteration_trace
                    and iteration_trace[-1]["center_shift"] <= float(tol)
                ),
                "labels": [int(value) for value in labels.tolist()],
                "inertia": float(inertia),
            }
        )

    return centers, weights, concentrations, posterior, labels, inertia


def _attempt_movmf_initialization(
    X,
    n_clusters,
    posterior_type,
    force_weights,
    max_iter,
    verbose,
    init,
    seed,
    tol,
    init_index,
    trace=None,
):
    """Attempt one seeded fit and attach its finite full mixture likelihood.

    Ordinary initialization or numerical failures are captured so the caller
    can exhaust its fixed budget. The returned index is the deterministic
    tie-break key; successful payloads retain the vendored six-array result.
    """
    # Isolate one initialization so serial and parallel scheduling share semantics.
    try:
        result = _movMF(
            X,
            n_clusters,
            posterior_type=posterior_type,
            force_weights=force_weights,
            max_iter=max_iter,
            verbose=verbose,
            init=init,
            random_state=int(seed),
            tol=tol,
            trace=trace,
        )
        centers, weights, concentrations, _, _, _ = result
        likelihood = vmf_mixture_log_likelihood(
            X, centers, weights, concentrations
        )
        if not np.isfinite(likelihood):
            raise FloatingPointError("full vMF mixture likelihood was non-finite.")
        if trace is not None:
            trace["status"] = "finite"
            trace["log_likelihood"] = float(likelihood)
        return init_index, result, float(likelihood), None, trace
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
        if trace is not None:
            trace["status"] = "failed"
            trace["error"] = f"{type(error).__name__}: {error}"
        return init_index, None, None, f"{type(error).__name__}: {error}", trace


def movMF(
    X,
    n_clusters,
    posterior_type="soft",
    force_weights=None,
    n_init=10,
    n_jobs=1,
    max_iter=300,
    verbose=False,
    init="random-class",
    random_state=None,
    tol=1e-6,
    copy_x=True,
    trace=None,
):
    """Run a fixed initialization budget and select the largest finite likelihood.

    All seeds are derived before any work starts, making serial and parallel
    execution scheduling-independent. Every initialization is attempted even
    when another raises or produces an invalid likelihood; equal finite
    likelihoods select the smaller initialization index.
    """
    # Validate controls and precompute the shared fit inputs.
    if n_init <= 0:
        raise ValueError(
            "Invalid number of initializations."
            " n_init=%d must be bigger than zero." % n_init
        )
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError(
            "Number of iterations should be a positive number,"
            " got %d instead" % max_iter
        )

    X = as_float_array(X, copy=copy_x)
    tol = _tolerance(X, tol)

    if hasattr(init, "__array__"):
        init = check_array(init, dtype=X.dtype.type, copy=True)
        _validate_center_shape(X, n_clusters, init)

    # Derive exactly one deterministic seed per initialization before scheduling.
    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
    attempt_traces = [
        {"init_index": int(index), "seed": int(seed)}
        for index, seed in enumerate(seeds)
    ]
    if n_jobs == 1:
        results = [
            _attempt_movmf_initialization(
                X,
                n_clusters,
                posterior_type,
                force_weights,
                max_iter,
                verbose,
                init,
                seed,
                tol,
                init_index,
                attempt_traces[init_index] if trace is not None else None,
            )
            for init_index, seed in enumerate(seeds)
        ]
    else:
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_attempt_movmf_initialization)(
                X,
                n_clusters,
                posterior_type,
                force_weights,
                max_iter,
                verbose,
                init,
                seed,
                tol,
                init_index,
                attempt_traces[init_index] if trace is not None else None,
            )
            for init_index, seed in enumerate(seeds)
        )

    # Select only finite candidates; strict improvement preserves the smaller tie index.
    best_result = None
    best_likelihood = -np.inf
    best_init_index = None
    for init_index, result, likelihood, _, _ in results:
        if result is not None and likelihood > best_likelihood:
            best_result = result
            best_likelihood = likelihood
            best_init_index = int(init_index)
    ordered_traces = [item[4] for item in results]
    if trace is not None:
        trace.update(
            {
                "backend": "dense_cpu",
                "n_rows": int(X.shape[0]),
                "ambient_dim": int(X.shape[1]),
                "mode_count": int(n_clusters),
                "n_init": int(n_init),
                "max_iter": int(max_iter),
                "scaled_tolerance": float(tol),
                "initializations": ordered_traces,
                "selected_init_index": best_init_index,
                "selected_log_likelihood": (
                    float(best_likelihood) if best_result is not None else None
                ),
                "status": "finite" if best_result is not None else "failed",
                "workload": {
                    "initialization_attempts": len(results),
                    "iterations": sum(
                        int(item.get("iteration_count", 0))
                        for item in ordered_traces
                        if item is not None
                    ),
                    "kmeans_selections": sum(
                        len(item.get("initialization", {}).get("selections", []))
                        for item in ordered_traces
                        if item is not None
                    ),
                    "fallback_components": sum(
                        int(component.get("fallback", False))
                        for item in ordered_traces
                        if item is not None
                        for iteration in item.get("iterations", [])
                        for component in iteration.get("maximization", {}).get(
                            "components", []
                        )
                    ),
                },
            }
        )

    if best_result is None:
        failures = "; ".join(
            "init {}: {}".format(init_index, error)
            for init_index, _, _, error, _ in results
        )
        if trace is not None:
            trace["error"] = failures
        raise FloatingPointError(
            "No initialization produced a finite full vMF mixture likelihood. "
            + failures
        )

    centers, weights, concentrations, posterior, labels, inertia = best_result
    return (
        centers,
        labels,
        inertia,
        weights,
        concentrations,
        posterior,
    )


class VonMisesFisherMixture(BaseEstimator, ClusterMixin, TransformerMixin):
    """Estimator for Mixture of von Mises Fisher clustering on the unit sphere.

    Implements the algorithms (i) and (ii) from

      "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions"
      by Banerjee, Dhillon, Ghosh, and Sra.

    TODO: Currently only supports Banerjee et al 2005 approximation of kappa,
          however, there are numerous other approximations see _update_params.

    Attribution
    ----------
    Approximation of log-vmf distribution function from movMF R-package.

    movMF: An R Package for Fitting Mixtures of von Mises-Fisher Distributions
    by Kurt Hornik, Bettina Grun, 2014

    Find more at:
      https://cran.r-project.org/web/packages/movMF/vignettes/movMF.pdf
      https://cran.r-project.org/web/packages/movMF/index.html

    Basic sklearn scaffolding from sklearn.cluster.KMeans.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    posterior_type: 'soft' or 'hard'
        Type of posterior computed in exepectation step.
        See note about attribute: self.posterior_

    force_weights : None or array [n_clusters, ]
        If None, the algorithm will estimate the weights.
        If an array of weights, algorithm will estimate concentrations and
        centers with given weights.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final result has the largest finite full vMF
        mixture log likelihood; ties retain the earlier initialization.

    init:  (string) one of
        random-class [default]: random class assignment & centroid computation
        k-means++ : uses sklearn k-means++ initialization algorithm
        spherical-k-means : use centroids from one pass of spherical k-means
        random : random unit norm vectors
        random-orthonormal : random orthonormal vectors
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    tol : float, default: 1e-6
        Relative tolerance with regards to inertia to declare convergence

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    verbose : int, default 0
        Verbosity mode.

    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    normalize : boolean, default True
        Normalize the input to have unnit norm.

    Attributes
    ----------

    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    weights_ : array, [n_clusters,]
        Weights of each cluster in vMF distribution (alpha).

    concentrations_ : array [n_clusters,]
        Concentration parameter for each cluster (kappa).
        Larger values correspond to more concentrated clusters.

    posterior_ : array, [n_clusters, n_examples]
        Each column corresponds to the posterio distribution for and example.

        If posterior_type='hard' is used, there will only be one non-zero per
        column, its index corresponding to the example's cluster label.

        If posterior_type='soft' is used, this matrix will be dense and the
        column values correspond to soft clustering weights.

    log_likelihood_ : float
        Full vMF mixture log likelihood of the selected initialization.
    """

    def __init__(
        self,
        n_clusters=5,
        posterior_type="soft",
        force_weights=None,
        n_init=10,
        n_jobs=1,
        max_iter=300,
        verbose=False,
        init="random-class",
        random_state=None,
        tol=1e-6,
        copy_x=True,
        normalize=True,
        trace=None,
    ):
        self.n_clusters = n_clusters
        self.posterior_type = posterior_type
        self.force_weights = force_weights
        self.n_init = n_init
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.verbose = verbose
        self.init = init
        self.random_state = random_state
        self.tol = tol
        self.copy_x = copy_x
        self.normalize = normalize
        self.trace = trace

    def _check_force_weights(self):
        if self.force_weights is None:
            return

        self.force_weights = _validate_force_weights(
            self.force_weights, self.n_clusters
        )

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
        n_samples, n_features = X.shape
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                "n_samples=%d should be >= n_clusters=%d"
                % (X.shape[0], self.n_clusters)
            )

        for ee in range(n_samples):
            if sp.issparse(X):
                n = sp.linalg.norm(X[ee, :])
            else:
                n = np.linalg.norm(X[ee, :])

            if np.abs(n - 1.0) > 1e-4:
                raise ValueError("Data l2-norm must be 1, found {}".format(n))

        return X

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse="csr", dtype=FLOAT_DTYPES)
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError(
                "Incorrect number of features. "
                "Got %d features, expected %d" % (n_features, expected_n_features)
            )

        for ee in range(n_samples):
            if sp.issparse(X):
                n = sp.linalg.norm(X[ee, :])
            else:
                n = np.linalg.norm(X[ee, :])

            if np.abs(n - 1.0) > 1e-4:
                raise ValueError("Data l2-norm must be 1, found {}".format(n))

        return X

    def fit(self, X, y=None):
        """Fit every configured initialization and retain the likelihood winner.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """
        # Normalize and validate once before running the fixed initialization budget.
        if self.normalize:
            X = normalize(X)

        self._check_force_weights()
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)

        (
            self.cluster_centers_,
            self.labels_,
            self.inertia_,
            self.weights_,
            self.concentrations_,
            self.posterior_,
        ) = movMF(
            X,
            self.n_clusters,
            posterior_type=self.posterior_type,
            force_weights=self.force_weights,
            n_init=self.n_init,
            n_jobs=self.n_jobs,
            max_iter=self.max_iter,
            verbose=self.verbose,
            init=self.init,
            random_state=random_state,
            tol=self.tol,
            copy_x=self.copy_x,
            trace=self.trace,
        )
        self.log_likelihood_ = vmf_mixture_log_likelihood(
            X, self.cluster_centers_, self.weights_, self.concentrations_
        )

        return self

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        return self.fit(X).labels_

    def fit_transform(self, X, y=None):
        """Compute clustering and transform X to cluster-distance space.
        Equivalent to fit(X).transform(X), but more efficiently implemented.
        """
        # Currently, this just skips a copy of the data if it is not in
        # np.array or CSR format already.
        # XXX This skips _check_test_data, which may change the dtype;
        # we should refactor the input validation.
        return self.fit(X)._transform(X)

    def transform(self, X, y=None):
        """Transform X to a cluster-distance space.
        In the new space, each dimension is the cosine distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        if self.normalize:
            X = normalize(X)

        check_is_fitted(self)
        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        """guts of transform method; no input validation"""
        return cosine_distances(X, self.cluster_centers_)

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Note:  Does not check that each point is on the sphere.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        if self.normalize:
            X = normalize(X)

        check_is_fitted(self)

        X = self._check_test_data(X)
        return _labels_inertia(X, self.cluster_centers_)[0]

    def score(self, X, y=None):
        """Inertia score (sum of all distances to closest cluster).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data.

        Returns
        -------
        score : float
            Larger score is better.
        """
        if self.normalize:
            X = normalize(X)

        check_is_fitted(self)
        X = self._check_test_data(X)
        return -_labels_inertia(X, self.cluster_centers_)[1]

    def log_likelihood(self, X):
        """Return the full mixture log likelihood under the fitted estimator.

        The score includes fitted mixture weights and the order-aware normalizer
        for every component; invalid or non-finite evaluations are surfaced by
        the shared numerical authority.
        """
        # Mirror score preprocessing before delegating the complete density score.
        if self.normalize:
            X = normalize(X)

        check_is_fitted(self)
        X = self._check_test_data(X)
        return vmf_mixture_log_likelihood(
            X, self.cluster_centers_, self.weights_, self.concentrations_
        )
