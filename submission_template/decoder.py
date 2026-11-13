"""BP + OSD decoder for the [[18,4,4]] code, written from scratch (numpy + numba only).

Pipeline for one shot with detector events D:
  1. Build the decoding problem from the circuit: check matrix H (detectors x mechanisms),
     observable matrix L (observables x mechanisms) and prior probabilities p.
  2. Belief propagation (normalised min-sum) on the Tanner graph of H gives a posterior
     log-likelihood ratio for every mechanism.
  3. Ordered statistics decoding (OSD): sort mechanisms from most to least likely, take
     the first linearly independent columns of H as a basis, and solve H e = D on that
     basis.  A combination sweep then also tries flipping each of the next most likely
     mechanisms, and keeps the lowest-cost solution.
  4. Prediction: O = L e mod 2.

Everything is deterministic (no randomness, fixed operation order), so predictions are
reproducible run after run.
"""
import numpy as np
import stim
from numba import njit, prange

# ----------------------------------------------------------------------------------------
# 1. Decoding problem from the circuit
# ----------------------------------------------------------------------------------------


def build_problem(circuit: stim.Circuit):
    """Return H (uint8, D x M), L (uint8, K x M), p (float64, M) from the circuit's DEM.

    Mechanisms with identical (detectors, observables) effect are merged: two independent
    events with probabilities a and b that have the same effect flip it with a(1-b)+b(1-a).
    """
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    merged = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        dets, obs = set(), set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs ^= {t.val}
        key = (tuple(sorted(dets)), tuple(sorted(obs)))
        if not key[0]:
            continue  # invisible to the detectors: no decoder can see it
        q = merged.get(key, 0.0)
        merged[key] = q * (1 - prob) + prob * (1 - q)
    keys = sorted(merged)
    nD, nK, nM = dem.num_detectors, dem.num_observables, len(keys)
    H = np.zeros((nD, nM), np.uint8)
    L = np.zeros((nK, nM), np.uint8)
    p = np.zeros(nM, np.float64)
    for j, (dets, obs) in enumerate(keys):
        H[list(dets), j] = 1
        L[list(obs), j] = 1
        p[j] = merged[(dets, obs)]
    return H, L, p


def pack_columns(M):
    """Pack each column of a 0/1 matrix (rows <= 128) into two uint64 words."""
    rows, cols = M.shape
    out = np.zeros((cols, 2), np.uint64)
    for r in range(rows):
        bit = np.uint64(1) << np.uint64(r % 64)
        out[M[r] == 1, r // 64] |= bit
    return out


# ----------------------------------------------------------------------------------------
# 2. Belief propagation (normalised min-sum)
# ----------------------------------------------------------------------------------------


@njit(cache=True)
def bp_min_sum(syn, chk_ptr, chk_var, var_ptr, var_edge, prior_llr, max_iter, alpha, llr_out):
    """Min-sum BP on one syndrome. Edges are listed check by check (chk_var[e] = variable
    of edge e). Writes the posterior LLR to llr_out; returns True if the hard decision
    satisfies the syndrome."""
    n_chk = len(chk_ptr) - 1
    n_var = len(prior_llr)
    n_edge = len(chk_var)
    v2c = np.empty(n_edge)
    c2v = np.zeros(n_edge)
    for e in range(n_edge):
        v2c[e] = prior_llr[chk_var[e]]
    hard = np.zeros(n_var, np.uint8)
    for v in range(n_var):
        llr_out[v] = prior_llr[v]
    for it in range(max_iter):
        # check -> variable
        for c in range(n_chk):
            sgn = -1.0 if syn[c] else 1.0
            min1 = 1e300
            min2 = 1e300
            arg = -1
            for e in range(chk_ptr[c], chk_ptr[c + 1]):
                m = v2c[e]
                if m < 0:
                    sgn = -sgn
                a = abs(m)
                if a < min1:
                    min2 = min1
                    min1 = a
                    arg = e
                elif a < min2:
                    min2 = a
            for e in range(chk_ptr[c], chk_ptr[c + 1]):
                m = v2c[e]
                s = sgn * (-1.0 if m < 0 else 1.0)
                mag = min2 if e == arg else min1
                c2v[e] = alpha * s * mag
        # variable -> check and posterior
        for v in range(n_var):
            tot = prior_llr[v]
            for k in range(var_ptr[v], var_ptr[v + 1]):
                tot += c2v[var_edge[k]]
            llr_out[v] = tot
            hard[v] = 1 if tot < 0 else 0
            for k in range(var_ptr[v], var_ptr[v + 1]):
                e = var_edge[k]
                v2c[e] = tot - c2v[e]
        # syndrome check
        ok = True
        for c in range(n_chk):
            par = 0
            for e in range(chk_ptr[c], chk_ptr[c + 1]):
                par ^= hard[chk_var[e]]
            if par != syn[c]:
                ok = False
                break
        if ok:
            return True
    return False


# ----------------------------------------------------------------------------------------
# 3. Ordered statistics decoding over GF(2), columns packed as 2 x uint64
# ----------------------------------------------------------------------------------------


@njit(cache=True)
def _bit(w0, w1, b):
    """Bit b of the 128-bit word (w0, w1)."""
    if b < 64:
        return (w0 >> np.uint64(b)) & np.uint64(1)
    return (w1 >> np.uint64(b - 64)) & np.uint64(1)


@njit(cache=True)
def osd_classes(syn_w0, syn_w1, order, H_cols, L_cols, rank_cost, prob_cost, sweep, sweep2, aggregate):
    """Ordered statistics decoding with a combination sweep.

    order:     mechanisms sorted from most to least likely to have fired (by rank_cost).
    rank_cost: posterior LLR from BP (lower = likelier); picks the best single solution.
    prob_cost: prior LLR; -sum(prob_cost) over an error is its log-probability (up to a
               constant), used to weigh solutions when aggregate is True.
    sweep:     number of non-basis mechanisms tried one at a time on top of OSD-0.
    sweep2:    the first sweep2 of those are also tried in pairs.

    Returns a score for each of the 16 logical classes (values of the 4 observables):
    if aggregate, log of the summed probability of all solutions found in that class;
    otherwise 0 for the class of the lowest-cost solution and -inf for the others.
    """
    n = len(order)
    # --- Gaussian elimination: first linearly independent columns in `order` form the basis
    piv_w = np.zeros((128, 2), np.uint64)   # reduced basis vectors
    piv_lo = np.zeros(128, np.int64)        # pivot (lowest set bit) of each basis vector
    comb = np.zeros((128, 2), np.uint64)    # which basis columns XOR to this vector
    basis = np.zeros(128, np.int64)         # mechanism index of each basis column
    rank = 0
    nonbasis = np.empty(max(sweep, 1), np.int64)
    n_nb = 0
    for idx in range(n):
        j = order[idx]
        w0, w1 = H_cols[j, 0], H_cols[j, 1]
        c0 = np.uint64(0)
        c1 = np.uint64(0)
        if rank < 64:
            c0 = np.uint64(1) << np.uint64(rank)
        else:
            c1 = np.uint64(1) << np.uint64(rank - 64)
        for b in range(rank):
            if _bit(w0, w1, piv_lo[b]):
                w0 ^= piv_w[b, 0]
                w1 ^= piv_w[b, 1]
                c0 ^= comb[b, 0]
                c1 ^= comb[b, 1]
        if w0 == 0 and w1 == 0:            # dependent on the basis: candidate for the sweep
            if n_nb < sweep:
                nonbasis[n_nb] = j
                n_nb += 1
            continue
        lo = 0
        while not _bit(w0, w1, lo):
            lo += 1
        piv_w[rank, 0] = w0
        piv_w[rank, 1] = w1
        piv_lo[rank] = lo
        comb[rank, 0] = c0
        comb[rank, 1] = c1
        basis[rank] = j
        rank += 1
        if n_nb >= sweep and rank >= 70:
            break

    scores = np.full(16, -np.inf)
    best_cost = np.inf
    n_pairs = sweep2 * (sweep2 - 1) // 2 if sweep2 <= n_nb else n_nb * (n_nb - 1) // 2
    m2 = min(sweep2, n_nb)
    total = 1 + n_nb + n_pairs
    for trial in range(total):
        # which extra (non-basis) mechanisms are flipped in this trial
        e1 = -1
        e2 = -1
        if trial >= 1 and trial <= n_nb:
            e1 = nonbasis[trial - 1]
        elif trial > n_nb:
            t = trial - 1 - n_nb
            a = 0
            while t >= m2 - 1 - a:
                t -= m2 - 1 - a
                a += 1
            e1 = nonbasis[a]
            e2 = nonbasis[a + 1 + t]
        s0, s1 = syn_w0, syn_w1
        rc = 0.0
        pc = 0.0
        obs = np.uint64(0)
        if e1 >= 0:
            s0 ^= H_cols[e1, 0]
            s1 ^= H_cols[e1, 1]
            rc += rank_cost[e1]
            pc += prob_cost[e1]
            obs ^= L_cols[e1, 0]
        if e2 >= 0:
            s0 ^= H_cols[e2, 0]
            s1 ^= H_cols[e2, 1]
            rc += rank_cost[e2]
            pc += prob_cost[e2]
            obs ^= L_cols[e2, 0]
        # solve H_basis x = remaining syndrome
        r0 = np.uint64(0)
        r1 = np.uint64(0)
        for b in range(rank):
            if _bit(s0, s1, piv_lo[b]):
                s0 ^= piv_w[b, 0]
                s1 ^= piv_w[b, 1]
                r0 ^= comb[b, 0]
                r1 ^= comb[b, 1]
        if s0 != 0 or s1 != 0:
            continue  # syndrome not reachable (does not happen for a valid syndrome)
        for b in range(rank):
            if _bit(r0, r1, b):
                j = basis[b]
                rc += rank_cost[j]
                pc += prob_cost[j]
                obs ^= L_cols[j, 0]
        k = int(obs)
        if aggregate:
            lp = -pc
            hi = max(scores[k], lp)
            scores[k] = hi + np.log(np.exp(scores[k] - hi) + np.exp(lp - hi))
        elif rc < best_cost:
            best_cost = rc
            scores[:] = -np.inf
            scores[k] = 0.0
    return scores


# ----------------------------------------------------------------------------------------
# 4. Batch decoding, parallel over shots
# ----------------------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def decode_batch(dets, chk_ptr, chk_var, var_ptr, var_edge, prior_llr, H_cols, L_cols,
                 max_iter, alpha, sweep, sweep2, aggregate):
    shots, n_det = dets.shape
    n_var = len(prior_llr)
    out = np.zeros((shots, 4), np.uint8)
    for s in prange(shots):
        syn = dets[s]
        w = 0
        for d in range(n_det):
            w += syn[d]
        if w == 0:
            continue                    # no detector fired: predict no logical flip
        llr = np.empty(n_var)
        bp_min_sum(syn, chk_ptr, chk_var, var_ptr, var_edge, prior_llr, max_iter, alpha, llr)
        s0 = np.uint64(0)
        s1 = np.uint64(0)
        for d in range(n_det):
            if syn[d]:
                if d < 64:
                    s0 |= np.uint64(1) << np.uint64(d)
                else:
                    s1 |= np.uint64(1) << np.uint64(d - 64)
        order = np.argsort(llr, kind="mergesort")
        scores = osd_classes(s0, s1, order, H_cols, L_cols, llr, prior_llr, sweep, sweep2, aggregate)
        # each observable bit: compare total probability of classes with that bit 1 vs 0
        mx = scores.max()
        for k in range(4):
            p1 = 0.0
            p0 = 0.0
            for c in range(16):
                v = np.exp(scores[c] - mx)
                if (c >> k) & 1:
                    p1 += v
                else:
                    p0 += v
            out[s, k] = 1 if p1 > p0 else 0
    return out


# ----------------------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------------------

MAX_ITER = 15
ALPHA = 0.3
SWEEP = 300
SWEEP2 = 80
AGGREGATE = True


def prepare(circuit: stim.Circuit):
    H, L, p = build_problem(circuit)
    n_det, n_var = H.shape
    rows, cols_ = np.nonzero(H)                       # edges sorted by check
    chk_ptr = np.zeros(n_det + 1, np.int64)
    np.add.at(chk_ptr, rows + 1, 1)
    chk_ptr = np.cumsum(chk_ptr)
    chk_var = cols_.astype(np.int64)
    edge_order = np.argsort(chk_var, kind="stable")   # edges grouped by variable
    var_ptr = np.zeros(n_var + 1, np.int64)
    np.add.at(var_ptr, chk_var + 1, 1)
    var_ptr = np.cumsum(var_ptr)
    var_edge = edge_order.astype(np.int64)
    prior_llr = np.log((1 - p) / p)
    return (chk_ptr, chk_var, var_ptr, var_edge, prior_llr, pack_columns(H), pack_columns(L))


_cache = {}


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """
    circuit: the X or Z circuit, loaded with stim.Circuit.from_file.
    detector_events: bool array of shape (shots, 70), columns D0 to D69.
    Returns a 0/1 array of shape (shots, 4), columns L0 to L3.
    """
    key = str(circuit)
    if key not in _cache:
        _cache[key] = prepare(circuit)
    args = _cache[key]
    dets = np.ascontiguousarray(detector_events, dtype=np.uint8)
    return decode_batch(dets, *args, MAX_ITER, ALPHA, SWEEP, SWEEP2, AGGREGATE)


# Compile the numba kernels at import time (import is not timed).
def _warmup():
    c = stim.Circuit.generated("repetition_code:memory", rounds=2, distance=3,
                               before_round_data_depolarization=0.01)
    args = prepare(c)
    dets = np.ones((2, c.num_detectors), np.uint8)
    decode_batch(dets, *args, MAX_ITER, ALPHA, SWEEP, SWEEP2, AGGREGATE)


_warmup()
