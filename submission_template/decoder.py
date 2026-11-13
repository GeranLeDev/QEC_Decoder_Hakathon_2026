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


def gf2_rank(M):
    """Rank of a 0/1 matrix over GF(2) (plain Gaussian elimination, done once per circuit)."""
    A = (np.array(M, dtype=np.uint8) & 1).copy()
    rank = 0
    for c in range(A.shape[1]):
        if rank == A.shape[0]:
            break
        hits = np.nonzero(A[rank:, c])[0]
        if len(hits) == 0:
            continue
        p = rank + hits[0]
        if p != rank:
            A[[rank, p]] = A[[p, rank]]
        rows = np.nonzero(A[:, c])[0]
        rows = rows[rows != rank]
        A[rows] ^= A[rank]
        rank += 1
    return rank


# ----------------------------------------------------------------------------------------
# 2. Belief propagation (normalised min-sum), W shots at a time
# ----------------------------------------------------------------------------------------
# BP does the same sequence of operations for every shot (only the numbers differ), so we
# run W shots side by side: every message is a row of W numbers ("lanes"), and the inner
# loops over the lanes compile to SIMD instructions.  Each lane performs exactly the same
# floating-point operations in the same order as a one-shot BP, so the result does not
# depend on W nor on which shots share a group.

W = 8   # shots processed together (8 doubles = one AVX-512 register)


@njit(cache=True)
def bp_min_sum_lanes(synS, chk_ptr, chk_var, var_ptr, var_edge, det_idx, prior_llr,
                     max_iter, alpha, llr_out, v2c, c2v, par):
    """synS[c, l]: syndrome of check c for shot (lane) l, stored as a sign: -1.0 if the
    detector fired, +1.0 otherwise.  llr_out[v, l]: posterior LLR.
    v2c, c2v: (n_edge, W) scratch; par: (n_chk, W) scratch.
    Everything inside the lane loops is float64 so that they map to plain SIMD code."""
    n_chk = len(chk_ptr) - 1
    n_var = len(prior_llr)
    n_edge = len(chk_var)
    sg = np.empty(W)
    m1 = np.empty(W)
    m2 = np.empty(W)
    tot = np.empty(W)
    done = np.zeros(W)                 # 1.0 = lane already converged: its llr_out is frozen
    for e in range(n_edge):
        p = prior_llr[chk_var[e]]
        for l in range(W):
            v2c[e, l] = p
    for v in range(n_var):
        for l in range(W):
            llr_out[v, l] = prior_llr[v]
    for it in range(max_iter):
        # check -> variable
        for c in range(n_chk):
            lo = chk_ptr[c]
            hi = chk_ptr[c + 1]
            for l in range(W):
                sg[l] = synS[c, l]
                m1[l] = 1e300
                m2[l] = 1e300
            for e in range(lo, hi):
                for l in range(W):
                    m = v2c[e, l]
                    a = abs(m)
                    sg[l] = -sg[l] if m < 0 else sg[l]
                    # keep the two smallest magnitudes (same as the if/elif of the scalar code)
                    m2[l] = min(m2[l], max(a, m1[l]))
                    m1[l] = min(m1[l], a)
            for e in range(lo, hi):
                for l in range(W):
                    m = v2c[e, l]
                    s = -sg[l] if m < 0 else sg[l]
                    # the edge holding the minimum gets the second minimum (if two edges tie
                    # for the minimum then m2 == m1, so this matches the scalar code)
                    mag = m2[l] if abs(m) == m1[l] else m1[l]
                    c2v[e, l] = alpha * s * mag
        # variable -> check, posterior, and parity of the hard decision on every check
        # (par starts at the syndrome sign and is negated by every variable decided "1":
        # the syndrome is satisfied when all par are +1)
        for c in range(n_chk):
            for l in range(W):
                par[c, l] = synS[c, l]
        for v in range(n_var):
            lo = var_ptr[v]
            hi = var_ptr[v + 1]
            for l in range(W):
                tot[l] = prior_llr[v]
            for k in range(lo, hi):
                e = var_edge[k]
                for l in range(W):
                    tot[l] += c2v[e, l]
            for l in range(W):
                llr_out[v, l] = llr_out[v, l] if done[l] != 0.0 else tot[l]
            for k in range(lo, hi):
                e = var_edge[k]
                c = det_idx[k]
                for l in range(W):
                    v2c[e, l] = tot[l] - c2v[e, l]
                    par[c, l] = -par[c, l] if tot[l] < 0 else par[c, l]
        # a lane whose hard decision satisfies its syndrome stops updating (early exit)
        n_done = 0
        for l in range(W):
            if done[l] == 0.0:
                ok = True
                for c in range(n_chk):
                    if par[c, l] < 0:
                        ok = False
                        break
                if ok:
                    done[l] = 1.0
            if done[l] != 0.0:
                n_done += 1
        if n_done == W:
            break


# ----------------------------------------------------------------------------------------
# 3. Ordered statistics decoding over GF(2)
# ----------------------------------------------------------------------------------------
# Bit vectors over the detectors (<= 128 of them) are stored as two uint64 words.
#
# Trick to list the set bits of a word quickly ("de Bruijn" bit scan): `x & -x` isolates
# the lowest set bit 2^i; multiplying by the constant below and keeping the top 6 bits
# gives a different number for each i, which a 64-entry table turns back into i.

_DEBRUIJN = 0x03F79D71B4CB0A89
BIT_INDEX = np.zeros(64, np.int64)
for _i in range(64):
    BIT_INDEX[((_DEBRUIJN << _i) & 0xFFFFFFFFFFFFFFFF) >> 58] = _i
assert sorted(BIT_INDEX.tolist()) == list(range(64))

DB = np.uint64(_DEBRUIJN)
ONE = np.uint64(1)
ZERO = np.uint64(0)
S58 = np.uint64(58)


@njit(cache=True)
def osd_classes(syn, order, n_order, det_ptr, det_idx, L_cols, rank_cost, prob_cost,
                sweep, sweep2, aggregate, full_rank, bit_index,
                Rc, row_rank, b_cost, b_obs, nb_mech, nb_sol, nb_cost, nb_obs, scores):
    """Ordered statistics decoding with a combination sweep.

    order[:n_order]: mechanisms sorted from most to least likely to have fired.
    rank_cost: posterior LLR from BP; prob_cost: prior LLR (-sum over an error = its
               log-probability up to a constant).
    sweep:     number of non-basis mechanisms tried one at a time on top of OSD-0.
    sweep2:    the first sweep2 of those are also tried in pairs.
    full_rank: rank of H (capped at 70): the basis is complete when it reaches this size.

    Fills scores[16], one per logical class (value of the 4 observables): if aggregate,
    log of the summed probability of all solutions found in that class; otherwise 0 for
    the class of the lowest-cost solution and -inf for the others.

    Returns False if order[:n_order] ran out before the basis and the sweep list were
    complete (the caller then retries with a longer list); True otherwise.

    How it works.  We keep a matrix R (stored column by column in Rc) such that
    R * (basis column number b) = unit vector on "pivot row" of b.  Then for ANY vector
    y in the span of the basis, R*y directly reads which basis columns XOR to y.  So
      * R*h_j for a new column j tells at once whether j is independent (a bit outside
        the pivot rows) and, if not, its expansion on the basis;
      * the OSD-0 solution is R*syndrome;
      * flipping non-basis mechanisms a (and b) changes the basis part of the solution
        to  R*syn ^ R*h_a (^ R*h_b): one or two XORs per trial instead of a new solve.
    """
    n_det = len(syn)
    # --- R = identity
    for d in range(n_det):
        if d < 64:
            Rc[d, 0] = ONE << np.uint64(d)
            Rc[d, 1] = ZERO
        else:
            Rc[d, 0] = ZERO
            Rc[d, 1] = ONE << np.uint64(d - 64)
    piv0 = ZERO                 # set of pivot rows (two words)
    piv1 = ZERO
    rank = 0
    n_nb = 0
    complete = False
    cost = prob_cost if aggregate else rank_cost
    # --- Gaussian elimination: first linearly independent columns in `order` form the basis
    for idx in range(n_order):
        j = order[idx]
        v0 = ZERO               # v = R * h_j  (XOR of the columns of R at the detectors of j)
        v1 = ZERO
        for k in range(det_ptr[j], det_ptr[j + 1]):
            d = det_idx[k]
            v0 ^= Rc[d, 0]
            v1 ^= Rc[d, 1]
        f0 = v0 & ~piv0         # part of v outside the pivot rows
        f1 = v1 & ~piv1
        if f0 == ZERO and f1 == ZERO:
            # dependent on the basis: candidate for the sweep; v is its basis expansion
            if n_nb < sweep:
                nb_mech[n_nb] = j
                nb_sol[n_nb, 0] = v0
                nb_sol[n_nb, 1] = v1
                n_nb += 1
        else:
            # independent: new basis column. Pick a new pivot row and update R with row
            # operations so that R * h_j becomes the unit vector on that row.
            if f0 != ZERO:
                low = f0 & (~f0 + ONE)
                row = bit_index[(low * DB) >> S58]
                piv0 |= low
                u0 = v0 ^ low
                u1 = v1
                for d in range(n_det):
                    if Rc[d, 0] & low:
                        Rc[d, 0] ^= u0
                        Rc[d, 1] ^= u1
            else:
                low = f1 & (~f1 + ONE)
                row = 64 + bit_index[(low * DB) >> S58]
                piv1 |= low
                u0 = v0
                u1 = v1 ^ low
                for d in range(n_det):
                    if Rc[d, 1] & low:
                        Rc[d, 0] ^= u0
                        Rc[d, 1] ^= u1
            row_rank[row] = rank
            b_cost[rank] = cost[j]
            b_obs[rank] = L_cols[j, 0]
            rank += 1
        if n_nb >= sweep and rank >= full_rank:
            complete = True
            break
    if not complete and n_order < len(cost):
        return False            # ran out of sorted mechanisms: caller must give more

    for k in range(16):
        scores[k] = -np.inf

    # --- OSD-0 solution = R * syndrome
    s0 = ZERO
    s1 = ZERO
    for d in range(n_det):
        if syn[d]:
            s0 ^= Rc[d, 0]
            s1 ^= Rc[d, 1]
    if (s0 & ~piv0) != ZERO or (s1 & ~piv1) != ZERO:
        return True             # syndrome not reachable (does not happen for a valid one)

    # --- Re-index every solution from "pivot row" to "position in the basis" (bit b <->
    # b-th basis column, b-th most likely first) and compute its observable flips.
    # Slot n_nb holds the OSD-0 solution, slots 0..n_nb-1 the sweep candidates.
    nb_sol[n_nb, 0] = s0
    nb_sol[n_nb, 1] = s1
    for i in range(n_nb + 1):
        q0 = ZERO
        q1 = ZERO
        obs = ZERO
        for w in range(2):
            x = nb_sol[i, w]
            while x != ZERO:
                low = x & (~x + ONE)
                x ^= low
                b = row_rank[64 * w + bit_index[(low * DB) >> S58]]
                obs ^= b_obs[b]
                if b < 64:
                    q0 |= ONE << np.uint64(b)
                else:
                    q1 |= ONE << np.uint64(b - 64)
        nb_sol[i, 0] = q0
        nb_sol[i, 1] = q1
        if i < n_nb:
            j = nb_mech[i]
            nb_cost[i] = cost[j]
            obs ^= L_cols[j, 0]
        nb_obs[i] = obs
    s0 = nb_sol[n_nb, 0]
    s1 = nb_sol[n_nb, 1]
    s_obs = nb_obs[n_nb]

    # --- Trials: OSD-0, then every single sweep candidate, then pairs among the first m2
    m2 = min(sweep2, n_nb)
    best_cost = np.inf
    best_k = -1
    a = -1                      # a = -1: OSD-0 and singles; a >= 0: pairs (a, i), i > a
    while a < m2:
        if a < 0:
            a0 = s0
            a1 = s1
            a_obs = s_obs
            a_cost = 0.0
            first = -1          # i = -1 stands for "no flip" (OSD-0 itself)
            last = n_nb
        else:
            a0 = s0 ^ nb_sol[a, 0]
            a1 = s1 ^ nb_sol[a, 1]
            a_obs = s_obs ^ nb_obs[a]
            a_cost = 0.0 + nb_cost[a]
            first = a + 1
            last = m2
        for i in range(first, last):
            if i < 0:
                r0 = a0
                r1 = a1
                k = np.int64(a_obs)
                c = a_cost
            else:
                r0 = a0 ^ nb_sol[i, 0]
                r1 = a1 ^ nb_sol[i, 1]
                k = np.int64(a_obs ^ nb_obs[i])
                c = a_cost + nb_cost[i]
            # add the cost of the basis mechanisms of this solution, most likely first
            # (same summation order as the reference implementation -> same rounding)
            while r0 != ZERO:
                low = r0 & (~r0 + ONE)
                r0 ^= low
                c += b_cost[bit_index[(low * DB) >> S58]]
            while r1 != ZERO:
                low = r1 & (~r1 + ONE)
                r1 ^= low
                c += b_cost[64 + bit_index[(low * DB) >> S58]]
            if aggregate:
                # scores[k] = log(exp(scores[k]) + exp(-c)), written as
                # max + log(1 + exp(-|difference|)).
                lp = -c
                sk = scores[k]
                if lp > sk:
                    diff = lp - sk
                    sk = lp
                else:
                    diff = sk - lp
                if diff < 40.0:     # beyond that 1 + exp(-diff) == 1 exactly in float64
                    sk += np.log(1.0 + np.exp(-diff))
                scores[k] = sk
            elif c < best_cost:
                best_cost = c
                best_k = k
        a += 1
    if not aggregate and best_k >= 0:
        scores[best_k] = 0.0
    return True


# ----------------------------------------------------------------------------------------
# 4. Batch decoding, parallel over shots
# ----------------------------------------------------------------------------------------

CHUNK = 128         # shots handled by one task; scratch buffers are allocated once per task


@njit(parallel=True, cache=True)
def decode_batch(dets, chk_ptr, chk_var, var_ptr, var_edge, prior_llr, det_ptr, det_idx,
                 L_cols, full_rank, bit_index, max_iter, alpha, sweep, sweep2, aggregate,
                 presort):
    """presort: how many of the most likely mechanisms are sorted first (OSD rarely needs
    more; when it does, that shot falls back to a full sort, so the result never changes)."""
    shots, n_det = dets.shape
    n_var = len(prior_llr)
    n_edge = len(chk_var)
    out = np.zeros((shots, 4), np.uint8)
    n_chunks = (shots + CHUNK - 1) // CHUNK
    for ch in prange(n_chunks):
        # scratch buffers of this task
        todo = np.empty(CHUNK, np.int64)
        synS = np.empty((n_det, W))
        llrW = np.empty((n_var, W))
        v2c = np.empty((n_edge, W))
        c2v = np.empty((n_edge, W))
        par = np.empty((n_det, W))
        llr = np.empty(n_var)
        tmp = np.empty(n_var)
        cand = np.empty(n_var, np.int64)
        cand_llr = np.empty(n_var)
        order = np.empty(n_var, np.int64)
        Rc = np.empty((n_det, 2), np.uint64)
        row_rank = np.empty(128, np.int64)
        b_cost = np.empty(128)
        b_obs = np.empty(128, np.uint64)
        nb_mech = np.empty(sweep + 1, np.int64)
        nb_sol = np.empty((sweep + 1, 2), np.uint64)
        nb_cost = np.empty(sweep + 1)
        nb_obs = np.empty(sweep + 1, np.uint64)
        scores = np.empty(16)
        w16 = np.empty(16)
        # shots of this task with at least one detector fired (the others: predict no flip)
        n_todo = 0
        for s in range(ch * CHUNK, min(shots, (ch + 1) * CHUNK)):
            w = 0
            for d in range(n_det):
                w += dets[s, d]
            if w != 0:
                todo[n_todo] = s
                n_todo += 1
        for g in range(0, n_todo, W):
            n_lane = min(W, n_todo - g)
            # --- BP on up to W shots at once (unused lanes get an all-zero syndrome)
            for l in range(W):
                for d in range(n_det):
                    synS[d, l] = -1.0 if (l < n_lane and dets[todo[g + l], d]) else 1.0
            bp_min_sum_lanes(synS, chk_ptr, chk_var, var_ptr, var_edge, det_idx, prior_llr,
                             max_iter, alpha, llrW, v2c, c2v, par)
            # --- OSD shot by shot
            for l in range(n_lane):
                s = todo[g + l]
                syn = dets[s]
                for v in range(n_var):
                    llr[v] = llrW[v, l]
                # sort only the `presort` most likely mechanisms (ties broken by index,
                # exactly like a stable sort of everything would do)
                done = False
                if presort < n_var:
                    tmp[:] = llr
                    thr = np.partition(tmp, presort - 1)[presort - 1]   # presort-th smallest LLR
                    m = 0
                    for v in range(n_var):
                        if llr[v] <= thr:
                            cand[m] = v
                            cand_llr[m] = llr[v]
                            m += 1
                    sub = np.argsort(cand_llr[:m], kind="mergesort")
                    for i in range(m):
                        order[i] = cand[sub[i]]
                    done = osd_classes(syn, order, m, det_ptr, det_idx, L_cols, llr, prior_llr,
                                       sweep, sweep2, aggregate, full_rank, bit_index,
                                       Rc, row_rank, b_cost, b_obs, nb_mech, nb_sol, nb_cost,
                                       nb_obs, scores)
                if not done:                # rare: needs the full ordering
                    full = np.argsort(llr, kind="mergesort")
                    osd_classes(syn, full, n_var, det_ptr, det_idx, L_cols, llr, prior_llr,
                                sweep, sweep2, aggregate, full_rank, bit_index,
                                Rc, row_rank, b_cost, b_obs, nb_mech, nb_sol, nb_cost,
                                nb_obs, scores)
                # each observable bit: total probability of classes with that bit 1 vs 0
                mx = scores[0]
                for c in range(1, 16):
                    if scores[c] > mx:
                        mx = scores[c]
                for c in range(16):
                    w16[c] = np.exp(scores[c] - mx)
                for k in range(4):
                    p1 = 0.0
                    p0 = 0.0
                    for c in range(16):
                        if (c >> k) & 1:
                            p1 += w16[c]
                        else:
                            p0 += w16[c]
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
PRESORT = 512       # speed only: predictions do not depend on it (see decode_batch)


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
    det_idx = rows[edge_order].astype(np.int64)       # detectors of each mechanism (CSR with var_ptr)
    full_rank = min(70, gf2_rank(H))
    return (chk_ptr, chk_var, var_ptr, var_edge, prior_llr, var_ptr, det_idx,
            pack_columns(L), full_rank, BIT_INDEX)


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
    return decode_batch(dets, *args, MAX_ITER, ALPHA, SWEEP, SWEEP2, AGGREGATE, PRESORT)


# Compile the numba kernels at import time (import is not timed).
def _warmup():
    c = stim.Circuit.generated("repetition_code:memory", rounds=2, distance=3,
                               before_round_data_depolarization=0.01)
    args = prepare(c)
    dets = np.ones((2, c.num_detectors), np.uint8)
    decode_batch(dets, *args, MAX_ITER, ALPHA, SWEEP, SWEEP2, AGGREGATE, PRESORT)


_warmup()
