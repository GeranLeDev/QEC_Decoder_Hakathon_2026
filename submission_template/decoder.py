"""Beam maximum-likelihood decoder, fast version (numpy + numba only, written from scratch).

We want, for the observed syndrome, the total probability of every logical class
(16 values of the 4 observables), summed over ALL error sets that explain the syndrome.

Sweep over the detectors in order.  Mechanisms are grouped by their FIRST detector d, so
once group d is processed nothing can change detector d any more.  A state is
      (syndrome bits still to explain, logical flip so far)  ->  weight
where weight = sum over the error sets leading there of prod p/(1-p)  (odds; the common
factor prod(1-p) is the same for everything and is dropped).
For group d each state must end with bit d explained:
  - bit d already 0: fire none of the group's mechanisms (or two of them),
  - bit d is 1:      fire exactly one of them (or three).
Children that reach the same state are merged (their weights add up: that is the sum over
equivalent errors that makes this a maximum-likelihood rather than a min-weight decoder).
After each group only the K best states are kept, ranked by
      weight x LOOK^(number of syndrome bits still unexplained)
(a state with more unexplained bits will need more errors later, so it is worth less).
At the end all surviving states have an empty syndrome; summing weights per logical value
gives the class probabilities, and each observable bit is predicted by its marginal.
Fully deterministic.
"""
import numpy as np
import stim
from numba import njit, prange

LBIT = 8  # logical bits live in bits 8..11 of the second key word (detectors 64..69 in bits 0..5)


def build_problem(circuit: stim.Circuit):
    """H (detectors x mechanisms), L (observables x mechanisms), p, from the circuit's DEM.
    Mechanisms with the same effect are merged (p = a(1-b) + b(1-a))."""
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
            continue
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


def prepare(circuit: stim.Circuit):
    """Pack every mechanism into two 64-bit words (detector bits + logical bits), group the
    mechanisms by first detector and sort each group from most to least likely."""
    H, L, p = build_problem(circuit)
    nD, nM = H.shape
    assert nD <= 64 + LBIT and L.shape[0] <= 4
    first = np.array([np.nonzero(H[:, j])[0][0] for j in range(nM)])
    odds = p / (1 - p)
    order = np.lexsort((-odds, first))          # by group, then by decreasing odds
    k0 = np.zeros(nM, np.uint64)
    k1 = np.zeros(nM, np.uint64)
    for d in range(nD):
        if d < 64:
            k0[H[d] == 1] |= np.uint64(1) << np.uint64(d)
        else:
            k1[H[d] == 1] |= np.uint64(1) << np.uint64(d - 64)
    for k in range(L.shape[0]):
        k1[L[k] == 1] |= np.uint64(1) << np.uint64(LBIT + k)
    grp_ptr = np.zeros(nD + 1, np.int64)
    np.add.at(grp_ptr, first + 1, 1)
    grp_ptr = np.cumsum(grp_ptr)
    return k0[order], k1[order], odds[order].copy(), grp_ptr


@njit(cache=True, inline='always')
def _popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


# ----------------------------------------------------------------------------------------
# Hash table of the states created for the current detector.
#
# The states themselves live in the plain arrays b0/b1/bw, in the order in which they were
# first created (state number 0, 1, 2, ...).  The hash table only answers "does this state
# already exist, and what is its number?".  One table slot is a single 64-bit integer:
#       high 32 bits = stamp,   low 32 bits = state number.
# A slot is in use only if its stamp equals the current stamp, so "emptying" the table is
# just taking a new stamp (no memory is cleared).
#
# Speed: the table is kept SMALL (a few times the number of states actually present) so it
# stays in the CPU cache; it is doubled and refilled when it gets half full.  The table
# layout has no influence on the result: state numbers and the order of the additions
# depend only on the order in which the children are generated.
# ----------------------------------------------------------------------------------------
IDX_MASK = 0xFFFFFFFF
PRESIZE = 4          # table slots reserved per expected child (emptier table = fewer collisions)


@njit(cache=True, inline='always')
def _hash(t0, t1):
    """64-bit hash of a state; its low bits give the table position.  Two states often
    differ only in a few HIGH detector bits, and a multiplication alone never carries high
    bits down to the low ones, so the high half is folded onto the low half before the
    multiplication and once more after it: every input bit then influences the low bits.
    (The table position has no effect on the result, only on the speed.)"""
    x = t0 ^ (t1 * np.uint64(0xC2B2AE3D27D4EB4F))
    x ^= x >> np.uint64(32)
    x *= np.uint64(0x9E3779B97F4A7C15)
    return x ^ (x >> np.uint64(32))


@njit(cache=True, inline='always')
def _add(t0, t1, wt, b0, b1, bw, m, table, cur, mask):
    """Add weight wt to state (t0, t1) (create it if absent). Returns the new number of
    states.  `cur` is the current stamp already shifted to the high 32 bits."""
    pos = np.int64(_hash(t0, t1) & mask)
    while True:
        e = table[pos]
        if (e & ~IDX_MASK) != cur:        # free slot: the state does not exist yet
            break
        i = e & IDX_MASK
        if b0[i] == t0 and b1[i] == t1:   # found: merge (weights add up)
            bw[i] += wt
            return m
        pos = (pos + 1) & np.int64(mask)
    if m < len(bw):
        table[pos] = cur | m
        b0[m] = t0
        b1[m] = t1
        bw[m] = wt
        return m + 1
    return m


@njit(cache=True)
def _new_table(size, b0, b1, m, table, ctx):
    """Start using a table of `size` slots under a NEW stamp and put the m existing states
    (all different) into it.  ctx[0] = last stamp used; ctx[1] = number of table slots
    already zeroed: the table is allocated uninitialised and zeroed lazily, only the part
    that is really used (stamp 0 = never used)."""
    if size > ctx[1]:
        table[ctx[1]:size] = 0
        ctx[1] = size
    ctx[0] += 1
    cur = ctx[0] << 32
    mask = np.uint64(size - 1)
    for i in range(m):
        pos = np.int64(_hash(b0[i], b1[i]) & mask)
        while (table[pos] & ~IDX_MASK) == cur:
            pos = (pos + 1) & np.int64(mask)
        table[pos] = cur | i


@njit(cache=True, inline='always')
def _room(m, need, size, tsize):
    """Smallest table size (doubling from `size`, at most tsize) that stays at most half
    full after `need` more states."""
    while 2 * (m + need) > size and size < tsize:
        size *= 2
    return size


@njit(cache=True)
def _max_group(grp_ptr):
    """Size of the largest group of mechanisms."""
    best = 1
    for d in range(len(grp_ptr) - 1):
        best = max(best, grp_ptr[d + 1] - grp_ptr[d])
    return best


@njit(cache=True)
def beam_one(s0, s1, m0, m1, odds, grp_ptr, K, thr1, thr2, keep, pow_look, max_group,
             a0, a1, w, b0, b1, bw, sc, tmp, table, ctx, P):
    """Fills P (16,) with the weights of the logical classes for one syndrome (s0, s1).
    The work arrays are passed in so that each thread allocates them only once.
    ctx = [last stamp used, number of table slots already zeroed] (see _new_table).
    Returns the number of states created (a measure of the work done)."""
    tsize = len(table)
    n = 1
    a0[0] = s0
    a1[0] = s1
    w[0] = 1.0
    work = 0
    nD = len(grp_ptr) - 1
    for d in range(nD):
        g0 = grp_ptr[d]
        g1 = grp_ptr[d + 1]
        wmax = 0.0
        n_set = 0                 # number of states whose bit d is set
        for i in range(n):
            if w[i] > wmax:
                wmax = w[i]
            if d < 64:
                n_set += np.int64((a0[i] >> np.uint64(d)) & np.uint64(1))
            else:
                n_set += np.int64((a1[i] >> np.uint64(d - 64)) & np.uint64(1))
        cut1 = thr1 * wmax        # one-mechanism children lighter than this are not created
        cut2 = thr2 * wmax        # same for two-mechanism children
        # fresh (empty) table for this detector, sized for the expected number of children:
        # one per state, plus the whole group for the states whose bit is set (pairs are
        # rare; if the guess is too small the table simply grows below)
        size = _room(0, PRESIZE * (n + n_set * (g1 - g0)), 256, tsize)
        _new_table(size, b0, b1, 0, table, ctx)
        cur = ctx[0] << 32
        mask = np.uint64(size - 1)
        m = 0
        for i in range(n):
            # Keep the table at most half full: before a burst of at most max_group + 1
            # insertions, double it if necessary (the states are re-entered under a new stamp).
            if 2 * (m + max_group + 1) > size and size < tsize:
                size = _room(m, max_group + 1, size, tsize)
                _new_table(size, b0, b1, m, table, ctx)
                cur = ctx[0] << 32
                mask = np.uint64(size - 1)
            x0 = a0[i]
            x1 = a1[i]
            wi = w[i]
            if d < 64:
                bit = (x0 >> np.uint64(d)) & np.uint64(1)
            else:
                bit = (x1 >> np.uint64(d - 64)) & np.uint64(1)
            if bit == 0:
                # nothing fires in this group ...
                m = _add(x0, x1, wi, b0, b1, bw, m, table, cur, mask)
                # ... or two mechanisms fire (rare; the group is sorted by decreasing
                # odds, so we can stop as soon as the product gets too small)
                for j in range(g0, g1 - 1):
                    wj = wi * odds[j]
                    if wj * odds[j + 1] < cut2:
                        break
                    if 2 * (m + max_group) > size and size < tsize:
                        size = _room(m, max_group, size, tsize)
                        _new_table(size, b0, b1, m, table, ctx)
                        cur = ctx[0] << 32
                        mask = np.uint64(size - 1)
                    y0 = x0 ^ m0[j]
                    y1 = x1 ^ m1[j]
                    for k in range(j + 1, g1):
                        wk = wj * odds[k]
                        if wk < cut2:
                            break
                        m = _add(y0 ^ m0[k], y1 ^ m1[k], wk, b0, b1, bw, m, table, cur, mask)
            else:
                # exactly one mechanism of the group fires
                for j in range(g0, g1):
                    wj = wi * odds[j]
                    if wj < cut1:
                        break
                    m = _add(x0 ^ m0[j], x1 ^ m1[j], wj, b0, b1, bw, m, table, cur, mask)
        work += m
        if m >= len(bw):
            work += 1000000000   # table full (never happens with sane settings): flag it
        # score = weight x LOOK^(unexplained bits); keep the states whose score is at
        # least KEEP x best score, and at most K of them
        best = 0.0
        for i in range(m):
            sc[i] = bw[i] * pow_look[_popcount(b0[i]) + _popcount(b1[i] & np.uint64(63))]
            if sc[i] > best:
                best = sc[i]
        floor = keep * best
        # usual case: at most K states pass the floor -> copy them directly (index order).
        # cnt counts ALL the states passing the floor, even those that no longer fit.
        cnt = 0
        for i in range(m):
            if sc[i] >= floor:
                if cnt < K:
                    a0[cnt] = b0[i]; a1[cnt] = b1[i]; w[cnt] = bw[i]
                cnt += 1
        if cnt <= K:
            n = cnt
        else:
            # more than K candidates: find the K-th largest score.  Only the cnt scores
            # passing the floor matter (the K-th largest is necessarily one of them).
            c = 0
            for i in range(m):
                if sc[i] >= floor:
                    tmp[c] = sc[i]
                    c += 1
            kth = np.partition(tmp[:cnt], cnt - K)[cnt - K]      # K-th largest score
            n = 0
            for i in range(m):
                if sc[i] > kth:
                    a0[n] = b0[i]; a1[n] = b1[i]; w[n] = bw[i]
                    n += 1
            for i in range(m):                              # ties, in index order
                if n >= K:
                    break
                if sc[i] == kth:
                    a0[n] = b0[i]; a1[n] = b1[i]; w[n] = bw[i]
                    n += 1
    P[:] = 0.0
    for i in range(n):
        P[np.int64(a1[i] >> np.uint64(LBIT)) & 15] += w[i]
    return work


@njit(parallel=True, cache=True)
def decode_batch(dets, m0, m1, odds, grp_ptr, K, thr1, thr2, keep, look, cap_mult):
    """Decode all shots (parallel over shots). Returns predictions (shots, 4) and the
    work done per shot (diagnostic only)."""
    shots, nD = dets.shape
    out = np.zeros((shots, 4), np.uint8)
    work = np.zeros(shots, np.int64)
    cap = K * cap_mult + 256            # max number of states created for one detector
    tsize = 1                           # max hash-table size (power of two >= 2 x cap)
    while tsize < 2 * cap:
        tsize *= 2
    pow_look = np.empty(80)
    for i in range(80):
        pow_look[i] = look ** i
    max_group = _max_group(grp_ptr)
    # Share the work evenly between the threads.  The cost of a shot grows with the number
    # of detectors that fired, so the shots are sorted by that number (counting sort) and
    # then dealt like cards: chunk ch takes the shots number ch, ch + n_chunks, ... of the
    # sorted list.  Every chunk gets the same mix of easy and hard shots.  The decoding of a
    # shot does not depend on which chunk handles it, so this changes nothing but the speed.
    fired = np.zeros(shots, np.int64)
    start = np.zeros(nD + 2, np.int64)
    for s in range(shots):
        c = 0
        for d in range(nD):
            c += dets[s, d] != 0
        fired[s] = c
        start[c + 1] += 1
    for c in range(nD + 1):
        start[c + 1] += start[c]
    order = np.empty(shots, np.int64)
    for s in range(shots):
        order[start[fired[s]]] = s
        start[fired[s]] += 1
    # 1920 = 48 x 40 is a multiple of the usual thread counts (2, 4, 8, 10, 12, 16, 24, 32,
    # 48, 64, 96...), so every thread receives the same number of chunks.
    n_chunks = min(shots, 1920)         # each chunk owns its work arrays
    for ch in prange(n_chunks):
        # np.empty costs nothing until the memory is really used: only the beginning of
        # these (generously sized) arrays is ever touched.
        a0 = np.empty(K, np.uint64); a1 = np.empty(K, np.uint64); w = np.empty(K)
        b0 = np.empty(cap, np.uint64); b1 = np.empty(cap, np.uint64); bw = np.empty(cap)
        sc = np.empty(cap); tmp = np.empty(cap)
        table = np.empty(tsize, np.int64)
        ctx = np.zeros(2, np.int64)     # [last stamp, number of table slots zeroed]
        P = np.empty(16)
        for q in range(ch, shots, n_chunks):
            s = order[q]
            s0 = np.uint64(0)
            s1 = np.uint64(0)
            for d in range(nD):
                if dets[s, d]:
                    if d < 64:
                        s0 |= np.uint64(1) << np.uint64(d)
                    else:
                        s1 |= np.uint64(1) << np.uint64(d - 64)
            if s0 == 0 and s1 == 0:
                continue                # no detector fired: predict no logical flip
            work[s] = beam_one(s0, s1, m0, m1, odds, grp_ptr, K, thr1, thr2, keep, pow_look,
                               max_group, a0, a1, w, b0, b1, bw, sc, tmp, table, ctx, P)
            tot = 0.0
            for c in range(16):
                tot += P[c]
            for k in range(4):
                p1 = 0.0
                for c in range(16):
                    if (c >> k) & 1:
                        p1 += P[c]
                out[s, k] = 1 if p1 > 0.5 * tot else 0
    return out, work


# ----------------------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------------------
K_BEAM = 300      # at most this many states kept after each detector
THR1 = 1e-8       # one-mechanism children lighter than THR1 x (heaviest state) are not created
THR2 = 1e-6       # same for children made by two mechanisms of the same group
KEEP = 1e-5        # states scoring below KEEP x (best score) are dropped
LOOK = 0.1        # look-ahead penalty per unexplained syndrome bit
CAP_MULT = 256    # work-table size = K_BEAM x CAP_MULT

_cache = {}


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """circuit: the X or Z circuit. detector_events: bool (shots, 70).
    Returns a 0/1 array (shots, 4)."""
    key = str(circuit)
    if key not in _cache:
        _cache[key] = prepare(circuit)
    dets = np.ascontiguousarray(detector_events, dtype=np.uint8)
    return decode_batch(dets, *_cache[key], K_BEAM, THR1, THR2, KEEP, LOOK, CAP_MULT)[0]


def _warmup():
    c = stim.Circuit.generated("repetition_code:memory", rounds=2, distance=3,
                               before_round_data_depolarization=0.01)
    decode_batch(np.ones((2, c.num_detectors), np.uint8), *prepare(c), 8, THR1, THR2, KEEP, LOOK, CAP_MULT)


_warmup()
