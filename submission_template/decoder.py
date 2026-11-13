"""Beam maximum-likelihood decoder with a confidence cascade (numpy + numba only, written from scratch).

CASCADE (see decode() at the bottom): every shot is first decoded with a cheap beam.  The
decoder knows how sure it is: for each observable it has the probability p of a flip, and
the margin of a shot is min over the 4 observables of |p - 0.5|.  Only the shots whose
margin is small (the decision could change with a more exact computation) are decoded
again with a wider beam and looser pruning, and the least certain of those once more with
the heaviest settings.  The beam itself is described below.

Beam maximum-likelihood decoder (one pass):

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
      score = weight x LOOK^(number of syndrome bits still unexplained)
(a state with more unexplained bits will need more errors later, so it is worth less).
The SAME score decides which children are created at all: a child whose score is below
CHILD x (best score of the current states) would be thrown away by the ranking anyway, so
it is never put in the table.  This is what makes two-mechanism children affordable: only
the pairs that really help (those that clear syndrome bits) are created.
At the end all surviving states have an empty syndrome; summing weights per logical value
gives the class probabilities, and each observable bit is predicted by its marginal.
Fully deterministic.
"""
import numpy as np
import stim
from numba import get_num_threads, njit, prange, types
from numba.extending import intrinsic

LBIT = 8  # logical bits live in bits 8..11 of the second key word (detectors 64..69 in bits 0..5)


def build_problem(circuit: stim.Circuit):
    """H (detectors x mechanisms), L (observables x mechanisms), p, from the circuit's DEM.
    Mechanisms with the same effect are merged (p = a(1-b) + b(1-a))."""
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    merged = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        # Round to 12 significant digits: stim derives these probabilities with library
        # functions (powers, roots) whose last bit may differ between machines.  After
        # rounding, every machine starts from exactly the same numbers.
        prob = float(f"{inst.args_copy()[0]:.12g}")
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


@intrinsic
def _ctpop(typingctx, x):
    """Number of set bits of a 64-bit integer, as one processor instruction (LLVM ctpop)."""
    def codegen(context, builder, signature, args):
        fn = builder.module.declare_intrinsic("llvm.ctpop", [args[0].type])
        return builder.call(fn, [args[0]])
    return types.uint64(types.uint64), codegen


@njit(cache=True, inline='always')
def _popcount(x):
    return np.int64(_ctpop(x))


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
def beam_one(s0, s1, m0, m1, odds, grp_ptr, K, thr1, thr2, child, keep, pow_look, max_group,
             ps, a0, a1, w, b0, b1, bw, sc, tmp, table, ctx, P):
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
        smax = 0.0                # best score among the current states
        n_set = 0                 # number of states whose bit d is set
        for i in range(n):
            # score of the state = weight x LOOK^(unexplained bits)
            ps[i] = w[i] * pow_look[_popcount(a0[i]) + _popcount(a1[i] & np.uint64(63))]
            if ps[i] > smax:
                smax = ps[i]
            if d < 64:
                n_set += np.int64((a0[i] >> np.uint64(d)) & np.uint64(1))
            else:
                n_set += np.int64((a1[i] >> np.uint64(d - 64)) & np.uint64(1))
        # Two levels of cuts, all relative to the best score:
        #  - cut1 / cut2: (score of the parent) x (odds of the fired mechanisms) below this
        #    -> stop scanning the group (it is sorted by decreasing odds).  These are loose
        #    bounds: a child can score higher than that if it clears syndrome bits.
        #  - cutc: exact score of the child below this -> the child is not created.
        cut1 = thr1 * smax
        cut2 = thr2 * smax
        cutc = child * smax
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
            si = ps[i]
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
                    sj = si * odds[j]
                    if sj * odds[j + 1] < cut2:
                        break
                    if 2 * (m + max_group) > size and size < tsize:
                        size = _room(m, max_group, size, tsize)
                        _new_table(size, b0, b1, m, table, ctx)
                        cur = ctx[0] << 32
                        mask = np.uint64(size - 1)
                    y0 = x0 ^ m0[j]
                    y1 = x1 ^ m1[j]
                    for k in range(j + 1, g1):
                        if sj * odds[k] < cut2:
                            break
                        z0 = y0 ^ m0[k]
                        z1 = y1 ^ m1[k]
                        wk = wi * odds[j] * odds[k]
                        # exact score of this child; skip it if the ranking would drop it
                        if wk * pow_look[_popcount(z0) + _popcount(z1 & np.uint64(63))] < cutc:
                            continue
                        m = _add(z0, z1, wk, b0, b1, bw, m, table, cur, mask)
            else:
                # exactly one mechanism of the group fires
                for j in range(g0, g1):
                    if si * odds[j] < cut1:
                        break
                    wj = wi * odds[j]
                    z0 = x0 ^ m0[j]
                    z1 = x1 ^ m1[j]
                    if wj * pow_look[_popcount(z0) + _popcount(z1 & np.uint64(63))] < cutc:
                        continue
                    m = _add(z0, z1, wj, b0, b1, bw, m, table, cur, mask)
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
def decode_batch(dets, m0, m1, odds, grp_ptr, K, thr1, thr2, child, keep, look, cap_mult, n_threads):
    """Decode all shots (parallel over shots). Returns predictions (shots, 4) and the
    work done per shot (diagnostic only)."""
    shots, nD = dets.shape
    out = np.zeros((shots, 4), np.uint8)
    work = np.zeros(shots, np.int64)
    marg = np.full(shots, 0.5)
    cap = K * cap_mult + 256            # max number of states created for one detector
    tsize = 1                           # max hash-table size (power of two >= 2 x cap)
    while tsize < 2 * cap:
        tsize *= 2
    # look ** i by repeated multiplication (not the library power function), so that the
    # result is bit-identical on every machine.
    pow_look = np.empty(80)
    pow_look[0] = 1.0
    for i in range(1, 80):
        pow_look[i] = pow_look[i - 1] * look
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
    # One chunk per thread: chunk ch takes the shots number ch, ch + n_chunks, ... of the sorted
    # list, so every THREAD gets the same mix of easy and hard shots, even when a tier of the
    # cascade only has a few thousand shots (with many more chunks than that, each thread
    # would receive a contiguous slice of the sorted list and the last ones all the hard shots).
    n_chunks = min(shots, n_threads)    # each chunk owns its work arrays
    for ch in prange(n_chunks):
        # np.empty costs nothing until the memory is really used: only the beginning of
        # these (generously sized) arrays is ever touched.
        a0 = np.empty(K, np.uint64); a1 = np.empty(K, np.uint64); w = np.empty(K)
        ps = np.empty(K)                # scores of the current states
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
            work[s] = beam_one(s0, s1, m0, m1, odds, grp_ptr, K, thr1, thr2, child, keep, pow_look,
                               max_group, ps, a0, a1, w, b0, b1, bw, sc, tmp, table, ctx, P)
            tot = 0.0
            for c in range(16):
                tot += P[c]
            mm = 0.5
            for k in range(4):
                p1 = 0.0
                for c in range(16):
                    if (c >> k) & 1:
                        p1 += P[c]
                out[s, k] = 1 if p1 > 0.5 * tot else 0
                mm = min(mm, abs(p1 / tot - 0.5))
            marg[s] = mm
    return out, work, marg


# ----------------------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------------------
LOOK = 0.15        # look-ahead penalty per unexplained syndrome bit
CAP_MULT = 256     # work-table size = K x CAP_MULT

# One line per tier: (K, THR1, THR2, CHILD, KEEP, NEXT)
#   K      at most this many states kept after each detector
#   THR1   stop scanning a group when parent score x odds < THR1 x best score
#   THR2   same with the product of the odds of two mechanisms of the group
#   CHILD  a child scoring below CHILD x (best score) is not created
#   KEEP   states scoring below KEEP x (best score) are dropped
#   NEXT   shots whose margin is below NEXT are decoded again by the next tier
TIERS = [
    (200, 1e-7, 1e-6, 3e-5, 1e-5, 0.45),      # cheap pass, every shot
    (1000, 1e-9, 1e-8, 3e-6, 1e-6, 0.1),      # the previous single-pass settings, ~17% of the shots
    (2000, 1e-10, 1e-9, 3e-7, 1e-7, 0.0),     # heaviest pass, ~3.5% of the shots (last tier: NEXT unused)
]

_cache = {}


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """circuit: the X or Z circuit. detector_events: bool (shots, 70).
    Returns a 0/1 array (shots, 4)."""
    key = str(circuit)
    if key not in _cache:
        _cache[key] = prepare(circuit)
    prep = _cache[key]
    dets = np.ascontiguousarray(detector_events, dtype=np.uint8)
    out = np.zeros((len(dets), 4), np.uint8)
    idx = np.arange(len(dets))           # shots handled by the current tier
    n_threads = get_num_threads()        # read here: decode_batch stays cacheable
    for K, thr1, thr2, child, keep, nxt in TIERS:
        if len(idx) == 0:
            break
        sub = dets if len(idx) == len(dets) else np.ascontiguousarray(dets[idx])
        o, _, marg = decode_batch(sub, *prep, K, thr1, thr2, child, keep, LOOK, CAP_MULT, n_threads)
        out[idx] = o
        idx = idx[marg < nxt]
    return out


def _warmup():
    c = stim.Circuit.generated("repetition_code:memory", rounds=2, distance=3,
                               before_round_data_depolarization=0.01)
    decode_batch(np.ones((2, c.num_detectors), np.uint8), *prepare(c), 8, 1e-9, 1e-8, 3e-6, 1e-6, LOOK, CAP_MULT, 2)


_warmup()
