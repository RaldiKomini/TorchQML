import torch


def weighted_usage_sample(usage, count: int) -> list[int]:
    """Sample less-used feature/parameter indices first."""
    if count <= 0:
        return []
    weights = torch.exp(-torch.as_tensor(usage, dtype=torch.float32))
    picks = []
    local_usage = list(usage)
    for _ in range(count):
        probs = weights / weights.sum()
        idx = int(torch.multinomial(probs, 1).item())
        picks.append(idx)
        local_usage[idx] += 1
        weights[idx] = torch.exp(torch.tensor(-float(local_usage[idx])))
    return picks


def _pattern_size(pattern, attr: str, nq: int) -> int:
    value = getattr(pattern, attr, 0)
    return value(nq) if callable(value) else value


def _pattern_name(pattern) -> str:
    return getattr(pattern, "name", getattr(pattern, "__name__", pattern.__class__.__name__))


def beam_search(c0, x_sym, t_sym, specs, patterns, criterion, *, depth: int = 5, beam_size: int = 8, verbose: bool = False):
    """Beam search over symbolic circuit-building patterns."""
    xlen = c0.specs.xlen
    tlen = c0.specs.tlen
    beams = [(c0, [0] * xlen, [0] * tlen, criterion(c0), [])]

    for level in range(depth):
        candidates = []
        for circ_prev, used_x, used_t, _, pattern_list in beams:
            for pattern in patterns:
                kx = _pattern_size(pattern, "kx", specs.num_qubits)
                kt = _pattern_size(pattern, "kt", specs.num_qubits)
                idx_x = weighted_usage_sample(used_x, kx)
                idx_t = weighted_usage_sample(used_t, kt)

                circ_try = circ_prev.copy()
                circ_try = pattern(circ_try, idx_x, idx_t, x_sym, t_sym)
                score = criterion(circ_try)

                next_used_x = used_x.copy()
                next_used_t = used_t.copy()
                for idx in idx_x:
                    next_used_x[idx] += 1
                for idx in idx_t:
                    next_used_t[idx] += 1

                candidates.append((circ_try, next_used_x, next_used_t, score, pattern_list + [pattern]))

        beams = sorted(candidates, key=lambda row: row[3], reverse=True)[:beam_size]
        if verbose:
            top = [(_pattern_name(p) for p in row[4]) for row in beams[:3]]
            print(f"depth {level}: {[list(names) for names in top]}")

    best_circ, _, best_used_t, best_score, best_patterns = beams[0]
    used_t_idx = [idx for idx, count in enumerate(best_used_t) if count > 0]
    actual_tlen = max(used_t_idx, default=-1) + 1
    return best_circ, best_patterns, actual_tlen, used_t_idx, best_score


search = beam_search

__all__ = ["beam_search", "search", "weighted_usage_sample"]
