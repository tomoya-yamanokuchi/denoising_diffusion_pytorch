


def closed_range(start: int, stop: int) -> range:
    """Return integers from start through stop, inclusive."""
    step = 1 if start <= stop else -1
    return range(start, stop + step, step)
