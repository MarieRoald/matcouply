import itertools


def all_combinations(*args):
    """All combinations of the input iterables.

    Each argument must be an iterable.

    Examples:
    ---------
    >>> all_combinations([1, 2], ["ab", "cd"])
    [(1, 'ab'), (1, 'cd'), (2, 'ab'), (2, 'cd')]
    """
    return list(itertools.product(*args))
