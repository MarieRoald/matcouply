# MIT License: Copyright (c) 2022, Marie Roald.
# See the LICENSE file in the root directory for full license text.


def random_length(rng, min=2, mean=5):
    """Generate a random dimension length.

    Use Poisson distribution since it is discrete and centered around the mean.
    """
    if min >= mean:
        raise ValueError("Min must be less than mean.")
    return min + round(rng.poisson(mean - min))
