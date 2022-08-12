# MIT License: Copyright (c) 2022, Marie Roald.
# See the LICENSE file in the root directory for full license text.

import numpy as np
import tensorly as tl

from .admm_penalty import *


def assert_allclose(actual, desired, *args, **kwargs):
    np.testing.assert_allclose(tl.to_numpy(actual), tl.to_numpy(desired), *args, **kwargs)
