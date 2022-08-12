# MIT License: Copyright (c) 2022, Marie Roald.
# See the LICENSE file in the root directory for full license text.

import tensorly as tl

if tl.get_backend() == "numpy":
    RTOL_SCALE = 1
else:
    RTOL_SCALE = 500  # Single precision backends need less strict tests
