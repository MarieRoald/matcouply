import os

# We set environment variables before we load the matcouply fixtures to avoid import-time side effects

# Disable JIT for unit tests
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Anaconda on Windows can have problems with multiple linked OpenMP dlls. This (unsafe) workaround makes it possible to run code with multiple linked OpenMP dlls.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

pytest_plugins = "matcouply.testing.fixtures"
