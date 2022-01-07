import tensorly as tl


def pytest_ignore_collect():
    if tl.get_backend() != "numpy":
        return True
    return False
