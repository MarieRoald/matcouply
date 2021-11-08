import tensorly as tl


def is_iterable(x):
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True


def get_svd(svd):
    if svd in tl.SVD_FUNS:
        return tl.SVD_FUNS[svd]
    else:
        message = "Got svd={}. However, for the current backend ({}), the possible choices are {}".format(
            svd, tl.get_backend(), tl.SVD_FUNS
        )
        raise ValueError(message)
