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


def get_shapes(matrices):
    return [tl.shape(matrix) for matrix in matrices]


def get_padded_tensor_shape(matrices):
    I = len(matrices)
    K = tl.shape(matrices[0])[1]

    # Compute max J and check that all matrices have the same number of columns
    J = -float("inf")
    for matrix in matrices:
        J_i, K_i = tl.shape(matrix)
        if K_i != K:
            raise ValueError("All matrices must have the same number of columns")

        J = max(J_i, J)

    return I, J, K


def create_padded_tensor(matrices):
    tensor = tl.zeros(get_padded_tensor_shape(matrices), **tl.context(matrices[0]))
    for i, matrix in enumerate(matrices):
        length = tl.shape(matrix)[0]
        tensor = tl.index_update(tensor, tl.index[i, :length], matrix)
    return tensor
