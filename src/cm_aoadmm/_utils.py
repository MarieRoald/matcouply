def is_iterable(x):
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True
    