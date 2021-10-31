from cm_aoadmm import _utils as utils


def test_is_iterable():
    # TODO: Make this test. 
    # TESTPLAN:
    # Use a list, tuple, str, generator and class with the __iter__ function
    # Test with some stuff that is not iterable too (int, float, class without __iter__)
    pass
    

def test_get_svd(rng):
    # TODO: Make this test. 
    # TESTPLAN:
    # Create random matrix, X
    # Compute SVD with both scipy.linalg.svd (U1, s1, Vh1) and utils.get_svd called with svd="truncated_svd" and svd="numpy_svd" (U2, s2, Vh2)
    # Compute U1.T@U2 and Vh1.T@Vh2
    # Assert that U1.T@U2 == Vh1.T@Vh2
    # Assert that U1.T@U2 * Vh1.T@Vh2 == identity
    # Assert that s1 == s2
    pass
