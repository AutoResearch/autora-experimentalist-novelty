from autora.experimentalist.sampler.novelty import novelty_sampler
import numpy as np

# Note: We encourage you to write more functionality tests for your sampler.

def test_output_dimensions():
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    X_ref = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
    n = 2
    X_new = novelty_sampler(X, n, X_ref)

    # Check that the sampler returns n experiment conditions
    assert X_new.shape == (n, X.shape[1])

def test_novelty_sampler_1D():

    num_samples = 2

    # define two matrices
    matrix1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    matrix2 = np.array([1, 2, 3])

    # reorder matrix1 according to its distances to matrix2
    reordered_matrix1 = novelty_sampler(X = matrix1, n = num_samples, X_ref = matrix2)

    assert reordered_matrix1.shape[0] == num_samples
    assert reordered_matrix1.shape[1] == 1
    assert np.array_equal(reordered_matrix1, np.array([[10], [9]]))


def test_novelty_sampler_ND():
    # define two matrices
    matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    matrix2 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    num_samples = 2

    # reorder matrix1 according to its distances to matrix2
    reordered_matrix1 = novelty_sampler(X = matrix1, n = num_samples, X_ref = matrix2)

    assert reordered_matrix1.shape[0] == 2
    assert reordered_matrix1.shape[1] == 3
    assert np.array_equal(reordered_matrix1, np.array([[10, 11, 12], [7, 8, 9]]))