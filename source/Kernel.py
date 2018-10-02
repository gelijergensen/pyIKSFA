"""
A Kernel object contains several functions which are related to the mathematical kernel function of the IKSFA.
First, each kernel requires the number of output dimensions given a particular shape input. Second, each kernel needs to provide the matrix resulting from the kernel of two matricies. NOTE: for this, the columns of each of the input matricies are considered to be the individual datapoints! Third, each kernel needs to provide the 3tensor gradient of the kernel at the points given by the columns of two matricies.

Additionally, for testing purposes, it is recommended that the expansion function also be attached to the kernel, as for all cases where the equivalent expansion function can actually be computed for the kernel, there is a nice relationship between the kernel/gradient matricies/tensors and the dot products of the expansion of the inputs.

@author G. Eli Jergensen <gelijergensen@ou.edu> - constructed following the
    paper by Stephen Liwicki et. al
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.special import binom


class Kernel(ABC):

    @abstractmethod
    def getMatrix(self, X, Y, only_diag=False):
        """Returns the result of computing the kernel function, k over X and Y, i.e. k(X,Y). This is equivalent to
        phi(X)^* phi(Y), where (.)^* denotes conjugate transpose, and phi is the implicit expansion function related to
        the kernel function

        :param X: a numpy array for the left parameter (shape (N, M))
        :param Y: a numpy array for the right parameter (shape (N, P))
        :param only_diag: set to True to only compute the diagonal of the matrix
        :returns: a numpy array of shape (M, P). If :only_diag:, returns a numpy array of shape (min(M, P), )
        """
        pass

    @abstractmethod
    def getDimension(self, input_shape):
        """Returns the dimension of the implicit space, which may be useful as a default value for the dimensionality
        reduction of the whitening step in KSFA

        :param input_shape: a tuple for the shape of the input data (dimension of datapoint, number of datapoints)
        :returns: dimension of the resulting vector
        """
        pass

    @abstractmethod
    def getGradient(self, Y, X, only_diag=False):
        """Returns the right-gradient vectors of the kernel function evaluated at each pair of columns of X and Y
        Given k(a1, ..., an, b1, ..., bn), define the right-gradient vector as
        Vr(k) = (del(k)/del(b1), ..., del(k)/del(bn)),
        i.e. the vector of the partial derivatives of k with regards to the components of the right vector

        For evaluation, each column of X is used once as the left vector and y is always used as the right vector
        i.e. Vr(k)(X = [x1 x2], Y = [y1 y2]) = [Vr(k)(x1, y1) Vr(k)(x1, y1)]
                                               [Vr(k)(x2, y1) Vr(k)(x2, y2)]

        :param Y: a matrix whose columns are inputs to the left input of the gradient of the kernel function
        :param X: a matrix whose columns are inputs to the right input of the gradient of the kernel function
        :param only_diag: if set to true, then only the diagonal of the resulting 3-Tensor is computed
        :returns: a numpy array of shape (n, X.shape[1], Y.shape[1]), where n is the dimension of the input vectors
            (i.e. X.shape[0]). Each of the elements along the first axis is the gradient evaluated at the pair
            (xi, yj), Vr(k)(xi, yj). If :only_diag: is set to True, then a numpy array of shape
            (n, min(X.shape[1], Y.shape[1])) is returned instead and each element along the first axis is the gradient
            evaluated at the pair (xi, yi)
        """
        pass

    @abstractmethod
    def expansion(self, X):
        """Returns the result of the equivalent expansion on the columns of X. If no expansion can be explicitly written, simply returns None

        param X: a matrix whose columns correspond to individual input variables
        returns: The matrix of expanded inputs or None if no expansion can be explicitly written
        """
        pass


class PolynomialKernel(Kernel):

    def __init__(self, degree, c=1):
        """Initializes the polynomial kernel with a particular degree and possibly a parameter to shift the weight of
        smaller and larger terms

        :param degree: degree of the polynomial kernel
        :param c: shift between 0 and +inf, where 1 indicates no bias to small or large terms
        """
        self.degree = degree
        self.c = c

    def getDimension(self, input_shape):
        """Returns the dimension of the implicit space (useful for determining how many dimensions to keep after
        whitening in KSFA)

        :param n: dimension of the input data
        :returns: an integer for the number of dimensions in the expansion of a n-dim vector
        """
        return self._getDimension(self.degree, input_shape)

    @staticmethod
    def _getDimension(degree, input_shape):
        """Returns the dimension of the implicit space (useful for determining how many dimensions to keep after
        whitening in KSFA)

        :param degree: degree of the polynomial kernel
        :param n: dimension of the input data
        :returns: an integer for the number of dimensions in the expansion of a n-dim vector
        """
        n = input_shape[0]
        num_of_degree = np.arange(degree)
        num_of_degree = binom(n + num_of_degree, n - 1)
        return int(num_of_degree.sum())

    def getMatrix(self, X, Y, only_diag=False):
        """Returns the Kernel matrix of X and Y (assumes that columns of X and Y are individual datapoints)

        :param X: the left matrix
        :param Y: the right matrix
        :returns: kernel matrix of X and Y
        """
        return self._getMatrix(self.degree, self.c, X, Y, only_diag)

    @staticmethod
    def _getMatrix(degree, c, X, Y, only_diag=False):
        """Returns the Kernel matrix of X and Y (assumes that columns of X and Y are individual datapoints)

        :param degree: degree of the polynomial kernel
        :param c: shift between 0 and +inf, where 1 indicates no bias to small or large terms
        :param X: the left matrix
        :param Y: the right matrix
        :returns: kernel matrix of X and Y
        """

        if only_diag:
            n = min(X.shape[1], Y.shape[1])
            dot_result = np.einsum('ij,ij->j', X[:, :n].conj(), Y[:, :n])
        else:
            dot_result = X.conj().T @ Y
        return (dot_result + c) ** degree

    def getGradient(self, Y, X, only_diag=False):
        """Returns the gradient vectors of the kernel function evaluated at each pair of columns of X and Y

        :param Y: a matrix whose columns are inputs to the left input of the gradient of the kernel function
        :param X: a matrix whose columns are inputs to the right input of the gradient of the kernel function
        :param only_diag: if set to true, then only the diagonal of the resulting 3-Tensor is computed
        :returns: a 3-Tensor of shape (n, X.shape[1], Y.shape[1]), where n is the dimension of the input vectors
            (i.e. X.shape[0]). Each of the elements along the first axis is the gradient evaluated at the pair
            (xi, yj), Vr(k)(xi, yj). If :only_diag: is set to True, then a 3-Tensor of shape
            (n, 1, min(X.shape[1], Y.shape[1])) is returned instead and each element along the first is the gradient
            evaluated at the pair (xi, yi)
        """
        return self._getGradient(self.degree, self.c, Y, X, only_diag)

    @staticmethod
    def _getGradient(degree, c, Y, X, only_diag=False):
        """Returns the gradient vectors of the kernel function evaluated at each pair of columns of X and Y

        :param degree: degree of the polynomial kernel
        :param c: shift between 0 and +inf, where 1 indicates no bias to small or large terms
        :param Y: a matrix whose columns are inputs to the left input of the gradient of the kernel function
        :param X: a matrix whose columns are inputs to the right input of the gradient of the kernel function
        :param only_diag: if set to true, then only the diagonal of the resulting 3-Tensor is computed
        :returns: a 3-Tensor of shape (n, X.shape[1], Y.shape[1]), where n is the dimension of the input vectors
            (i.e. X.shape[0]). Each of the elements along the first axis is the gradient evaluated at the pair
            (xi, yj), Vr(k)(xi, yj). If :only_diag: is set to True, then a 3-Tensor of shape
            (n, 1, min(X.shape[1], Y.shape[1])) is returned instead and each element along the first is the gradient
            evaluated at the pair (xi, yi)
        """

        if only_diag:
            n = min(X.shape[1], Y.shape[1])
            dot_result = np.einsum(
                'ij,ij->j', Y[:, :n].conj(), X[:, :n])  # (n, )
        else:
            n = X.shape[1]
            dot_result = Y.conj().T @ X  # (n, p)

        dot_result += c
        dot_result **= degree - 1
        dot_result *= degree

        # np.einsum('ij,ki->ijk', dot_result, X[:, :n])
        # (m, p, n) or (m, n)
        return np.einsum('ij,...j->i...j', X[:, :n], dot_result)

    def expansion(self, X):
        """Returns the result of the equivalent expansion on the columns of X. If no expansion can be explicitly written, simply returns None

        param X: a matrix whose columns correspond to individual input variables
        returns: The matrix of expanded inputs or None if no expansion can be explicitly written
        """
        return self._expansion(self.degree, self.c, X)

    @staticmethod
    def _expansion(degree, c, X):
        """Returns the result of the equivalent expansion on the columns of X. If no expansion can be explicitly written, simply returns None

        param X: a matrix whose columns correspond to individual input variables
        returns: The matrix of expanded inputs or None if no expansion can be explicitly written
        """
        return NotImplementedError


def linearExpansion(c, X):
    return np.r_[c * np.ones((1, X.shape[1])), X]


def quadraticExpansion(c, X):
    dtype = np.dtype('complex64') if np.any(
        np.iscomplex(X)) else np.dtype('float64')
    Y = np.zeros((PolynomialKernel._getDimension(
        2, X.shape) + 1, X.shape[1]), dtype=dtype)
    Y[0, :] = c
    Y[1:1+X.shape[0], :] = X.copy() * np.sqrt(2)
    base = X.shape[0]+1
    for i in range(X.shape[0]):
        inc = X.shape[0] - i
        Y[base:base+inc, :] = X[i:] * X[i]
        if inc > 0:
            Y[base+1:base+inc, :] *= np.sqrt(2)
        base += inc
    return Y


QuadraticKernel = PolynomialKernel(2)
QuadraticKernel._expansion = lambda degree, c, X: quadraticExpansion(c, X)
LinearKernel = PolynomialKernel(1)
LinearKernel._expansion = lambda degree, c, X: linearExpansion(c, X)
