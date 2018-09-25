"""
TODO finish the documentation once you are completely sure how it all works

REMEMBER TO MENTION COLUMN MAJOR!

@author G. Eli Jergensen <gelijergensen@ou.edu> - constructed following the
    paper by Stephen Liwicki et. al
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.special import binom

# class Kernel(ABC):

#     @abstractmethod
#     def getMatrix(self, X, Y):
#         """Returns the result of computing the kernel function, k
#         over X and Y, i.e. k(X,Y). This is equivalent to
#         X^t * K * Y, where (.)^t denotes transpose, (.) * (.) denotes
#         matrix multiplication, and K is the matrix version of the
#         kernel

#         :param X: a numpy array for the left parameter (shape (N, M))
#         :param Y: a numpy array for the right parameter (shape (N, P))
#         :returns: a numpy array (shape (M, P))
#         """
#         pass

#     @abstractmethod
#     def getPreimage(self, alphas, data, dif):
#         """TODO

#         :param alphas: a numpy array
#         :param data: a numpy array
#         :param dif: a scalar
#         :returns: a numpy array
#         """
#         pass

#     @abstractmethod
#     def tolerance(self, dif, dim):
#         """Returns whether the difference value is suitably
#         small given the dimension

#         :param dif: a scalar for the difference value
#         :param dim: an integer for the dimension of the (input) data
#         :returns: a numpy array with truth values for each element
#         """
#         pass


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


# class LinearKernel(Kernel):
#     """An instance of the Kernel metaclass which handles the kernel
#     for data with a "none expansion". Namely, the matrix for this
#     kernel is the identity matrix
#     """

#     def getMatrix(self, X, Y):
#         """For the linear kernel, K is the identity matrix"""
#         return X.conj().T @ Y

#     def getPreimage(self, alphas, data, dif=None):
#         """

#         :param dif: ignored
#         """
#         return data @ alphas

#     def tolerance(self, dif, dim):
#         """Compare to a tolerance of 1e-7. No expansion occurs,
#         so the dimension is the same
#         """
#         return (dif / dim) < 1e-7


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

        # if self.degree == 1:
        #     print("dot_result", dot_result)
        #     print("X[:, :n]", X[:, :n])
        #     print(np.einsum('ij,...j->i...j', X[:, :n], dot_result))

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


<<<<<<< HEAD
def quadraticExpansion(c, X):
    Y = np.zeros((PolynomialKernel._getDimension(2, X.shape) + 1, X.shape[1]))
    Y[0, :] = c
    Y[1:1+X.shape[0], :] = X.copy()
    base = X.shape[0]+1
    for i in range(X.shape[0]):
=======
def quadraticExpansion(X):
    Y = np.zeros((PolynomialKernel._getDimension(2, X.shape), X.shape[1]))
    Y[:X.shape[0], :] = X.copy()
    base = X.shape[0]
    for i in range(X.shape[0]):
        print(Y)
>>>>>>> dfff9f094225b06b8026889973563260184db691
        inc = X.shape[0] - i
        Y[base:base+inc, :] = X[i:] * X[i]
        base += inc
    return Y


QuadraticKernel = PolynomialKernel(2)
<<<<<<< HEAD
QuadraticKernel._expansion = lambda degree, c, X: quadraticExpansion(c, X)
=======
QuadraticKernel._expansion = lambda degree, c, X: quadraticExpansion(X)
>>>>>>> dfff9f094225b06b8026889973563260184db691
LinearKernel = PolynomialKernel(1)
LinearKernel._expansion = lambda degree, c, X: linearExpansion(c, X)

# class QuadraticKernel(Kernel):

#     def _derivative(self, y, alphas, data):
#         """Helper method which .......

#         :param y: a numpy array representing a row vector
#             the point at which the derivative is taken
#         :param alphas: a numpy array
#         :param data: a numpy array
#         :returns: a numpy array
#         """
#         # print("Start...")
#         # determine the dimensions of the data
#         d, n = data.shape

#         # calculate parts of derivative
#         # print("A")
#         Gy = self.getMatrix(y, y)
#         # print("B")
#         # print(alphas.conj().T)
#         # print(self.getMatrix(data, y))
#         # print("C")
#         Fy = alphas.conj().T @ self.getMatrix(data, y)
#         # print("Fy", Fy)

#         # calculate derivative of Fy
#         # print("Do we get here?")
#         rep_y = np.matlib.repmat(y, 1, n)
#         rep_dat = np.matlib.repmat(y.conj().T  @ data, d, 1)
#         dFy = (data * (1 + rep_dat + rep_y * data)) @ alphas  # optimized a bit
#         # print("But I bet we don't get here")
#         # calculate derivative of Gy
#         T = y.conj().T @ y + 1
#         dGy = 2*(y*y + np.matlib.repmat(T, d, 1)) * y  # optimized a tiny bit

#         # calculate final derivative
#         # print(Fy)
#         N = Fy @ (2 * Gy @ dFy - Fy @ dGy)  # numerator
#         D = np.linalg.matrix_power(Gy, 2)  # denominator

#         # B / A = (A' \ B')'
#         # C \ D = np.linalg.lstsq(C, D)[0]
#         # or, if they are square, then we can do
#         # C \ D = np.linalg.solve(C, D)
#         # "Divide N by D"
#         return np.linalg.solve(D.conj().T, N.conj().T).conj().T
#         # if N.shape[0] == N.shape[1] and D.shape[0] == D.shape[1]:
#         #     print("Yeah, that's square")
#         #     return np.linalg.solve(D.conj().T, N.conj().T).conj().T
#         # else:
#         #     print("Nope, that's not square")
#         #     return np.linalg.lstsq(D.conj().T, N.conj().T)[0].conj().T

#     def expansion(self, data):
#         """Assumes that the data has rows which are variables and columns which
#         are independent occurences and computes the expansion which we expect
#         """

#         n = len(data)
#         expanded = np.r_[data, data ** 2]
#         for i in range(n):
#             for j in range(i):
#                 expanded = np.vstack((expanded, data[i, :] * data[j, :]))
#         return expanded

#     def getMatrix(self, X, Y):

#         c = 1
#         # c = 0.9
#         a = 1e1
#         p = 3
#         return (a * (X.conj().T @ Y) + c) ** p

#     def getMatrixOLD(self, X, Y):

#         # get the shapes
#         m1, n1 = X.shape
#         m2, n2 = Y.shape

#         # for(j=0;j<n1;j++)
#         # {
#         #     for(i=0;i<n2;i++)
#         #     {
#         #         accum = 0;
#         #         for(n=0;n<m1;n++)
#         #         {
#         #             accum += *(A1+j*m1+n) *  *(A2+i*m2+n);
#         #             for(t=n; t<m1;t++){

#         #                 accum += *(A1+j*m1+t) * *(A1+j*m1+n) * *(A2+i*m2+t) * *(A2+i*m2+n);

#         #             }
#         #         }
#         #         K[ j + i*n1 ] = (accum );
#         #     }
#         # }

#         # TODO I am not certain that the above math is truly correct for the quadratic kernel...
#         # TODO I believe it should be L = np.linalg.matrix_power(X.conj().T @ Y, 2)
#         # TODO look at the difference between K and K2 (and acc and acc2) K2 = L, but K != K2
#         K_root = X.conj().T @ Y
#         print( K_root ** 2 )
#         L = K_root ** 2  # element-wise squaring
#         M = K_root.conj() * K_root
#         print(M)
#         N = (K_root + 1) ** 2
#         O = (K_root + np.sqrt(2)) ** 2
#         P = (K_root + np.sqrt(2) / 2) ** 2

#         K2 = np.zeros((n1, n2))
#         for j in range(n1):
#             for i in range(n2):
#                 acc = 0
#                 acc2 = 0
#                 for n in range(m1):
#                     acc += X[n,j] * Y[n,i]
#                     acc2 += 0
#                     for t in range(n, m1):
#                         acc += X[t,j] * X[n,j] * Y[t,i] * Y[n,i]
#                         acc2 += X[t,j] * X[n,j] * Y[t,i] * Y[n,i]

#                 K2[j,i] = acc2

#         # Okay, so the original code ignores the complex part of a matrix
#         X = X.real
#         Y = Y.real

#         K = np.zeros((n1, n2))  # this implicitly forces a real-valued matrix, which we want
#         K2 = np.zeros((n1, n2))
#         for j in range(n1):
#             for i in range(n2):
#                 acc = 0
#                 acc2 = 0
#                 for n in range(m1):
#                     acc += X[n,j] * Y[n,i]
#                     acc2 += 0
#                     for t in range(n, m1):
#                         acc += X[t,j] * X[n,j] * Y[t,i] * Y[n,i]
#                         acc2 += X[t,j] * X[n,j] * Y[t,i] * Y[n,i]

#                 K[j,i] = acc
#                 K2[j,i] = acc2


#         # A faster way? Probably not
#         # K[i,j] = X[j,n]*Y[i,n]*(1+(X[j,n:m1]*Y[i,n:m1]).sum()) for n in range(m1)

#         print("Here, compare:")
#         print(K2 - L)
#         print(K - L)
#         print(K2 - M)
#         print(K - M)
#         print(K2 - N)
#         print(K - N)
#         print(K2 - O)
#         print(K - O)
#         print(K2 - P)
#         print(K - P)
#         print()
#         return K

#     def getPreimage(self, alphas, data, dif):

#         # determine the dimension after the expansion
#         dim = data.shape[0]
#         d = dim + dim*(dim+1)/2

#         print("start")
#         rate = 0.0001 / ((dif/float(d)) ** 2)
#         maxIt = 10
#         print("mid")
#         y = np.random.rand(dim, 1) * np.max(data, axis=1)

#         for _ in range(maxIt):
#             y += rate * self._derivative(y, alphas, data)
#             print(".", end='', flush=True)

#         print("\nend")
#         return y


#     def tolerance(self, dif, dim):
#         """Compare to a tolerance of 1e-5. Expansion occurs, so the
#         dimension is increased by dim*(dim+1)/2
#         """
#         d = dim + dim*(dim+1)/2
#         return (dif / d) < 1e-5


# from oct2py import octave
# from oct2py.utils import Oct2PyError
# import os
# def initializeOctaveIKSFA(basedir):
#     """Initializes and returns an octave object which has all the IKSFA functions

#     :param basedir: string for the path where all the Matlab files sit (above all the @-directories)
#     :returns: an octave object which hopefully has all of the necessary functions for running an (D-)(I)KSFA
#     """
#     octave.restart()
#     for direc in [x[0] for x in os.walk(basedir)]:
#         print(direc)
#         octave.addpath(direc)
#     return octave
# OCTDIR = "/home/eli/Programming/Projects/episodic-driven-semantic-learning/source/core/IKSFA/Matlab_v2/"
# octave = initializeOctaveIKSFA(OCTDIR)
# class OLDQuadraticKernel(Kernel):

#     def getMatrix(self, X, Y):
#         return octave.QuadraticKernel_getMatrix(X, Y)

#     def getPreimage(self, alphas, data, dif):
#         return octave.QuadraticKernel_getPreimage(alphas, data, dif)

#     def tolerance(self, dif, dim):
#         return octave.QuadraticKernel_tolerance(dif, dim)

#     def dimension(self, n):
#         """Returns the dimension of the implicit space (useful for determining
#         how many dimensions to keep after whitening in KSFA)

#         :param n: dimension of the input data
#         :returns: an integer for the number of dimensions in the expansion of a
#             n-dim vector
#         """
#         return int((n+1)*(n+2)/2) - 1
