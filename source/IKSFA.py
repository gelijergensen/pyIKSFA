"""Godspeed Mathematician! 
//You are capable// --indeed 

Based on the paper and code by Stephan Liwicki and colleagues:
S. Liwicki, S. P. Zafeiriou and M. Pantic, "Online Kernel Slow Feature Analysis 
for Temporal Video Segmentation and Tracking," in IEEE Transactions on Image 
Processing, vol. 24, no. 10, pp. 2955-2970, Oct. 2015.
doi: 10.1109/TIP.2015.2428052
retrieved from
https://ieeexplore.ieee.org/document/7097728/
on 01 July 2018

Additionally based on the paper by Tat-Jun Chin and David Suter:
T.-J. Chin, D. Suter, "Incremental kernel principal component analysis", 
IEEE Trans. Image Process., vol. 16, no. 6, pp. 1662-1674, Jun. 2007
retrieved from
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4200753
on 01 August 2018
@author G. Eli Jergensen <gelijergensen@ou.edu>
"""

import numpy as np
from scipy.sparse import linalg as splinalg
from scipy import linalg
import time
import source.core.IKSFA.Python_version.Difference as Difference
import source.core.IKSFA.Python_version.Kernel as Kernel
import sys

class IKSFA(object):

    def __init__(self, kernel="linear", derivative="backwards", num_whitening_components=None, 
                 num_output_components=None, num_reduced_set_components=None, row_major = True, epsilon=1e-12,
                 RS_reconstruction_eps=1e-12, RS_gradient_ascent_learning_rate = 0.01, 
                 RS_gradient_ascent_max_iters = 10000, RS_gradient_ascent_threshold = 1e-10,
                 RS_gradient_ascent_parallelization = (64,16)):
        if kernel == "linear":
            self.kernel = Kernel.LinearKernel
        elif kernel == "quadratic":
            self.kernel = Kernel.QuadraticKernel
        elif isinstance(kernel, Kernel.Kernel):
            self.kernel = kernel
        else:
            print("WARNING! Invalid kernel specified!\nDefaulting to linear...")
            self.kernel = Kernel.LinearKernel
        
        if derivative == "backwards":
            self.derivative = Difference.BackwardDifference()
        elif derivative == "forwards":
            self.derivative = Difference.ForwardDifference()
        elif derivative == "central":
            self.derivative = Difference.CentralDifference()
        elif isinstance(derivative, Difference.Difference):
            self.derivative = derivative
        else:
            print("WARNING! Invalid derivative specified!\nDefaulting to the backwards difference...")
            self.derivative = Difference.BackwardDifference()
        
        # Determine how many components to keep after whitening
        if num_whitening_components is None:
            self.num_whitens = self.kernel.dimension
        elif callable(num_whitening_components):
            self.num_whitens = num_whitening_components
        elif int(num_whitening_components) == num_whitening_components:
            # They gave us a number, so make a lambda which just returns that
            self.num_whitens = lambda n: num_whitening_components
        else:
            print("WARNING! Invalid num_whitening_components specified!\n",
                  "Defaulting to the dimension of the input data (probably too small)...")
            self.num_whitens = lambda n: n[0]
        
        # Determine how many output components to return
        if num_output_components is None:
            # Default to the number of input components
            self.num_outputs = lambda n: n[0]
        elif callable(num_output_components):
            self.num_outputs = num_output_components
        elif int(num_output_components) == num_output_components:
            # They gave us a number, so make a lambda which just returns that
            self.num_outputs = lambda n: num_output_components
        else:
            print("WARNING! Invalid num_output_components specified!\n",
                  "Defaulting to the dimension of the input data (probably too large)...")
            self.num_outputs = lambda n: n[0]

        # Determine how many components to keep in the reduced set
        if num_reduced_set_components is None:
            # Default to returning none (which will break things!) TODO fix...
            self.num_reduced_set_components = lambda n: None
        elif callable(num_reduced_set_components):
            self.num_reduced_set_components = num_reduced_set_components
        elif int(num_reduced_set_components) == num_reduced_set_components:
            self.num_reduced_set_components = lambda n: num_reduced_set_components
        else:
            print("WARNING! Invalid num_reduced_set_components specified!\n",
                  "Defaulting to None, which will break things!")
            self.num_reduced_set_components = lambda n: None
        
        self.row_major = row_major
        # Thresholds
        self.eps = epsilon
        self.RS_reconstruction_eps = RS_reconstruction_eps
        self.RS_gradient_ascent_learning_rate = RS_gradient_ascent_learning_rate
        self.RS_gradient_ascent_max_iters = RS_gradient_ascent_max_iters
        self.RS_gradient_ascent_threshold = RS_gradient_ascent_threshold
        self.RS_gradient_ascent_parallelization = RS_gradient_ascent_parallelization

        # Set to ensure we do the right KPCA the first time
        self.ikpca_initialized = False
        self.ikpca_augmented = False

        self.iksfa_initialized = False

    def _get_centering_matrix(self, n):
        """Convenience method which returns the centering matrix of size n

        :param n: the size of the square matrix to return
        :returns: an (n by n) symmetric real matrix
        """
        return np.identity(n) - np.ones((n,n)) / n

    def _eigen_decomp(self, A, num=None, sparse=False, abs_val=True):
        """Computes the eigenvalue decomposition of a hermitian matrix, returning the top :num: non-zero components

        :param A: a hermitian matrix
        :param num: number of components to return. Defaults to all
        :param sparse: whether to use the sparse algorithm or not
        :param abs_val: whether to return the absolute value of the eigenvalues
        :returns: (a (N,) numpy array of the magnitudes of the eigenvalues, the corresponding (A.shape[0], N) array of 
            eigenvectors), where N = min(# non-zero components, num)
        """
        if sparse:
            assert num is not None, "Specify number of components when using sparse eigen_decomp"
            L, M = splinalg.eigsh(A, num, which="LM")
            idxs = np.argsort(np.abs(L))[::-1]
            L = L[idxs]  # these are copying...
            M = M[:, idxs] # these are copying...
        else:
            if num is None:
                num = A.shape[0]
            L, M = linalg.eigh(A)
            L = L[:-1-num:-1]
            M = M[:, :-1-num:-1]
        # Now remove all zero values
        last = np.searchsorted(np.abs(L[::-1]), self.eps, side='left')
        L_size = L.shape[0]
        L = L[:L_size-last]
        M = M[:, :L_size-last]
        if abs_val:
            np.abs(L, L)
        return L, M

    def BatchTrain(self, input_data, derivative_data=None):
        """Kernel SFA (KSFA) training, given possibly different input data and data to use for calculating the 
        derivative

        :param input_data: data to train over
        :param derivative_data: data to use for computing derivatives (defaults to the input_data)

        Inputs:
            Z in IC^{P' x N} --input data
            D in IR^{N x N} --derivative matrix
            r --# of components kept after whitening
            f --# of output dimensions
            k: IC^{P'} x IC^{P'} -> IC --kernel function

        Stores:
            Z: --input data
            V: --projection matrix
            signal_shift: --used for shifting data when transforming

        KSFA attempts to find the projection matrix, which is given by
        V = argmin{tr((`V`^* S `V`)^{-1} `V`^* S. `V`)}
        After some substitutions, including
        S = C X X^* C^*
        S. = C X. X.^* C^*
        X = phi(Z) <phi is the implicit expansion function hiding in the kernel>
        X. = phi(P) D (usually X D)
        C^* = C = C^2
        |L|^{-1}^{H} = |L|^{-1}
        and two assumptions, which can be written in one as:
        `V` = X C M |L|^{-1} `G`, where
            C^* X^* X C = M L M^H, where
                L is diagonal
                M^H M = Id
        our new optimization problem is to find
        V = X C M |L|^{-1} G, where
        G = argmin{tr((`G`^H `G`)^{-1} `G`^H `Q` C `Q`^H `G`}, where
        `Q` = |L|^{-1} M^H C^* X^* phi(P) D
        The solution to this optimization, G, is such that 
            G^H G = Id and 
            Q C Q^H = G E G^H (E is also diagonal)

        If we employ the kernel trick, i.e. phi(A)^* phi(B) = k(A, B), we can convert all this to two eigen. decomps.
        G E G^H = |L|^{-1} M^H C k(Z, P) D C D^H k(Z, P)^H C M |L|^{-1}, where
        M L M^H = C k(Z, Z) C

        For convenience, here is a list of variable names for the code below
        Z: --input data
        P: --input derivative data
        C: --centering matrix
        D: --derivative matrix
        K: --kernel of input: K = k(Z, Z)
        R: --kernel of input and derivative (times derivative): R = k(Z, P) D
        L: --eigenvalues of C K C
        M: --(left) eigenvectors of C K C
        W: --convenience for M |L|^{-1}
        Q: --convenience for |L|^{-1} M^H C k(Z, P) D
        E: --eigenvalues of Q C Q^H
        G: --(left) eigenvectors of Q C Q^H
        Vh: --projection matrix: (C M |L|^{-1} G)^H  = V^H
        signal_shift: --average{V (k(Z, Z) + k(Z, P) D)} along data axis
        """

        special_derivative = (derivative_data is not None)
        # the math is done for column major
        self.Z = input_data.T.copy() if self.row_major else input_data.copy()

        # Compute K = X^* X using the kernel
        K = self.kernel.getMatrix(self.Z, self.Z)

        # Compute the centering matrix for this size
        C = self._get_centering_matrix(K.shape[0])

        # Compute the backward difference matrix for this size
        D = self.derivative.getMatrix(K.shape[0])
        # Figure out the data to use for computing the derivative
        P = derivative_data.T if special_derivative else self.Z

        # Find the eigen. decomp. of C^* K C
        L, M = self._eigen_decomp(C @ K @ C, self.num_whitens(self.Z.shape[0]), sparse=True)
        # Compute W = M |L|^{-1}
        np.reciprocal(L, L)
        W = M * L  # broadcasting means we can skip making the diag. matrix
        
        # Compute R = kernel(Z, P) @ D
        R = (self.kernel.getMatrix(self.Z, P) if special_derivative else K) @ D

        # Compute Q = W^H C R
        Q = W.conj().T @ C @ R

        # Compute the eigen. decomp. of Q^H C Q
        E, G = self._eigen_decomp(Q @ C @ Q.conj().T, sparse=False)
        # Order from slowest to fastest
        E = E[::-1]
        G = G[:, ::-1]

        # Compute Vh = (C W G)^H and remove the fast components
        self.Vh = (C @ (W @ G))[:, :self.num_outputs(self.Z.shape[0])].conj().T

        # Lastly, we compute the signal shift for the output projection
        self.signal_shift = self.Vh @ (np.average(K + R, axis=1)[:, np.newaxis])

    def BatchTransform(self, input_data):
        """Kernel SFA (KSFA) projection of the input_data to the slowness space

        :param input_data: data to transform
        :returns: transformed data

        Given the definitions above in the training function, the projection of KSFA is given by
        O = (G^H |L|^{-1} M^H C^* X^*)(phi(Y) - avg{phi(Z) + phi(P) D}), where Y is the new data to transform
        Noticing that the avg{} piece does not depend on Y, we have
        O = G^H |L|^{-1} M^H C^* k(Z, Y) - signal_shift, where
            signal_shift = G^H |L|^{-1} M^H C^* avg{(k(Z, Z) + k(Z, P) D)}
        or, more simply
        O = Vh k(Z, Y) - signal_shift

        For convenience, here is a list of variable names for the code below
        Z: --training data
        Vh: --projection matrix
        signal_shift: --average{V (k(Z, Z) + k(Z, P) D)} along data axis
        Y: --data to transform
        """
        # Use V to compute the result after the kernel evaluation
        Y = input_data.T if self.row_major else input_data
        O = self.Vh @ self.kernel.getMatrix(self.Z, Y) - self.signal_shift
        return O.T if self.row_major else O

    def GreedyIKPCA(self, update_data):
        """

        """
        # If we haven't set up anything yet, then we need to do the initial KPCA
        if not self.ikpca_initialized:
            # make sure data is column major
            self.Z = update_data.T.copy() if self.row_major \
                     else update_data.copy()
            print("Z_0", self.Z)

            # Compute K_0 = X_0^* X_0 using the kernel
            K_0 = self.kernel.getMatrix(self.Z, self.Z)
            print("K_0", K_0)
            self.K_t_avg = np.average(K_0)  # store these for later
            self.K_t_col_avg = np.average(K_0, axis=1)
            print("K_t_avg", self.K_t_avg)
            print("K_t_col_avg", self.K_t_col_avg)

            # Get the right centering matrix for this size
            C = self._get_centering_matrix(K_0.shape[0])
            print("C_0", C)

            # Find the complete eigen. decomp. of C^* K_0 C
            self.L_t, self.M_t = self._eigen_decomp(C @ K_0 @ C, sparse=False)
            print("|L_0|", self.L_t)
            print("M_0", self.M_t)

            # Set a few other useful numbers
            self.q_last = 0
            self.n_t_last = 0

            self.ikpca_initialized = True
            self.ikpca_augmented = False
        else:
            print("Z_t", self.Z)
            # make sure input data is column major
            Z_d = update_data.T.copy() if self.row_major else update_data.copy()
            print("Z_d", Z_d)
            # Compute q, the weighted constant for the change of mean
            n_t = self.Z.shape[1]
            print("n_t", n_t)
            n_d = Z_d.shape[1]
            print("n_d", n_d)
            q = (n_d * n_t) / (n_d + n_t)
            print("q", q)

            # Compute K_d = X_d^* X_d using the kernel
            K_d = self.kernel.getMatrix(Z_d, Z_d)
            print("K_d", K_d)
            K_d_avg = np.average(K_d)
            print("K_d_avg", K_d_avg)
            K_d_col_avg = np.average(K_d, axis=1)
            print("K_d_col_avg", K_d_col_avg)

            # Also compute K_mix = X_t^* X_d using the kernel
            K_mix = self.kernel.getMatrix(self.Z, Z_d)
            print("K_mix", K_mix)
            K_mix_avg = np.average(np.real(K_mix))
            print("K_mix_avg", K_mix_avg)
            K_mix_col_avg = np.average(K_mix, axis=1)
            print("K_mix_col_avg", K_mix_col_avg)
            K_mix_row_avg = np.average(K_mix, axis=0)
            print("K_mix_row_avg", K_mix_row_avg)

            # Get the right centering matrix for this size
            C_d = self._get_centering_matrix(K_d.shape[0])
            print("C_d", C_d)

            # Find the complete eigen. decomp. of C_d^* K_d C_d
            L_d, M_d = self._eigen_decomp(C_d @ K_d @ C_d, sparse=False)
            # To account for the changing mean, we need to augment L_d and M_d
            L_d = np.append(L_d, 
                            np.abs(q*(self.K_t_avg + K_d_avg - 2 * K_mix_avg)))
            n, m = M_d.shape
            M_d = np.block([[M_d, np.zeros((n, 1))], [np.zeros((1, m)), 1]])
            print("L_d", L_d)
            print("M_d", M_d)

            """ Archived
            # Form K_aug = X_t^* X'_d
            mean_aug = self.K_t_col_avg - np.average(K_mix, axis=1)
            print((K_mix @ C_d).shape)
            K_aug = np.block([K_mix @ C_d, np.sqrt(q)*mean_aug[:, np.newaxis]])
            print("K_aug", K_aug)
            """


            # Form K_aug = X'_t^* X'_d
            C_t = self._get_centering_matrix(K_mix.shape[0])
            if self.ikpca_augmented:
                n, m = K_mix.shape
                K_aug = np.zeros((n+1, m+1))
                K_aug[:-1, :-1] = C_t @ K_mix @ C_d
                # This number is computed from the column avgs for efficiency
                p = np.average(self.K_t_col_avg[:self.n_t_last]) - \
                    np.average(self.K_t_col_avg[self.n_t_last:]) + \
                    np.average(K_mix_col_avg[:self.n_t_last]) - \
                    np.average(K_mix_col_avg[self.n_t_last:])
                K_aug[-1: -1] = np.sqrt(self.q_last) * np.sqrt(q) * p
            else:
                # We don't actually have to augment this
                K_aug = C_t @ K_mix @ C_d
            print("K_aug", K_aug)

            # Compute Q = M_t^H C_t^T K_aug M_d
            C_t = self._get_centering_matrix(K_aug.shape[0])
            print("C_t", C_t)
            print(self.M_t.conj().T.shape)
            print(C_t.shape)
            print(K_aug.shape)
            print(M_d.shape)
            Q = self.M_t.conj().T @ (C_t @ (K_aug @ M_d))
            print("Q", Q)
            
            # Form R = [|L_t|   Q  ]
            #          [ Q^H  |L_d|]
            print(np.diag(self.L_t).shape)
            print(Q.shape)
            print(Q.conj().T.shape)
            print(np.diag(L_d).shape)
            R = np.block([[np.diag(self.L_t), Q], [Q.conj().T, np.diag(L_d)]])
            print("R", R)

            # Take the complete eigen. decomp. of R
            self.L_t, M_R = linalg.eigh(R)
            n_a, m_a = self.M_t.shape
            n_b, m_b = M_d.shape
            M_t_helper = np.zeros((n_a + n_b, m_a + m_b))
            M_t_helper[:n_a, :m_a] = self.M_t
            M_t_helper[n_a:, m_a:] = M_d
            print("M_t_helper", M_t_helper)
            self.M_t = M_t_helper @ M_R
            print("L_t", self.L_t)
            print("M_t", self.M_t)

            # Sort the eigenvalues
            np.abs(self.L_t, self.L_t)  # abs val before sort
            sorted_idxs = np.argsort(self.L_t)[::-1]
            large_idxs = self.L_t > self.eps
            self.L_t = self.L_t[sorted_idxs][large_idxs]  # these are copying...
            self.M_t = self.M_t[:, sorted_idxs][:, large_idxs] # these too...
            print("L_t", self.L_t)
            print("M_t", self.M_t)

            # Updated the stored data
            self.Z = np.block([self.Z, Z_d])
            print("Z_t+1", self.Z)

            # Update the stored "q_last" value
            self.q_last = q
            self.n_t_last = n_t

            # Compute the new average over all K_{t+1}
            self.K_t_avg = (n_t ** 2 * self.K_t_avg + n_d ** 2 * K_d_avg + \
                            2 * n_d * n_t * K_mix_avg) / ((n_t + n_d) ** 2)
            print("K_t_avg", self.K_t_avg)

            # Compute the new average of the columns of K_{t+1}
            new_K_t_col_avg = np.zeros(n_t + n_d)
            new_K_t_col_avg[:n_t] = (n_t * self.K_t_col_avg + \
                                     n_d * K_mix_row_avg.conj()) / (n_t + n_d)
            new_K_t_col_avg[n_t:] = (n_d * K_d_col_avg + \
                                     n_t * K_mix_col_avg) / (n_t + n_d)
            self.K_t_col_avg = new_K_t_col_avg
            print("K_t_col_avg", self.K_t_col_avg)

            # Set this so that we know for next time
            self.ikpca_augmented = True

        # Return the projection matrix
        W = self.M_t * np.reciprocal(self.L_t)
        return W.T if self.row_major else W

    def ExactIKPCATrain(self, update_data):
        """Incremental Kernel PCA (IKPCA) training, given the data to train or update over

        :param update_data: data to add to the training set

        Inputs:
            Z_d in IC^{P' x n_d} --update data
        
        Stores:
            Z_{t+1} = [Z_t Z_d]: --total input data
            K_t: --kernel matrix of all data
            U_t: --projection matrix of SVD
            E_t: --eigenvalues of SVD
            Vh_t = V_t^*: --"anti-projection" matrix of SVD
            pca_signal_shift: --used for shifting data when transforming

        KPCA attempts to find the projection matrix, U_t, which is given by
        X_t C_t = X_t C_t U_t E_t V_t^*, where
            (X_t C_t U_t)^* X_t C_t U_t = Id
            E_t is diagonal
            V_t^* V_t = Id
            X_t = phi(Z_t) <phi is the implicit expansion function of the kernel>
        Generally, given the centered kernel matrix
        C_t^* K_t C_t = C_t^* X_t^* X_t C_t, we can take the eigen decomposition: C_t^* K_t C_t = M_t L_t M_t^*, where
            L_t is diagonal
            M_t^* M_t = Id
        Using this, we can produce the desired projection and scaling matrices:
        U_t = M_t L_t^{-1/2}
        E_t = K_t^{1/2}
        V_t^* = M_t^*

        IKPCA performs this in a different, but algebraically equivalent manner (after the initial step, as above):
        [X_t C_t X_d x_+] = [X_t X_d] U_{t+1} E_{t+1} V_{t+1}^*, where
            x_+ is a special point to handle changing means: 
                x_+ = sqrt(n_t * n_d / (n_t + n_d)) * (avg{X_t} - avg{X_d}) = sqrt(q) * (avg{X_t} - avg{X_d}) 
                (averages along data axis)
            U_{t+1} = ([U_t P] `U`)[:, :r]
                       [ 0   | ] (0 indicates 0-padding, | indicates matrix extends full height of block matrix)
            E_{t+1} = `E`[:r, :r]
            V_{t+1}^* = (`V`^* [V_t^* 0])[:r, :]
                               [ 0   Id] (Id here is of size n_d + 1), where
                `U` `E` `V`^* = SVD{[E_t Q]}
                                    [ 0  R]
                P = B M_H L_H^{-1/2}, where
                    Q = U_t^* X_t^* [X_t X_d] G
                    R = L_H^{1/2} M_H
                    B = [G[:n_t, :]-(U_t Q)]
                        [    G[n_t:, :]    ]
                    M_H L_H M_H^* = eigen. decomp.{B^* [X_t X_d]^* [X_t X_d] B}, where
                        G = [ 0   sqrt(c)/n_t*1_{n_t,1}]
                            [C_d -sqrt(c)/n_d*1_{n_d,1}] 1_{a,b} = (a,b) matrix of only ones
        Notice that this only requires a SVD (singular value decomposition) of a (<(r + n_d + 1), r + n_d + 1) matrix 
        and a eigen. decomp. of a (n_d + 1, n_d + 1) matrix. Additionally, we can use the kernel trick to evaluate 
        X_{t or d}^* X_{t or d}. Sadly, because this requires all of X_{t+1} at every step, the memory requirement will
        grow indefinitely, as will the time requirement (because of computing the kernel matrix)

        For convenience, here is a list of variable names for the code below
        Z_t: --previous input data
        Z_d: --update data
        Z_{t+1}: --total input data. Z_{t+1} = [Z_t Z_d]
        n_t: --number of datapoints in Z_t
        n_d: --number of datapoints in Z_d
        r: --dimensionality of output
        K_t: --kernel of total input. K_t=k(Z_{t+1}, Z_{t+1}) --slight misnomer, I know
        C_t: --centering matrix of size n_t
        C_d: --centering matrix of size n_d
        G: --helper matrix which is computing (among other things) the extra datapoint, x_+ (or the mean correction)
        B: --helper matrix which is "centering" the new information in the mean correction and new datapoints
        P: --helper matrix which is the orthogonal basis for the new information in the mean correction and new data
        Q: --submatrix which handles the (old) projection of the mean correction and new datapoints
        R: --submatrix which is the projection of orthogonal space of the mean correction and new datapoints
        S: --block matrix for the updating of the SVD to include the new information. S = [[E_t, Q] [0, R]]
        U: --projection matrix of the SVD update
        E: --scaling matrix of the SVD update
        Vh: --"antiprojection" matrix of the SVD update Vh = V^*
        self.U: --entire projection matrix after SVD update. self.U = U_{t+1} --sorry for the namespace collision
        self.E: --entire diagonal scaling matrix after SVD update. self.E = E_{t+1}
        self.Vh: --entire "antiprojection" matrix after SVD update. self.Vh = V_{t+1}^*
        pca_signal_shift: --average{U_t^* k(Z_t, Z_t)} along data axis
        """

        if not self.ikpca_initialized:
            # Store the initial input data
            self.Z = update_data.T.copy() if self.row_major else update_data.copy()

            # Compute and store the kernel matrix
            self.K_t = self.kernel.getMatrix(self.Z, self.Z)

            # Get the corresponding centering matrix
            C = self._get_centering_matrix(self.K_t.shape[0])

            # Determine the SVD of the centered input, C^* X^* X C
            L, M = self._eigen_decomp(C @ self.K_t @ C, num=self.num_whitens(self.Z.shape))
            np.sqrt(L, L)  # This will not elevate it to a complex matrix (since we took the abs. val before)
            self.U = C @ M * np.reciprocal(L)
            self.E = L
            self.Vh = M.conj().T

            self.ikpca_initialized = True
        else:
            # Store the new data as well as the old and determine how many datapoints there are
            n_t = self.Z.shape[1]
            self.Z = np.block([self.Z, update_data.T.copy() if self.row_major else update_data.copy()])
            n_d = self.Z.shape[1] - n_t

            # Compute and store the bigger kernel matrix
            K_new = self.kernel.getMatrix(self.Z, self.Z[:, n_t:])
            K_t_new = np.zeros((n_t + n_d, n_t + n_d), dtype=K_new.dtype) # this could be complex
            K_t_new[:n_t, :n_t] = self.K_t
            K_t_new[:, n_t:] = K_new
            K_t_new[n_t:, :n_t] = K_t_new[:n_t, n_t:].conj().T
            self.K_t = K_t_new

            # Form the helper matrix G
            G = np.zeros((n_t + n_d, n_d + 1))  # definitely real, so don't need to worry
            G[n_t:, :n_d] = self._get_centering_matrix(n_d)
            G[:n_t, -1] = 1.0 / n_t
            G[n_t:, -1] = -1.0 / n_d
            G[:, -1] *= np.sqrt(n_t * n_d / (n_t + n_d))

            # Compute the submatrix Q
            Q = self.U.conj().T @ self.K_t[:n_t, :] @ G

            # Compute the helper matrix B
            B = G.astype(dtype=self.U.dtype)
            B[:n_t, :] -=  self.U @ Q

            # Take the eigen. decomp. of K_H = B^* K_t B
            L_H, M_H = self._eigen_decomp(B.conj().T @ self.K_t @ B)
            np.sqrt(L_H, L_H)  # This will not elevate it to a complex matrix (since we took the abs. val before)

            # Compute the submatrix R
            R = (M_H * L_H).conj().T

            # Compute the helper matrix P
            P = B @ M_H * L_H

            # Form the block matrix S
            S = np.block([[np.diag(self.E), Q], [np.zeros((R.shape[0], self.E.shape[0])), R]])

            # Take the SVD of S
            U, E, Vh = np.linalg.svd(S)

            # Produce our final SVD
            r = self.num_whitens(self.Z.shape)
            U_new = P @ U[r:, :r]
            U_new[:n_t, :] += self.U @ U[:r, :r]
            self.U = U_new
            self.E = E[:r]
            V_new_dtype = Vh.dtype if np.iscomplexobj(Vh) else self.Vh.dtype
            V_new = np.zeros((r, self.Vh.shape[1] + n_d + 1), dtype=V_new_dtype)
            V_new[:, self.Vh.shape[1]:] = Vh[:r, r:]
            V_new[:, :self.Vh.shape[1]] = Vh[:r, :r] @ self.Vh
            self.Vh = V_new

            # print("U", self.U[:5, :])
            # print("E", self.E)
            # print("Vh", self.Vh)
            # print("K_t", self.K_t)
            # print("Z", self.Z)
            # print("mu", n_t / (n_t + n_d), n_d / (n_t + n_d))
        
        # For convenience, also store the PCA signal shift (so we don't need to compute it each time we transform)
        self.pca_signal_shift = self.U.conj().T @ (np.average(self.K_t, axis=1)[:, np.newaxis])

    def ExactIKPCATransform(self, input_data):
        """Incremental Kernel PCA (IKPCA) projection of the input_data to the lower-dim space

        :param input_data: data to transform
        :returns: transformed data

        Given the definitions above in the training function, the projection of IKPCA is given by
        O = U_t^* X_t^* (phi(Y) - avg{phi(Z)}), where Y is the new data to transform
        With some simplification, this becomes
        O = U_t^* k(Z, Y) - U_t^* avg{k(Z, Z)}
        Since the second term does not depend on Y, we precalculate it
        pca_signal_shift = U_t^* avg{k(Z, Z)}
        and have
        O = U_t^* k(Z, Y) - pca_signal_shift

        For convenience, here is a list of variable names for the code below
        Z: --traning data
        U: --projection matrix
        pca_signal_shift: --average{U_t^* k(Z, Z)} along data axis
        Y: --data to transform
        """
        Y = input_data.T if self.row_major else input_data
        O = self.U.conj().T @ self.kernel.getMatrix(self.Z, Y) - self.pca_signal_shift
        return O.T if self.row_major else O

    def IKPCARSTrain(self, update_data):
        """Incremental Kernel PCA with Reduced-Set expansion (IKPCARS) training, given the data to train or update over

        :param update_data: data to add to the training set

        Inputs:
            Z_d in IC^{P' x n_d} --update data
        
        Stores:
            Z_{t+1} = [Z_t Z_d]: --total input data
            K_t: --kernel matrix of all data
            U_t: --projection matrix of SVD
            E_t: --eigenvalues of SVD
            Vh_t = V_t^*: --"anti-projection" matrix of SVD
            pca_signal_shift: --used for shifting data when transforming

        KPCA attempts to find the projection matrix, U_t, which is given by
        X_t C_t = X_t C_t U_t E_t V_t^*, where
            (X_t C_t U_t)^* X_t C_t U_t = Id
            E_t is diagonal
            V_t^* V_t = Id
            X_t = phi(Z_t) <phi is the implicit expansion function of the kernel>
        Generally, given the centered kernel matrix
        C_t^* K_t C_t = C_t^* X_t^* X_t C_t, we can take the eigen decomposition: C_t^* K_t C_t = M_t L_t M_t^*, where
            L_t is diagonal
            M_t^* M_t = Id
        Using this, we can produce the desired projection and scaling matrices:
        U_t = M_t L_t^{-1/2}
        E_t = K_t^{1/2}
        V_t^* = M_t^*

        IKPCA performs this in a different, but algebraically equivalent manner (after the initial step, as above):
        [X_t C_t X_d x_+] = [X_t X_d] U_{t+1} E_{t+1} V_{t+1}^*, where
            x_+ is a special point to handle changing means: 
                x_+ = sqrt(n_t * n_d / (n_t + n_d)) * (avg{X_t} - avg{X_d}) = sqrt(q) * (avg{X_t} - avg{X_d}) 
                (averages along data axis)
            U_{t+1} = ([U_t P] `U`)[:, :r]
                       [ 0   | ] (0 indicates 0-padding, | indicates matrix extends full height of block matrix)
            E_{t+1} = `E`[:r, :r]
            V_{t+1}^* = (`V`^* [V_t^* 0])[:r, :]
                               [ 0   Id] (Id here is of size n_d + 1), where
                `U` `E` `V`^* = SVD{[E_t Q]}
                                    [ 0  R]
                P = B M_H L_H^{-1/2}, where
                    Q = U_t^* X_t^* [X_t X_d] G
                    R = L_H^{1/2} M_H
                    B = [G[:n_t, :]-(U_t Q)]
                        [    G[n_t:, :]    ]
                    M_H L_H M_H^* = eigen. decomp.{B^* [X_t X_d]^* [X_t X_d] B}, where
                        G = [ 0   sqrt(c)/n_t*1_{n_t,1}]
                            [C_d -sqrt(c)/n_d*1_{n_d,1}] 1_{a,b} = (a,b) matrix of only ones
        Notice that this only requires a SVD (singular value decomposition) of a (<(r + n_d + 1), r + n_d + 1) matrix 
        and a eigen. decomp. of a (n_d + 1, n_d + 1) matrix. Additionally, we can use the kernel trick to evaluate 
        X_{t or d}^* X_{t or d}. Sadly, because this requires all of X_{t+1} at every step, the memory requirement will
        grow indefinitely, as will the time requirement (because of computing the kernel matrix)

        For convenience, here is a list of variable names for the code below
        Z_t: --previous input data
        Z_d: --update data
        Z_{t+1}: --total input data. Z_{t+1} = [Z_t Z_d]
        n_t: --number of datapoints in Z_t
        n_d: --number of datapoints in Z_d
        r: --dimensionality of output
        K_t: --kernel of total input. K_t=k(Z_{t+1}, Z_{t+1}) --slight misnomer, I know
        C_t: --centering matrix of size n_t
        C_d: --centering matrix of size n_d
        G: --helper matrix which is computing (among other things) the extra datapoint, x_+ (or the mean correction)
        B: --helper matrix which is "centering" the new information in the mean correction and new datapoints
        P: --helper matrix which is the orthogonal basis for the new information in the mean correction and new data
        Q: --submatrix which handles the (old) projection of the mean correction and new datapoints
        R: --submatrix which is the projection of orthogonal space of the mean correction and new datapoints
        S: --block matrix for the updating of the SVD to include the new information. S = [[E_t, Q] [0, R]]
        U: --projection matrix of the SVD update
        E: --scaling matrix of the SVD update
        Vh: --"antiprojection" matrix of the SVD update Vh = V^*
        self.U: --entire projection matrix after SVD update. self.U = U_{t+1} --sorry for the namespace collision
        self.E: --entire diagonal scaling matrix after SVD update. self.E = E_{t+1}
        self.Vh: --entire "antiprojection" matrix after SVD update. self.Vh = V_{t+1}^*
        pca_signal_shift: --average{U_t^* k(Z_t, Z_t)} along data axis
        """

        if not self.ikpca_initialized:
            # Store the initial input data
            self.Z = update_data.T.copy() if self.row_major else update_data.copy()
            self.n_total = self.Z.shape[1]
            self.mu_coef = np.ones((self.Z.shape[1],)) / self.Z.shape[1]  # After RS-expans. this will change

            # Compute and store the kernel matrix
            self.K_t = self.kernel.getMatrix(self.Z, self.Z)

            # Get the corresponding centering matrix
            C = self._get_centering_matrix(self.K_t.shape[0])

            # Determine the SVD of the centered input, C^* X^* X C
            L, M = self._eigen_decomp(C @ self.K_t @ C, num=self.num_whitens(self.Z.shape))
            np.sqrt(L, L)  # This will not elevate it to a complex matrix (since we took the abs. val before)
            self.U = C @ M * np.reciprocal(L)
            self.E = L
            self.Vh = M.conj().T

            self.ikpca_initialized = True


            self.U_old = self.U.copy()
            self.Z_old = self.Z.copy()
            self.K_t_old = self.K_t.copy()
            self.mu_coef_old = self.mu_coef.copy()

        else:
            # Store the new data as well as the old and determine how many datapoints there are
            n_t = self.Z.shape[1]
            self.Z = np.block([self.Z, update_data.T.copy() if self.row_major else update_data.copy()])
            n_d = self.Z.shape[1] - n_t
            self.n_total += n_d

            # Compute and store the bigger kernel matrix
            K_new = self.kernel.getMatrix(self.Z, self.Z[:, n_t:])
            K_t_new = np.zeros((n_t + n_d, n_t + n_d), dtype=K_new.dtype) # this could be complex
            K_t_new[:n_t, :n_t] = self.K_t
            K_t_new[:, n_t:] = K_new
            K_t_new[n_t:, :n_t] = K_t_new[:n_t, n_t:].conj().T
            self.K_t = K_t_new

            # Form the helper matrix G
            G = np.zeros((n_t + n_d, n_d + 1))  # definitely real, so don't need to worry
            G[n_t:, :n_d] = self._get_centering_matrix(n_d)
            # The last column of G is how to compute the "mean" from the rest of the data
            G[:n_t, -1] = self.mu_coef
            G[n_t:, -1] = -1.0 / n_d
            G[:, -1] *= np.sqrt((self.n_total - n_d) * n_d / self.n_total)
            # G[:, -1] *= np.sqrt(n_t * n_d / (n_t + n_d))

            
            # Update how to calculate the mean so that we can use it for RS-expansion
            self.mu_coef *= (self.n_total - n_d) / (self.n_total)
            new_mu_coef = np.ones((n_t + n_d,))
            new_mu_coef[:n_t] = self.mu_coef
            new_mu_coef[n_t:] /= self.n_total
            self.mu_coef = new_mu_coef

            # Compute the submatrix Q
            Q = self.U.conj().T @ self.K_t[:n_t, :] @ G

            # Compute the helper matrix B
            B = G.astype(dtype=self.U.dtype)
            B[:n_t, :] -=  self.U @ Q

            # Take the eigen. decomp. of K_H = B^* K_t B
            L_H, M_H = self._eigen_decomp(B.conj().T @ self.K_t @ B)
            np.sqrt(L_H, L_H)  # This will not elevate it to a complex matrix (since we took the abs. val before)

            # Compute the submatrix R
            R = (M_H * L_H).conj().T

            # Compute the helper matrix P
            P = B @ M_H * L_H

            # Form the block matrix S
            S = np.block([[np.diag(self.E), Q], [np.zeros((R.shape[0], self.E.shape[0])), R]])

            # Take the SVD of S
            U, E, Vh = np.linalg.svd(S)

            # Produce our final SVD
            r = self.num_whitens(self.Z.shape)
            p = self.num_reduced_set_components(self.Z.shape)
            U_new = P @ U[r:, :r]
            U_new[:n_t, :] += self.U @ U[:r, :r]
            self.U = U_new
            self.E = E[:r]
            V_new_dtype = Vh.dtype if np.iscomplexobj(Vh) else self.Vh.dtype
            V_new = np.zeros((r, self.Vh.shape[1] + n_d + 1), dtype=V_new_dtype)
            V_new[:, self.Vh.shape[1]:] = Vh[:r, r:]
            V_new[:, :self.Vh.shape[1]] = Vh[:r, :r] @ self.Vh
            self.Vh = V_new

            self.U_old = self.U.copy()
            self.Z_old = self.Z.copy()
            self.K_t_old = self.K_t.copy()
            self.mu_coef_old = self.mu_coef.copy()
            # print("U", self.U[:5, :])
            # print("E", self.E)
            # print("Vh", self.Vh)
            # print("K_t", self.K_t)
            # print("Z", self.Z)
            # print("mu", self.mu_coef)

            # If we have too many data values, then we need to perform RS
            if self.Z.shape[1] > p * (r + 1):
                self._performRS(r, p)

            # TODO figure out if / why / how Vh won't explode. I can recover it!
        
        # For convenience, also store the PCA signal shift (so we don't need to compute it each time we transform)

        # print("Z.shape", self.Z.shape)
        # print("U.shape", self.U.shape)
        # print("E.shape", self.E.shape)
        print("Vh.shape", self.Vh.shape)
        # print("mu.shape", self.mu_coef.shape)
        # print("Z_old.shape", self.Z_old.shape)
        # print("U_old.shape", self.U_old.shape)

        self.pca_signal_shift = self.U.conj().T @ self.K_t @ self.mu_coef
        self.pca_signal_shift_old = self.U_old.conj().T @ self.K_t_old @ self.mu_coef_old

    def _performRS(self, num_whitens, num_reduced_set_components):
        """
        """
        r = num_whitens
        p = num_reduced_set_components
        # Assume that we have somehow been given Z_{RS} and S
        # U ~=~ phi(Z_{RS}) S

        Z_RS, S, mu_coef, K_Y, K_YZ = self._findPreimages(r, p)

        TEST = np.random.rand(self.Z.shape[0], 10)
        A = self.kernel.getMatrix(Z_RS, TEST)
        B = self.kernel.getMatrix(self.Z, TEST)
        C = self.kernel.getMatrix(TEST, self.Z)
        D = self.kernel.getMatrix(TEST, Z_RS)
        U_1 = S.conj().T @ A
        U_2 = self.U.conj().T @ B
        V_1 = mu_coef.conj().T @ A
        V_2 = self.mu_coef.conj().T @ B
        if np.allclose(U_1, U_2) and np.allclose(V_1, V_2):
            print("\033[32mReconstruction Successful!\033[39m")
        else:
            print("\033[31mReconstruction Failed!\033[39m")
            print("U_1", U_1)
            print("U_2", U_2)
            print("max_diff", np.max(np.abs(U_1 - U_2)))
            print("V_1", V_1)
            print("V_2", V_2)
            print("max_diff", np.max(np.abs(V_1 - V_2)))
        

        A = S.conj().T @ K_Y @ S
        B = self.U.conj().T @ self.K_t @ self.U
        if np.allclose(A, B):
            # print("S*Y*YS == U*X*XU")
            pass
        else:
            print("Whoa! That isn't right!")

        self.Z = Z_RS
        self.K_t = K_Y
        self.mu_coef = mu_coef
        # print("Z_RS.shape", self.Z.shape)
        # print("S.shape", S.shape)

        # Take the eigen. decomp of K_RS = S^* K_t S
        L_RS, M_RS = self._eigen_decomp(S.conj().T @ self.K_t @ S)
        np.sqrt(L_RS, L_RS)

        # print("M_RS.shape", M_RS.shape)
        # print("L_RS.shape", L_RS.shape)
        # print("L_RS", L_RS)
        if np.allclose(M_RS @ np.diag(L_RS ** 2) @ M_RS.conj().T, S.conj().T @ self.K_t @ S):
            # print("M L M* = S* Y* Y S")
            pass
        else:
            print("Check this:")
            print("A", M_RS @ np.diag(L_RS ** 2) @ M_RS.conj().T)
            print("B", S.conj().T @ self.K_t @ S)

        P_1 = (S @ M_RS * np.reciprocal(L_RS)).conj().T @ self.K_t @ S

        # Form the helper matrix P
        P = (M_RS * L_RS).conj().T

        if np.allclose(P_1, P):
            # print("Yeah, these are the same")
            pass
        else:
            print("Hooray! Hopefully that was the problem")

        # Normalize the columns of P
        N = np.reciprocal(np.linalg.norm(P, axis=1))
        print("N", N)
        print("diag(S* Y* Y S", np.diag(S.conj().T @ K_Y @ S))


        P_norm = P * N
        # print("Is this 1?", np.sum(P[:, 0]))  # This had better be 1

        # print("P.shape", P.shape)

        # Form the new U
        U_new = S @ (M_RS * np.reciprocal(L_RS)) @ P_norm
        # print("U_new.shape", U_new.shape)

        I = U_new.conj().T @ self.K_t @ U_new
        if np.allclose(I, np.eye(I.shape[0])):
            # print("That's good too...")
            pass
        else:
            print("Hey! That's wrong!")
            print("I", I)
            print("L_RS", L_RS)
            print("L_RS ** 2", L_RS ** 2)
            print("L_RS ** 0.5", np.sqrt(L_RS))

        TEST = np.random.rand(self.Z.shape[0], 10)
        
        A = (C @ (self.U * self.E))

        # Compute the new E and store the new U
        self.E = np.diag(U_new.conj().T @ K_YZ @ self.U * self.E)
        self.U = U_new
        self.Vh = np.reciprocal(self.E) * P_norm.conj().T @ P

        B = (D @ (self.U * self.E))
        if np.allclose(A, B):
            # print("Okay, now I really don't know what could be wrong, except the mean")
            pass
        else:
            print("A difference!")
            print("old", A)
            print("new", B)

    def _findPreimages(self, num_whitens, num_reduced_set_components):
        """
        """
        m, n = self.Z.shape
        r = num_whitens
        p = num_reduced_set_components
        # This will hold our preimages (the vectors)
        Z_RS = np.zeros((m, (r + 1) * p))
        # This will hold their coefficients
        S = np.zeros(((r + 1) * p, r))
        # This will hold the new coefficients for the mean
        mu_coef = np.zeros(((r + 1) * p,))

        # Storage for the kernels
        K_YZ = np.zeros(((r + 1) * p, n))
        K_Y = np.zeros(((r + 1) * p, (r + 1) * p))

        
        # Index for the current feature space vector (-1 means the mean)
        j = -1
        max_allowed_preimage_index = -1
        # Whether or not we have just moved to the next feature space vector
        new_fs_vector = True
        for i in range((r + 1) * p):  # Build one preimage per loop
            if j == -1: # We are actually dealing with approximating the mean
                # We know all of the previous coefficients for sure, but need to compute the reconstruction sum
                if new_fs_vector:
                    max_allowed_preimage_index += p
                    new_fs_vector = False

                # Produce a set of vectors which make decent initial guesses
                initial_guesses = np.random.rand(m, np.prod(self.RS_gradient_ascent_parallelization))
                # First one is the "mean" of the library of vectors (which is also a blind guess)
                initial_guesses[:, 0] = self.Z @ self.mu_coef
                # The rest we simply ensure are orthogonal to the last preimage (if we have one)
                if i != 0:
                    initial_guesses[:, 1:] -= (initial_guesses[:, 1:].conj().T @ Z_RS[:, i-1]) / \
                                              (Z_RS[:, i-1].conj().T @ Z_RS[:, i-1]) * Z_RS[:, i-1, np.newaxis]

                Z_RS[:, i], mu_coef[i], K_YZ[i, :], K_Y[i, :i], K_Y[i, i], frac_recon_err = \
                    self._incrementally_solve(i, j, K_Y, K_YZ, Z_RS, mu_coef, initial_guesses)
                K_Y[:i, i] = K_Y[i, :i]  # Symmetry!
            else:
                # We may need to determine the first several coefficients and the reconstruction error
                if new_fs_vector:
                    S[:i, j] = np.linalg.pinv(K_Y[:i, :i]) @ K_YZ[:i, :] @ self.U[:, j]
                    max_allowed_preimage_index += p
                    err_base = self._reconstruction_error_base(i, K_Y, K_YZ, self.U[:, j], S[:i, j])
                    norm = self.U[:, j].conj().T @ self.K_t @ self.U[:, j]
                    if err_base / norm < self.RS_reconstruction_eps:
                        # By chance, this vector is already good enough, so skip trying to reconstruct it
                        print("\033[34mSkipping a vector...\033[39m")
                        continue
                    
                    new_fs_vector = False

                # Produce a set of random vectors
                initial_guesses = np.random.rand(m, sum(self.RS_gradient_ascent_parallelization))
                # Ensure they are all orthogonal to the last preimage
                initial_guesses -= (initial_guesses.conj().T @ Z_RS[:, i-1]) / (Z_RS[:, i-1].conj().T @ Z_RS[:, i-1]) *\
                                   Z_RS[:, i-1, np.newaxis]

                Z_RS[:, i], S[i, j], K_YZ[i, :], K_Y[i, :i], K_Y[i, i], frac_recon_err = \
                    self._incrementally_solve(i, j, K_Y, K_YZ, Z_RS, S, initial_guesses)
                K_Y[:i, i] = K_Y[i, :i]  # Symmetry!
            # determine whether we have sufficiently approximated this feature space vector and should move on
            if i >= max_allowed_preimage_index or frac_recon_err < self.RS_reconstruction_eps:
                # Notice that if we ever manage to approximate a feature space vector with fewer than r+1 preimages,
                # we use the remainder to assist in making the next one
                # print("Moving on to the next feature vector")
                print("i, j:", i, ",", j)
                # print("max_allowed_preimage_index", max_allowed_preimage_index)
                print("reconstruction_err", frac_recon_err)
                # print("RS_reconstruction_eps", self.RS_reconstruction_eps)

                # sys.exit(-1)

                j += 1
                new_fs_vector = True
                if j == r:  # This will only happen if we can reconstruct with fewer than expected
                    break
        
        # Find the most optimal coefficients (back fill the empty 0's with more precise values)
        S = np.linalg.pinv(K_Y) @ K_YZ @ self.U  # pseudo-inverse because there's a tiny chance it's not full-rank
        mu_coef = np.linalg.pinv(K_Y) @ K_YZ @ self.mu_coef

        return Z_RS, S, mu_coef, K_Y, K_YZ

    def _incrementally_solve(self, i, j, K_Y, K_YZ, preimages, pre_coefs, initial_guesses):
        """Incrementally determines the preimage which best approximates u = phi(Z) U[:, i] - phi(preimages) neg_coefs
        through gradient ascent.

        m = dimension of input data
        r = dimension of ikpca output data
        p = number of preimages to use per

        :param i: the index of the preimage to find
        :param j: the index of the current feature space vector to approximate (-1 indicates the mean)
        :param preimages: a matrix containing all the preimages as columns (and space to store all the others)
        :param pre_coegs: coefficients for the preimage vectors in the reconstruction of the feature space vector with 
            this vectors' coefficients in the j-th column (or 0th if j == -1)
        :param initial_guesses: a matrix containing up to RS_gradient_ascent_parallelization columns with each a guess
            at a preimage. More columns will be ignored and fewer columns will be supplemented with random values
        :returns: a matrix containing as a single column the preimage vector, the coefficient corresponding to the new
            preimage, a matrix containing the Kernels of all data vectors and the new preimage, a matrix containing the 
            Kernels of all previous preimage vectors and the new preimage, a matrix containing the Kernel of the new
            preimage with itself, and the fractional reconstruction error of the feature vector
        """
        if j == -1:
            pos_coefs = self.mu_coef
            neg_coefs = pre_coefs[:i]
        else:
            pos_coefs = self.U[:, j]
            neg_coefs = pre_coefs[:i, j]
        
        ovector = self.Z @ pos_coefs
        reconst = preimages[:, :i] @ neg_coefs
        print("Original Vector:", ovector)
        print("Current Reconstruction:", reconst)
        print("Vector to reconstruct:", ovector - reconst)
        
        # Start with the initial guess(es) (copy to make sure nothing breaks)
        y = np.zeros((self.Z.shape[0], np.prod(self.RS_gradient_ascent_parallelization)))
        copy_idx = min(y.shape[1], initial_guesses.shape[1])
        y[:, :copy_idx] = initial_guesses[:, :copy_idx].copy()
        # Supplement with random values in [0,1] if necessary
        y[:, copy_idx:] = np.random.rand(y.shape[0], max(y.shape[1] - copy_idx, 0))

        # Space for storing the best guesses
        y_best = np.zeros((y.shape[0], self.RS_gradient_ascent_parallelization[0]))
        # Space for storing the last best value
        y_last = np.zeros((y.shape[0], ))

        # Parameters for gradient descent
        learning_rates = np.power(10, np.linspace(np.log10(self.RS_gradient_ascent_threshold), 8, 
                                                  num=self.RS_gradient_ascent_parallelization[1]))
        print("lr", learning_rates)
        learning_rates = np.array([1e-3])
        temperature = 1
        norms = 1  # just a placeholder
        iteration = 0
        
        err_base = self._reconstruction_error_base(i, K_Y, K_YZ, pos_coefs, neg_coefs)

        non_improvement = 0
        while np.min(norms) > self.RS_gradient_ascent_threshold and iteration < self.RS_gradient_ascent_max_iters:
            iteration += 1
            if iteration > 200:
                sys.exit(-1)
            # Kernel function of original vectors, all preimages, and the new preimage(s)
            K_yZ = self.kernel.getMatrix(y, self.Z)  # (16, n)
            K_yY = self.kernel.getMatrix(y, preimages[:, :i])  # (16, i)
            K_yy = self.kernel.getMatrix(y, y, only_diag=True)  # (16, )

            # Compute the new ideal coefficients
            new_coefs = (K_yZ @ pos_coefs - K_yY @ neg_coefs) / K_yy

            # Compute the reconstruction error after the new preimage
            err = self._reconstruction_error(err_base, K_yZ, K_yY, K_yy, pos_coefs, neg_coefs, new_coefs)

            # print("err", np.min(err))
            if np.min(err) < -self.RS_gradient_ascent_threshold:
                print("\033[33mSOMETHING WENT WRONG!\033[39m")
                print("err_base", err_base)
                print("err", err)
    

            np.abs(err, out=err)  # To ensure that if floating point errors take it below zero, we ignore

            # Determine which of our current guesses are best
            min_idxs = np.argsort(err)

            y_best = y[:, min_idxs[:y_best.shape[1]]]  # (m, 2)
            new_coefs_best = new_coefs[min_idxs[:y_best.shape[1]]]  # (2, )
            K_yZ_best = K_yZ[min_idxs[:y_best.shape[1]], :]
            K_yY_best = K_yY[min_idxs[:y_best.shape[1]], :]
            K_yy_best = K_yy[min_idxs[:y_best.shape[1]]]

            
            y_diff = np.linalg.norm(y_best[:, 0] - y_last)
            # print("y_diff", y_diff)
            y_last = y_best[:, 0]
            print(y_last, y_diff, np.min(err))

            if np.min(err) < self.RS_gradient_ascent_threshold:  # Our best is already good enough
                print("Error is small enough")
                break
            if y_diff < self.RS_gradient_ascent_threshold:  # We aren't going to get any better
                non_improvement += 1
                if non_improvement > 5:
                    print("We didn't see any improvement")
                    break
            else:
                non_improvement = 0
            
            # Gradients of kernel function with orignal vectors, all preimages, and the new best preimage(s)
            Kprime_yZ = self.kernel.getGradient(y_best, self.Z) # (m, 2, n)
            Kprime_yY = self.kernel.getGradient(y_best, preimages[:, :i])  # (m, 2, i)
            Kprime_yy = self.kernel.getGradient(y_best, y_best, only_diag=True) # (m, 2)

            # Gradient of the gradient ascent
            gradient = self._projection_gradient(Kprime_yZ, Kprime_yY, Kprime_yy, pos_coefs, neg_coefs, new_coefs_best)
            print("old gradient:", gradient[:, 0])
            gradient = ((ovector - reconst) - y_best[:, 0])[:, np.newaxis]
            print("new gradient:", gradient[:, 0])
            # update y
            # Use a sigmoid to ensure we start quickly and then focus in
            temperature = 1.0 #/ (1.0 + np.power(np.e, -4*(1 - 2 * iteration / self.RS_gradient_ascent_max_iters)))
            print(y_best)
            print(gradient)
            print(learning_rates)
            # y = (y_best[..., np.newaxis] + \
            #     temperature * gradient[..., np.newaxis] * learning_rates[np.newaxis, :]).reshape(y.shape)
            y = y_best + learning_rates * gradient
        
        # Figure out the most optimal y one last time
        K_yZ = self.kernel.getMatrix(y, self.Z)  # (16, n)
        K_yY = self.kernel.getMatrix(y, preimages[:, :i])  # (16, i)
        K_yy = self.kernel.getMatrix(y, y, only_diag=True)  # (16, )

        # Compute the new ideal coefficients
        new_coefs = (K_yZ @ pos_coefs - K_yY @ neg_coefs) / K_yy

        # Compute the reconstruction error after the new preimage
        err = self._reconstruction_error(err_base, K_yZ, K_yY, K_yy, pos_coefs, neg_coefs, new_coefs)
        best_idx = np.argmin(err)
        # print("err:", err[best_idx])
        print("num iters:", iteration)

        # print("Return", y[:, best_idx], c[best_idx] / K_yy[best_idx], K_yZ[best_idx, :], K_yY[best_idx, :], K_yy[best_idx])
        return y[:, best_idx], new_coefs[best_idx], K_yZ[best_idx, :], K_yY[best_idx, :], K_yy[best_idx], err[best_idx]

    def _reconstruction_error_base(self, i, K_Y, K_YZ, pos_coefs, neg_coefs):
        """Convenience method which computes the consistent part reconstruction error of the jth feature space vector 
        using up to the ith preimage (which is the reconstruction error of the jth feature space vector using up to the
        (i-1)th preimage)

        :param i: index for the last preimage to use
        :param K_Y: kernel matrix between the preimages
        :param K_YZ: kernel matrix between the preimages and the data library
        :param pos_coefs: coefficients for the vector to reconstruct in terms of the data library
        :param neg_coefs: coefficients of the reconstruction of the vector in terms of the preimages
        :returns: the Frobenius difference between the jth feature vector and its reconstruction up to the (i-1)th
            preimage
        """
        err_base = pos_coefs.conj().T @ self.K_t @ pos_coefs + neg_coefs.conj().T @ K_Y[:i, :i] @ neg_coefs
        if i != 0:
            err_base -= 2 * np.real(neg_coefs[:, np.newaxis].conj().T @ K_YZ[:i, :] @ pos_coefs)
        return err_base

    def _reconstruction_error(self, err_base, K_yZ, K_yY, K_yy, pos_coefs, neg_coefs, new_coef):
        """Convenience method which computes the full reconstruction error of the jth feature space vector using up to 
        the ith preimage

        :param err_base: the Frobenius difference between the jth feature vector and its reconstruction up to the 
            (i-1)th preimage
        :param K_yZ: kernel matrix between the new preimage and the data library
        :param K_yY: kernel matrix between the new preimage and the other preimages
        :param K_yy: kernel matrix between the new preimage and itself
        :param pos_coefs: coefficients for the vector to reconstruct in terms of the data library
        :param neg_coefs: coefficients of the reconstruction of the vector in terms of the preimages
        :param new_coef: the coefficient for the new preimage
        :returns: the Frobenius difference between the jth feature vector and its reconstruction
        """
        err = err_base + new_coef.conj().T * K_yy * new_coef
        err += 2 * np.real(new_coef.conj().T * (- K_yZ @ pos_coefs + K_yY @ neg_coefs))
        return err


    def _reconstruction_gradient(self, K_yZ, K_yY, K_yy, Kprime_yZ, Kprime_yY, Kprime_yy, pos_coefs,neg_coefs,new_coef):
        """Convenience method which computes the gradient of the reconstruction error of the jth feature space vector 
        using up to the ith preimage

        :param K_yZ: kernel matrix between the new preimage and the data library
        :param K_yY: kernel matrix between the new preimage and the other preimages
        :param K_yy: kernel matrix between the new preimage and itself
        :param Kprime_yZ: kernel gradient matrix between the new preimage and the data library
        :param Kprime_yY: kernel gradient matrix between the new preimage and the other preimages
        :param Kprime_yy: kernel gradient matrix between the new preimage and itself
        :param pos_coefs: coefficients for the vector to reconstruct in terms of the data library
        :param neg_coefs: coefficients of the reconstruction of the vector in terms of the preimages
        :param new_coef: the coefficient for the new preimage
        :returns: the gradient of the reconstruction at this particular preimage
        """
        # helper value
        grad_new_coef = (Kprime_yZ @ pos_coefs - Kprime_yY @ neg_coefs - 2 * Kprime_yy * new_coef) / K_yy
        gradient = grad_new_coef.conj() * K_yy * new_coef + 2 * new_coef.conj() * Kprime_yy * new_coef + \
                   new_coef.conj() * K_yy * grad_new_coef
        gradient += 2*np.real(-grad_new_coef.conj() * (K_yZ @ pos_coefs) - new_coef.conj() * (Kprime_yZ @ pos_coefs) + \
                               grad_new_coef.conj() * (K_yY @ neg_coefs) + new_coef.conj() * (Kprime_yY @ neg_coefs))
        return gradient

    def _projection_gradient(self, Kprime_yZ, Kprime_yY, Kprime_yy, pos_coefs, neg_coefs, new_coef):
        """Convenience method which computes the gradient of the reconstruction error of the projection of the jth 
        feature space vector using up to the ith preimage

        :param Kprime_yZ: kernel gradient matrix between the new preimage and the data library
        :param Kprime_yY: kernel gradient matrix between the new preimage and the other preimages
        :param Kprime_yy: kernel gradient matrix between the new preimage and itself
        :param pos_coefs: coefficients for the vector to reconstruct in terms of the data library
        :param neg_coefs: coefficients of the reconstruction of the vector in terms of the preimages
        :param new_coef: the coefficient for the new preimage
        :returns: the gradient of the reconstruction at this particular preimage
        """
        gradient = 2 * np.real(new_coef.conj().T * (Kprime_yZ @ pos_coefs - Kprime_yY @ neg_coefs))
        gradient -= 2 * new_coef.conj().T * new_coef * Kprime_yy
        return gradient


    def IKPCARSTransform(self, input_data):
        """Incremental Kernel PCA (IKPCA) projection of the input_data to the lower-dim space

        :param input_data: data to transform
        :returns: transformed data

        Given the definitions above in the training function, the projection of IKPCA is given by
        O = U_t^* X_t^* (phi(Y) - avg{phi(Z)}), where Y is the new data to transform
        With some simplification, this becomes
        O = U_t^* k(Z, Y) - U_t^* avg{k(Z, Z)}
        Since the second term does not depend on Y, we precalculate it
        pca_signal_shift = U_t^* avg{k(Z, Z)}
        and have
        O = U_t^* k(Z, Y) - pca_signal_shift

        For convenience, here is a list of variable names for the code below
        Z: --traning data
        U: --projection matrix
        pca_signal_shift: --average{U_t^* k(Z, Z)} along data axis
        Y: --data to transform
        """
        Y = input_data.T if self.row_major else input_data
        O_new = self.U.conj().T @ self.kernel.getMatrix(self.Z, Y) - self.pca_signal_shift[:, np.newaxis]
        O_old = self.U_old.conj().T @ self.kernel.getMatrix(self.Z_old, Y) - self.pca_signal_shift_old[:, np.newaxis]
        if np.allclose(O_new, O_old):
            # print("Okay, wtf! These agree too!?!")
            pass
        else:
            print("They don't match")
            print("O_new", O_new)
            print("O_old", O_old)
        O = O_new
        return O.T if self.row_major else O
