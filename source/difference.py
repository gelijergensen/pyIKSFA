"""TODO

@author G. Eli Jergensen <gelijergensen@ou.edu> - modified from MATLAB code by 
    Stephen Liwicki
"""

import numpy as np

class Difference(object):
    """
    TODO
    """

    @staticmethod
    def _Taylor(self, n):
        """Computes 

        :param n:
        :returns:
        """

        taylorMat = np.zeros((n, 2*n+1))

        for row in range(n):
            # f'
            taylorMat[row, 0] = 1

            # f




