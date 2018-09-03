"""
TODO finish the documentation once you are completely sure how it all works


REMEMBER TO MENTION COLUMN MAJOR!

@author G. Eli Jergensen <gelijergensen@ou.edu> - constructed following the 
    paper by Stephen Liwicki et. al
"""

from abc import ABC, abstractmethod
import numpy as np

class Difference(ABC):

    @abstractmethod
    def getMatrix(self, n):
        """Returns the difference matrix of a particular size

        :param n: the dimension of the square matrix to return
        :returns: an (n by n) matrix
        """
        pass

class BackwardDifference(Difference):

    def getMatrix(self, n):
        """Backward difference subtracts the previous data point from the 
        current

        :param n: the size of the square matrix to return
        :returns: an (n by n) real matrix
        """
        D = np.identity(n) - np.eye(n, k=1)
        D[0, 0] = 0 
        return D

class ForwardDifference(Difference):

    def getMatrix(self, n):
        """Forward difference subtracts the current data point from the next

        :param n: the size of the square matrix to return
        :returns: an (n by n) real matrix
        """
        D = np.eye(n, k=-1) - np.identity(n)
        D[-1, -1] = 0
        return D

class CentralDifference(Difference):

    def getMatrix(self, n):
        """Forward difference subtracts half the previous data point from half
        of the next

        :param n: the size of the square matrix to return
        :returns: an (n by n) real matrix
        """
        D = np.eye(n, k=-1) - np.eye(n, k=1)
        D *= 0.5
        return D