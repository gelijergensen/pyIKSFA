"""
TODO




@author G. Eli Jergensen - modified from MATLAB code by Stephen Liwicki
"""


from abc import ABC, abstractmethod
import numpy as np


class Expansion(ABC):

    @abstractmethod
    def getExpansion(self, data):
        """Performs an arbitrary mapping on the data and returns the result.
        This is typically an expansion of the data. For instance, the quadratic
        expansion for data like ( a b ) is ( a b a^2 b^2 ab)
                                ( c d )    ( c d c^2 d^2 cd)
                                ( ... )    (       ...     )

        :param data: A matrix whose rows consist of data entries
        :returns: A matrix whose rows consist of the same data entries after the
            mapping
        """
        pass

    def __call__(self, data):
        """Convenience method which overrides the call method to call the 
        getExpansion function

        :param data: A matrix whose rows consist of data entries
        :returns: A matrix whose rows consist of the same data entries after the
            expansion mapping
        """
        return self.getExpansion(data)


class NoneExpansion(Expansion):
    """Simply returns 