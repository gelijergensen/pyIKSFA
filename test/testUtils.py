"""A quick utility function which prints a statement if the expected result does not match the actual result"""

import numpy as np

def testFunction(statement, expected_result, actual_result):
    if isinstance(expected_result, np.ndarray):
        if not np.allclose(expected_result, actual_result):
            print(statement)
            print("Expected:", expected_result)
            print("Acutal:", actual_result)
    else:
        if expected_result != actual_result:
            print(statement)
            print("Expected:", expected_result)
            print("Acutal:", actual_result)