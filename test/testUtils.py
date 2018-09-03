"""A quick utility function which prints a statement if the expected result does not match the actual result"""

def testFunction(statement, expected_result, actual_result):
    if expected_result != actual_result:
        print(statement)
        print("Expected:", expected_result)
        print("Acutal:", actual_result)