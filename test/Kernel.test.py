import source.Kernel as Kernel
from testUtils import testFunction

def test_LinearKernel_getDimension():
    kernel = Kernel.LinearKernel

    statement = "The method _getDimension on LinearKernel will return the dimension of the data after the linear " + \
                "expansion function is applied"
    
    expected_result = 5
    actual_result = kernel._getDimension(1, (5, 0))
    testFunction(statement, expected_result, actual_result)

    expected_result = 7
    actual_result = kernel._getDimension(1, (7, 20))
    testFunction(statement, expected_result, actual_result)
    
if __name__ == "__main__":
    test_LinearKernel_getDimension()