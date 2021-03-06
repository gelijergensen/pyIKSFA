import source.Kernel as Kernel
import numpy as np
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


def test_LinearKernel_getMatrix():
    kernel = Kernel.LinearKernel

    statement = "The method _getMatrix on LinearKernel will return a matrix(i, j) containing 1 + the result of dot " + \
                "products of the ith column of the left matrix with the jth column of the right matrix"

    expected_result = "ValueError"
    try:
        actual_result = kernel._getMatrix(1, 1, np.array([[1], [2]]), np.array([[1], [2], [3]]))
    except ValueError:
        actual_result = "ValueError"
    testFunction(statement, expected_result, actual_result)

    expected_result = 1 + np.array([[14]])
    actual_result = kernel._getMatrix(1, 1, np.array([[1], [2], [3]]), np.array([[1], [2], [3]]))
    testFunction(statement, expected_result, actual_result)

    A = np.arange(200).reshape(10, 20)
    B = np.arange(20, 180).reshape(10, 16)
    C = np.array([[109200,110100,111000,111900,112800,113700,114600,115500,116400,117300
                  ,118200,119100,120000,120900,121800,122700],
                  [110120,111030,111940,112850,113760,114670,115580,116490,117400,118310
                  ,119220,120130,121040,121950,122860,123770],
                  [111040,111960,112880,113800,114720,115640,116560,117480,118400,119320
                  ,120240,121160,122080,123000,123920,124840],
                  [111960,112890,113820,114750,115680,116610,117540,118470,119400,120330
                  ,121260,122190,123120,124050,124980,125910],
                  [112880,113820,114760,115700,116640,117580,118520,119460,120400,121340
                  ,122280,123220,124160,125100,126040,126980],
                  [113800,114750,115700,116650,117600,118550,119500,120450,121400,122350
                  ,123300,124250,125200,126150,127100,128050],
                  [114720,115680,116640,117600,118560,119520,120480,121440,122400,123360
                  ,124320,125280,126240,127200,128160,129120],
                  [115640,116610,117580,118550,119520,120490,121460,122430,123400,124370
                  ,125340,126310,127280,128250,129220,130190],
                  [116560,117540,118520,119500,120480,121460,122440,123420,124400,125380
                  ,126360,127340,128320,129300,130280,131260],
                  [117480,118470,119460,120450,121440,122430,123420,124410,125400,126390
                  ,127380,128370,129360,130350,131340,132330],
                  [118400,119400,120400,121400,122400,123400,124400,125400,126400,127400
                  ,128400,129400,130400,131400,132400,133400],
                  [119320,120330,121340,122350,123360,124370,125380,126390,127400,128410
                  ,129420,130430,131440,132450,133460,134470],
                  [120240,121260,122280,123300,124320,125340,126360,127380,128400,129420
                  ,130440,131460,132480,133500,134520,135540],
                  [121160,122190,123220,124250,125280,126310,127340,128370,129400,130430
                  ,131460,132490,133520,134550,135580,136610],
                  [122080,123120,124160,125200,126240,127280,128320,129360,130400,131440
                  ,132480,133520,134560,135600,136640,137680],
                  [123000,124050,125100,126150,127200,128250,129300,130350,131400,132450
                  ,133500,134550,135600,136650,137700,138750],
                  [123920,124980,126040,127100,128160,129220,130280,131340,132400,133460
                  ,134520,135580,136640,137700,138760,139820],
                  [124840,125910,126980,128050,129120,130190,131260,132330,133400,134470
                  ,135540,136610,137680,138750,139820,140890],
                  [125760,126840,127920,129000,130080,131160,132240,133320,134400,135480
                  ,136560,137640,138720,139800,140880,141960],
                  [126680,127770,128860,129950,131040,132130,133220,134310,135400,136490
                  ,137580,138670,139760,140850,141940,143030]])
    expected_result = 1 + C
    actual_result = kernel._getMatrix(1, 1, A, B)

    A = np.arange(200).reshape(10, 20) + 1.0j * np.arange(200, 400).reshape(10, 20)
    B = np.arange(20, 180).reshape(10, 16) + 1.0j * np.arange(160, 320).reshape(10, 16)
    C = np.array([[808400.-58000.j, 812200.-60000.j, 816000.-62000.j, 819800.-64000.j,
                   823600.-66000.j, 827400.-68000.j, 831200.-70000.j, 835000.-72000.j,
                   838800.-74000.j, 842600.-76000.j, 846400.-78000.j, 850200.-80000.j,
                   854000.-82000.j, 857800.-84000.j, 861600.-86000.j, 865400.-88000.j],
                   [811640.-56600.j, 815460.-58600.j, 819280.-60600.j, 823100.-62600.j,
                   826920.-64600.j, 830740.-66600.j, 834560.-68600.j, 838380.-70600.j,
                   842200.-72600.j, 846020.-74600.j, 849840.-76600.j, 853660.-78600.j,
                   857480.-80600.j, 861300.-82600.j, 865120.-84600.j, 868940.-86600.j],
                   [814880.-55200.j, 818720.-57200.j, 822560.-59200.j, 826400.-61200.j,
                   830240.-63200.j, 834080.-65200.j, 837920.-67200.j, 841760.-69200.j,
                   845600.-71200.j, 849440.-73200.j, 853280.-75200.j, 857120.-77200.j,
                   860960.-79200.j, 864800.-81200.j, 868640.-83200.j, 872480.-85200.j],
                   [818120.-53800.j, 821980.-55800.j, 825840.-57800.j, 829700.-59800.j,
                   833560.-61800.j, 837420.-63800.j, 841280.-65800.j, 845140.-67800.j,
                   849000.-69800.j, 852860.-71800.j, 856720.-73800.j, 860580.-75800.j,
                   864440.-77800.j, 868300.-79800.j, 872160.-81800.j, 876020.-83800.j],
                   [821360.-52400.j, 825240.-54400.j, 829120.-56400.j, 833000.-58400.j,
                   836880.-60400.j, 840760.-62400.j, 844640.-64400.j, 848520.-66400.j,
                   852400.-68400.j, 856280.-70400.j, 860160.-72400.j, 864040.-74400.j,
                   867920.-76400.j, 871800.-78400.j, 875680.-80400.j, 879560.-82400.j],
                   [824600.-51000.j, 828500.-53000.j, 832400.-55000.j, 836300.-57000.j,
                   840200.-59000.j, 844100.-61000.j, 848000.-63000.j, 851900.-65000.j,
                   855800.-67000.j, 859700.-69000.j, 863600.-71000.j, 867500.-73000.j,
                   871400.-75000.j, 875300.-77000.j, 879200.-79000.j, 883100.-81000.j],
                   [827840.-49600.j, 831760.-51600.j, 835680.-53600.j, 839600.-55600.j,
                   843520.-57600.j, 847440.-59600.j, 851360.-61600.j, 855280.-63600.j,
                   859200.-65600.j, 863120.-67600.j, 867040.-69600.j, 870960.-71600.j,
                   874880.-73600.j, 878800.-75600.j, 882720.-77600.j, 886640.-79600.j],
                   [831080.-48200.j, 835020.-50200.j, 838960.-52200.j, 842900.-54200.j,
                   846840.-56200.j, 850780.-58200.j, 854720.-60200.j, 858660.-62200.j,
                   862600.-64200.j, 866540.-66200.j, 870480.-68200.j, 874420.-70200.j,
                   878360.-72200.j, 882300.-74200.j, 886240.-76200.j, 890180.-78200.j],
                   [834320.-46800.j, 838280.-48800.j, 842240.-50800.j, 846200.-52800.j,
                   850160.-54800.j, 854120.-56800.j, 858080.-58800.j, 862040.-60800.j,
                   866000.-62800.j, 869960.-64800.j, 873920.-66800.j, 877880.-68800.j,
                   881840.-70800.j, 885800.-72800.j, 889760.-74800.j, 893720.-76800.j],
                   [837560.-45400.j, 841540.-47400.j, 845520.-49400.j, 849500.-51400.j,
                   853480.-53400.j, 857460.-55400.j, 861440.-57400.j, 865420.-59400.j,
                   869400.-61400.j, 873380.-63400.j, 877360.-65400.j, 881340.-67400.j,
                   885320.-69400.j, 889300.-71400.j, 893280.-73400.j, 897260.-75400.j],
                   [840800.-44000.j, 844800.-46000.j, 848800.-48000.j, 852800.-50000.j,
                   856800.-52000.j, 860800.-54000.j, 864800.-56000.j, 868800.-58000.j,
                   872800.-60000.j, 876800.-62000.j, 880800.-64000.j, 884800.-66000.j,
                   888800.-68000.j, 892800.-70000.j, 896800.-72000.j, 900800.-74000.j],
                   [844040.-42600.j, 848060.-44600.j, 852080.-46600.j, 856100.-48600.j,
                   860120.-50600.j, 864140.-52600.j, 868160.-54600.j, 872180.-56600.j,
                   876200.-58600.j, 880220.-60600.j, 884240.-62600.j, 888260.-64600.j,
                   892280.-66600.j, 896300.-68600.j, 900320.-70600.j, 904340.-72600.j],
                   [847280.-41200.j, 851320.-43200.j, 855360.-45200.j, 859400.-47200.j,
                   863440.-49200.j, 867480.-51200.j, 871520.-53200.j, 875560.-55200.j,
                   879600.-57200.j, 883640.-59200.j, 887680.-61200.j, 891720.-63200.j,
                   895760.-65200.j, 899800.-67200.j, 903840.-69200.j, 907880.-71200.j],
                   [850520.-39800.j, 854580.-41800.j, 858640.-43800.j, 862700.-45800.j,
                   866760.-47800.j, 870820.-49800.j, 874880.-51800.j, 878940.-53800.j,
                   883000.-55800.j, 887060.-57800.j, 891120.-59800.j, 895180.-61800.j,
                   899240.-63800.j, 903300.-65800.j, 907360.-67800.j, 911420.-69800.j],
                   [853760.-38400.j, 857840.-40400.j, 861920.-42400.j, 866000.-44400.j,
                   870080.-46400.j, 874160.-48400.j, 878240.-50400.j, 882320.-52400.j,
                   886400.-54400.j, 890480.-56400.j, 894560.-58400.j, 898640.-60400.j,
                   902720.-62400.j, 906800.-64400.j, 910880.-66400.j, 914960.-68400.j],
                   [857000.-37000.j, 861100.-39000.j, 865200.-41000.j, 869300.-43000.j,
                   873400.-45000.j, 877500.-47000.j, 881600.-49000.j, 885700.-51000.j,
                   889800.-53000.j, 893900.-55000.j, 898000.-57000.j, 902100.-59000.j,
                   906200.-61000.j, 910300.-63000.j, 914400.-65000.j, 918500.-67000.j],
                   [860240.-35600.j, 864360.-37600.j, 868480.-39600.j, 872600.-41600.j,
                   876720.-43600.j, 880840.-45600.j, 884960.-47600.j, 889080.-49600.j,
                   893200.-51600.j, 897320.-53600.j, 901440.-55600.j, 905560.-57600.j,
                   909680.-59600.j, 913800.-61600.j, 917920.-63600.j, 922040.-65600.j],
                   [863480.-34200.j, 867620.-36200.j, 871760.-38200.j, 875900.-40200.j,
                   880040.-42200.j, 884180.-44200.j, 888320.-46200.j, 892460.-48200.j,
                   896600.-50200.j, 900740.-52200.j, 904880.-54200.j, 909020.-56200.j,
                   913160.-58200.j, 917300.-60200.j, 921440.-62200.j, 925580.-64200.j],
                   [866720.-32800.j, 870880.-34800.j, 875040.-36800.j, 879200.-38800.j,
                   883360.-40800.j, 887520.-42800.j, 891680.-44800.j, 895840.-46800.j,
                   900000.-48800.j, 904160.-50800.j, 908320.-52800.j, 912480.-54800.j,
                   916640.-56800.j, 920800.-58800.j, 924960.-60800.j, 929120.-62800.j],
                   [869960.-31400.j, 874140.-33400.j, 878320.-35400.j, 882500.-37400.j,
                   886680.-39400.j, 890860.-41400.j, 895040.-43400.j, 899220.-45400.j,
                   903400.-47400.j, 907580.-49400.j, 911760.-51400.j, 915940.-53400.j,
                   920120.-55400.j, 924300.-57400.j, 928480.-59400.j, 932660.-61400.j]])
    expected_result = 1 + C
    actual_result = kernel._getMatrix(1, 1, A, B)
    testFunction(statement, expected_result, actual_result)

    A = np.arange(200).reshape(10, 20)
    B = np.arange(20, 180).reshape(10, 16)
    C = np.array([[109200,110100,111000,111900,112800,113700,114600,115500,116400,117300
                  ,118200,119100,120000,120900,121800,122700],
                  [110120,111030,111940,112850,113760,114670,115580,116490,117400,118310
                  ,119220,120130,121040,121950,122860,123770],
                  [111040,111960,112880,113800,114720,115640,116560,117480,118400,119320
                  ,120240,121160,122080,123000,123920,124840],
                  [111960,112890,113820,114750,115680,116610,117540,118470,119400,120330
                  ,121260,122190,123120,124050,124980,125910],
                  [112880,113820,114760,115700,116640,117580,118520,119460,120400,121340
                  ,122280,123220,124160,125100,126040,126980],
                  [113800,114750,115700,116650,117600,118550,119500,120450,121400,122350
                  ,123300,124250,125200,126150,127100,128050],
                  [114720,115680,116640,117600,118560,119520,120480,121440,122400,123360
                  ,124320,125280,126240,127200,128160,129120],
                  [115640,116610,117580,118550,119520,120490,121460,122430,123400,124370
                  ,125340,126310,127280,128250,129220,130190],
                  [116560,117540,118520,119500,120480,121460,122440,123420,124400,125380
                  ,126360,127340,128320,129300,130280,131260],
                  [117480,118470,119460,120450,121440,122430,123420,124410,125400,126390
                  ,127380,128370,129360,130350,131340,132330],
                  [118400,119400,120400,121400,122400,123400,124400,125400,126400,127400
                  ,128400,129400,130400,131400,132400,133400],
                  [119320,120330,121340,122350,123360,124370,125380,126390,127400,128410
                  ,129420,130430,131440,132450,133460,134470],
                  [120240,121260,122280,123300,124320,125340,126360,127380,128400,129420
                  ,130440,131460,132480,133500,134520,135540],
                  [121160,122190,123220,124250,125280,126310,127340,128370,129400,130430
                  ,131460,132490,133520,134550,135580,136610],
                  [122080,123120,124160,125200,126240,127280,128320,129360,130400,131440
                  ,132480,133520,134560,135600,136640,137680],
                  [123000,124050,125100,126150,127200,128250,129300,130350,131400,132450
                  ,133500,134550,135600,136650,137700,138750],
                  [123920,124980,126040,127100,128160,129220,130280,131340,132400,133460
                  ,134520,135580,136640,137700,138760,139820],
                  [124840,125910,126980,128050,129120,130190,131260,132330,133400,134470
                  ,135540,136610,137680,138750,139820,140890],
                  [125760,126840,127920,129000,130080,131160,132240,133320,134400,135480
                  ,136560,137640,138720,139800,140880,141960],
                  [126680,127770,128860,129950,131040,132130,133220,134310,135400,136490
                  ,137580,138670,139760,140850,141940,143030]])
    expected_result = 1 + np.diag(C)
    actual_result = kernel._getMatrix(1, 1, A, B, only_diag=True)
    testFunction(statement, expected_result, actual_result)


def test_LinearKernel_getGradient():
    kernel = Kernel.LinearKernel

    statement = "The method _getGradient on LinearKernel will return a 3tensor containing one gradient vector for " + \
                "each pair of columns of the two input matrices, i and j, respectively (tensor[:, i, j])"
    
    expected_result = "ValueError"
    try:
        actual_result = kernel._getGradient(1, 1, np.array([[1], [2]]), np.array([[1], [2], [3]]))
    except ValueError:
        actual_result = "ValueError"
    testFunction(statement, expected_result, actual_result)

    expected_result = np.array([[[1]], [[2]], [[3]]])
    actual_result = kernel._getGradient(1, 1, np.array([[4], [5], [6]]), np.array([[1], [2], [3]]))
    testFunction(statement, expected_result, actual_result)
    
    A = np.arange(200).reshape(10, 20)
    B = np.arange(20, 180).reshape(10, 16)
    expected_result = np.repeat(B.copy(), 20, axis=0).reshape(10, 20, 16)
    actual_result = kernel._getGradient(1, 1, A, B)
    testFunction(statement, expected_result, actual_result)

    A = np.arange(200).reshape(10, 20) + 1.0j * np.arange(200, 400).reshape(10, 20)
    B = np.arange(20, 180).reshape(10, 16) + 1.0j * np.arange(160, 320).reshape(10, 16)
    expected_result = np.repeat(B.copy(), 20, axis=0).reshape(10, 20, 16)
    actual_result = kernel._getGradient(1, 1, A, B)
    testFunction(statement, expected_result, actual_result)

    A = np.arange(200).reshape(10, 20)
    B = np.arange(20, 180).reshape(10, 16)
    expected_result = B.copy().reshape(10, 1, 16)
    actual_result = kernel._getGradient(1, 1, A, B, only_diag=True)


def test_LinearKernel_expansion():
    kernel = Kernel.LinearKernel

    statement = "The method _expansion on LinearKernel will augment the input with a constant and the matrix of " + \
                "the kernel of two matrices will agree with the dot product of their expansions"

    A = np.array([[1], [2], [3]])
    expected_result = np.r_[[[1]], A]
    actual_result = kernel._expansion(1, 1, A)
    testFunction(statement, expected_result, actual_result)

    A = np.arange(200).reshape(10, 20) / 10
    B = np.arange(20, 180).reshape(10, 16) / 10
    expected_result = (kernel._expansion(1, 1, A)).T @ (kernel._expansion(1, 1, B))
    actual_result = kernel._getMatrix(1, 1, A, B)
    testFunction(statement, expected_result, actual_result)


def test_LinearKernel():
    test_LinearKernel_getDimension()
    test_LinearKernel_getMatrix()
    test_LinearKernel_getGradient()
    test_LinearKernel_expansion()
    return 4


def test_QuadraticKernel_getDimension():
    kernel = Kernel.QuadraticKernel

    statement = "The method _getDimension on QuadraticKernel will return the dimension of the data after a " + \
                "quadratic expansion function is applied"
    
    expected_result = 5
    actual_result = kernel._getDimension(2, (2, 0))
    testFunction(statement, expected_result, actual_result)

    expected_result = (7 + 6 + 5 + 4 + 3 + 2 + 1) + 7
    actual_result = kernel._getDimension(2, (7, 20))
    testFunction(statement, expected_result, actual_result)


def test_QuadraticKernel_getMatrix():
    kernel = Kernel.QuadraticKernel

    statement = "The method _getMatrix on QuadraticKernel will return a matrix(i, j) containing the element-wise " + \
                "square of 1 + the dot products of the ith column of the left matrix with the jth column of the " + \
                "right matrix"

    expected_result = "ValueError"
    try:
        actual_result = kernel._getMatrix(2, 1, np.array([[1], [2]]), np.array([[1], [2], [3]]))
    except ValueError:
        actual_result = "ValueError"
    testFunction(statement, expected_result, actual_result)

    expected_result = (1 + np.array([[14]])) ** 2
    actual_result = kernel._getMatrix(2, 1, np.array([[1], [2], [3]]), np.array([[1], [2], [3]]))
    testFunction(statement, expected_result, actual_result)

    A = np.arange(200).reshape(10, 20)
    B = np.arange(20, 180).reshape(10, 16)
    expected_result = (kernel._expansion(2, 1, A)).T @ (kernel._expansion(2, 1, B))
    actual_result = kernel._getMatrix(2, 1, A, B)
    testFunction(statement, expected_result, actual_result)

    A = np.arange(200).reshape(10, 20) + 1.0j * np.arange(200, 400).reshape(10, 20)
    B = np.arange(20, 180).reshape(10, 16) + 1.0j * np.arange(160, 320).reshape(10, 16)
    expected_result = (kernel._expansion(2, 1, A)).conj().T @ (kernel._expansion(2, 1, B))
    actual_result = kernel._getMatrix(2, 1, A, B)
    testFunction(statement, expected_result, actual_result)

    A = np.arange(200).reshape(10, 20) + 1.0j * np.arange(200, 400).reshape(10, 20)
    B = np.arange(20, 180).reshape(10, 16) + 1.0j * np.arange(160, 320).reshape(10, 16)
    expected_result = np.diag((kernel._expansion(2, 1, A)).conj().T @ (kernel._expansion(2, 1, B)))
    actual_result = kernel._getMatrix(2, 1, A, B, only_diag=True)
    testFunction(statement, expected_result, actual_result)


def test_QuadraticKernel_getGradient():
    kernel = Kernel.QuadraticKernel

    statement = "The method _getGradient on Quadratic will return a 3tensor containing one gradient vector for " + \
                "each pair of columns of the two input matrices, i and j, respectively (tensor[:, i, j])"

    expected_result = "ValueError"
    try:
        actual_result = kernel._getGradient(2, 1, np.array([[1], [2]]), np.array([[1], [2], [3]]))
    except ValueError:
        actual_result = "ValueError"
    testFunction(statement, expected_result, actual_result)

    A = np.array([[4], [5], [6]])
    B = np.array([[1], [2], [3]])
    expected_result = np.einsum('ij,...j->i...j', B, 2*(1 + A.T @ B))
    actual_result = kernel._getGradient(2, 1, A, B)
    testFunction(statement, expected_result, actual_result)

    A = np.arange(200).reshape(10, 20)
    B = np.arange(20, 180).reshape(10, 16)
    expected_result = np.einsum('ij,...j->i...j', B, 2*(1 + A.T @ B))
    actual_result = kernel._getGradient(2, 1, A, B)
    testFunction(statement, expected_result, actual_result)

    A = np.arange(200).reshape(10, 20) + 1.0j * np.arange(200, 400).reshape(10, 20)
    B = np.arange(20, 180).reshape(10, 16) + 1.0j * np.arange(160, 320).reshape(10, 16)
    expected_result = np.einsum('ij,...j->i...j', B, 2*(1 + A.conj().T @ B))
    actual_result = kernel._getGradient(2, 1, A, B)
    testFunction(statement, expected_result, actual_result)

    A = np.arange(200).reshape(10, 20) + 1.0j * np.arange(200, 400).reshape(10, 20)
    B = np.arange(20, 180).reshape(10, 16) + 1.0j * np.arange(160, 320).reshape(10, 16)
    expected_result = np.einsum('ij,...j->i...j', B, np.diag(2*(1 + A.conj().T @ B)))
    actual_result = kernel._getGradient(2, 1, A, B, only_diag=True)
    testFunction(statement, expected_result, actual_result)


def test_QuadraticKernel_expansion():
    kernel = Kernel.QuadraticKernel

    statement = "The quadratic expansion will return a constant in the first row, and then the original data, and " + \
                "then all combinations of two rows in the original data. All combinations except the constant and " + \
                "the perfect squares are multiplied by the constant sqrt(2)"
    A = np.array([[1, 2, 3]])
    expected_result = np.array([[1, 1, 1], (np.array([1, 2, 3]) * np.sqrt(2)), [1, 4, 9]])
    actual_result = kernel._expansion(2, 1, A)
    testFunction(statement, expected_result, actual_result)

    A = np.array([[2, 4, 6], [1, 0, 9]])
    expected_result = np.array([[1, 1, 1], (np.array([2, 4, 6]) * np.sqrt(2)), np.array([1, 0, 9]) * np.sqrt(2), [4, 16, 36], np.array([2, 0, 54]) * np.sqrt(2), [1, 0, 81]])
    actual_result = kernel._expansion(2, 1, A)
    testFunction(statement, expected_result, actual_result)


def test_QuadraticKernel():
    test_QuadraticKernel_getDimension()
    test_QuadraticKernel_getMatrix()
    test_QuadraticKernel_getGradient()
    test_QuadraticKernel_expansion()
    return 4

if __name__ == "__main__":
    testCount = 0
    testCount += test_LinearKernel()
    testCount += test_QuadraticKernel()

    print("All %d Tests Complete." % testCount)
