
import numpy as np
from source.core.IKSFA.Python_version import Kernel
from source.core.IKSFA.Python_version.IKSFA import *
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def test_linear():
    # # make some test data (which is linear)
    # num = 2000
    # v = np.linspace(0, 4*np.pi, num)
    # x1 = np.sin(v) + np.cos(11*v)
    # x2 = np.cos(11*v)
    # data = np.c_[x1, x2]

    # make some test data (which is quadratic, but contains the linear piece)
    num = 2000
    v = np.linspace(0, 4*np.pi, num)
    x1 = np.sin(v) + np.cos(11*v) ** 2
    x2 = np.cos(11*v)
    data = np.c_[x1, x2, x1 ** 2, x2 ** 2, x1 * x2]

    # Test my version of IKSFA
    linearKernel = Kernel.LinearKernel
    sfa = IKSFA(linearKernel, num_output_components=2)

    sfa.BatchTrain(data)
    result = sfa.BatchTransform(data)

    # Use scaling to restore the results to a reasonable range
    scaler = StandardScaler()
    scaler.fit(result)
    result = scaler.transform(result)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
    # fig = plt.figure(figsize=(3, 12))
    ax1.plot(v, x1, c="red")
    ax1.set_ylabel(r"$x_1$", rotation=0, ha="right", va="center")
    ax1.text(0.997, 0.98, r"$\sin(v) + \cos(11v)$", 
             bbox={'facecolor':'w', 'alpha':0.5, 'edgecolor':'black', 'pad':1},
             ha="right", va="top", transform=ax1.transAxes)
    ax1.axhline(0, color="black", linestyle="--")
    ax1.set_yticks([])
    ax2.plot(v, x2, c="blue")
    ax2.set_ylabel(r"$x_2$", rotation=0, ha="right", va="center")
    ax2.text(0.997, 0.98, r"$\cos(11v)$", 
             bbox={'facecolor':'w', 'alpha':0.5, 'edgecolor':'black', 'pad':1},
             ha="right", va="top", transform=ax2.transAxes)
    ax2.axhline(0, color="black", linestyle="--")
    ax2.set_yticks([])


    ax3.plot(v, result[:, 0], c="green")
    ax3.set_ylabel(r"$SFA_0$", rotation=0, ha="right", va="center")
    ax3.text(0.997, 0.98, r"$\sim\alpha*\sin(v)$", 
             bbox={'facecolor':'w', 'alpha':0.5, 'edgecolor':'black', 'pad':1},
             ha="right", va="top", transform=ax3.transAxes)
    ax3.axhline(0, color="black", linestyle="--")
    ax3.set_yticks([])
    ax4.plot(v, result[:, 1], c="orange")
    ax4.set_ylabel(r"$SFA_1$", rotation=0, ha="right", va="center")
    ax4.text(0.997, 0.98, r"$\sim\beta*\cos(11v)$", 
             bbox={'facecolor':'w', 'alpha':0.5, 'edgecolor':'black', 'pad':1},
             ha="right", va="top", transform=ax4.transAxes)
    ax4.axhline(0, color="black", linestyle="--")
    ax4.set_yticks([])
    plt.xlabel("v")
    plt.xlim([0, 4*np.pi])
    plt.ylim([-2.2, 2.2])
    plt.xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi],
               ["0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"])
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    fig.suptitle("Test Batch KSFA (Linear)")

    filename = "../../plots/test-batchksfa-linear.png"
    print("Saving to {}".format(filename))
    plt.savefig(filename, dpi=300)


def test_quadratic():
    # make some test data (which is linear)
    # num = 2000
    # v = np.linspace(0, 4*np.pi, num)
    # x1 = np.sin(v) + np.cos(11*v)
    # x2 = np.cos(11*v)
    # data = np.c_[x1, x2]

    # # make some test data (which is quadratic)
    num = 2000
    v = np.linspace(0, 4*np.pi, num)
    x1 = np.sin(v) + np.cos(11*v) ** 2
    x2 = np.cos(11*v)
    # data = np.c_[x1, x2, x1 ** 2, x2 ** 2, x1 * x2]
    data = np.c_[x1, x2]

    # Test my version of IKSFA
    quadraticKernel = Kernel.QuadraticKernel
    sfa = IKSFA(quadraticKernel, "forwards", num_output_components=2)

    sfa.BatchTrain(data)
    result = sfa.BatchTransform(data)
    # Use scaling to restore the results to a reasonable range
    scaler = StandardScaler()
    scaler.fit(result)
    result = scaler.transform(result)


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
    # fig = plt.figure(figsize=(3, 12))
    ax1.plot(v, x1, c="red")
    ax1.set_ylabel(r"$x_1$", rotation=0, ha="right", va="center")
    ax1.text(0.997, 0.98, r"$\sin(v) + \cos(11v)^2$", 
             bbox={'facecolor':'w', 'alpha':0.5, 'edgecolor':'black', 'pad':1},
             ha="right", va="top", transform=ax1.transAxes)
    ax1.axhline(0, color="black", linestyle="--")
    ax1.set_yticks([])
    ax2.plot(v, x2, c="blue")
    ax2.set_ylabel(r"$x_2$", rotation=0, ha="right", va="center")
    ax2.text(0.997, 0.98, r"$\cos(11v)$", 
             bbox={'facecolor':'w', 'alpha':0.5, 'edgecolor':'black', 'pad':1},
             ha="right", va="top", transform=ax2.transAxes)
    ax2.axhline(0, color="black", linestyle="--")
    ax2.set_yticks([])


    ax3.plot(v, result[:, 0], c="green")
    ax3.set_ylabel(r"$SFA_0$", rotation=0, ha="right", va="center")
    ax3.text(0.997, 0.98, r"$\sim\alpha*\sin(v)$", 
             bbox={'facecolor':'w', 'alpha':0.5, 'edgecolor':'black', 'pad':1},
             ha="right", va="top", transform=ax3.transAxes)
    ax3.axhline(0, color="black", linestyle="--")
    ax3.set_yticks([])
    ax4.plot(v, result[:, 1], c="orange")
    ax4.set_ylabel(r"$SFA_1$", rotation=0, ha="right", va="center")
    ax4.text(0.997, 0.98, r"$\sim\beta*\cos(11v)$", 
             bbox={'facecolor':'w', 'alpha':0.5, 'edgecolor':'black', 'pad':1},
             ha="right", va="top", transform=ax4.transAxes)
    ax4.axhline(0, color="black", linestyle="--")
    ax4.set_yticks([])
    plt.xlabel("v")
    plt.xlim([0, 4*np.pi])
    plt.ylim([-2.2, 2.2])
    plt.xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi],
               ["0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"])
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    fig.suptitle("Test Batch KSFA (Quadratic)")

    filename = "../../plots/test-batchksfa-quadratic.png"
    print("Saving to {}".format(filename))
    plt.savefig(filename, dpi=300)


def test_quadraticOLD():
    # make some test data (which is linear)
    # num = 2000
    # v = np.linspace(0, 4*np.pi, num)
    # x1 = np.sin(v) + np.cos(11*v)
    # x2 = np.cos(11*v)
    # data = np.c_[x1, x2]

    # # make some test data (which is quadratic)
    num = 2000
    v = np.linspace(0, 4*np.pi, num)
    x1 = np.sin(v) + np.cos(11*v) ** 2
    x2 = np.cos(11*v)
    data = np.c_[x1, x2]
    # data = np.c_[x1, x2, x1 ** 2, x2 ** 2, x1 * x2]

    # Test my version of IKSFA
    quadraticKernel = Kernel.OLDQuadraticKernel()
    sfa = IKSFA(quadraticKernel, "central", num_output_components=2)

    sfa.BatchTrain(data)
    result = sfa.BatchTransform(data)
    # Use scaling to restore the results to a reasonable range
    scaler = StandardScaler()
    scaler.fit(result)
    result = scaler.transform(result)


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
    # fig = plt.figure(figsize=(3, 12))
    ax1.plot(v, x1, c="red")
    ax1.set_ylabel(r"$x_1$", rotation=0, ha="right", va="center")
    ax1.text(0.997, 0.98, r"$\sin(v) + \cos(11v)^2$", 
             bbox={'facecolor':'w', 'alpha':0.5, 'edgecolor':'black', 'pad':1},
             ha="right", va="top", transform=ax1.transAxes)
    ax1.axhline(0, color="black", linestyle="--")
    ax1.set_yticks([])
    ax2.plot(v, x2, c="blue")
    ax2.set_ylabel(r"$x_2$", rotation=0, ha="right", va="center")
    ax2.text(0.997, 0.98, r"$\cos(11v)$", 
             bbox={'facecolor':'w', 'alpha':0.5, 'edgecolor':'black', 'pad':1},
             ha="right", va="top", transform=ax2.transAxes)
    ax2.axhline(0, color="black", linestyle="--")
    ax2.set_yticks([])


    ax3.plot(v, result[:, 0], c="green")
    ax3.set_ylabel(r"$SFA_0$", rotation=0, ha="right", va="center")
    ax3.text(0.997, 0.98, r"$\sim\alpha*\sin(v)$", 
             bbox={'facecolor':'w', 'alpha':0.5, 'edgecolor':'black', 'pad':1},
             ha="right", va="top", transform=ax3.transAxes)
    ax3.axhline(0, color="black", linestyle="--")
    ax3.set_yticks([])
    ax4.plot(v, result[:, 1], c="orange")
    ax4.set_ylabel(r"$SFA_1$", rotation=0, ha="right", va="center")
    ax4.text(0.997, 0.98, r"$\sim\beta*\cos(11v)$", 
             bbox={'facecolor':'w', 'alpha':0.5, 'edgecolor':'black', 'pad':1},
             ha="right", va="top", transform=ax4.transAxes)
    ax4.axhline(0, color="black", linestyle="--")
    ax4.set_yticks([])
    plt.xlabel("v")
    plt.xlim([0, 4*np.pi])
    plt.ylim([-2.2, 2.2])
    plt.xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi],
               ["0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"])
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    fig.suptitle("Test Batch KSFA (OLDQuadratic)")

    filename = "../../plots/test-batchksfa-OLDquadratic.png"
    print("Saving to {}".format(filename))
    plt.savefig(filename, dpi=300)


def test_greedyIKPCA_linear():

    # make some test data (which is quadratic, but contains the linear piece)
    num = 2000
    v = np.linspace(0, 4*np.pi, num)
    x1 = np.sin(v) + np.cos(11*v) ** 2
    x2 = np.cos(11*v)
    data = np.c_[x1, x2, x1 ** 2, x2 ** 2, x1 * x2]

    # Test my version of IKSFA
    linearKernel = Kernel.LinearKernel
    sfa = IKSFA(linearKernel, num_output_components=2)

    sfa.GreedyIKPCA(data[:10])
    sfa.GreedyIKPCA(data[10:20])


def test_greedyIKPCA_quadratic():

     # # make some test data (which is quadratic)
    num = 2000
    v = np.linspace(0, 4*np.pi, num)
    x1 = np.sin(v) + np.cos(11*v) ** 2
    x2 = np.cos(11*v)
    # data = np.c_[x1, x2, x1 ** 2, x2 ** 2, x1 * x2]
    data = np.c_[x1, x2]

    # Test my version of IKPCA
    quadraticKernel = Kernel.QuadraticKernel
    sfa = IKSFA(quadraticKernel, num_output_components=2)
    sfa2 = IKSFA(quadraticKernel, num_output_components=2)

    # Compare doing IKPCA over everything at once to doing it in several pieces
    total = sfa.GreedyIKPCA(data)

    for i in range(int(num/100)):
        piecewise = sfa2.GreedyIKPCA(data[100*i:100*(i+1)])

    print("total:", total)
    print("piecewise:", piecewise)

    print(np.allclose(total, piecewise))


def test_exactIKPCA(name, data, kernel):
    sfa = IKSFA(kernel, num_output_components=2)
    sfa2 = IKSFA(kernel, num_output_components=2)

    # Compare doing IKPCA over everything at once to doing it in several pieces
    sfa.ExactIKPCATrain(data)
    total = sfa.ExactIKPCATransform(data)

    for i in range(int(data.shape[0]/100)):
        sfa2.ExactIKPCATrain(data[100*i:100*(i+1)])
    piecewise = sfa2.ExactIKPCATransform(data)

    # Because the columns might be -1 * each other, we multiply by a correction
    signs = total[0, :] / piecewise[0, :]
    piecewise *= signs

    if not np.allclose(total, piecewise):
        print("Test Failed!")
        print("total:", total)
        print("piecewise:", piecewise)
    else:
        print("Test Passed ({})".format(name))


def time_exactIKPCA(data, kernel, num=5):
    
    times = np.zeros((num, ))
    
    for j in range(num):
        start_time = time.time()
        sfa = IKSFA(kernel, num_output_components=2)
        for i in range(int(data.shape[0]/100)):
            sfa.ExactIKPCATrain(data[100*i:100*(i+1)])
        end_time = time.time()
        times[j] = end_time - start_time
    print("Average:", np.average(times), "+/-", np.std(times))


# def test_IKPCARS_2(name, data, kernel):
#     sfa_exact = IKSFA(kernel, num_output_components=2)
#     sfa_inc = IKSFA(kernel, num_output_components=2)
#     sfa2 = IKSFA(kernel, num_output_components=2, num_reduced_set_components=2)

#     # Compare doing IKPCA over everything at once to doing it in several pieces
#     sfa_exact.ExactIKPCATrain(data)
#     total = sfa_exact.ExactIKPCATransform(data)

#     for i in range(int(data.shape[0]/100)):
#         sfa2.IKPCARS_2_Train(data[100*i:100*(i+1)])
#         sfa_inc.ExactIKPCATrain(data[100*i:100*(i+1)])

#         piecewise = sfa2.IKPCARS_2_Transform(data)
#         inc_out = sfa_inc.ExactIKPCATransform(data)
#         scaling = inc_out[0, :] / piecewise[0, :]
#         piecewise *= scaling

#         if not np.allclose(inc_out, piecewise):
#             print("Test %d Failed!" % i)
#             print("inc_out:", inc_out)
#             print("piecewise:", piecewise)
#             print("inc - piece", inc_out - piecewise)
#             # sys.exit(-1)
#         else:
#             print(i)
#     piecewise = sfa2.IKPCARS_2_Transform(data)

#     # Because the columns might be -1 * each other, we multiply by a correction
#     signs = total[0, :] / piecewise[0, :]
#     piecewise *= signs

#     if not np.allclose(total, piecewise):
#         print("Test Failed!")
#         print("total:", total)
#         print("piecewise:", piecewise)
#     else:
#         print("Test Passed ({})".format(name))

def test_IKPCARS(name, data, kernel):
    sfa = IKSFA(kernel, num_output_components=2)
    sfa_inc = IKSFA(kernel, num_output_components=2)
    sfa2 = IKSFA(kernel, num_output_components=2, num_reduced_set_components=10, RS_gradient_ascent_parallelization = (1, 1))

    # Compare doing IKPCA over everything at once to doing it in several pieces
    sfa.ExactIKPCATrain(data)
    total = sfa.ExactIKPCATransform(data)

    for i in range(int(data.shape[0]/100)):
        sfa3 = IKSFA(kernel, num_output_components=2)

        sfa2.IKPCARSTrain(data[100*i:100*(i+1)])
        sfa3.IKPCARSTrain(data[:100*(i+1)])
        sfa_inc.ExactIKPCATrain(data[100*i:100*(i+1)])

        piecewise = sfa2.IKPCARSTransform(data)
        p3 = sfa3.IKPCARSTransform(data)
        inc_out = sfa_inc.ExactIKPCATransform(data)
        

        scaling = p3[0, :] / piecewise[0, :]
        piecewise *= scaling

        if not np.allclose(p3, piecewise):
            print("Test %d Failed!" % i)
            print("p3:", p3)
            print("piecewise:", piecewise)
            print("p3 - piece", p3 - piecewise)

        scaling = inc_out[0, :] / piecewise[0, :]
        piecewise *= scaling

        if not np.allclose(inc_out, piecewise):
            print("Test %d Failed!" % i)
            print("inc_out:", inc_out)
            print("piecewise:", piecewise)
            print("inc - piece", inc_out - piecewise)
            # sys.exit(-1)
        else:
            print(i)
    piecewise = sfa2.IKPCARSTransform(data)

    # Because the columns might be -1 * each other, we multiply by a correction
    signs = total[0, :] / piecewise[0, :]
    piecewise *= signs

    if not np.allclose(total, piecewise):
        print("Test Failed!")
        print("total:", total)
        print("piecewise:", piecewise)
    else:
        print("Test Passed ({})".format(name))

def data(data_type):

    if data_type == "linear":
        # make some test data (which is quadratic, but contains the linear piece)
        num = 2000
        v = np.linspace(0, 4*np.pi, num)
        x1 = np.sin(v) + np.cos(11*v) ** 2
        x2 = np.cos(11*v)
        data = np.c_[x1, x2, x1 ** 2, x2 ** 2, x1 * x2]
    elif data_type == "quadratic":
        # make some test data (which is quadratic)
        num = 2000
        v = np.linspace(0, 4*np.pi, num)
        x1 = np.sin(v) + np.cos(11*v) ** 2
        x2 = np.cos(11*v)
        data = np.c_[x1, x2]
    elif data_type == "complex":
        # make some test data (which is quadratic)
        num = 2000
        v = np.linspace(0, 4*np.pi, num)
        x1 = (np.sin(v) + np.cos(11*v) ** 2) * (1+1j)
        x2 = np.cos(11*v) * (1+1j)
        data = np.c_[x1, x2]
    return data


if __name__ == "__main__":

    # From calculation, it is expected that the local minimum occurs at x=9/4

    cur_x = 6 # The algorithm starts at x=6
    gamma = 0.01 # step size multiplier
    precision = 1e-10
    previous_step_size = 1 
    max_iters = 10000 # maximum number of iterations
    iters = 0 #iteration counter

    df = lambda x: 4 * x**3 - 9 * x**2

    while (previous_step_size > precision) & (iters < max_iters):
        print(cur_x)
        prev_x = cur_x
        cur_x -= gamma * df(prev_x)
        previous_step_size = abs(cur_x - prev_x)
        iters+=1

    print("I took", iters, "iterations")
    print("The local minimum occurs at", cur_x)
    #The output for the above will be: ('The local minimum occurs at', 2.2499646074278457)





    # test_linear()
    # test_quadratic()
    # test_quadraticOLD()

    # test_greedyIKPCA_linear()
    # test_greedyIKPCA_quadratic()

    # test_exactIKPCA("exact-linear", data("linear"), Kernel.LinearKernel)
    ###test_exactIKPCA("exact-quadratic", data("quadratic"), Kernel.QuadraticKernel)
    # test_exactIKPCA("exact-complex", data("complex"), Kernel.QuadraticKernel)
    # time_exactIKPCA(data("quadratic"), Kernel.QuadraticKernel, num=25)
    # time_exactIKPCA(data("complex"), Kernel.QuadraticKernel, num=5)

    # test_IKPCARS_2("rs2-linear", data("linear"), Kernel.LinearKernel)

    test_IKPCARS("rs-linear", data("linear"), Kernel.LinearKernel)
    test_IKPCARS("rs-quadratic", data("quadratic"), Kernel.QuadraticKernel)
    
    print("done.")
