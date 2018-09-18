# pyIKSFA

## Theory of Incremental Kernel Slow Feature Analysis

Slow Feature Analysis (SFA) is an unsupervised learning method similar to
Principal Component Analysis (PCA). It operates on the principle that important
information within a signal varies more slowly than the signal itself, due to
either the nature of the signal itself or noise found within the received
signal. For example, in a video of a ball rolling across a field, the individual
pixels may change quite rapidly due to subtle lighting changes, the grass
blowing in the wind, or the movement of the ball across the image. Despite this,
the important information can primarily be carried in a single number: the
location of the ball along the width of the image. Informally, Slow Feature
Analysis extracts the slowest-changing features (other than the constant
feature) in an image in the hopes that these features carry important structure.
To be more precise, given a particular class of functions, SFA find those
functions of the inputs which minimize the average of the square of the time
derivative of the inputs (coined the "slowness"). More information on the
original formulation of SFA can be found on the
[scholarpedia article](http://www.scholarpedia.org/article/Slow_feature_analysis).

Sadly, the formulation of SFA is all but analytically intractable except for a
select number of cases. The most important case is where the class of functions
is linear. Under this assumption (plus a few other reasonable ones), SFA reduces
to little more than a few eigenvalue decompositions (or singular value
decompositions, depending on the specifics), making it very similar to PCA. In
fact, SFA requires PCA as an orthogonalization step. However, this restriction
greatly limits the power of SFA. One method of overcoming this limitation is to
first expand the input variables through some expansion function before
performing SFA. (e.g. The quadratic expansion would take the vector (x, y) to
(x<sup>2</sup>, y<sup>2</sup>, xy, x, y).) Unfortunately, this causes an
exponential explosion in the time complexity of SFA and is therefore unfeasible
for all but the smallest datasets.

To circumvent this, we take advantage of Mercer's Theorem, which informally
states that, given an expansion function and an inner product, there exists a
direct function, called a kernel function, of pairs of the inputs which is
equivalent to the inner product of the expansion of those inputs. SFA can be
alternatively formulated as Kernel SFA, which accepts a particular kernel
function and performs SFA on the data through that kernel function. _Liwicki et
al. (2015)_ provide an implementation of Kernel SFA which also works online[1].
Rather than requiring the entire dataset at once, their formulation can take the
dataset incrementally in smaller chunks, which provides it much greater
flexibility and the possibility of handling datasets orders of magnitude larger.
On account of the incremental nature of their implementation of SFA, I refer to
it as Incremental Kernel Slow Feature Analysis, or IKSFA for short. This is the
primary source for the implementation here.

There are some subtle modifications which I made to the original implementation
put forward in [1]. Namely, because of the lack of necessary details for the
implementation of Incremental Kernel Principal Component Analysis (IKPCA), I
refer to the paper from _Chin and Suter (2007)_ on this subject [2]. They, in
turn, discuss methods for producing a constant-size Reduced Set representation
of the projection space of the IKPCA. The iterative method I have chosen draws
primarily from their method, but is influenced by the work of _Schölkopf et al.
(1998)_ [3] and _Schölkopf et al. (1999)_ [4]

TODO: figures to explain kernel, math, etc. would be nice

<sup>1</sup> S. Liwicki, S. P. Zafeiriou and M. Pantic, "Online Kernel Slow
Feature Analysis for Temporal Video Segmentation and Tracking," in IEEE
Transactions on Image Processing, vol. 24, no. 10, pp. 2955-2970, Oct. 2015.
doi: https://doi.org/10.1109/TIP.2015.2428052

<sup>2</sup> T.-J. Chin, D. Suter, "Incremental kernel principal component
analysis", IEEE Trans. Image Process., vol. 16, no. 6, pp. 1662-1674, Jun. 2007.
doi: https://doi.org/10.1109/TIP.2007.896668

<sup>3</sup> B. Schlkopf, P. Knirsch, A. Smola, C. Burges, "Fast approximation
of support vector kernel expansions and an interpretation of clustering as
approximation in feature spaces", Proc. DAGM Symp., pp. 124-132, 1998. doi:
https://doi.org/10.1007/978-3-642-72282-0_12

<sup>4</sup> B. Schlkopf, S. Mika, C. Burges, P. Knirsch, K.-R. Mller, G. Rtsch,
A. Smola, "Input space vs feature space in kernel-based methods", IEEE Trans.
Neural Netw., vol. 10, no. 5, pp. 1000-1017, May 1999. doi:
https://doi.org/10.1109/72.788641
