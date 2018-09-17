# pyIKSFA

## Incremental Kernel Slow Feature Analysis

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
function and performs SFA on the data through that kernel function.

TODO: Link all the papers in TODO: add figures to explain kernel, math, etc.
