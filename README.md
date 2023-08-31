Branch: features/parts_of_code

MLOps class 3: Digit classification problem

How to setup:

install conda
conda create -n digit python=3.11.4
conda activate digit


Rrequired Libraries:

matplotlib=3.7.2
scikit-learn=1.3.0
Add them to requirements.txt
pip install -r requirements.txt

How to Run:

python digiclf.py


Meaning of failure:

- poor performance matrix
- code/runtime error
- adjust model hyperparameters
- model giving poor/wrong prediction on unseen data


Problem Definition:

================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

Digits dataset
--------------

The digits dataset consists of 8x8
pixel images of digits. The ``images`` attribute of the dataset stores
8x8 arrays of grayscale values for each image. We will use these arrays to
visualize the first 4 images. The ``target`` attribute of the dataset stores
the digit each image represents and this is included in the title of the 4
plots below.

Note: if we were working from image files (e.g., 'png' files), we would load
them using :func:`matplotlib.pyplot.imread`.


Classification
--------------

To apply a classifier on this data, we need to flatten the images, turning
each 2-D array of grayscale values from shape ``(8, 8)`` into shape
``(64,)``. Subsequently, the entire dataset will be of shape
``(n_samples, n_features)``, where ``n_samples`` is the number of images and
``n_features`` is the total number of pixels in each image.

We can then split the data into train and test subsets and fit a support
vector classifier on the train samples. The fitted classifier can
subsequently be used to predict the value of the digit for the samples
in the test subset.






