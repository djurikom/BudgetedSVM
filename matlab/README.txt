------------------------------------------------------------------------------------------
-- Matlab/Octave interface of BudgetedSVM, a Toolbox for Large-scale SVM Approximations --
------------------------------------------------------------------------------------------

Table of Contents
=================
- Table of Contents
- Introduction
- Version history
- The implemented algorithms
- Installation
- BudgetedSVM Usage - Training
- BudgetedSVM Usage - Testing
- Other Utilities
- Examples
- Returned Model Structure
- Additional Information
- Acknowledgments


Introduction
============
This tool provides a simple interface to BudgetedSVM, a library for large-scale
non-linear, multi-class problems. It is very easy to use as the usage and 
the way of specifying parameters are the same as that of LIBSVM or LIBLINEAR.

Please read the "./LICENSE.txt" license file before using the BudgetedSVM toolbox.


Version history
===============
The current version of the software is v1.1. Compared to the previous v1.0 version,
we have added the following changes:
	- (OBSOLETE) The software is no longer published under GPL v3 license, instead we
		publish BudgetedSVM under less restrictive LGPL v3 license.
        - The software is no longer published under LGPL v3 license, instead we
		publish BudgetedSVM under less restrictive Modified BSD license.
	- We changed the way certain parameters are set in the option string, please
		be very careful when using earlier scripts for running BudgetedSVM. Refer
		to "'budgetedsvm-train' Usage" and "'Budgetedsvm-predict' Usage" sections of
		this readme file, or to the command prompt help for more details. For example,
		kernel width is no longer specified with '-G', instead use '-g', etc.		
	- Added '-r' option for turning on/off randomization of the training data.
	- Added a number of kernel functions to be used with kernel-based algorithm, such
		as exponential, sigmoid, polynomial kernels. The parameters of the kernels
		are controlled with '-g', '-d', and '-i' options. Please refer to 
		"'budgetedsvm-train' Usage" section of this readme file, or to the command 
		prompt help for more details.
	- No longer necessary to specify the dimensionality of the data. However, note 
		that directly specifying the dimensionality through '-D' option can result
		in faster loading times.
	- In BSGD, we check if there are two identical vectors in the support vector set and
		either merge those two in the case of merging strategy, or remove the newer
		one in the case of random removal strategy, leading to better performance.
	- Fixed the randomization of LLSVM, previously if randomization was switched off
		you could still obtain different results between runs.
	- Added class scores to the outputs, please see "BudgetedSVM Usage - Testing" for
		more details regarding these output scores for different algorithms.
	- Many smaller code changes, that hopefully resulted in better, more readable code.
	- Bug fixes (Thanks to all users who reported bugs and provided feedback!).


The implemented algorithms
==========================
The BudgetedSVM toolbox implements Pegasos, Adaptive Multi-hyperplane Machines (AMM), 
Low-rank Linearization SVM (LLSVM), and Budgeted Stochastic Gradient Descent (BSGD)
algorithms. An overview of the algorithm properties is given in the table below:

	--------------------------------------------------------------------------------------------------------------
	| Algorithm | Classifier type | Multi-class? |                     Available kernels                         |
	==============================================================================================================
	|  Pegasos  |   Linear        | Multi-class  | Linear                                                        |
	|  AMM      |   Non-linear    | Multi-class  | Linear                                                        |
	|  LLSVM    |   Non-linear    | Binary       | Any                                                           |
	|  BSGD     |   Non-linear    | Multi-class  | Any for random removal, Gaussian when merging support vectors |
	--------------------------------------------------------------------------------------------------------------

For more details, please see their respective published papers. In particular, 
the publications can be found here:
*** "Pegasos: primal estimated sub-gradient solver for SVM" (Pegasos, found at
http://link.springer.com/article/10.1007/s10107-010-0420-4)
*** "Trading Representability for Scalability: Adaptive Multi-Hyperplane Machine for 
Nonlinear Classification" (AMM, found at "../doc/pdfs_of_algorithm_papers/AMM_paper.pdf")
*** "Scaling up Kernel SVM on Limited Resources: A Low-rank Linearization Approach"
(LLSVM, found at "../doc/pdfs_of_algorithm_papers/LLSVM_paper.pdf")
*** "Breaking the Curse of Kernelization: Budgeted Stochastic Gradient Descent for 
Large-Scale SVM Training" (BSGD, found at "../doc/pdfs_of_algorithm_papers/BSGD_paper.pdf")

For our BudgetedSVM paper, which gives a brief overview of the toolbox and summarizes its
main features, please see a PDF file at "../doc/pdfs_of_algorithm_papers/BudgetedSVM_paper.pdf".


Installation
============
We provide binary files for 64-bit Matlab on Windows. If you would like 
to re-build the package or there are problems with running the precompiled
files, please rely on the following steps.

We recommend using make.m on both Matlab and Octave. Simply type 'make'
to build 'libsvmread.mex', 'libsvmwrite.mex', 'budgetedsvm_train.mex', and
'budgetedsvm_predict.mex'.

On Matlab or Octave type:
	>> make

If make.m does not work on Matlab (especially for Windows), try 'mex
-setup' to choose a suitable compiler for mex. Make sure your compiler
is accessible and workable. Then type 'make' to start the installation.

Example from the author's computer:

	>> mex -setup
	(ps: Matlab will show the following messages to setup default compiler.)
	Please choose your compiler for building external interface (MEX) files:
	Would you like mex to locate installed compilers [y]/n? y
	Select a compiler:
	[1] Microsoft Visual C/C++ version 7.1 in C:\Program Files\Microsoft Visual Studio
	[0] None
	Compiler: 1
	Please verify your choices:
	Compiler: Microsoft Visual C/C++ 7.1
	Location: C:\Program Files\Microsoft Visual Studio
	Are these correct?([y]/n): y

	>> make

For a list of supported/compatible compilers for Matlab, please check
the following page:

http://www.mathworks.com/support/compilers/current_release/


BudgetedSVM Usage - Training
============================
In order to train the classification model, run in the Matlab prompt:

	>> model = budgetedsvm_train(label_vector, instance_matrix, parameter_string = '');

Inputs:
	label_vector		- label vector of size (NUM_POINTS x 1), a label set can include any integer
							representing a class, such as 0/1 or +1/-1 in the case of binary-class
							problems; in the case of multi-class problems it can be any set of integers
	instance_matrix		- instance matrix of size (NUM_POINTS x DIMENSIONALITY),
							where each row represents one example
	parameter_string	- parameters of the model, if not provided default empty string is assumed

Output:
	model				- structure that holds the trained model
	

Since the previous call to budgetedsvm_train() function requires the data set to be loaded to Matlab,
which can be infeasible for large data, we provide another variant of the call to the training procedure:

	>> budgetedsvm_train(train_file, model_file, parameter_string = '')

Inputs:
	train_file			- filename of .txt file containing training data set in LIBSVM format
	model_file			- filename of .txt file that will contain trained model
	parameter_string	- parameters of the model, defaults to empty string if not provided
	

Parameter string is of the same format for both versions, specified as follows:

	'-OPTION1 VALUE1 -OPTION2 VALUE2 ...'
	
	
	Following options are available; affected algorithm and default
	values in parentheses (algorithm not specified if option affects all):
	A - algorithm, which large-scale SVM approximation to use (2):
		0 - Pegasos
		1 - AMM batch
		2 - AMM online
		3 - LLSVM
		4 - BSGD
	D - dimensionality (faster loading if set, if omitted inferred from the data)
	B - limit on the number of weights per class in AMM, OR
			total SV set budget in BSGD, OR number of landmark points in LLSVM (50)
	L - lambda regularization parameter; high value -> less complex model (0.00010)
	b - bias term, if 0 no bias added (1.0)
	e - number of training epochs (AMM, BSGD; 5)
	s - number of subepochs (AMM batch; 1)
	k - pruning frequency, after how many observed examples is pruning done (AMM; 10000)
	c - pruning threshold; high value -> less complex model (AMM; 10.00)
	K - kernel function (0 - RBF; 1 - exponential, 2 - polynomial; 3 - linear,
			4 - sigmoid; 5 - user-defined) (LLSVM, BSGD; 0)
	g - RBF or exponential kernel width gamma (LLSVM, BSGD; 1/DIMENSIONALITY)
	d - polynomial kernel degree or sigmoid kernel slope (LLSVM, BSGD; 2.00)
	i - polynomial or sigmoid kernel intercept (LLSVM, BSGD; 1.00)
	m - budget maintenance in BSGD (0 - removal; 1 - merging, uses Gaussian kernel), OR
			landmark sampling strategy in LLSVM (0 - random; 1 - k-means; 2 - k-medoids) (1)

	z - training and test file are loaded in chunks so that the algorithm can 
			handle budget files on weaker computers; z specifies number of examples loaded in
			a single chunk of data, ONLY when inputs are .txt files (50000)
	w - model weights are split in chunks, so that the algorithm can handle
			highly dimensional data on weaker computers; w specifies number of dimensions stored
			in one chunk, ONLY when inputs are .txt files (1000)
	S - if set to 1 data is assumed sparse, if 0 data is assumed non-sparse, used to
			speed up kernel computations (default is 1 when percentage of non-zero
			features is less than 5%, and 0 when percentage is larger than 5%)
	r - randomize the algorithms; 1 to randomize, 0 not to randomize (1)
	v - verbose output: 1 to show the algorithm steps (epoch ended, training started, ...), 0 for quiet mode (0)
	--------------------------------------------


BudgetedSVM Usage - Testing
===========================
In order to evaluate the learned model, run in the Matlab prompt the following command:

	>> [error_rate, pred_labels, pred_scores] = budgetedsvm_predict(labelVector, instanceMatrix, model, parameter_string);

Inputs:
	labelVector			- label vector of the data set of size (NUM_POINTS x 1), a label can be any number							
							representing a class, such as 0/1, or +1/-1, or, in the
							case of multi-class problems, any set of integers
	instanceMatrix		- instance matrix of size (NUM_POINTS x DIMENSIONALITY),
							where each row represents one example
	model				- structure holding the model trained using budgetedsvm_train()
	parameter_string	- parameters of the model, if not provided default empty string is assumed

Output:
	error_rate			- error rate on the test set
	pred_labels			- vector of predicted labels of size (NUM_POINTS x 1)
	pred_scores			- vector of predicted scores of size (NUM_POINTS x 1)


Since the previous call to budgetedsvm_predict() function requires the data set to be loaded to Matlab,
we also provide another variant of the call to the testing procedure:

	>> [error_rate, pred_labels, pred_scores] = budgetedsvm_predict(test_file, model_file, parameter_string = '')

	Inputs:
		test_file			- filename of .txt file containing test data set in LIBSVM format
		model_file			- filename of .txt file containing model trained through budgetedsvm_train()
		parameter_string	- parameters of the model, defaults to empty string if not provided

	Output:
		error_rate			- error rate on the test set
		pred_labels			- vector of predicted labels of size (N x 1)
		pred_scores			- vector of predicted scores of size (NUM_POINTS x 1)


Parameter string is of the same format for both versions, specified as follows:

	'-OPTION1 VALUE1 -OPTION2 VALUE2 ...'

	The following options are available (default values in parentheses):
	z - the training and test file are loaded in chunks so that the algorithm can 
			handle budget files on weaker computers; z specifies number of examples loaded in
			a single chunk of data, ONLY when inputs are .txt files (50000)
	w - the model weight is split in parts, so that the algorithm can handle
			highly dimensional data on weaker computers; w specifies number of dimensions stored
			in one chunk, ONLY when inputs are .txt files (1000)
	S - if set to 1 data is assumed sparse, if 0 data assumed non-sparse, used to
			speed up kernel computations (default is 1 when percentage of non-zero
		    features is less than 5%, and 0 when percentage is larger than 5%)
	v - verbose output: 1 to show algorithm steps, 0 for quiet mode (0)
	--------------------------------------------

The function budgetedsvm_predict has three outputs. The first output,
accuracy, is a classification accuracy on the provided testing set.
The second output, pred_labels, is a vector of predicted labels.
The third output, pred_scores, is a vector of predicted scores. For
LLSVM the score represents the distance of test example from the separating
hyperplane; for AMM and BSGD this score represents difference between the
winning-class score and the score of a class that had the second-best score.


Other Utilities
===============
Matlab function libsvmread reads files in LIBSVM format: 

	>> [label_vector, instance_matrix] = libsvmread('data.txt'); 

Two outputs are labels and instances, which can then be used as inputs
of budgetedsvm_train or budgetedsvm_predict functions.

Matlab function libsvmwrite writes Matlab matrix to a file in LIBSVM format:

	>> libsvmwrite('data.txt', label_vector, instance_matrix);

The instance_matrix must be a sparse matrix (type must be double).
For 32-bit Matlab on Windows, pre-built binary files are ready in the directory `../matlab'.

These codes were prepared by Rong-En Fan and Kai-Wei Chang from National Taiwan University.


Examples
========
Here we show a simple example on how to train and test a classifier on the provided adult9a data set.
We first give an example where inputs to budgetedsvm_train and budgetedsvm_predict are data sets first
loaded into Matlab memory, and then provided to BudgetedSVM as Matlab variables:

	>> % first load the data into Matlab
	>> [a9a_label, a9a_inst] = libsvmread('../a9a_train.txt');

	>> % train an AMM model on the training set
	>> model = budgetedsvm_train(a9a_label, a9a_inst, '-A 2 -L 0.001 -v 1 -e 5');
	>> % evaluate the trained model on the training data
	>> [accuracy, predict_label, predict_score] = budgetedsvm_predict(a9a_label, a9a_inst, model, '-v 1');

	>> % train a LLSVM model on the training set
	>> model = budgetedsvm_train(a9a_label, a9a_inst, '-A 3 -L 0.1 -g 0.01 -B 100 -m 1 -v 1');
	>> % evaluate the trained model on the training data
	>> [accuracy, predict_label, predict_score] = budgetedsvm_predict(a9a_label, a9a_inst, model, '-v 1');

	>> % train a BSGD model on the training set
	>> model = budgetedsvm_train(a9a_label, a9a_inst, '-A 4 -g 0.01 -e 5 -L 0.0001 -B 200 -m 1 -v 1');
	>> % evaluate the trained model on the training data
	>> [accuracy, predict_label, predict_score] = budgetedsvm_predict(a9a_label, a9a_inst, model, '-v 1');

Next, we give an example when the inputs to budgetedsvm_train and budgetedsvm_predict are specified
as filenames of files containing training and test data sets, and the model is saved to .txt file:

	>> % train an AMM model on the training set
	>> budgetedsvm_train('../a9a_train.txt', '../a9a_model.txt', '-A 2 -L 0.001 -v 1 -e 5 -D 123');
	>> % evaluate the trained model on the testing data
	>> [accuracy, predict_label, predict_score] = budgetedsvm_predict('../a9a_test.txt', '../a9a_model.txt', '-v 1');

	>> % train a LLSVM model on the training set
	>> budgetedsvm_train('../a9a_train.txt', '../a9a_model.txt', '-A 3 -L 0.1 -g 0.01 -B 100 -m 1 -v 1 -D 123');
	>> % evaluate the trained model on the testing data
	>> [accuracy, predict_label, predict_score] = budgetedsvm_predict('../a9a_test.txt', '../a9a_model.txt', '-v 1');

	>> % train a BSGD model on the training set
	>> budgetedsvm_train('../a9a_train.txt', '../a9a_model.txt', '-A 4 -g 0.01 -e 5 -L 0.0001 -B 200 -m 1 -v 1 -D 123');
	>> % evaluate the trained model on the testing data
	>> [accuracy, predict_label, predict_score] = budgetedsvm_predict('../a9a_test.txt', '../a9a_model.txt', '-v 1');

After running the above examples in Matlab prompt, algorithms should return accuracy roughly around 15%.


Returned Model Structure
========================
The budgetedsvm_train function returns a model which can be used for future
classification. It is a structure organized as follows ["algorithm", "dimension",
"numClasses", " labels", " numWeights", " paramBias", " kernel", "kernelGammaParam", 
"kernelDegreeParam", "kernelInterceptParam", "model"]:

	- algorithm				: algorithm used to train a classification model
	- dimension				: dimensionality of the data set
	- numClasses			: number of classes in the data set
	- labels				: label of each class
	- numWeights			: number of weights for each class
	- paramBias				: bias term
	- kernel				: used kernel function
	- kernelGammaParam		: width of the Gaussian or exponential kernel
	- kernelDegreeParam		: degree of polynomial kernel or slope of sigmoid kernel
	- kernelInterceptParam	: coefficient of polynomial kernel or intercept of sigmoid kernel
	- model					: the learned model

In order to compress memory and to use the memory efficiently, we coded the model in the following way:

AMM online, AMM batch, and Pegasos: The model is stored as (("dimension" + 1) x "numWeights") matrix. The 
first element of each weight is the degradation of the weight, followed by values of the weight for each 
feature of the data set. If bias term is non-zero, then the final element of each weight corresponds to bias
term, and the matrix is of size (("dimension" + 2) x "numWeights"). By looking at "labels" and "numWeights"
members of Matlab structure we can find out which weights belong to which class. For example, first numWeights[0]
weights belong to labels[0] class, next numWeights[1] weights belong to labels[1] class, and so on.

BSGD: The model is stored as (("numClasses" + "dimension") x "numWeights") matrix. The first "numClasses" 
elements of each weight correspond to alpha parameters for each class, given in order of "labels" member of
the Matlab structure. This is followed by elements of the weights (or support vectors) for each feature of 
the data set.

LLSVM: The model is stored as ((1 + "dimension") x "numWeights") matrix. Each row corresponds to one landmark
point. The first element of each row corresponds to element of linear SVM hyperplane for that particular 
landmark point. This is followed by features of the landmark point in the original feature space. 

More details about the implementation can be found in BudgetedSVM implementation manual
"../doc/BudgetedSVM_reference_manual.pdf" or by openning "../doc/html/index.html" in your browser.


Additional Information
======================
The toolbox was written by Nemanja Djuric, Liang Lan, and Slobodan Vucetic
from the Department of Computer and Information Sciences, Temple University,
together with Zhuang Wang from IBM Global Business Services.
BudgetedSVM webpage is located at http://www.dabi.temple.edu/budgetedsvm/.

If you found our work useful, please cite us as:
Djuric, N., Lan, L., Vucetic, S., & Wang, Z. (2014). BudgetedSVM: A Toolbox for Scalable 
SVM Approximations. Journal of Machine Learning Research, 14, 3813-3817.

For any questions or comments, please contact Nemanja Djuric at <nemanja@temple.edu>.
Last updated: August 5th, 2014


Acknowledgments
===============
This work was supported by the National Science Foundation via grants IIS-0546155 and IIS-1117433.