---------------------------------------------------------------
-- BudgetedSVM: A Toolbox for Large-scale SVM Approximations --
---------------------------------------------------------------

BudgetedSVM is a simple, easy-to-use, and efficient software for large-scale,
non-linear, budgeted classification through approximations of SVM models.
This document explains the use of BudgetedSVM. For Matlab/Octave interface 
help please see "./matlab/readme.txt"

Please read the "./LICENSE.txt" license file before using BudgetedSVM.

Also, the toolbox includes two source files ("./src/libsvmwrite.c" and 
"./src/libsvmread.c") and uses some code from LibSVM package, please 
read "./COPYRIGHT.txt" before using BudgetedSVM for terms and conditions
pertaining to these parts of the toolbox.


Table of Contents
=================
- Table of Contents
- Version history
- The implemented algorithms
- Installation and Data Format
- "budgetedsvm-train" Usage
- "budgetedsvm-predict" Usage
- Examples
- Library Usage
- Matlab/Octave Interface
- Additional Information
- Acknowledgments


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
	- Added class scores to the outputs; please see '"budgetedsvm-predict" Usage' for
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
Nonlinear Classification" (AMM, found at "./doc/pdfs_of_algorithm_papers/AMM_paper.pdf")
*** "Scaling up Kernel SVM on Limited Resources: A Low-rank Linearization Approach"
(LLSVM, found at "./doc/pdfs_of_algorithm_papers/LLSVM_paper.pdf")
*** "Breaking the Curse of Kernelization: Budgeted Stochastic Gradient Descent for 
Large-Scale SVM Training" (BSGD, found at "./doc/pdfs_of_algorithm_papers/BSGD_paper.pdf")

For our BudgetedSVM paper, which gives a brief overview of the toolbox and summarizes its
main features, please see a PDF file at "./doc/pdfs_of_algorithm_papers/BudgetedSVM_paper.pdf".


Installation and Data Format
============================
On Unix systems, type `make' to build the `budgetedsvm-train' and `budgetedsvm-predict'
programs. Type 'make clean' to delete the generated files. Run the programs without 
arguments for description on how to use them.

We note that the authors have tested the toolbox on the following platform with success:
>> gcc -v
gcc version 4.6.3 (Ubuntu/Linaro 4.6.3-1ubuntu5)

Data format
-----------
The format of training and testing data file is as follows:

<label> <index1>:<value1> <index2>:<value2> ...
.
.
.

Each line contains an instance and is ended by a '\n' character.  For
classification, <label> is an integer indicating the class label
(multi-class is supported). See './a9a_train.txt' for an example. 
For further details about LIBSVM format please see the following webpage
http://www.csie.ntu.edu.tw/~cjlin/libsvm


`budgetedsvm-train' Usage
=========================
In order to get the detailed usage description, run the budgetedsvm-train function
without providing any arguments to obtain the following instructions:

Usage:
budgetedsvm-train [options] train_file [model_file]

Inputs:
options        - parameters of the model
train_file     - url of training file in LIBSVM format
model_file     - file that will hold a learned model
	--------------------------------------------
Options are specified in the following format:
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

	z - training and test file are loaded in chunks so that the algorithms can
			handle budget files on weaker computers; z specifies number of examples
			loaded in a single chunk of data (50000)
	w - model weights are split in chunks, so that the algorithm can handle
			highly dimensional data on weaker computers; w specifies number of
			dimensions stored in one chunk (1000)
	S - if set to 1 data is assumed sparse, if 0 data assumed non-sparse; used to
			speed up kernel computations (default is 1 when percentage of non-zero
			features is less than 5%, and 0 when percentage is larger than 5%)
	r - randomize the algorithms; 1 to randomize, 0 not to randomize (1)
	v - verbose output; 1 to show the algorithm steps, 0 for quiet mode (0)
	--------------------------------------------
 
The model is saved in a text file which has the following rows:
[ALGORITHM, DIMENSION, NUMBER OF CLASSES, LABELS, NUMBER OF WEIGHTS, BIAS TERM, KERNEL WIDTH, MODEL] 
In order to compress memory and to use the memory efficiently, we coded the model in the following way:

For AMM batch, AMM online, PEGASOS:	The model is stored so that each row of the text file corresponds 
to one weight. The first element of each weight is the class of the weight, followed by the degradation 
of the weight. The rest of the row corresponds to non-zero elements of the weight, given as 
feature_index:feature_value, in a standard LIBSVM format.

For BSGD: The model is stored so that each row corresponds to one support vector (or weight). The 
first elements of each weight correspond to alpha parameters for each class, given in order of 
"labels" member of the Matlab structure. However, since alpha can be equal to 0, we use LIBSVM format
to store alphas, as -class_index:class-specific_alpha, where we added '-' (minus sign) in front of 
the class index to differentiate between class indices and feature indices that follow. After the 
alphas, in the same row the elements of the weights (or support vectors) for each feature are given 
in LIBSVM format.

For LLSVM: The model is stored so that each row corresponds to one landmark point. The first element of 
each row corresponds to element of linear SVM hyperplane for that particular landmark point. This is 
followed by features of the landmark point in the original feature space of the data set in LIBSVM format. 


`budgetedsvm-predict' Usage
===========================
In order to get the detailed usage description, run the budgetedsvm-predict function
without providing any arguments to obtain the following instructions:

Usage:
budgetedsvm-predict [options] test_file model_file output_file

Inputs:
options        - parameters of the model
test_file      - url of test file in LIBSVM format
model_file     - file that holds a learned model
output_file    - url of file where output will be written
--------------------------------------------
Options are specified in the following format:
'-OPTION1 VALUE1 -OPTION2 VALUE2 ...'

The following options are available (default values in brackets):

	z - the training and test file are loaded in chunks so that the algorithm can
			handle budget files on weaker computers; z specifies number of examples
			loaded in a single chunk of data (50000)
	w - the model weight is split in parts, so that the algorithm can handle
			highly dimensional data on weaker computers; w specifies number of
			dimensions stored in one chunk (1000)
	S - if set to 1 data is assumed sparse, if 0 data assumed non-sparse, used to
			speed up kernel computations (default is 1 when percentage of non-zero
			features is less than 5%, and 0 when percentage is larger than 5%)
	o - if set to 1, the output file will contain not only the class predictions,
			but also tab-delimited scores of the winning class (0)
	v - verbose output; 1 to show algorithm steps, 0 for quiet mode (0)
--------------------------------------------

When setting the '-o' option, the scores should be interpreted as follows. For
LLSVM the score represents the distance of test example from the separating
hyperplane; for AMM and BSGD this score represents difference between the
winning-class score and the score of a class that had the second-best score.


Examples
========
Here is a simple example on how to train and test a classifier on the provided adult9a data set,
after budgetedsvm-train and budgetedsvm-predict functions were compiled by running 'make'.
Note that the train and predict programs will be created in the "./bin" folder, which is
why we need to append "bin/" to the calls to the functions. If the programs are run in Windows,
a user should use "\" (back-slash) instead of "/" (forward-slash) when specifying the path
to the programs in the command prompt. In all examples below, algorithms should return accuracy 
roughly around 15%.

	How to train and test AMM:
	--------------------------
>> bin/budgetedsvm-train -A 1 -e 5 -L 0.001 -B 20 -D 123 -v 1 -k 10000 -c 10 a9a_train.txt a9a_model.txt
>> bin/budgetedsvm-predict -v 1 a9a_test.txt a9a_model.txt a9a_preds.txt

The first command uses AMM batch ("-A 1") algorithm to train multi-hyperplane machine for 5 epochs ("-e 5"),
using regularization parameter lambda of 0.001 ("-L 0.001", larger values result in less complex model, or,
in other words, more regularization) and setting the maximum number of weights per class to 20 ("-B 20").
As adult9a data set is of dimensionality 123, we also write "-D 123", and choose verbose output ("-v 1")
which prints detailed steps of the algorithm. Lastly, we specify that pruning of weigths will be performed
every 10,000 iterations ("-k 10000", smaller values results in more aggressive pruning), and the pruning
parameter is set to 10 ("-c 10", larger values result in more aggressive pruning). If you did not specify
a name for the model file, it will be created such that suffix '.model' is appended to the filename of the
training file (note that we did include the model filename in the above example, namely 'a9a_model.txt').
The second command tests the model on testing data set, and prints the accuracy on the testing set while
saving the predictions to 'a9a_preds.txt'. We also set verbose output by writing "-v 1".

	How to train and test LLSVM:
	----------------------------
>> bin/budgetedsvm-train -A 3 -L 0.1 -K 0 -g 0.01 -B 100 -m 1 -D 123 -v 1 a9a_train.txt a9a_model.txt
>> bin/budgetedsvm-predict -v 1 a9a_test.txt a9a_model.txt a9a_predictions.txt

The first command uses LLSVM ("-A 3") algorithm to train a classification model, setting
regularization parameter to 0.1 ("-L 0.1", larger values result in less complex model, or,
in other words, more regularization), which result in higher regularization than in
the AMM case described above. We use Gaussian kernel ("-K 0") with kernel width 0.01 ("-g 0.01").
With "-B 100" option we set the budget, specifying that the model will comprise 100 landmark
points which will be chosen by running k-means on the loaded training data ("-m 1"). As adult9a
data set is of dimensionality 123, we also write "-D 123", and choose verbose output ("-v 1")
which prints detailed steps of the algorithm. If you did not specify a name for the model file,
it will be created such that suffix '.model' is appended to the filename of the training file
(note that we did include the model filename in the above example, namely 'a9a_model.txt').
The second command evaluates the model on testing data set, and prints the accuracy on the
testing set while saving the predictions to 'a9a_predictions.txt'.

	How to train and test BSGD:
	---------------------------
>> bin/budgetedsvm-train -A 4 -g 0.01 -e 5 -L 0.0001 -B 200 -m 1 -D 123 -v 1 a9a_train.txt a9a_model.txt
>> bin/budgetedsvm-predict -v 1 a9a_test.txt a9a_model.txt a9a_predictions.txt

The first command uses BSGD ("-A 4") algorithm to train a classification model for 5 epochs
("-e 5"), using learning rate lambda of 0.0001 ("-L 0.0001", larger values result in less complex model,
or, in other words, more regularization) and kernel width of Gaussian kernel of 0.01 ("-g 0.01").
 With "-B 200" option we specifying that the model will comprise 200 support vectors,
and in the case of a budget overflow two support vectors will be merged ("-m 1") to maintain
the budget. As adult9a data set is of dimensionality 123, we also write "-D 123", and choose
verbose output ("-v 1") which prints detailed steps of the algorithm. If you did not specify a name
for the model file, it will be created such that suffix '.model' is appended to the filename of the
training file (note that we did include the model filename in the above example, namely 'a9a_model.txt').
The second command evaluates the model on testing data set, and prints the accuracy
on the testing set while saving the predictions to 'a9a_predictions.txt'.


Library Usage
=============
See the "./doc/BudgetedSVM_reference_manual.pdf" or open "./doc/html/index.html" in your browser 
for details about the implementation.


Matlab/Octave Interface
=======================
Please check the README.txt file in the "./matlab" directory for more information.


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
This work was supported by the National Science Foundation via the grants IIS-0546155 and IIS-1117433.