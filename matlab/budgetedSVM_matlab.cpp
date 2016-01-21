/*
	\file budgetedSVM_matlab.cpp
	\brief Implements classes and functions that are used to communicate between C++ and Matlab environment.
*/
/*
	Copyright (c) 2013-2014 Nemanja Djuric, Liang Lan, Slobodan Vucetic, and Zhuang Wang
	All rights reserved.
	
	Author	:	Nemanja Djuric, with some parts influenced by LIBSVM C++ code
	Name	:	budgetedSVM_matlab.cpp
	Date	:	December 10th, 2012
	Desc.	:	Implements classes and functions that are used to communicate between C++ and Matlab environment.
	Version	:	v1.01
*/

#include "../Eigen/Dense"
using namespace Eigen;

#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>
using namespace std;

#include "mex.h"

#include "../src/budgetedSVM.h"
#include "../src/mm_algs.h"
#include "../src/bsgd.h"
#include "../src/llsvm.h"
#include "budgetedSVM_matlab.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

/*!
    \brief Expected number of fields in Matlab structure from which the model is loaded.
*/
#define NUM_OF_RETURN_FIELD 11

/*!
    \brief Labels of the fields in Matlab structure.
*/
static const char *fieldNames[] = 
{
	"algorithm",
	"dimension",
	"numClasses",
	"labels",
	"numWeights",
	"paramBias",
	"kernel",
	"kernelGammaParam",
	"kernelDegreeParam",
	"kernelInterceptParam",
	"model",
};
	
/* \fn static int getAlgorithm(const mxArray *matlabStruct)
	\brief Get algorithm from the trained model stored in Matlab structure.
	\param [in] matlabStruct Pointer to Matlab structure.
	\return -1 if error, otherwise returns algorithm code from the model file.
*/
int budgetedModelMatlab::getAlgorithm(const mxArray *matlabStruct)
{	
	if (mxGetNumberOfFields(matlabStruct) != NUM_OF_RETURN_FIELD)
		return -1;
	
	// get algorithm
	return (int)(*(mxGetPr(mxGetFieldByNumber(matlabStruct, 0, 0))));
}

/* \fn void budgetedDataMatlab::readDataFromMatlab(const mxArray *labelVec, const mxArray *instanceMat, parameters *param)
	\brief Loads the data from Matlab.
	\param [in] labelVec Vector of labels.
	\param [in] instanceMat Matrix of data points, each row is a single data point.
	\param [in] param The parameters of the algorithm.
*/	
void budgetedDataMatlab::readDataFromMatlab(const mxArray *labelVec, const mxArray *instanceMat, parameters *param)
{
	long start = clock();
	unsigned int i, j, k, labelVectorRowNum;
	long unsigned int low, high;
	mwIndex *ir, *jc;
	double *samples, *labels;
	bool labelFound;
	mxArray *instanceMatCol; // transposed instance sparse matrix
	bool warningWritten = false;
	char str[256];
	
	// otherwise load the data, given below	
	// transpose instance matrix
	{
		mxArray *prhs[1], *plhs[1];
		prhs[0] = mxDuplicateArray(instanceMat);
		if (mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
			mexErrMsgTxt("Error: Cannot transpose training instance matrix.\n");
		
		instanceMatCol = plhs[0];
		mxDestroyArray(prhs[0]);
	}

	// each column is one instance
	labels = mxGetPr(labelVec);
	samples = mxGetPr(instanceMatCol);

	// get number of instances
	labelVectorRowNum = (int)mxGetM(labelVec);
	if (labelVectorRowNum != (int)mxGetN(instanceMatCol))
		mexErrMsgTxt("Length of label vector does not match number of instances.\n");

	// set the dimension and the number of data points
	this->N = labelVectorRowNum;
	if ((*param).DIMENSION == 0)
	{
		// it is 0 when loading training data set
		this->dimensionHighestSeen = (*param).DIMENSION = (int)mxGetM(instanceMatCol);
		if ((*param).BIAS_TERM != 0.0)
			(*param).DIMENSION++;
		
		// set KERNEL_GAMMA_PARAM here if needed, done during loading of training set
		if ((*param).KERNEL_GAMMA_PARAM == 0.0)
			(*param).KERNEL_GAMMA_PARAM = 1.0 / (double) (*param).DIMENSION;
	}
	else
	{
		// it is non-zero only when loading testing data set, no need to set GAMMA parameter as it is read from the model structure from Matlab
		this->dimensionHighestSeen = (*param).DIMENSION;	
		
		// if bias term is non-zero, then the actual dimensionality of data is one less than DIMENSION
		if ((*param).BIAS_TERM != 0.0)
			this->dimensionHighestSeen--;					
	}

	// allocate memory for labels
	this->al = new (nothrow) unsigned char[this->N];
	if (this->al == NULL)
		mexErrMsgTxt("Memory allocation error (readDataFromMatlab function)! Restart MATLAB and try again.");
	
	if (mxIsSparse(instanceMat))
	{
		ir = mxGetIr(instanceMatCol);
		jc = mxGetJc(instanceMatCol);
		
		j = 0;				
		for (i = 0; i < labelVectorRowNum; i++)
		{
			// where the instance starts
			ai.push_back(j);
			
			// get yLabels, if label not seen before add it in the label array
			labelFound = false;
			for (k = 0; k < (int) yLabels.size(); k++)
			{
				if (yLabels[k] == (int)labels[i])
				{
					al[i] = k;
					labelFound = true;
					break;
				}
			}
			if (!labelFound)
			{
				if (isTrainingSet)
				{
					yLabels.push_back((int)labels[i]);
					al[i] = (unsigned char) (yLabels.size() - 1);
				}
				else
				{
					// so unseen label detected during testing phase, issue a warning
					if (!warningWritten)
					{
						sprintf(str, "Warning: Testing label '%d' detected that was not seen during training.\n", (int)labels[i]);
						mexPrintf(str);
						mexEvalString("drawnow;");
						
						warningWritten = true;
					}
					
					// give an example a label index that can never be predicted
					al[i] = (unsigned char) yLabels.size();
				}
			}
			
			// get features
			low = (int) jc[i], high = (int) jc[i + 1];
			for (k = low; k < high; k++)
			{
				// we save the actual feature no. in aj, and the value in an
				aj.push_back((int) ir[k] + 1);
				an.push_back((float) samples[k]);
				j++;					
			}
		}
	}
	else
	{
		j = 0;
		low = 0;
		for (i = 0; i < labelVectorRowNum; i++)
		{
			// where the instance starts
			ai.push_back(j);
			
			// get yLabels, if label not seen before add it in the label array
			labelFound = false;
			for (k = 0; k < (int) yLabels.size(); k++)
			{
				if (yLabels[k] == (int) labels[i])
				{
					al[i] = k;
					labelFound = true;
					break;
				}
			}
			if (!labelFound)
			{
				if (isTrainingSet)
				{
					yLabels.push_back((int)labels[i]);
					al[i] = (unsigned char) (yLabels.size() - 1);
				}
				else
				{
					// so unseen label detected during testing phase, issue a warning
					if (!warningWritten)
					{
						sprintf(str, "Warning: Testing label '%d' detected that was not seen during training.\n", (int) labels[i]);
						mexPrintf(str);
						mexEvalString("drawnow;");
						
						warningWritten = true;
					}
					
					// give an example a label index that can never be predicted
					al[i] = (unsigned char) yLabels.size();
				}
			}
			
			// get features
			for (k = 0; k < (int)mxGetM(instanceMatCol); k++)
			{
				if (samples[low] != 0.0)
				{
					// we save the actual feature no. in aj, and the value in an
					aj.push_back(k + 1);
					an.push_back((float) samples[low]);
					j++;
				}
				low++;
			}
		}
	}
	
	// if very beginning, just allocate memory for assignments
	if (keepAssignments)
		this->assignments = new (nothrow) unsigned int[this->N];
	
	loadTime += (clock() - start);
};

/* \fn void budgetedModelMatlab::saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param)
	\brief Save the trained model to Matlab, by creating Matlab structure.
	\param [out] plhs Pointer to Matlab output.
	\param [in] yLabels Vector of possible labels.
	\param [in] param The parameters of the algorithm.
*/
void budgetedModelMatlabAMM::saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param)
{
	unsigned int i, j, numWeights = 0, cnt;
	double *ptr;
	mxArray *returnModel, **rhs;
	int outID = 0;
	
	rhs = (mxArray **) mxMalloc(sizeof(mxArray *) * NUM_OF_RETURN_FIELD);

	// algorithm type
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->ALGORITHM;
	outID++;
	
	// dimension
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->DIMENSION;
	outID++;
	
	// number of classes
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = (double) (*yLabels).size();
	outID++;
	
	// labels
	rhs[outID] = mxCreateDoubleMatrix((*yLabels).size(), 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	for (i = 0; i < (*yLabels).size(); i++)
		ptr[i] = (*yLabels)[i];
	outID++;
	
	// total number of weights
	rhs[outID] = mxCreateDoubleMatrix((*yLabels).size(), 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	for (i = 0; i < (*modelMM).size(); i++)
	{
		ptr[i] = (double) (*modelMM)[i].size();
		numWeights += (unsigned int) (*modelMM)[i].size();
	}
	outID++;
	
	// bias param
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->BIAS_TERM;
	outID++;
	
	// kernel choice
	rhs[outID] = mxCreateDoubleMatrix(0, 0, mxREAL);
	outID++;
	
	// kernel width gamma
	rhs[outID] = mxCreateDoubleMatrix(0, 0, mxREAL);
	outID++;
	
	// kernel degree/slope param
	rhs[outID] = mxCreateDoubleMatrix(0, 0, mxREAL);
	outID++;
	
	// kernel intercept param
	rhs[outID] = mxCreateDoubleMatrix(0, 0, mxREAL);
	outID++;
	
	// weights
	int irIndex, nonZeroElement;
	mwIndex *ir, *jc;
	
	// find how many non-zero elements there are
	nonZeroElement = 0;
	for (i = 0; i < (*modelMM).size(); i++) 
	{
		for (j = 0; j < (*modelMM)[i].size(); j++)
		{
			for (unsigned int k = 0; k < (*param).DIMENSION; k++)              // for every feature
			{
				if ((*((*modelMM)[i][j]))[k] != 0.0)
					nonZeroElement++;
			}
		}
	}
	
	// +1 is for degradation of AMM algorithms, it will be the first number in the row representing a weight
	if (param->ALGORITHM == PEGASOS)
		rhs[outID] = mxCreateSparse(param->DIMENSION, numWeights, nonZeroElement, mxREAL);
	else if ((param->ALGORITHM == AMM_BATCH) || (param->ALGORITHM == AMM_ONLINE))
		rhs[outID] = mxCreateSparse(param->DIMENSION + 1, numWeights, nonZeroElement + numWeights, mxREAL);
	ir = mxGetIr(rhs[outID]);
	jc = mxGetJc(rhs[outID]);
	ptr = mxGetPr(rhs[outID]);
	jc[0] = irIndex = cnt = 0;		
	for (i = 0; i < (*modelMM).size(); i++)
	{
		for (j = 0; j < (*modelMM)[i].size(); j++)
		{
			int xIndex = 0;
			
			// this adds degradation to the beginning of a vector, more compact 
			if ((param->ALGORITHM == AMM_BATCH) || (param->ALGORITHM == AMM_ONLINE))
			{
				ir[irIndex] = 0; 
				ptr[irIndex] = (*modelMM)[i][j]->getDegradation();
				irIndex++, xIndex++;
			}
			
			// add the actual features
			for (unsigned int k = 0; k < (*param).DIMENSION; k++)              // for every feature
			{
				if ((*((*modelMM)[i][j]))[k] != 0.0)
				{
					if ((param->ALGORITHM == AMM_BATCH) || (param->ALGORITHM == AMM_ONLINE))
						ir[irIndex] = k + 1;
					else if (param->ALGORITHM == PEGASOS)
						ir[irIndex] = k;
					ptr[irIndex] = (*((*modelMM)[i][j]))[k];
					irIndex++, xIndex++;
				}
			}
			jc[cnt + 1] = jc[cnt] + xIndex;
			cnt++;
		}			
	}
	// commented, since now it is appended to the weight matrix
	/*// degradations
	cnt = 0;
	rhs[outID] = mxCreateDoubleMatrix(numWeights, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	for (i = 0; i < (*modelMM).size(); i++)
		for (j = 0; j < (*modelMM)[i].size(); j++)
			ptr[cnt++] = (*modelMM)[i][j]->degradation;
	outID++;*/
		
	/* Create a struct matrix contains NUM_OF_RETURN_FIELD fields */
	returnModel = mxCreateStructMatrix(1, 1, NUM_OF_RETURN_FIELD, fieldNames);
	
	/* Fill struct matrix with input arguments */
	for(i = 0; i < NUM_OF_RETURN_FIELD; i++)
		mxSetField(returnModel, 0, fieldNames[i], mxDuplicateArray(rhs[i]));
	
	plhs[0] = returnModel;
	mxFree(rhs);
}

/* \fn bool budgetedModelMatlabAMM::loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg);
	\brief Loads the trained model from Matlab structure.
	\param [in] matlabStruct Pointer to Matlab structure.
	\param [in] yLabels Vector of possible labels.
	\param [in] param The parameters of the algorithm.
	\param [out] msg Error message, if error encountered.
	\return Returns false if error encountered, otherwise true.
*/
bool budgetedModelMatlabAMM::loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg)
{
	int i, j, numOfFields, numClasses, currClass, classCounter;
	double *ptr;
	int id = 0;
	mxArray **rhs;
	vector <unsigned int> numWeights;
	double sqrNorm;
	
	numOfFields = mxGetNumberOfFields(matlabStruct);
	if (numOfFields != NUM_OF_RETURN_FIELD) 
	{
		*msg = "number of return fields is not correct";
		return false;
	}
	rhs = (mxArray **) mxMalloc(sizeof(mxArray *) * numOfFields);

	for (i = 0; i < numOfFields; i++)
		rhs[i] = mxGetFieldByNumber(matlabStruct, 0, i);	
	
	// algorithm
	ptr = mxGetPr(rhs[id]);
	param->ALGORITHM = (unsigned int)ptr[0];
	id++;
	
	// dimension
	ptr = mxGetPr(rhs[id]);
	param->DIMENSION = (unsigned int)ptr[0];
	id++;
	
	// numClasses
	ptr = mxGetPr(rhs[id]);
	numClasses = (unsigned int)ptr[0];
	id++;
	
	// labels
	if (mxIsEmpty(rhs[id]) == 0)
	{
		ptr = mxGetPr(rhs[id]);
		for(i = 0; i < numClasses; i++)
		{
			(*yLabels).push_back((int)ptr[i]);
			
			// add to model empty weight vector for each class
			vector <budgetedVectorAMM*> tempV;
			(*modelMM).push_back(tempV);
		}
	}
	id++;
	
	// numWeights
	if (mxIsEmpty(rhs[id]) == 0)
	{
		ptr = mxGetPr(rhs[id]);
		for(i = 0; i < numClasses; i++)
		{
			numWeights.push_back((int)ptr[i]);
		}
	}
	id++;
	
	// bias term
	ptr = mxGetPr(rhs[id]);
	param->BIAS_TERM = (double)ptr[0];
	id++;
	
	// kernel choice, just skip
	id++;
	
	// kernel width gamma
	id++;
	
	// kernel degree/slope param
	id++;
	
	// kernel intercept param
	id++;
	
	// weights
	int sr, sc;
	mwIndex *ir, *jc;

	sr = (int)mxGetN(rhs[id]);
	sc = (int)mxGetM(rhs[id]);

	ptr = mxGetPr(rhs[id]);
	ir = mxGetIr(rhs[id]);
	jc = mxGetJc(rhs[id]);
	
	// weights are in columns
	currClass = classCounter = 0;
	for (i = 0; i < sr; i++)
	{
		int low = (int)jc[i], high = (int)jc[i + 1];			
		budgetedVectorAMM *eNew = new budgetedVectorAMM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
		sqrNorm = 0.0;
		
		for (j = low; j < high; j++)
		{
			if (param->ALGORITHM == PEGASOS)
				((*eNew)[(int)ir[j]]) = (float)ptr[j];
			else if ((param->ALGORITHM == AMM_BATCH) || (param->ALGORITHM == AMM_ONLINE))
			{
				if (j == low)
					eNew->setDegradation(ptr[j]);
				else
				{
					((*eNew)[(int)ir[j] - 1]) = (float)ptr[j];
					sqrNorm += (ptr[j] * ptr[j]);
				}
			}
		}
		eNew->setSqrL2norm(sqrNorm);
		(*modelMM)[currClass].push_back(eNew);			
		eNew = NULL;
		
		// increment weight counter and check if new class is starting
		if (++classCounter == numWeights[currClass])
		{
			classCounter = 0;
			currClass++;		
		}
	}
	id++;
	
	mxFree(rhs);
	return true;
}

/* \fn void budgetedModelMatlabBSGD::saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param)
	\brief Save the trained model to Matlab, by creating Matlab structure.
	\param [out] plhs Pointer to Matlab output.
	\param [in] yLabels Vector of possible labels.
	\param [in] param The parameters of the algorithm.
*/
void budgetedModelMatlabBSGD::saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param)
{
	unsigned int i, j, numWeights = 0, cnt;
	double *ptr;
	mxArray *returnModel, **rhs;
	int outID = 0;
	
	rhs = (mxArray **) mxMalloc(sizeof(mxArray *) * NUM_OF_RETURN_FIELD);

	// algorithm type
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->ALGORITHM;
	outID++;
	
	// dimension
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->DIMENSION;
	outID++;
	
	// number of classes
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = (double) (*yLabels).size();
	outID++;
	
	// labels
	rhs[outID] = mxCreateDoubleMatrix((*yLabels).size(), 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	for (i = 0; i < (*yLabels).size(); i++)
		ptr[i] = (*yLabels)[i];
	outID++;
	
	// total number of weights
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = (double) (*modelBSGD).size();
	numWeights = (unsigned int) (*modelBSGD).size();
	outID++;
	
	// bias param
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->BIAS_TERM;
	outID++;
	
	// kernel choice
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->KERNEL;
	outID++;
	
	// kernel width gamma
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->KERNEL_GAMMA_PARAM;
	outID++;
	
	// kernel degree/slope param
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->KERNEL_DEGREE_PARAM;
	outID++;
	
	// kernel intercept param
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->KERNEL_COEF_PARAM;
	outID++;
	
	// weights, different for MM algorithms, BSGD and LLSVM
	int irIndex, nonZeroElement;
	mwIndex *ir, *jc;
	
	// find how many non-zero elements there are
	nonZeroElement = 0;
	for (i = 0; i < (*modelBSGD).size(); i++) 
	{
		// count non-zero features
		for (j = 0; j < (*param).DIMENSION; j++)
		{
			if ((*((*modelBSGD)[i]))[j] != 0.0)
				nonZeroElement++;
		}
		
		// count non-zero alphas also
		for (j = 0; j < (*yLabels).size(); j++)
		{
			if ((*((*modelBSGD)[i])).alphas[j] != 0.0)
				nonZeroElement++;
		}
	}

	//  +(*yLabels).size() is for the alpha parameters of each BSGD weight
	rhs[outID] = mxCreateSparse(param->DIMENSION + (*yLabels).size(), numWeights, nonZeroElement, mxREAL);
	ir = mxGetIr(rhs[outID]);
	jc = mxGetJc(rhs[outID]);
	ptr = mxGetPr(rhs[outID]);
	jc[0] = irIndex = cnt = 0;	
	for (i = 0; i < (*modelBSGD).size(); i++)
	{
		int xIndex = 0;
		
		// this adds alpha weights to the beginning of a vector, more compact
		for (j = 0; j < (*yLabels).size(); j++)
		{
			if ((*((*modelBSGD)[i])).alphas[j] != 0.0)
			{
				ir[irIndex] = j; 
				ptr[irIndex] = (*((*modelBSGD)[i])).alphas[j];
				irIndex++, xIndex++;
			}
		}
		
		// add the actual features
		for (j = 0; j < (*param).DIMENSION; j++)              // for every feature
		{
			if ((*((*modelBSGD)[i]))[j] != 0.0)
			{
				ir[irIndex] = j + (*yLabels).size();		// shift it to accomodate alpha weights
				ptr[irIndex] = (*((*modelBSGD)[i]))[j];
				irIndex++, xIndex++;
			}
		}
		jc[cnt + 1] = jc[cnt] + xIndex;
		cnt++;
	}
	
	/* Create a struct matrix contains NUM_OF_RETURN_FIELD fields */
	returnModel = mxCreateStructMatrix(1, 1, NUM_OF_RETURN_FIELD, fieldNames);
	
	/* Fill struct matrix with input arguments */
	for(i = 0; i < NUM_OF_RETURN_FIELD; i++)
		mxSetField(returnModel, 0, fieldNames[i], mxDuplicateArray(rhs[i]));
	
	plhs[0] = returnModel;
	mxFree(rhs);
}

/* \fn bool budgetedModelMatlabBSGD::loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg);
	\brief Loads the trained model from Matlab structure.
	\param [in] matlabStruct Pointer to Matlab structure.
	\param [in] yLabels Vector of possible labels.
	\param [in] param The parameters of the algorithm.
	\param [out] msg Error message, if error encountered.
	\return Returns false if error encountered, otherwise true.
*/
bool budgetedModelMatlabBSGD::loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg)
{
	int i, j, numOfFields, numClasses, currClass, classCounter;
	double *ptr;
	int id = 0;
	mxArray **rhs;
	vector <unsigned int> numWeights;
	double sqrNorm;
	
	numOfFields = mxGetNumberOfFields(matlabStruct);
	if (numOfFields != NUM_OF_RETURN_FIELD) 
	{
		*msg = "number of return fields is not correct";
		return false;
	}
	rhs = (mxArray **) mxMalloc(sizeof(mxArray *) * numOfFields);

	for (i = 0; i < numOfFields; i++)
		rhs[i] = mxGetFieldByNumber(matlabStruct, 0, i);	
	
	// algorithm
	ptr = mxGetPr(rhs[id]);
	param->ALGORITHM = (unsigned int)ptr[0];
	id++;
	
	// dimension
	ptr = mxGetPr(rhs[id]);
	param->DIMENSION = (unsigned int)ptr[0];
	id++;
	
	// numClasses
	ptr = mxGetPr(rhs[id]);
	numClasses = (unsigned int)ptr[0];
	id++;
	
	// labels
	if (mxIsEmpty(rhs[id]) == 0)
	{
		ptr = mxGetPr(rhs[id]);
		for(i = 0; i < numClasses; i++)
			(*yLabels).push_back((int)ptr[i]);
	}
	id++;
	
	// numWeights, just skip
	id++;
	
	// bias term
	ptr = mxGetPr(rhs[id]);
	param->BIAS_TERM = (double)ptr[0];
	id++;
	
	// kernel choice, just skip
	ptr = mxGetPr(rhs[id]);
	param->KERNEL = (unsigned int) ptr[0];
	id++;
	
	// kernel width gamma
	ptr = mxGetPr(rhs[id]);
	param->KERNEL_GAMMA_PARAM = (double)ptr[0];
	id++;
	
	// kernel degree/slope param
	ptr = mxGetPr(rhs[id]);
	param->KERNEL_DEGREE_PARAM = (double)ptr[0];
	id++;
	
	// kernel intercept param
	ptr = mxGetPr(rhs[id]);
	param->KERNEL_COEF_PARAM = (double)ptr[0];
	id++;
	
	// weights
	int sr, sc;
	mwIndex *ir, *jc;

	sr = (int)mxGetN(rhs[id]);
	sc = (int)mxGetM(rhs[id]);

	ptr = mxGetPr(rhs[id]);
	ir = mxGetIr(rhs[id]);
	jc = mxGetJc(rhs[id]);
	
	// weights are in columns
	currClass = classCounter = 0;
	for (i = 0; i < sr; i++)
	{
		int low = (int)jc[i], high = (int)jc[i + 1];			
		budgetedVectorBSGD *eNew = new budgetedVectorBSGD((*param).DIMENSION, (*param).CHUNK_WEIGHT, numClasses);
		sqrNorm = 0.0;
		
		for (j = low; j < high; j++)
		{
			if ((unsigned int)ir[j] < (*yLabels).size())
			{
				// get alpha values
				eNew->alphas[(int)ir[j]] = ptr[j];
			}
			else
			{
				// get features
				((*eNew)[(int)ir[j] - (int) (*yLabels).size()]) = (float)ptr[j];
				sqrNorm += (ptr[j] * ptr[j]);
			}
		}
		eNew->setSqrL2norm(sqrNorm);
		(*modelBSGD).push_back(eNew);			
		eNew = NULL;
	}
	id++;
	
	mxFree(rhs);
	return true;
}

/* \fn void budgetedModelMatlabLLSVM::saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param)
	\brief Save the trained model to Matlab, by creating Matlab structure.
	\param [out] plhs Pointer to Matlab output.
	\param [in] yLabels Vector of possible labels.
	\param [in] param The parameters of the algorithm.
*/
void budgetedModelMatlabLLSVM::saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param)
{
	unsigned int i, j, numWeights = 0, cnt;
	double *ptr;
	mxArray *returnModel, **rhs;
	int outID = 0;
	
	rhs = (mxArray **) mxMalloc(sizeof(mxArray *) * NUM_OF_RETURN_FIELD);

	// algorithm type
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->ALGORITHM;
	outID++;
	
	// dimension
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->DIMENSION;
	outID++;
	
	// number of classes
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = (double) (*yLabels).size();
	outID++;
	
	// labels
	rhs[outID] = mxCreateDoubleMatrix((*yLabels).size(), 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	for (i = 0; i < (*yLabels).size(); i++)
		ptr[i] = (*yLabels)[i];
	outID++;
	
	// total number of weights
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = (double) (*modelLLSVMlandmarks).size();
	numWeights = (unsigned int) (*modelLLSVMlandmarks).size();
	outID++;
	
	// bias param
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->BIAS_TERM;
	outID++;
	
	// kernel choice
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->KERNEL;
	outID++;
	
	// kernel width gammma
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->KERNEL_GAMMA_PARAM;
	outID++;
	
	// kernel degree/slope param
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->KERNEL_DEGREE_PARAM;
	outID++;
	
	// kernel intercept param
	rhs[outID] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[outID]);
	ptr[0] = param->KERNEL_COEF_PARAM;
	outID++;
	
	// weights, different for MM algorithms, BSGD and LLSVM
	int irIndex, nonZeroElement;
	mwIndex *ir, *jc;
	
	// find how many non-zero elements there are
	nonZeroElement = 0;
	for (i = 0; i < (*modelLLSVMlandmarks).size(); i++) 
	{
		// count non-zero features
		for (j = 0; j < (*param).DIMENSION; j++)
		{
			if ((*((*modelLLSVMlandmarks)[i]))[j] != 0.0)
				nonZeroElement++;
		}
		
		// count all elements of modelLLSVMmatrixW also
		nonZeroElement += (numWeights * numWeights);
		
		// count linear SVM length also
		nonZeroElement += numWeights;
	}

	//  +(*yLabels).size() is for the alpha parameters of each BSGD weight
	rhs[outID] = mxCreateSparse(param->DIMENSION + numWeights + 1, numWeights, nonZeroElement, mxREAL);
	ir = mxGetIr(rhs[outID]);
	jc = mxGetJc(rhs[outID]);
	ptr = mxGetPr(rhs[outID]);
	jc[0] = irIndex = cnt = 0;	
	for (i = 0; i < (*modelLLSVMlandmarks).size(); i++)
	{
		int xIndex = 0;
		
		// this adds alpha weights to the beginning of a vector, more compact
		ir[irIndex] = 0; 
		ptr[irIndex] = modelLLSVMweightVector(i, 0);
		irIndex++, xIndex++;
		
		// this adds row of modelLLSVMmatrixW next, more compact
		for (j = 0; j < numWeights; j++)
		{
			ir[irIndex] = j + 1;		// shift it to accomodate linear weight
			ptr[irIndex] = modelLLSVMmatrixW(i, j);
			irIndex++, xIndex++;
		}
		
		// add the actual features
		for (j = 0; j < (*param).DIMENSION; j++)
		{
			if ((*((*modelLLSVMlandmarks)[i]))[j] != 0.0)
			{
				ir[irIndex] = j + numWeights + 1;		// shift it to accomodate linear weight and modelLLSVMmatrixW
				ptr[irIndex] = (*((*modelLLSVMlandmarks)[i]))[j];
				irIndex++, xIndex++;
			}
		}
		jc[cnt + 1] = jc[cnt] + xIndex;
		cnt++;
	}
	
	/* Create a struct matrix contains NUM_OF_RETURN_FIELD fields */
	returnModel = mxCreateStructMatrix(1, 1, NUM_OF_RETURN_FIELD, fieldNames);
	
	/* Fill struct matrix with input arguments */
	for(i = 0; i < NUM_OF_RETURN_FIELD; i++)
		mxSetField(returnModel, 0, fieldNames[i], mxDuplicateArray(rhs[i]));
	
	plhs[0] = returnModel;
	mxFree(rhs);
}

/* \fn bool budgetedModelMatlabLLSVM::loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg);
	\brief Loads the trained model from Matlab structure.
	\param [in] matlabStruct Pointer to Matlab structure.
	\param [in] yLabels Vector of possible labels.
	\param [in] param The parameters of the algorithm.
	\param [out] msg Error message, if error encountered.
	\return Returns false if error encountered, otherwise true.
*/
bool budgetedModelMatlabLLSVM::loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg)
{
	unsigned int i, j, numOfFields, numClasses;
	double *ptr, sqrNorm;
	int id = 0;
	mxArray **rhs;
	
	numOfFields = mxGetNumberOfFields(matlabStruct);
	if (numOfFields != NUM_OF_RETURN_FIELD) 
	{
		*msg = "Number of return fields is not correct.";
		return false;
	}
	rhs = (mxArray **) mxMalloc(sizeof(mxArray *) * numOfFields);

	for (i = 0; i < numOfFields; i++)
		rhs[i] = mxGetFieldByNumber(matlabStruct, 0, i);	
	
	// algorithm
	ptr = mxGetPr(rhs[id]);
	param->ALGORITHM = (unsigned int)ptr[0];
	id++;
	
	// dimension
	ptr = mxGetPr(rhs[id]);
	param->DIMENSION = (unsigned int)ptr[0];
	id++;
	
	// numClasses
	ptr = mxGetPr(rhs[id]);
	numClasses = (unsigned int)ptr[0];
	id++;
	
	// labels
	if (mxIsEmpty(rhs[id]) == 0)
	{
		ptr = mxGetPr(rhs[id]);
		for(i = 0; i < numClasses; i++)
			(*yLabels).push_back((int)ptr[i]);
	}
	id++;
	
	// numWeights
	ptr = mxGetPr(rhs[id]);
	param->BUDGET_SIZE = (unsigned int) ptr[0];
	id++;
	
	// bias term
	ptr = mxGetPr(rhs[id]);
	param->BIAS_TERM = (double) ptr[0];
	id++;
	
	// kernel choice
	ptr = mxGetPr(rhs[id]);
	param->KERNEL = (unsigned int) ptr[0];
	id++;
	
	// kernel width gamma
	ptr = mxGetPr(rhs[id]);
	param->KERNEL_GAMMA_PARAM = (double) ptr[0];
	id++;
	
	// kernel degree/slope param
	ptr = mxGetPr(rhs[id]);
	param->KERNEL_DEGREE_PARAM = (double) ptr[0];
	id++;
	
	// kernel intercept param
	ptr = mxGetPr(rhs[id]);
	param->KERNEL_COEF_PARAM = (double) ptr[0];
	id++;
	
	// weights
	unsigned int sr, sc;
	mwIndex *ir, *jc;

	sr = (int)mxGetN(rhs[id]);
	sc = (int)mxGetM(rhs[id]);

	ptr = mxGetPr(rhs[id]);
	ir = mxGetIr(rhs[id]);
	jc = mxGetJc(rhs[id]);
	
	// allocate memory for model
	modelLLSVMmatrixW.resize((*param).BUDGET_SIZE, (*param).BUDGET_SIZE);
	modelLLSVMweightVector.resize((*param).BUDGET_SIZE, 1);
	
	// weight-vectors are in columns
	for (i = 0; i < sr; i++)
	{
		unsigned int low = (int)jc[i], high = (int)jc[i + 1];			
		budgetedVectorLLSVM *eNew = new budgetedVectorLLSVM((*param).DIMENSION, (*param).CHUNK_WEIGHT);
		sqrNorm = 0.0;
		
		// get the linear weight
		modelLLSVMweightVector(i, 0) = ptr[low];
		
		// get the modelLLSVMmatrixW
		for (j = low + 1; j < low + (*param).BUDGET_SIZE + 1; j++)
			modelLLSVMmatrixW(i, j - low - 1) = ptr[j];
		
		// get the features
		for (j = low + (*param).BUDGET_SIZE + 1; j < high; j++)
		{
			((*eNew)[(int)ir[j] - (*param).BUDGET_SIZE - 1]) = (float)ptr[j];
			sqrNorm += (ptr[j] * ptr[j]);
		}
		eNew->setSqrL2norm(sqrNorm);
		(*modelLLSVMlandmarks).push_back(eNew);			
		eNew = NULL;
	}
	id++;
	
	mxFree(rhs);
	return true;
}

/* \fn void printStringMatlab(const char *s) 
	\brief Prints string to Matlab, used to modify callback found in budgetedSVM.cpp
	\param [in] s Text to be printed.
*/
void printStringMatlab(const char *s) 
{
	mexPrintf(s);
	mexEvalString("drawnow;");
}

/* \fn void printErrorStringMatlab(const char *s) 
	\brief Prints error string to Matlab, used to modify callback found in budgetedSVM.cpp
	\param [in] s Text to be printed.
*/
void printErrorStringMatlab(const char *s) 
{
	mexErrMsgTxt(s);
}

/* \fn void fakeAnswer(mxArray *plhs[])
	\brief Returns empty matrix to Matlab.
	\param [out] plhs Pointer to Matlab output.
*/
void fakeAnswer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

/* \fn void printUsageMatlab(bool trainingPhase)
	\brief Prints to standard output the instructions on how to use the software.
	\param [in] trainingPhase Indicator if training or testing phase.
*/
void printUsageMatlab(bool trainingPhase, parameters *param)
{
	if (trainingPhase)
	{
		mexPrintf("\n\tUsage:\n");
		mexPrintf("\t\tmodel = budgetedsvm_train(label_vector, instance_matrix, parameter_string = '')\n\n");
		mexPrintf("\tInputs:\n");
		mexPrintf("\t\tlabel_vector\t\t- label vector of size (NUM_POINTS x 1), a label set can include any integer\n");
		mexPrintf("\t\t\t\t\t              representing a class, such as 0/1 or +1/-1 in the case of binary-class\n");
		mexPrintf("\t\t\t\t\t              problems; in the case of multi-class problems it can be any set of integers\n");
		mexPrintf("\t\tinstance_matrix\t\t- instance matrix of size (NUM_POINTS x DIMENSIONALITY),\n");
		mexPrintf("\t\t\t\t                  where each row represents one example\n");
		mexPrintf("\t\tparameter_string\t- parameters of the model, defaults to empty string if not provided\n\n");
		mexPrintf("\tOutput:\n");
		mexPrintf("\t\tmodel\t\t\t\t- structure that holds the learned model\n\n");
		mexPrintf("\t--------------------------------------------\n\n");
		mexPrintf("\tIf the data set cannot be fully loaded to Matlab, another variant can be used:\n");
		mexPrintf("\t\tbudgetedsvm_train(train_file, model_file, parameter_string = '')\n\n");
		mexPrintf("\tInputs:\n");
		mexPrintf("\t\ttrain_file\t\t\t- filename of .txt file containing training data set in LIBSVM format\n");
		mexPrintf("\t\tmodel_file\t\t\t- filename of .txt file that will contain trained model\n");
		mexPrintf("\t\tparameter_string\t- parameters of the model, defaults to empty string if not provided\n\n");
		mexPrintf("\t--------------------------------------------\n\n");
		mexPrintf("\tParameter string is of the following format:\n");
		mexPrintf("\t'-OPTION1 VALUE1 -OPTION2 VALUE2 ...'\n\n");
		mexPrintf("\tFollowing options are available; affected algorithm and default values\n");
		mexPrintf("\tare given in parentheses (algorithm not specified if option affects all):\n");
		mexPrintf("\t A - algorithm, which large-scale SVM to use (%d):\n", (*param).ALGORITHM);
		mexPrintf("\t\t     0 - Pegasos\n");
		mexPrintf("\t\t     1 - AMM batch\n");
		mexPrintf("\t\t     2 - AMM online\n");
		mexPrintf("\t\t     3 - LLSVM\n");
		mexPrintf("\t\t     4 - BSGD\n");
		mexPrintf("\t D - dimensionality (faster loading if set, if omitted inferred from the data)\n");
		mexPrintf("\t B - limit on the number of weights per class in AMM, OR\n");
		mexPrintf("\t\t     total SV set budget in BSGD, OR number of landmark points in LLSVM (%d)\n", (*param).BUDGET_SIZE);
		mexPrintf("\t L - lambda regularization parameter; high value -> less complex model (%.5f)\n", (*param).LAMBDA_PARAM);
		mexPrintf("\t b - bias term, if 0 no bias added (%.1f)\n", (*param).BIAS_TERM);
		mexPrintf("\t e - number of training epochs (AMM, BSGD; %d)\n", (*param).NUM_EPOCHS);
		mexPrintf("\t s - number of subepochs (AMM batch; %d)\n", (*param).NUM_SUBEPOCHS);
		mexPrintf("\t k - pruning frequency, after how many observed examples is pruning done (AMM; %d)\n", (*param).K_PARAM);
		mexPrintf("\t c - pruning threshold; high value -> less complex model (AMM; %.2f)\n", (*param).C_PARAM);
		mexPrintf("\t K - kernel function (0 - RBF; 1 - exponential, 2 - polynomial; 3 - linear, \n");
		mexPrintf("\t\t     4 - sigmoid; 5 - user-defined) (LLSVM, BSGD; %d)\n", (*param).KERNEL);
		mexPrintf("\t g - RBF or exponential kernel width gamma (LLSVM, BSGD; 1/DIMENSIONALITY)\n");
		mexPrintf("\t d - polynomial kernel degree or sigmoid kernel slope (LLSVM, BSGD; %.2f)\n", (*param).KERNEL_DEGREE_PARAM);
		mexPrintf("\t i - polynomial or sigmoid kernel intercept (LLSVM, BSGD; %.2f)\n", (*param).KERNEL_COEF_PARAM);		
		mexPrintf("\t m - budget maintenance in BSGD (0 - removal; 1 - merging, uses Gaussian kernel), OR\n");
		mexPrintf("\t\t     landmark sampling strategy in LLSVM (0 - random; 1 - k-means; 2 - k-medoids) (%d)\n\n", (*param).MAINTENANCE_SAMPLING_STRATEGY);
		
		mexPrintf("\t z - training and test file are loaded in chunks so that the algorithm can \n");
		mexPrintf("\t\t     handle budget files on weaker computers; z specifies number of examples loaded in\n");
		mexPrintf("\t\t     a single chunk of data, ONLY when inputs are .txt files (%d)\n", (*param).CHUNK_SIZE);
		mexPrintf("\t w - model weights are split in chunks, so that the algorithm can handle\n");
		mexPrintf("\t\t     highly dimensional data on weaker computers; w specifies number of dimensions stored\n");
		mexPrintf("\t\t     in one chunk, ONLY when inputs are .txt files (%d)\n", (*param).CHUNK_WEIGHT);
		mexPrintf("\t S - if set to 1 data is assumed sparse, if 0 data is assumed non-sparse, used to\n");
		mexPrintf("\t\t     speed up kernel computations (default is 1 when percentage of non-zero\n");
		mexPrintf("\t\t     features is less than 5%%, and 0 when percentage is larger than 5%%)\n");
		mexPrintf("\t r - randomize the algorithms; 1 to randomize, 0 not to randomize (%d)\n", (*param).RANDOMIZE);
		mexPrintf("\t v - verbose output: 1 to show the algorithm steps (epoch ended, training started, ...), 0 for quiet mode (%d)\n", (*param).VERBOSE);
		mexPrintf("\t--------------------------------------------\n");
		mexPrintf("\tInstructions on how to convert data to and from the LIBSVM format can be found on <a href=\"http://www.csie.ntu.edu.tw/~cjlin/libsvm/\">LIBSVM website</a>.\n");
	}
	else
	{
		mexPrintf("\n\tUsage:\n");
		mexPrintf("\t\t[error_rate, pred_labels, pred_scores] = budgetedsvm_predict(label_vector, instance_matrix, model, parameter_string = '')\n\n");
		mexPrintf("\tInputs:\n");
		mexPrintf("\t\tlabel_vector\t\t- label vector of size (NUM_POINTS x 1), a label set can include any integer\n");
		mexPrintf("\t\t\t\t\t              representing a class, such as 0/1 or +1/-1 in the case of binary-class\n");
		mexPrintf("\t\t\t\t\t              problems; in the case of multi-class problems it can be any set of integers\n");
		mexPrintf("\t\tinstance_matrix\t\t- instance matrix of size (NUM_POINTS x DIMENSIONALITY),\n");
		mexPrintf("\t\t\t\t                  where each row represents one example\n");
		mexPrintf("\t\tmodel\t\t\t\t- structure holding the model learned through budgetedsvm_train()\n");
		mexPrintf("\t\tparameter_string\t- parameters of the model, defaults to empty string if not provided\n\n");
		mexPrintf("\tOutput:\n");
		mexPrintf("\t\terror_rate\t\t\t- error rate on the test set\n");
		mexPrintf("\t\tpred_labels\t\t\t- vector of predicted labels of size (NUM_POINTS x 1)\n");
		mexPrintf("\t\tpred_scores\t\t\t- vector of predicted scores of size (NUM_POINTS x 1)\n\n");		
		mexPrintf("\t--------------------------------------------\n\n");
		
		mexPrintf("\tIf the data set cannot be fully loaded to Matlab, another variant can be used:\n");
		mexPrintf("\t\t[error_rate, pred_labels, pred_scores] = budgetedsvm_predict(test_file, model_file, parameter_string = '')\n\n");
		mexPrintf("\tInputs:\n");
		mexPrintf("\t\ttest_file\t\t\t- filename of .txt file containing test data set in LIBSVM format\n");
		mexPrintf("\t\tmodel_file\t\t\t- filename of .txt file containing model trained through budgetedsvm_train()\n");
		mexPrintf("\t\tparameter_string\t- parameters of the model, defaults to empty string if not provided\n\n");
		mexPrintf("\tOutput:\n");
		mexPrintf("\t\terror_rate\t\t\t- error rate on the test set\n");
		mexPrintf("\t\tpred_labels\t\t\t- vector of predicted labels of size (NUM_POINTS x 1)\n");
		mexPrintf("\t\tpred_scores\t\t\t- vector of predicted scores of size (NUM_POINTS x 1)\n\n");
		
		mexPrintf("\t--------------------------------------------\n\n");
		mexPrintf("\tParameter string is of the following format:\n");
		mexPrintf("\t'-OPTION1 VALUE1 -OPTION2 VALUE2 ...'\n\n");
		mexPrintf("\tThe following options are available (default values in parentheses):\n");
		mexPrintf("\tz - the training and test file are loaded in chunks so that the algorithm can\n");
		mexPrintf("\t\t    handle budget files on weaker computers; z specifies number of examples loaded in\n");
		mexPrintf("\t\t    a single chunk of data, ONLY when inputs are .txt files (%d)\n", (*param).CHUNK_SIZE);
		mexPrintf("\tw - the model weight is split in parts, so that the algorithm can handle\n");
		mexPrintf("\t\t    highly dimensional data on weaker computers; w specifies number of dimensions stored\n");
		mexPrintf("\t\t    in one chunk, ONLY when inputs are .txt files (%d)\n", (*param).CHUNK_WEIGHT);
		mexPrintf("\tS - if set to 1 data is assumed sparse, if 0 data is assumed non-sparse, used to\n");
		mexPrintf("\t\t    speed up kernel computations (default is 1 when percentage of non-zero\n");
		mexPrintf("\t\t    features is less than 5%%, and 0 when percentage is larger than 5%%)\n");
		mexPrintf("\tv - verbose output: 1 to show algorithm steps, 0 for quiet mode (%d)\n", (*param).VERBOSE);
		mexPrintf("\t--------------------------------------------\n");
		mexPrintf("\tInstructions on how to convert data to and from the LIBSVM format can be found on <a href=\"http://www.csie.ntu.edu.tw/~cjlin/libsvm/\">LIBSVM website</a>.\n");		
	}
}

/* \fn void parseInputMatlab(parameters *param, const char *paramString, bool trainingPhase, const char *inputFileName, const char *modelFileName)
	\brief Parses the user input and modifies parameter settings as necessary.
	\param [out] param Parameter object modified by user input.
	\param [in] paramString User-provided parameter string, can be NULL in which case default parameters are used.
	\param [in] trainingPhase Indicator if training or testing phase.
	\param [in] inputFileName Filename of the file that holds the data.
	\param [in] modelFileName Filename of the file that will hold the model (if trainingPhase = 1), or that holds the model (if trainingPhase = 0).
*/
void parseInputMatlab(parameters *param, const char *paramString, bool trainingPhase, const char *inputFileName, const char *modelFileName)
{	
	int pos = 0, tempPos = 0, len;
	char str[256];
	vector <char> option;
	vector <float> value;
	FILE *pFile = NULL;
	
	if (paramString == NULL)
		len = 0;
	else
		len = (int) strlen(paramString);
	
	// check if the input data file exists only if input data filename is provided
	if (inputFileName)
	{
		if (!readableFileExists(inputFileName))
		{
			sprintf(str, "Can't open input file %s!\n", inputFileName);
			mexErrMsgTxt(str);
		}
	}
	
	while (pos < len)
	{
		if (paramString[pos++] == '-')
		{
			option.push_back(paramString[pos]);
			pos += 2;

			tempPos = 0;
			while ((paramString[pos] != ' ') && (paramString[pos] != '\0'))
			{
				str[tempPos++] = paramString[pos++];
			}
			str[tempPos++] = '\0';
			value.push_back((float) atof(str));
		}
	}
		
	if (trainingPhase)
	{
		// check if the model file exists only if model filename is provided
		if (modelFileName)
		{
			pFile = fopen(modelFileName, "w");
			if (pFile == NULL)
			{
				sprintf(str, "Can't create model file %s!\n", modelFileName);
				mexErrMsgTxt(str);
			}
			else
			{
				fclose(pFile);
				pFile = NULL;
			}
		}
		
		// modify parameters
		for (unsigned int i = 0; i < option.size(); i++)
		{
			switch (option[i])
			{
				case 'A':
					(*param).ALGORITHM = (unsigned int) value[i];
					if ((*param).ALGORITHM > 4)
					{
						sprintf(str, "Input parameter '-A %d' out of bounds!\nRun 'budgetedsvm_train()' for help.", (*param).ALGORITHM);
						mexErrMsgTxt(str);
					}
					break;
				case 'e':
					(*param).NUM_EPOCHS = (unsigned int) value[i];
					break;
				case 's':
					(*param).NUM_SUBEPOCHS = (unsigned int) value[i];
					break;
				case 'k':
					(*param).K_PARAM = (unsigned int) value[i];
					break;
				case 'c':
					(*param).C_PARAM = value[i];
					if ((*param).C_PARAM <= 0.0)
					{
						sprintf(str, "Input parameter '-c' should be a positive real number!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
				case 'L':
					(*param).LAMBDA_PARAM = (double) value[i];
					if ((*param).LAMBDA_PARAM <= 0.0)
					{
						sprintf(str, "Input parameter '-L' should be a positive real number!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
				
				case 'K':
					(*param).KERNEL = (unsigned int) value[i];
					break;
				
				case 'g':
					(*param).KERNEL_GAMMA_PARAM = (long double) value[i];
					if ((*param).KERNEL_GAMMA_PARAM <= 0.0)
					{
						sprintf(str, "Input parameter '-g' should be a positive real number!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
					
				case 'd':
					(*param).KERNEL_DEGREE_PARAM = (double) value[i];
					if ((*param).KERNEL_DEGREE_PARAM <= 0.0)
					{
						sprintf(str, "Input parameter '-d' should be a positive real number!\nRun 'budgetedsvm-train()' for help.\n");
						mexErrMsgTxt(str);
					}
					break;
				
				case 'i':
					(*param).KERNEL_COEF_PARAM = (double) value[i];
					break;
				
				case 'm':
					(*param).MAINTENANCE_SAMPLING_STRATEGY = (unsigned int) value[i];
					break;  
				
				case 'b':
					(*param).BIAS_TERM = (double) value[i];
					break;
				case 'v':
					(*param).VERBOSE = (value[i] != 0);
					break;
				case 'r':
					(*param).RANDOMIZE = (value[i] != 0);
					break;
				case 'B':
					(*param).BUDGET_SIZE = (unsigned int) value[i];
					if ((*param).BUDGET_SIZE < 1)
					{
						sprintf(str, "Input parameter '-B' should be a positive integer!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
				case 'D':
					// a user explicitly assigns dimensionality only if data set is given in .txt file, otherwise dimensionality is found directly from Matlab, no need for a user to specify it
					if (inputFileName)
					{
						(*param).DIMENSION = (unsigned int) value[i];
					}
					else
					{
						//sprintf(str, "Warning, if data loaded to Matlab no need to set '-D %d' option.\nRun 'budgetedsvm_train()' for help.\n", (int) value[i]);
						//mexPrintf(str);
					}
					break;				
				
				case 'z':
					(*param).CHUNK_SIZE = (unsigned int) value[i];
					if ((*param).CHUNK_SIZE < 1)
					{
						sprintf(str, "Input parameter '-z' should be an integer larger than 0!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
				case 'w':
					(*param).CHUNK_WEIGHT = (unsigned int) value[i];
					if ((*param).CHUNK_WEIGHT < 1)
					{
						sprintf(str, "Input parameter '-w' should be an integer larger than 0!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
				case 'S':
					(*param).VERY_SPARSE_DATA = (unsigned int) (value[i] != 0);
					break;

				default:
					sprintf(str, "Error, unknown input parameter '-%c'!\nRun 'budgetedsvm_train()' for help.", option[i]);
					mexErrMsgTxt(str);
					break;
			}
		}
		
		// for BSGD, when we use merging budget maintenance strategy then only Gaussian kernel can be used,
		//	due to the nature of merging; here check if user specified some other kernel while merging
		if (((*param).ALGORITHM == BSGD) && ((*param).KERNEL != KERNEL_FUNC_GAUSSIAN) && ((*param).MAINTENANCE_SAMPLING_STRATEGY == BUDGET_MAINTAIN_MERGE))
		{
			mexPrintf("Warning, BSGD with merging strategy can only use Gaussian kernel!\nKernel function switched to Gaussian.\n");
			(*param).KERNEL = KERNEL_FUNC_GAUSSIAN;
		}
		
		// check the MAINTENANCE_SAMPLING_STRATEGY validity
		if ((*param).ALGORITHM == LLSVM)
		{
			if ((*param).MAINTENANCE_SAMPLING_STRATEGY > 2)
			{
				// 0 - random removal, 1 - k-means, 2 - k-medoids
				sprintf(str, "Error, unknown input parameter '-m %d'!\nRun 'budgetedsvm_train()' for help.\n", (*param).MAINTENANCE_SAMPLING_STRATEGY);
				mexErrMsgTxt(str);
			}
		}
		else if ((*param).ALGORITHM == BSGD)
		{
			if ((*param).MAINTENANCE_SAMPLING_STRATEGY > 1)
			{
				// 0 - smallest removal, 1 - merging
				sprintf(str, "Error, unknown input parameter '-m %d'!\nRun 'budgetedsvm_train()' for help.\n", (*param).MAINTENANCE_SAMPLING_STRATEGY);
				mexErrMsgTxt(str);
			}
		}
		
		// no bias term for LLSVM and BSGD functions
		if (((*param).ALGORITHM == LLSVM) || ((*param).ALGORITHM == BSGD))
			(*param).BIAS_TERM = 0.0;
		
		if ((*param).VERBOSE)
		{
			mexPrintf("*** Training started with the following parameters:\n");
			switch ((*param).ALGORITHM)
			{
				case PEGASOS:
					mexPrintf("Algorithm \t\t\t\t: Pegasos\n");
					break;
				case AMM_ONLINE:
					mexPrintf("Algorithm \t\t\t\t: AMM online\n");
					break;
				case AMM_BATCH:
					mexPrintf("Algorithm \t\t\t\t: AMM batch\n");
					break;
				case BSGD:
					mexPrintf("Algorithm \t\t\t\t\t: BSGD\n");
					break;
				case LLSVM:
					mexPrintf("Algorithm \t\t\t\t\t: LLSVM\n");
					break;
			}
			
			if (((*param).ALGORITHM == PEGASOS) || ((*param).ALGORITHM == AMM_BATCH) || ((*param).ALGORITHM == AMM_ONLINE))
			{
				mexPrintf("Lambda parameter \t\t: %f\n", (*param).LAMBDA_PARAM);
				mexPrintf("Bias term \t\t\t\t: %f\n", (*param).BIAS_TERM);
				if ((*param).ALGORITHM != PEGASOS)
				{
					mexPrintf("Pruning frequency k \t: %d\n", (*param).K_PARAM);
					mexPrintf("Pruning threshold c \t: %f\n", (*param).C_PARAM);
					mexPrintf("Num. weights per class\t: %d\n", (*param).BUDGET_SIZE);
					mexPrintf("Number of epochs \t\t: %d\n\n", (*param).NUM_EPOCHS);
				}
				else
					mexPrintf("\n");
			}
			else if (((*param).ALGORITHM == BSGD) || ((*param).ALGORITHM == LLSVM))
			{
				if ((*param).ALGORITHM == BSGD)
				{
					mexPrintf("Number of epochs \t\t\t: %d\n", (*param).NUM_EPOCHS);
					mexPrintf("Size of the budget \t\t\t: %d\n", (*param).BUDGET_SIZE);
					if ((*param).MAINTENANCE_SAMPLING_STRATEGY == BUDGET_MAINTAIN_REMOVE)
						mexPrintf("Maintenance strategy \t\t: smallest removal)n");
					else if ((*param).MAINTENANCE_SAMPLING_STRATEGY == BUDGET_MAINTAIN_MERGE)
						mexPrintf("Maintenance strategy \t\t: merging\n");
					else
						mexErrMsgTxt("Error, unknown budget maintenance set. Run 'budgetedsvm_train()' for help.\n");
					
					mexPrintf("Lambda regularization param.: %f\n", (*param).LAMBDA_PARAM);
				}
				else if ((*param).ALGORITHM == LLSVM)
				{
					switch ((*param).MAINTENANCE_SAMPLING_STRATEGY)
					{
						case LANDMARK_SAMPLE_RANDOM:
							mexPrintf("Landmark sampling \t\t\t: random sampling\n");
							break;
							
						case LANDMARK_SAMPLE_KMEANS:
							mexPrintf("Landmark sampling \t\t\t: k-means initialization\n");
							break;
							
						case LANDMARK_SAMPLE_KMEDOIDS:
							mexPrintf("Landmark sampling \t\t\t: k-medoids initialization\n");
							break;
							
						default:
							mexErrMsgTxt("Error, unknown landmark sampling set. Run 'budgetedsvm_train()' for help.\n");
							break;
					}
					mexPrintf("Number of landmark points \t: %d\n", (*param).BUDGET_SIZE);
					mexPrintf("Lambda regularization param.: %f\n", (*param).LAMBDA_PARAM);
				}
				
				// print common parameters
				switch ((*param).KERNEL)
				{
					case KERNEL_FUNC_GAUSSIAN:
						mexPrintf("Gaussian kernel used \t\t: K(x, y) = exp(-0.5 * gamma * ||x - y||^2)\n");
						if ((*param).KERNEL_GAMMA_PARAM != 0.0)
						{
							sprintf(str, "Gaussian kernel width \t\t: %f\n\n", (*param).KERNEL_GAMMA_PARAM);
							mexPrintf(str);
						}
						else
							mexPrintf("Gaussian kernel width \t\t: 1 / DIMENSIONALITY\n\n");
						break;
					
					case KERNEL_FUNC_EXPONENTIAL:
						mexPrintf("Exponential kernel used \t: K(x, y) = exp(-0.5 * gamma * ||x - y||)\n");
						if ((*param).KERNEL_GAMMA_PARAM != 0.0)
						{
							sprintf(str, "Exponential kernel width \t: %f\n\n", (*param).KERNEL_GAMMA_PARAM);
							mexPrintf(str);
						}
						else
							mexPrintf("Exponential kernel width \t: 1 / DIMENSIONALITY\n\n");
						break;
					
					case KERNEL_FUNC_POLYNOMIAL:
						sprintf(str, "Polynomial kernel used \t\t: K(x, y) = (x^T * y + %.2f)^%.2f\n\n", (*param).KERNEL_COEF_PARAM, (*param).KERNEL_DEGREE_PARAM);
						mexPrintf(str);
						break;
						
					case KERNEL_FUNC_SIGMOID:
						sprintf(str, "Sigmoid kernel used \t\t: K(x, y) = tanh(%.2f * x^T * y + %.2f)\n\n", (*param).KERNEL_DEGREE_PARAM, (*param).KERNEL_COEF_PARAM);
						mexPrintf(str);
						break;
					
					case KERNEL_FUNC_LINEAR:
						mexPrintf("Linear kernel used \t\t\t: K(x, y) = (x^T * y)\n\n");
						break;
					
					case KERNEL_FUNC_USER_DEFINED:
						mexPrintf("User-defined kernel function used.\n\n");
						break;
					
					default:
						sprintf(str, "Input parameter '-K %d' out of bounds!\nRun 'budgetedsvm_train()' for help.\n", (*param).KERNEL);
						mexErrMsgTxt(str);
						break;
				}
			}
			mexEvalString("drawnow;");
		}
		
		// if inputs to training phase are .txt files, then also increase dimensionality due to added bias term, and update KERNEL_GAMMA_PARAM if not set by a user;
		//	NOTE that we do not execute this part if inputs are Matlab variables, as we still do not know the dimensionality, therefore BIAS_TERM and
		//	KERNEL_GAMMA_PARAM are adjusted in budgetedDataMatlab::readDataFromMatlab() function, after we find out the dimensionality of the considered data set
		if (inputFileName)
		{
			// signal error if a user wants to use an RBF kernel, but didn't specify either data dimension or kernel width
			if ((((*param).ALGORITHM == LLSVM) || ((*param).ALGORITHM == BSGD)) && (((*param).KERNEL == KERNEL_FUNC_GAUSSIAN) || ((*param).KERNEL == KERNEL_FUNC_EXPONENTIAL)))
			{
				if (((*param).KERNEL_GAMMA_PARAM == 0.0) && ((*param).DIMENSION == 0))
				{
					// this means that both RBF kernel width and dimension were not set by the user in the input string to the toolbox
					//	since in this case the default value of RBF kernel is 1/dimensionality, report error to the user
					mexErrMsgTxt("Error, RBF kernel in use, please set either kernel width or dimensionality!\nRun 'budgetedsvm_train()' for help.\n");
				}
			}
			
			// increase dimensionality if bias term included
			if ((*param).BIAS_TERM != 0.0)
			{
				(*param).DIMENSION++;
			}
			
			// set gamma to default value of dimensionality
			if ((*param).KERNEL_GAMMA_PARAM == 0.0)
				(*param).KERNEL_GAMMA_PARAM = 1.0 / (double) (*param).DIMENSION;
		} 
	}
	else
	{
		// check if the model file exists only if model filename is provided
		if (modelFileName)
		{
			if (!readableFileExists(modelFileName))
			{
				sprintf(str, "Can't open model file %s!\n", modelFileName);
				mexErrMsgTxt(str);
			}
		}
		
		// modify parameters
		for (unsigned int i = 0; i < option.size(); i++)
		{
			switch (option[i])
			{
				/*case 'p':
					(*param).SAVE_PREDS = (value[i] != 0);
					break;*/
				case 'v':
					(*param).VERBOSE = (value[i] != 0);
					break;			
				
				case 'z':
					(*param).CHUNK_SIZE = (unsigned int) value[i];					
					if ((*param).CHUNK_SIZE < 1)
					{
						sprintf(str, "Input parameter '-z' should be an integer larger than 0!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
				case 'w':
					(*param).CHUNK_WEIGHT = (unsigned int) value[i];
					if ((*param).CHUNK_WEIGHT < 1)
					{
						sprintf(str, "Input parameter '-w' should be an integer larger than 0!\nRun 'budgetedsvm_train()' for help.");
						mexErrMsgTxt(str);
					}
					break;
				case 'S':
					(*param).VERY_SPARSE_DATA = (unsigned int) (value[i] != 0);
					break;

				default:
					sprintf(str, "Error, unknown input parameter '-%c'!\nRun 'budgetedsvm_predict()' for help.", option[i]);
					mexErrMsgTxt(str);
					break;
			}
		}
		
		/*if ((*param).VERBOSE)
		{
			mexPrintf("\n*** Testing with the following parameters:\n");
			switch ((*param).ALGORITHM)
			{
				case PEGASOS:
					mexPrintf("Algorithm: \t\t\t\tPEGASOS\n");
					break;
				case AMM_ONLINE:
					mexPrintf("Algorithm: \t\t\t\tAMM online\n");
					break;
				case AMM_BATCH:
					mexPrintf("Algorithm: \t\t\t\tAMM batch\n");
					break;
				case BSGD:
					mexPrintf("Algorithm: \t\t\t\tBSGD\n");
					break;
			}
			
			if (((*param).ALGORITHM == PEGASOS) || ((*param).ALGORITHM == AMM_BATCH) || ((*param).ALGORITHM == AMM_ONLINE))
			{
				mexPrintf("Bias term: \t\t\t\t%f\n\n", (*param).BIAS_TERM);
			}
			else if ((*param).ALGORITHM == BSGD)
			{
				mexPrintf("Gaussian kernel width: \t%f\n\n", (*param).GAMMA_PARAM);	
			}
			mexEvalString("drawnow;");
		}*/
	}
	
	setPrintErrorStringFunction(&printErrorStringMatlab);	
	if ((*param).VERBOSE)
		setPrintStringFunction(&printStringMatlab);
	else
		setPrintStringFunction(NULL);
}
