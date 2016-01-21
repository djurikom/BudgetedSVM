/*
	\file budgetedsvm_predict.cpp
	\brief Source file implementing commmand-prompt interface for testing phase of budgetedSVM toolbox.
*/
/*
	Copyright (c) 2013-2014 Nemanja Djuric, Liang Lan, Slobodan Vucetic, and Zhuang Wang
	All rights reserved.
	
	Author	:	Nemanja Djuric
	Name	:	budgetedsvm_train.cpp
	Date	:	November 19th, 2012
	Desc.	:	Source file implementing commmand-prompt interface for testing phase of budgetedSVM toolbox.
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
#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#include "../src/budgetedSVM.h"
#include "../src/mm_algs.h"
#include "../src/bsgd.h"
#include "../src/llsvm.h"
#include "budgetedSVM_matlab.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
	vector <int> yLabels;
	vector <int> labels;
	vector <float> scores;
	float errRate = -1.0;	
	const char *errorMessage;	
	parameters param;
	char tempStr[256];
	
	char inputParamString[1024];
	char testFileName[1024];
	char modelFileName[1024];
	
	if (nrhs == 0)
	{
		printUsageMatlab(false, &param);
		fakeAnswer(plhs);
		return;
	}
	
	// check if a user wants the class scores
	if (nlhs == 3)
		param.OUTPUT_SCORES = 1;
    
	// here we first check what kind of inputs are we working with, .txt files or Matlab variables; we will check that
	// 	by checking the first input parameter, if it is a double or sparse vector then it's a variant with Matlab variables
	if ((!mxIsDouble(prhs[0])) && (!mxIsSparse(prhs[0])))
	{
		// there need to be 3 inputs params, filename of the test file, matlab struct of a model, and parameter string.
		if ((nrhs != 2) && (nrhs != 3))
			mexErrMsgTxt("Error, wrong number of input parameters!\nRun 'budgetedsvm_predict()' for help.");
		
		budgetedData *testData = NULL;
		budgetedModel *model = NULL;
		
		// input 1 - testing file
		mxGetString(prhs[0], testFileName, mxGetN(prhs[0]) + 1);
		
		// input 2 - model file
		mxGetString(prhs[1], modelFileName, mxGetN(prhs[1]) + 1);
		
		// input 3 - parameters		
		if (nrhs == 2)
		{
			parseInputMatlab(&param, NULL, false, testFileName, modelFileName);
		}
		else
		{
			mxGetString(prhs[2], inputParamString, mxGetN(prhs[2]) + 1);
			parseInputMatlab(&param, inputParamString, false, testFileName, modelFileName);
		}
		param.ALGORITHM = budgetedModel::getAlgorithm(modelFileName);
		
		// init random number generator
		srand((unsigned)time(NULL));
		
		// initialize test data and run trained model, return Matlab outputs in plhs pointer
		switch (param.ALGORITHM)
		{
			case PEGASOS:
			case AMM_BATCH:
			case AMM_ONLINE:
				// input 3 - Matlab model, read and populate c++ model
				/*model = new budgetedModelMatlabAMM;
				if (model->loadFromMatlabStruct(prhs[1], &yLabels, &param, &errorMessage) == 0)
				{
					mexPrintf("Error, can't read model: %s\n", errorMessage);
					delete model;
					return;
				}*/
				model = new budgetedModelAMM;
				if (!model->loadFromTextFile(modelFileName, &yLabels, &param))
				{
					delete model;
					sprintf(tempStr, "Error: can't read model from file %s.\n", modelFileName);
					mexErrMsgTxt(tempStr);
				}
				testData = new budgetedData(testFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE, false, &yLabels);

				if (param.OUTPUT_SCORES)
					errRate = predictAMM(testData, &param, (budgetedModelAMM*) model, &labels, &scores);
				else
					errRate = predictAMM(testData, &param, (budgetedModelAMM*) model, &labels);
				//errRate = predictAMM(testData, &param, (budgetedModelMatlabAMM*) model, &labels);
				break;
				
			case LLSVM:
				// input 3 - Matlab model, read and populate c++ model
				/*model = new budgetedModelMatlabLLSVM;
				if (model->loadFromMatlabStruct(prhs[1], &yLabels, &param, &errorMessage) == 0)
				{
					mexPrintf("Error, can't read model: %s\n", errorMessage);
					delete model;
					return;
				}*/
				model = new budgetedModelLLSVM;
				if (!model->loadFromTextFile(modelFileName, &yLabels, &param))
				{
					delete model;
					sprintf(tempStr, "Error: can't read model from file %s.\n", modelFileName);
					mexErrMsgTxt(tempStr);
				}
				testData = new budgetedData(testFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE, false, &yLabels);

				if (param.OUTPUT_SCORES)
					errRate = predictLLSVM(testData, &param, (budgetedModelLLSVM*) model, &labels, &scores);
				else
					errRate = predictLLSVM(testData, &param, (budgetedModelLLSVM*) model, &labels);
				//errRate = predictLLSVM(testData, &param, (budgetedModelMatlabLLSVM*) model, &labels);
				break;
				
			case BSGD:
				// input 3 - Matlab model, read and populate c++ model
				/*model = new budgetedModelMatlabBSGD;
				if (model->loadFromMatlabStruct(prhs[1], &yLabels, &param, &errorMessage) == 0)
				{
					mexPrintf("Error, can't read model: %s\n", errorMessage);
					delete model;
					return;
				}*/
				model = new budgetedModelBSGD;
				if (!model->loadFromTextFile(modelFileName, &yLabels, &param))
				{
					delete model;
					sprintf(tempStr, "Error: can't read model from file %s.\n", modelFileName);
					mexErrMsgTxt(tempStr);
				}
				testData = new budgetedData(testFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE, false, &yLabels);

				if (param.OUTPUT_SCORES)
					errRate = predictBSGD(testData, &param, (budgetedModelBSGD*) model, &labels, &scores);
				else
					errRate = predictBSGD(testData, &param, (budgetedModelBSGD*) model, &labels);
				//errRate = predictBSGD(testData, &param, (budgetedModelMatlabBSGD*) model, &labels);
				break;
		}
		
		delete model;
		delete testData;
	}
	else
	{
		// there need to be 4 inputs params, filename of the test file, matlab struct of a model, and parameter string.
		if ((nrhs != 3) && (nrhs != 4))
			mexErrMsgTxt("Error, wrong number of input parameters!\nRun 'budgetedsvm_predict()' for help.");
		
		budgetedDataMatlab *testData = NULL;
		budgetedModelMatlab *model = NULL;
		
		// GET ALL INPUTS	
		// input 4 - parameters
		if (nrhs == 3)
			parseInputMatlab(&param, NULL, false);
		else
		{
			mxGetString(prhs[3], inputParamString, mxGetN(prhs[3]) + 1);
			parseInputMatlab(&param, inputParamString, false);
		}
		param.ALGORITHM = budgetedModelMatlab::getAlgorithm(prhs[2]);
		
		// init random number generator
		srand((unsigned)time(NULL));
		
		// initialize test data and run trained model, return Matlab outputs in plhs pointer	
		// inputs 1 and 2 - label vector and data points
		switch (param.ALGORITHM)
		{
			case PEGASOS:
			case AMM_BATCH:
			case AMM_ONLINE:
				// input 3 - Matlab model, read and populate c++ model
				model = new budgetedModelMatlabAMM;
				if (model->loadFromMatlabStruct(prhs[2], &yLabels, &param, &errorMessage) == 0)
				{	
					delete model;
					sprintf(tempStr, "Error: can't read model: %s.\n", errorMessage);
					mexErrMsgTxt(tempStr);
				}
				testData = new budgetedDataMatlab(prhs[0], prhs[1], &param, false, &yLabels);

				if (param.OUTPUT_SCORES)
					errRate = predictAMM(testData, &param, (budgetedModelMatlabAMM*) model, &labels, &scores);
				else
					errRate = predictAMM(testData, &param, (budgetedModelMatlabAMM*) model, &labels);
				break;
				
			case LLSVM:
				// input 3 - Matlab model, read and populate c++ model
				model = new budgetedModelMatlabLLSVM;
				if (model->loadFromMatlabStruct(prhs[2], &yLabels, &param, &errorMessage) == 0)
				{
					delete model;
					sprintf(tempStr, "Error: can't read model: %s.\n", errorMessage);
					mexErrMsgTxt(tempStr);
				}
				testData = new budgetedDataMatlab(prhs[0], prhs[1], &param, false, &yLabels);

				if (param.OUTPUT_SCORES)			
					errRate = predictLLSVM(testData, &param, (budgetedModelMatlabLLSVM*) model, &labels, &scores);
				else
					errRate = predictLLSVM(testData, &param, (budgetedModelMatlabLLSVM*) model, &labels);
				break;
				
			case BSGD:
				// input 3 - Matlab model, read and populate c++ model
				model = new budgetedModelMatlabBSGD;
				if (model->loadFromMatlabStruct(prhs[2], &yLabels, &param, &errorMessage) == 0)
				{
					delete model;
					sprintf(tempStr, "Error: can't read model: %s.\n", errorMessage);
					mexErrMsgTxt(tempStr);
				}
				testData = new budgetedDataMatlab(prhs[0], prhs[1], &param, false, &yLabels);

				if (param.OUTPUT_SCORES)				
					errRate = predictBSGD(testData, &param, (budgetedModelMatlabBSGD*) model, &labels, &scores);
				else
					errRate = predictBSGD(testData, &param, (budgetedModelMatlabBSGD*) model, &labels);
				break;
		}
		
		delete model;
		delete testData;
	}
		
	// here send to output accuracy ...
	plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *ptr = mxGetPr(plhs[0]);
	*ptr = errRate;	
	// ... and labels ...
	plhs[1] = mxCreateNumericMatrix(labels.size(), 1, mxINT32_CLASS, mxREAL);
	int *ptrPreds = (int*) mxGetPr(plhs[1]);
	for (unsigned int i = 0; i < labels.size(); i++)
		ptrPreds[i] = labels[i];
	// ... and scores
	if (param.OUTPUT_SCORES)
	{
		plhs[2] = mxCreateNumericMatrix(scores.size(), 1, mxSINGLE_CLASS, mxREAL);
		float *ptrScores = (float*) mxGetPr(plhs[2]);
		for (unsigned int i = 0; i < scores.size(); i++)
			ptrScores[i] = scores[i];
	}
	
	// defragment the MATLAB space, not necessary but helps when working with large data,
	// Matlab memory space gets very fragmented
	mexCallMATLAB(0, 0, 0, 0, "pack");
}
