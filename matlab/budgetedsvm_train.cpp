/*
	\file budgetedsvm_train.cpp
	\brief Source file implementing Matlab interface for training phase of budgetedSVM toolbox.
*/
/*
	Copyright (c) 2013-2014 Nemanja Djuric, Liang Lan, Slobodan Vucetic, and Zhuang Wang
	All rights reserved.
	
	Author	:	Nemanja Djuric
	Name	:	budgetedsvm_train.cpp
	Date	:	November 19th, 2012
	Desc	:	Source file implementing Matlab interface for training phase of BudgetedSVM toolbox.
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
	parameters param;	
	char trainFileName[1024];
	char modelFileName[1024];
	char inputParamString[1024];
	
	if (nrhs == 0)
	{
		printUsageMatlab(true, &param);
		fakeAnswer(plhs);
		return;
	}
	
	// Check the number of inputs
	if ((nrhs != 2) && (nrhs != 3))
		mexErrMsgTxt("Error, wrong number of input parameters!\nRun 'budgetedsvm_train()' for help.");
	
	// init random number generator
	srand((unsigned)time(NULL));
	
	// here we first check what kind of inputs are we working with, .txt files or Matlab variables; we will check that
	// 	by checking the first input parameter, if it is text file then we conclude that we are working with text inputs	
	if ((!mxIsDouble(prhs[0])) && (!mxIsSparse(prhs[0])))
	{
		budgetedModel *model;
		budgetedData *trainData;

		// input 1 - training data, loaded in the switch below
		mxGetString(prhs[0], trainFileName, mxGetN(prhs[0]) + 1);
		
		// input 2 - model file
		mxGetString(prhs[1], modelFileName, mxGetN(prhs[1]) + 1);
		
		// input 3 - parameters
		if (nrhs == 2)
			parseInputMatlab(&param, NULL, true, trainFileName, modelFileName);
		else
		{
			mxGetString(prhs[2], inputParamString, mxGetN(prhs[2]) + 1);
			parseInputMatlab(&param, inputParamString, true, trainFileName, modelFileName);
		}
		
		if (param.RANDOMIZE == 0)
			srand(0);
		
		// train a model
		switch (param.ALGORITHM)
		{
			case PEGASOS:
				model = new budgetedModelAMM;
				trainData = new budgetedData(trainFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE);
				trainPegasos(trainData, &param, (budgetedModelAMM*) model);
				break;
			case AMM_BATCH:
				model = new budgetedModelAMM;
				trainData = new budgetedData(trainFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE, true);
				trainAMMbatch(trainData, &param, (budgetedModelAMM*) model);
				break;
			case AMM_ONLINE:
				model = new budgetedModelAMM;
				trainData = new budgetedData(trainFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE);
				trainAMMonline(trainData, &param, (budgetedModelAMM*) model);
				break;
			case LLSVM:
				model = new budgetedModelLLSVM;
				trainData = new budgetedData(trainFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE);
				trainLLSVM(trainData, &param, (budgetedModelLLSVM*) model);
				break;
			case BSGD:
				model = new budgetedModelBSGD;
				trainData = new budgetedData(trainFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE);
				trainBSGD(trainData, &param, (budgetedModelBSGD*) model);
				break;
		}
		
		// save model to .txt file
		model->saveToTextFile(modelFileName, &(trainData->yLabels), &param);
		delete model;
		
		// delete trainData, no more need for it
		delete trainData;
	}
	else
	{
		budgetedModelMatlab *model;
		budgetedDataMatlab *trainData = NULL;
		
		// GET ALL INPUTS
		// inputs 1 and 2 - label vector and data points, loaded in the switch below
		
		// input 3 - parameters
		if (nrhs == 2)
			parseInputMatlab(&param, NULL, true);
		else
		{
			mxGetString(prhs[2], inputParamString, mxGetN(prhs[2]) + 1);
			parseInputMatlab(&param, inputParamString, true);
		}
		
		// if randomization is switched off seed the RNG with a constant
		if (param.RANDOMIZE == 0)
			srand(0);
		
		// train a model
		switch (param.ALGORITHM)
		{
			case PEGASOS:
				model = new budgetedModelMatlabAMM;
				trainData = new budgetedDataMatlab(prhs[0], prhs[1], &param);
				trainPegasos(trainData, &param, (budgetedModelMatlabAMM*) model);
				break;
			case AMM_BATCH:
				model = new budgetedModelMatlabAMM;
				trainData = new budgetedDataMatlab(prhs[0], prhs[1], &param, true);
				trainAMMbatch(trainData, &param, (budgetedModelMatlabAMM*) model);
				break;
			case AMM_ONLINE:
				model = new budgetedModelMatlabAMM;
				trainData = new budgetedDataMatlab(prhs[0], prhs[1], &param);
				trainAMMonline(trainData, &param, (budgetedModelMatlabAMM*) model);
				break;
			case LLSVM:
				model = new budgetedModelMatlabLLSVM;
				trainData = new budgetedDataMatlab(prhs[0], prhs[1], &param);
				trainLLSVM(trainData, &param, (budgetedModelMatlabLLSVM*) model);
				break;
			case BSGD:
				model = new budgetedModelMatlabBSGD;
				trainData = new budgetedDataMatlab(prhs[0], prhs[1], &param);
				trainBSGD(trainData, &param, (budgetedModelMatlabBSGD*) model);
				break;
		}
		
		// save model to matlab structure
		model->saveToMatlabStruct(plhs, &(trainData->yLabels), &param);
		delete model;
		
		// delete trainData, no more need for it
		delete trainData;
	}
	
	// defragment the MATLAB space, not necessary but helps when working with large data,
	// Matlab memory space gets very fragmented
	mexCallMATLAB(0, 0, 0, 0, "pack");
}
