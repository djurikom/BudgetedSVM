/*
	\file budgetedsvm-train.cpp
	\brief Source file implementing commmand-prompt interface for training phase of budgetedSVM library.
*/
/*
	Copyright (c) 2013-2014 Nemanja Djuric, Liang Lan, Slobodan Vucetic, and Zhuang Wang
	All rights reserved.
	
	Author	:	Nemanja Djuric
	Name	:	budgetedsvm_train.cpp
	Date	:	November 30th, 2012
	Desc.	:	Source file implementing commmand-prompt interface for training phase of budgetedSVM library.
*/

#include "../Eigen/Dense"
using namespace Eigen;

#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>
#include <stdio.h>
using namespace std;

#include "budgetedSVM.h"
#include "mm_algs.h"
#include "bsgd.h"
#include "llsvm.h"

int main(int argc, char **argv)
{	
	parameters param;
	if (argc == 1)
	{
		printUsagePrompt(true, &param);
		return 0;
	}
	
	// vars
	char inputFileName[1024];
	char modelFileName[1024];
	budgetedModel *model = NULL;
	
	// parse input string
	parseInputPrompt(argc, argv, true, inputFileName, modelFileName, NULL, &param);
	
	// init random number generator, if randomization is switched off seed the RNG with a constant
	if (param.RANDOMIZE)
		srand((unsigned)time(NULL));
	else
		srand(0);
	
	// train a model
	budgetedData *trainData = NULL;
	switch (param.ALGORITHM)
	{
		case PEGASOS:
			model = new budgetedModelAMM;
			trainData = new budgetedData(inputFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE);
			trainPegasos(trainData, &param, (budgetedModelAMM*) model);
			break;
		case AMM_BATCH:
			model = new budgetedModelAMM;
			trainData = new budgetedData(inputFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE, true);
			trainAMMbatch(trainData, &param, (budgetedModelAMM*) model);
			break;
		case AMM_ONLINE:
			model = new budgetedModelAMM;
			trainData = new budgetedData(inputFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE);
			trainAMMonline(trainData, &param, (budgetedModelAMM*) model);
			break;
		case LLSVM:
			model = new budgetedModelLLSVM;
			trainData = new budgetedData(inputFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE);
			trainLLSVM(trainData, &param, (budgetedModelLLSVM*) model);
			break;
		case BSGD:
			model = new budgetedModelBSGD;
			trainData = new budgetedData(inputFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE);
			trainBSGD(trainData, &param, (budgetedModelBSGD*) model);
			break;
	}
	
	// save model to .txt file
	model->saveToTextFile(modelFileName, &(trainData->yLabels), &param);
	delete model;
	
	// delete trainData, no more need for it
	delete trainData;
}
