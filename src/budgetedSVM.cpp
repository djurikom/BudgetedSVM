/*
	\file budgetedSVM.cpp
	\brief Implementation of classes used throughout the budgetedSVM toolbox.
*/
/* 
	Copyright (c) 2013-2014 Nemanja Djuric, Liang Lan, Slobodan Vucetic, and Zhuang Wang
	All rights reserved.
	
	Authors	:	Nemanja Djuric
	Name	:	budgetedSVM.cpp
	Date	:	November 29th, 2012
	Desc.	:	Implementation of classes used throughout the budgetedSVM toolbox.
	Version	:	v1.01
*/

#include <vector>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <cmath>
using namespace std;

#include "budgetedSVM.h"

unsigned int budgetedVector::id = 0;

/* \fn bool fgetWord(FILE *fHandle, char *str);
	\brief Reads one word string from an input file.
	\param [in] fHandle Handle to an open file from which one word is read.
	\param [out] str A character string that will hold the read word.
	\return True if end-of-line or end-of-file encountered after reading a word string, otherwise false.
	
	The function is similar to C++ functions fgetc() and getline(), only that it reads a single word from a file. A word is defined as a sequence of characters that does not contain a white-space character or new-line character '\n'. As a model in BudgetedSVM is stored in a text file where each line may corresponds to a single support vector, it is also useful to know if we reached the end of the line or the end of the file, which is indicated by the return value of the function.
*/
bool fgetWord(FILE *fHandle, char *str)
{	
	char temp;
	unsigned char index = 0;
	bool wordStarted = false;
	while (1)//for (int i = 0; i < 20; i++)
	{
		temp = (char) fgetc(fHandle);
		
		if (temp == EOF)
		{
			str[index++] = '\0';
			return true;
		}
		
		switch (temp)
		{
			case ' ':
				if (wordStarted)
				{
					str[index++] = '\0';
					return false;
				}
				break;
				
			case '\n':
				str[index++] = '\0';
				return true;
				break;
			
			default:
				wordStarted = true;
				str[index++] = temp;
				break;
		}
	}	
}

/* \fn static void printNull(const char *s)
	\brief Delibarately empty print function, used to turn off printing.
	\param [in] text Text to be (not) printed.
*/
static void printNull(const char *text)
{
	// deliberately empty
}

/* \fn static void printNull(const char *s)
	\brief Default error print function.
	\param [in] text Text to be printed.
*/
static void printErrorDefault(const char *text)
{
	// the function prints an error message and quits the program
	fputs(text, stderr);
	fflush(stderr);
	exit(1);
}

/* \fn static void printNull(const char *s)
	\brief Default print function.
	\param [in] text Text to be printed.
*/
static void printStringStdoutDefault(const char *text)
{
	fputs(text, stdout);
	fflush(stdout);
}
static funcPtr svmPrintStringStatic = &printStringStdoutDefault;
static funcPtr svmPrintErrorStringStatic = &printErrorDefault;

/* \fn void svmPrintString(const char* text)
	\brief Prints string to the output.
	\param [in] text Text to be printed.
	
	Prints string to the output. Exactly to which output should be specified by \link setPrintStringFunction\endlink, which modifies the callback that is invoked for printing. This is convinient when simple printf() can not be used, for example if we want to print to Matlab prompt. For example on how to set the printing function in Matlab environment, see the implementation of \link parseInputMatlab\endlink.
*/
void svmPrintString(const char* text)
{
	svmPrintStringStatic(text);
}

/* \fn void setPrintStringFunction(funcPtr printFunc)
	\brief Modifies a callback that prints a string.
	\param [in] printFunc New text-printing function.
	
	This function is used to modify the function that is used to print to standard output. 
	After calling this function, which modifies the callback function for printing, the text is printed simply by invoking \link svmPrintString\endlink. \sa funcPtr
*/
void setPrintStringFunction(funcPtr printFunc)
{
	if (printFunc == NULL)
		svmPrintStringStatic = &printNull;
	else
		svmPrintStringStatic = printFunc;
}

/* \fn void svmPrintErrorString(const char* text)
	\brief Prints error string to the output.
	\param [in] text Text to be printed.
	
	Prints error string to the output. Exactly to which output should be specified by \link setPrintErrorStringFunction\endlink, which modifies the callback that is invoked for printing. This is convinient when an error is detected and, prior to printing appropriate message to a user, we want to exit the program. For example on how to set the printing function in Matlab environment, see the implementation of \link parseInputMatlab\endlink.
*/
void svmPrintErrorString(const char* text)
{
	svmPrintErrorStringStatic(text);
}

/* \fn void setPrintErrorStringFunction(funcPtr printFunc)
	\brief Modifies a callback that prints an error string.
	\param [in] printFunc New text-printing function.
	
	This function is used to modify the function that is used to print to error output. 
	After calling this function, which modifies the callback function for printing error string, the text is printed simply by invoking \link svmPrintErrorString\endlink. \sa funcPtr
*/
void setPrintErrorStringFunction(funcPtr printFunc)
{
	if (printFunc == NULL)
		svmPrintErrorStringStatic = &printErrorDefault;
	else
		svmPrintErrorStringStatic = printFunc;
}

/* \fn bool readableFileExists(const char fileName[])
	\brief Checks if the file, identified by the input parameter, exists and is available for reading.
	\param [in] fileName Handle to an open file from which one word is read.
	\return True if the file exists and is available for reading, otherwise false.
*/
bool readableFileExists(const char fileName[])
{
	FILE *pFile = NULL;
	if (fileName)
	{
		pFile = fopen(fileName, "r");
		if (pFile != NULL)
		{
			fclose(pFile);
			return true;
		}
	}
	return false;
}

/* \fn static int getAlgorithm(const char *filename)
	\brief Get algorithm from the trained model stored in .txt file.
	\param [in] filename Filename of the .txt file where the model is saved.
	\return -1 if error, otherwise returns algorithm code from the model file.
*/
int budgetedModel::getAlgorithm(const char *filename)
{
	FILE *fModel = NULL;
	int temp;
	fModel = fopen(filename, "rt");			
	if (!fModel)
		return -1;
	
	if (!fscanf(fModel, "ALGORITHM: %d\n", &temp))
	{
		svmPrintErrorString("Error reading algorithm type from the model file!\n");
	}
	return temp;
}

// vanilla initialization, just set everything to NULL
// if labels provided then initialize the labels array, used when testing
/* \fn budgetedData::budgetedData(bool keepAssignments, vector <int> *yLabels)
	\brief Vanilla constructor, just initializes the variables.
	\param [in] keepAssignments True for AMM batch, otherwise false.
	\param [in] yLabels Possible labels in the classification problem, for training data is NULL since inferred from data.
*/
budgetedData::budgetedData(bool keepAssignments, vector <int> *yLabels)
{
	this->ifileName = NULL;
	this->ifileNameAssign = NULL;
	this->dimensionHighestSeen = 0;
	this->ifile = NULL;
	this->assignments = NULL;
	this->al = NULL;
	this->keepAssignments = keepAssignments;
	this->loadTime = 0;
	this->N = 0;
	this->dataPartiallyLoaded = false;
	this->loadedDataPointsSoFar = 0;
	this->numNonZeroFeatures = 0;
	this->isTrainingSet = true;
	
	// if labels provided use them, this happens in the case of testing data
	if (yLabels)
	{
		for (unsigned int i = 0; i < (*yLabels).size(); i++)
		{
			this->yLabels.push_back((*yLabels)[i]);
		}
		
		this->isTrainingSet = false;
	}
}

/* \fn budgetedData(const char fileName[], int dimension, unsigned int chunkSize, bool keepAssignments = false, vector <int> *yLabels = NULL)
	\brief Constructor that takes the data from LIBSVM-style .txt file.
	\param [in] fileName Path to the input .txt file.
	\param [in] dimension Dimensionality of the classification problem.
	\param [in] chunkSize Size of the input data chunk that is loaded.
	\param [in] keepAssignments True for AMM batch, otherwise false.
	\param [in] yLabels Possible labels in the classification problem, for training data is NULL since inferred from data.
*/		
budgetedData::budgetedData(const char fileName[], int dimension, unsigned int chunkSize, bool keepAssignments, vector <int> *yLabels)
{
	this->isTrainingSet = true;
	this->ifileName = strdup(fileName);
	if (dimension < 1)
		// if the input data dimensinality is incorrectly set, then we will infer the data dimensionality during data loading
		this->dimensionHighestSeen = 0;
	else
		this->dimensionHighestSeen = dimension;
	
	this->al = new (nothrow) unsigned char[chunkSize];
	if (this->al == NULL)
	{
		svmPrintErrorString("Memory allocation error (budgetedData Constructor)!");
	}
	
	// keepAssignments is used for AMM_batch, where we hold the epoch assignments of data points to hyperplanes
	this->keepAssignments = keepAssignments;
	if (keepAssignments)
	{
		this->ifileNameAssign = strdup("temp_assigns.txt");		// here we set name of the file in which the temporary assignments are kept; it will be removed after the training is completed
		this->assignments = new (nothrow) unsigned int[chunkSize];
	}
	else
		this->assignments = NULL;
	
	// if labels provided use them, this happens in the case of testing data
	if (yLabels)
	{
		for (unsigned int i = 0; i < (*yLabels).size(); i++)
		{
			this->yLabels.push_back((*yLabels)[i]);
		}
		this->isTrainingSet = false;
	}
	
	this->fileOpened = false;
	this->fileAssignOpened = false;
	this->loadTime = 0;
	this->N = 0;
	this->dataPartiallyLoaded = true;
	this->loadedDataPointsSoFar = 0;
	this->numNonZeroFeatures = 0;
}

/* \fn ~budgetedData(void)
	\brief Destructor, cleans up the memory.
*/		
budgetedData::~budgetedData(void)
{
	delete [] this->al;
	flushData();
	
	if (this->assignments)
	{
		// delete all memory taken for keeping epoch assignments
		delete [] this->assignments;
		
		// if data is partially loaded (e.g., since it is too big to fit the memory), then also remove the file used to keep the assignments of the loaded data points
		if (dataPartiallyLoaded)
			remove(this->ifileNameAssign);
	}
}

/* \fn void saveAssignment(unsigned int *assigns)
	\brief Used for AMM batch to save current assignments.
	\param [in] assigns Current assignments.
*/		
void budgetedData::saveAssignment(unsigned int *assigns)
{
	// no need for saving and loading to file, if data is fully (i.e., not partially) loaded, then everything is in the workspace (e.g., in the case of Matlab interface this can happen)
	if (!dataPartiallyLoaded)
	{
		if (assignments == NULL)
			assignments = new (nothrow) unsigned int[N];
			
		for (unsigned int i = 0; i < N; i++)            
			*(assignments + i) = *(assigns + i);
			
		return;
	}
	
	fAssignFile = fopen(ifileNameAssign, "at");
	
	for (unsigned int i = 0; i < N; i++)            
		fprintf(fAssignFile, "%d\n", *(assigns + i));
	
	fclose(fAssignFile);
};

/* \fn void readChunkAssignments(bool endOfFile)
	\brief Reads assignments for the current chunk.
	\param [in] endOfFile If the final chunk, close the assignment file.
*/		
void budgetedData::readChunkAssignments(bool endOfFile)
{
	// if data is fully loaded from the beginning then just exit (e.g., can happen when BudgetedSVM is called from Matlab interface)
	if (!dataPartiallyLoaded)
		return;
	
	int tempInt;
	if (!fileAssignOpened)
	{
		fileAssignOpened = true;
		fAssignFile = fopen(ifileNameAssign, "rt");
	}
	
	for (unsigned int i = 0; i < N; i++)
	{
		// get the assignments (as opposed to initial iteration and reassignment phase
		// where we write the assignments, here we read them)
		if (!fscanf(fAssignFile, "%d\n", &tempInt))
		{
			svmPrintErrorString("Error reading assignments from the text file!\n");
		}
		*(assignments + i) = (unsigned int) tempInt;
	} 

	if (endOfFile)
	{              
		fileAssignOpened = false;
		fclose(fAssignFile);
	}    
};

/* \fn void flushData(void)
	\brief Clears all data taken up by the current chunk.
*/	
void budgetedData::flushData(void)
{
	ai.clear();
	aj.clear();    
	an.clear();
	N = 0;
};    

/* \fn bool readChunk(int size, bool assign = false)
	\brief Reads the next data chunk.
	\param [in] size Size of the chunk to be loaded.
	\param [in] assign True if assignment should be saved, false otherwise.
	\return True if just read the last data chunk, false otherwise.
*/	
bool budgetedData::readChunk(unsigned int size, bool assign)
{
	string text;

	char line[262143];	// maximum length of the line to be read is set to 262143
	char str[256];
	int pos, label;
	unsigned int counter = 0, dimSeen, pointIndex = 0;
	unsigned long start = clock();
	bool labelFound, warningWritten = false;
	
	// if not loaded from .txt file just exit
	if (!dataPartiallyLoaded)
		return false;
	
	flushData();
	if (!fileOpened)
	{
		this->ifile = fopen(ifileName, "rt");
		this->fileOpened = true;		
		this->loadedDataPointsSoFar = 0;
		this->numNonZeroFeatures = 0;
		
		// if the very beginning, just create the assignment file if necessary
		if ((!assign) && (keepAssignments))
		{
			fAssignFile = fopen(ifileNameAssign, "wt");
			fclose(fAssignFile);
		}
	}
	
	// load chunk
	while (fgets(line, 262143, ifile))
	{      
		N++;
		loadedDataPointsSoFar++;
		
		stringstream ss;
		ss << line;
		
		// get label
		if (ss >> text) 
		{
			label = atoi(text.c_str());
			ai.push_back(pointIndex);
			
			// get yLabels, if label not seen before add it into the label array
			labelFound = false;
			for (unsigned int i = 0; i < yLabels.size(); i++)
			{
				if (yLabels[i] == label)
				{
					al[counter++] = (char) i;
					labelFound = true;
					break;
				}
			}
			
			if (!labelFound)
			{
				if (isTrainingSet)
				{
					yLabels.push_back(label);
					al[counter++] = (char) (yLabels.size() - 1);
				}
				else
				{
					// so unseen label detected during testing phase, issue a warning
					if (!warningWritten)
					{
						sprintf(str, "Warning: Testing label '%d' detected during loading that was not seen in training.\n", label);
						svmPrintString(str);
						warningWritten = true;
					}
					
					// give an example a label index that can never be predicted
					al[counter++] = (char) yLabels.size();
				}
			}
		}

		// get feature values
		while (ss >> text)
		{
			if ((pos = (int) text.find(":")))
			{
				dimSeen = atoi(text.substr(0, pos).c_str());					
				aj.push_back(dimSeen);
				an.push_back((float) atof(text.substr(pos + 1, text.length()).c_str()));
				pointIndex++;
				numNonZeroFeatures++;
				
				// if more features found than specified, print error message
				/*if (dimensionHighestSeen < dimSeen)
				{
					sprintf(line, "Found more features than specified with '-D' option (specified: %d, found %d)!\nPlease check your settings.\n", dimension, dimSeen);
					svmPrintErrorString(line);
				}*/
				
				if (dimensionHighestSeen < dimSeen)
					dimensionHighestSeen = dimSeen;
			}
		}

		// check the size of chunk
		if (N == size)
		{
			// still data left to load, keep working
			loadTime += (clock() - start);
			return true;
		}
	}
	
	// got to the end of file, no more data left to load, exit nicely
	fclose(ifile);
	fileOpened = false;
	loadTime += (clock() - start);
	
	return false;      
}

/* \fn float getElementOfVector(unsigned int vector, unsigned int element)
	\brief Returns an element of a vector stored in\link budgetedData\endlink structure.
	\param [in] vector Index of the vector (C-style indexing used, starting from 0; note that LibSVM format indices start from 1).
	\param [in] element Index of the element of the vector (C-style indexing used, starting from 0; note that LibSVM format indices start from 1).
	\return Element of the vector specified as an input.
*/
float budgetedData::getElementOfVector(unsigned int vector, unsigned int element)
{
	unsigned int maxPointIndex, pointIndexPointer;
	
	// check if vector index too big
	if (vector >= this->N)
	{
		svmPrintString("Warning: Vector index in getElementOfVector() function out of bounds, returning default value of 0.\n");
		return 0.0;
	}
	// check if element index too big
	if (element >= this->dimensionHighestSeen)
	{
		svmPrintString("Warning: Element index in getElementOfVector() function out of bounds, returning default value of 0.\n");
		return 0.0;
	}
	
	pointIndexPointer = this->ai[vector];
	maxPointIndex = ((unsigned int)(vector + 1) == this->N) ? (unsigned int) (this->aj.size()) : this->ai[vector + 1];
	
	for (unsigned int i = pointIndexPointer; i < maxPointIndex; i++)
	{
		// if we found the element return its value
		if (this->aj[i] == element + 1)
			return this->an[i];
		
		// if we went over the index of the wanted element, then the element is equal to 0
		if (this->aj[i] > element + 1)
			return 0.0;
	}
	// if the wanted element is indexed higher than all non-zero elements, then it is equal to 0
	return 0.0;
}

/* \fn long double getVectorSqrL2Norm(unsigned int vector, parameters *param)
	\brief Returns a squared L2-norm of a vector stored in \link budgetedData\endlink structure.
	\param [in] vector Index of the vector (C-style indexing used, starting from 0; note that LibSVM format indices start from 1).
	\param [in] param The parameters of the algorithm.
	\return Squared L2-norm of a vector.
	
	This function returns squared L2-norm of a vector stored in the \link budgetedData\endlink structure. In particular, it is used to speed up the computation of Gaussian kernel.
*/		
long double budgetedData::getVectorSqrL2Norm(unsigned int vector, parameters *param)
{
	unsigned int maxPointIndex, pointIndexPointer;
	long double result = 0.0;
	
	// check if vector index too big
	if (vector >= this->N)
	{
		svmPrintString("Warning: Vector index in getElementOfVector() function out of bounds, returning default value of 0.\n");
		return 0.0;
	}
	
	pointIndexPointer = this->ai[vector];
	maxPointIndex = ((unsigned int)(vector + 1) == this->N) ? (unsigned int)(this->aj.size()) : this->ai[vector + 1];
	
	for (unsigned int i = pointIndexPointer; i < maxPointIndex; i++)
		result += (this->an[i] * this->an[i]);
	if (param->BIAS_TERM != 0.0)
		result += (param->BIAS_TERM * param->BIAS_TERM);
	
	return result;
}

/* \fn double distanceBetweenTwoPoints(unsigned int index1, unsigned int index2)
	\brief Computes Euclidean distance between two data points from the input data.
	\param [in] index1 Index of the first data point.
	\param [in] index2 Index of the second data point.
	\return Euclidean distance between the two points.
*/
double budgetedData::distanceBetweenTwoPoints(unsigned int index1, unsigned int index2)
{	
	// if distance to itself, return 0.0
	if (index1 == index2)
		return 0.0;		
	
	long icurrent1 = ai[index1];    
	long iend1 = (index1 == ai.size() - 1) ? aj.size() : ai[index1 + 1];
	long icurrent2 = ai[index2];    
	long iend2 = (index2 == ai.size() - 1) ? aj.size() : ai[index2 + 1];
	double dotxx = 0.0, dotyy = 0.0, dotxy = 0.0;
	
	double currFeat1, currFeat2;
	while (1)
	{	
		// traverse the vectors non-zero feature by non-zero feature
		if (icurrent1 < iend1)
			currFeat1 = (double) aj[icurrent1];
		else
			currFeat1 = INF;
		if (icurrent2 < iend2)
			currFeat2 = (double) aj[icurrent2];
		else
			currFeat2 = INF;		
		
		if (currFeat1 == currFeat2)
		{
			dotxy += (an[icurrent1] * an[icurrent2]);
			dotxx += (an[icurrent1] * an[icurrent1]);
			dotyy += (an[icurrent2] * an[icurrent2]);
			
			icurrent1++;
			icurrent2++;
		}
		else 
		{
			if (currFeat1 < currFeat2)
			{
				dotxx += (an[icurrent1] * an[icurrent1]);
				icurrent1++;
			}
			else
			{
				dotyy += (an[icurrent2] * an[icurrent2]);
				icurrent2++;
			}
		}
		
		if ((icurrent1 >= iend1) && (icurrent2 >= iend2))
			break;
	}	
	return dotxx + dotyy - 2.0 * dotxy;
}

/* \fn const float operator[](int idx) const 
	\brief Overloaded [] operator that returns value.
	\param [in] idx Index of vector element that is retrieved.
	\return Value of the element of the vector.
*/	
const float budgetedVector::operator[](int idx) const 
{
	unsigned int vectorInd = (unsigned int) (idx / (int) chunkWeight);
	unsigned int arrayInd = (unsigned int) (idx % (int) chunkWeight);
	
	// this means that all elements of this chunk are 0
	if (array[vectorInd] == NULL)
		return 0.0;
	else
		return *(array[vectorInd] + arrayInd);
}

/*! fn virtual void extendDimensionality(unsigned int newDim, parameters* param)
	\brief Extend dimensionality of the vector.
	\param [in] newDim New dimensionality of the vector.
	\param [in] param The parameters of the learning algorithm.
	
	Extends the dimensionality of the existing vector to some larger number. We might want to do this due to a variaty of reasons, but the introduction of this method was motivated by this situation: it can happen that the user did not correctly specify the number of data dimensions as an input to BudgetedSVM, in which case this parameter is inferred during loading of the data. As in the first version of BudgetedSVM it was mandatory to specify data dimensionality, to remove this restriction we use this function to extend the dimensionality of the existing model vectors to some larger dimensionality. Since the last element of the vector might be a bias term, we also need param object as an input to locate the bias term and move it to a final element of a new, extended vector.
*/
void budgetedVector::extendDimensionality(unsigned int newDim, parameters* param)
{
	if (dimension > newDim)
	{
		svmPrintErrorString("In extendDimensionality(), extended vector dimensionality smaller than the old one!\n");
	}
	else
	{
		/*char text[127];
		sprintf(text, "In the func, current: %d\tnew: %d!\n", dimension, newDim);
		svmPrintString(text);*/
	}
	
	// when extending the vector, only the last element of the chunk array is modified,
	//	and possibly more zero-chunks are added after the last array element
	unsigned int newArrayLength = (unsigned int)((newDim - 1) / chunkWeight) + 1;
	
	float biasTerm = 0.0;
	if (param->BIAS_TERM != 0.0)
	{
		biasTerm = (*this)[dimension - 1];
	}
	
	unsigned int lastElementLength = dimension % chunkWeight;
	if (lastElementLength == 0)
		lastElementLength = chunkWeight;
	
	unsigned int newLastElementLength = newDim % chunkWeight;
	if (newLastElementLength == 0)
		newLastElementLength = chunkWeight;
	
	float *temp = NULL;	
	if (newArrayLength == arrayLength)
	{
		// just extend the current last array element by some number of elements, create a new array and copy the previous, shorter one to the larger one
		// if the new and the old array lengths are the same, then possibly the new chunk element is also smaller than chunkWeight
		temp = new float[newLastElementLength];
		for (unsigned int i = 0; i < newLastElementLength; i++)
		{
			if (i < (lastElementLength - (int)(param->BIAS_TERM != 0.0)))	// -1 to not copy the bias term
			{
				temp[i] = array[arrayLength - 1][i];	// copy the entire last element of chunk-array
			}
			else
			{
				temp[i] = 0.0;							// set the remaining elements to zero
			}
		}
	}
	else if (newArrayLength > arrayLength)
	{
		// in this case, pad the rest of the current last element with zeros, and new NULL weights will be created
		temp = new float[chunkWeight];
		for (unsigned int i = 0; i < chunkWeight; i++)
		{
			if (i < (lastElementLength - (int)(param->BIAS_TERM != 0.0)))	// -1 to not copy the bias term
			{
				temp[i] = array[arrayLength - 1][i];	// copy the entire last element of chunk-array
			}
			else
			{
				temp[i] = 0.0;							// set the remaining elements to zero
			}
		}
		
		// initialize the additional elements of array to NULL
		for (unsigned int i = 0; i < newArrayLength - arrayLength; i++)
			array.push_back(NULL);
	}
	else
	{
		// just a sanity check
		svmPrintErrorString("Error in extendDimensionality(): New array length shorter than old one, should never happen!");
	}
	
	// put the new, longer chunk instead of the old one
	delete [] array[arrayLength - 1];
	array[arrayLength - 1] = temp;
	temp = NULL;
	
	// set the static parameters of the budgetedVector class to new values
	arrayLength = newArrayLength;
	dimension = newDim;
	
	// put the bias term to the end if it exists
	if (param->BIAS_TERM != 0.0)
	{
		(*this)[dimension - 1] = biasTerm;
	}
}

/* \fn float& operator[](int idx) 
	\brief Overloaded [] operator that assigns value.
	\param [in] idx Index of vector element that is modified.
	\return Value of the modified element of the vector.
*/	
float& budgetedVector::operator[](int idx) 
{
	unsigned int vectorInd = (unsigned int)(idx / (int) chunkWeight);
	unsigned int arrayInd = (unsigned int) (idx % (int) chunkWeight);
	
	// if the input vector is longer than the budgeted vector, this can happen when the test data has
	//	vectors with dimensionality that is longer than previously seen during training, check your test data!
	if (vectorInd >= arrayLength)
	{
		svmPrintErrorString("Error, input vector is longer than the budgeted vector in function budgetedVector::operator[], check your input data.\n");
	}
	
	// if all elements were zero, then first create the array and only
	//    then return the reference
	if (array[vectorInd] == NULL)
	{
		float *tempArray = NULL;
		unsigned long arraySize = chunkWeight;

		// if the last chunk, then it might be smaller than the rest                  
		if (vectorInd == (arrayLength - 1))
		{
			arraySize = dimension % chunkWeight;
			if (arraySize == 0)
				arraySize = chunkWeight;
			tempArray = new (nothrow) float[arraySize];
		}
		else
			tempArray = new (nothrow) float[chunkWeight];

		if (tempArray == NULL)
		{
			svmPrintErrorString("Memory allocation error (budgetedVector assignment)!");
		}
		
		// null the array
		for (unsigned int j = 0; j < arraySize; j++)
			*(tempArray + j) = 0;

		array[vectorInd] = tempArray;
	}

	return *(array[vectorInd] + arrayInd);
}

/* \fn virtual void createVectorUsingDataPoint(budgetedData* inputData, unsigned int t, parameters* param)
	\brief Create new vector from training data point.
	\param [in] inputData Input data from which t-th vector is considered.
	\param [in] t Index of the input vector in the input data.
	\param [in] param The parameters of the algorithm.
	
	Initializes elements of a vector using a data point. Simply copies non-zero elements of the data point stored in budgetedData to the vector. If the vector already had non-zero elements, it is first cleared to become a zero-vector before copying the elements of a data point.
*/
void budgetedVector::createVectorUsingDataPoint(budgetedData* inputData, unsigned int t, parameters* param)
{
	unsigned int ibegin = inputData->ai[t];    
	unsigned int iend = (t == (unsigned int) (inputData->ai.size() - 1)) ? (unsigned int) (inputData->aj.size()) : inputData->ai[t + 1];
	
	this->clear();
	for (unsigned int i = ibegin; i < iend; i++)
	{
		((*this)[inputData->aj[i] - 1]) = inputData->an[i];
		sqrL2norm += (inputData->an[i] * inputData->an[i]);
	}
	
	if ((*param).BIAS_TERM != 0)
	{
		((*this)[(*param).DIMENSION - 1]) = (float)((long double)(*param).BIAS_TERM);
		sqrL2norm += ((*param).BIAS_TERM * (*param).BIAS_TERM);
	}
};

/* \fn long double budgetedVector::sqrNorm(void)
	\brief Calculates a squared norm of the vector.
	\return Squared norm of the vector.
*/	
long double budgetedVector::sqrNorm(void)
{
	long double tempSum = 0.0;
	unsigned long chunkSize = chunkWeight;

	for (unsigned int i = 0; i < arrayLength; i++)
	{
		if (array[i] != NULL)
		{
			if ((i == (arrayLength - 1)) && (dimension != chunkWeight))
				chunkSize = dimension % chunkWeight;
			
			for (unsigned int j = 0; j < chunkSize; j++)
				tempSum += ((long double)array[i][j] * (long double)array[i][j]);
		}     
	}
	return tempSum;
}

/* \fn long double budgetedVector::gaussianKernel(budgetedVector* otherVector, parameters *param)
	\brief Computes Gaussian kernel between this and some other vector.
	\param [in] otherVector The second input vector to RBF kernel.
	\param [in] param The parameters of the algorithm.
	\return Value of RBF kernel between two vectors.
	
	Function computes the value of Gaussian kernel between two vectors. The computation is very fast for sparse data, being only linear in a number of non-zero features. We use the fact that ||x - y||^2 = ||x||^2 - 2 * x^T * y + ||y||^2, where all right-hand side elements can be computed efficiently.
*/
long double budgetedVector::gaussianKernel(budgetedVector* otherVector, parameters *param)
{
	return exp(-0.5L * (long double)((*param).KERNEL_GAMMA_PARAM) * (sqrL2norm + otherVector->getSqrL2norm() - 2.0L * this->linearKernel(otherVector)));
}

/* \fn virtual long double budgetedVector::gaussianKernel(unsigned int t, budgetedData* inputData, parameters *param, inputVectorSqrNorm)
	\brief Computes Gaussian kernel between this and other vector from input data stored in \link budgetedData\endlink.
	\param [in] t Index of the input vector in the input data.
	\param [in] inputData Input data from which t-th vector is considered.
	\param [in] param The parameters of the algorithm.
	\param [in] inputVectorSqrNorm If equal to zero or not provided, the norm of the t-th vector from inputData is computed on-the-fly.
	\return Value of RBF kernel between two vectors.
	
	Function computes the value of Gaussian kernel between two vectors. The computation is very fast for sparse data, being only linear in a number of non-zero features. We use the fact that ||x - y||^2 = ||x||^2 - 2 * x^T * y + ||y||^2, where all right-hand side elements can be computed efficiently.
*/
long double budgetedVector::gaussianKernel(unsigned int t, budgetedData* inputData, parameters *param, long double inputVectorSqrNorm)
{
	if (inputVectorSqrNorm == 0.0)
		inputVectorSqrNorm = inputData->getVectorSqrL2Norm(t, param);
	return exp(-0.5L * (long double)((*param).KERNEL_GAMMA_PARAM) * (this->sqrL2norm + inputVectorSqrNorm - 2.0L * this->linearKernel(t, inputData, param)));
}

/* \fn long double budgetedVector::exponentialKernel(budgetedVector* otherVector, parameters *param)
	\brief Computes exponential kernel between this and some other vector.
	\param [in] otherVector The second input vector to RBF kernel.
	\param [in] param The parameters of the algorithm.
	\return Value of exponential kernel between two vectors.
	
	Function computes the value of exponential kernel between two vectors. The computation is very fast for sparse data, being only linear in a number of non-zero features. We use the fact that ||x - y|| = sqrt(||x||^2 - 2 * x^T * y + ||y||^2), where all right-hand side elements can be computed efficiently.
*/
long double budgetedVector::exponentialKernel(budgetedVector* otherVector, parameters *param)
{
	long double temp = sqrt((long double) (sqrL2norm + otherVector->getSqrL2norm() - 2.0L * this->linearKernel(otherVector)));
	if (temp >= 0.0)
		return exp(-0.5L * (long double)((*param).KERNEL_GAMMA_PARAM) * temp);
	else
		return 0.0L;
}

/* \fn virtual long double budgetedVector::exponentialKernel(unsigned int t, budgetedData* inputData, parameters *param, inputVectorSqrNorm)
	\brief Computes exponential kernel between this and other vector from input data stored in \link budgetedData\endlink.
	\param [in] t Index of the input vector in the input data.
	\param [in] inputData Input data from which t-th vector is considered.
	\param [in] param The parameters of the algorithm.
	\param [in] inputVectorSqrNorm If equal to zero or not provided, the norm of the t-th vector from inputData is computed on-the-fly.
	\return Value of exponential kernel between two vectors.
	
	Function computes the value of exponential kernel between two vectors. The computation is very fast for sparse data, being only linear in a number of non-zero features. We use the fact that ||x - y|| = sqrt(||x||^2 - 2 * x^T * y + ||y||^2), where all right-hand side elements can be computed efficiently.
*/
long double budgetedVector::exponentialKernel(unsigned int t, budgetedData* inputData, parameters *param, long double inputVectorSqrNorm)
{
	if (inputVectorSqrNorm == 0.0)
		inputVectorSqrNorm = inputData->getVectorSqrL2Norm(t, param);
	
	long double temp = sqrt((long double) (this->sqrL2norm + inputVectorSqrNorm - 2.0L * this->linearKernel(t, inputData, param)));
	if (temp >= 0)
		return exp(-0.5L * (long double)((*param).KERNEL_GAMMA_PARAM) * temp);
	else
		return 0.0L;
}

/* \fn long double budgetedVector::sigmoidKernel(budgetedVector* otherVector, parameters *param)
	\brief Computes sigmoid kernel between this and some other vector.
	\param [in] otherVector The second input vector to RBF kernel.
	\param [in] param The parameters of the algorithm.
	\return Value of sigmoid kernel between two vectors.
	
	Function computes the value of sigmoid kernel between two vectors. The computation is very fast for sparse data, being only linear in a number of non-zero features.
*/
long double budgetedVector::sigmoidKernel(budgetedVector* otherVector, parameters *param)
{
	return (long double) tanh((long double) (param->KERNEL_COEF_PARAM + param->KERNEL_DEGREE_PARAM * linearKernel(otherVector)));
}

/* \fn virtual long double budgetedVector::sigmoidKernel(unsigned int t, budgetedData* inputData, parameters *param, inputVectorSqrNorm)
	\brief Computes sigmoid kernel between this and other vector from input data stored in \link budgetedData\endlink.
	\param [in] t Index of the input vector in the input data.
	\param [in] inputData Input data from which t-th vector is considered.
	\param [in] param The parameters of the algorithm.
	\return Value of sigmoid kernel between two vectors.
	
	Function computes the value of sigmoid kernel between two vectors. The computation is very fast for sparse data, being only linear in a number of non-zero features.
*/
long double budgetedVector::sigmoidKernel(unsigned int t, budgetedData* inputData, parameters *param)
{
	return (long double) tanh((long double) (param->KERNEL_COEF_PARAM + param->KERNEL_DEGREE_PARAM * linearKernel(t, inputData, param)));
}

/* \fn virtual long double polyKernel(unsigned int t, budgetedData* inputData, parameters *param)
	\brief Computes polynomial kernel between this budgetedVector vector and another vector from input data stored in budgetedData.
	\param [in] t Index of the input vector in the input data.
	\param [in] inputData Input data from which t-th vector is considered.
	\param [in] param The parameters of the algorithm.
	\return Value of polynomial kernel between two vectors.
	
	Function computes the value of polynomial kernel between budgetedVector vector, and the input data point stored in budgetedData. The computation is very fast for sparse data, being only linear in a number of non-zero features.
*/
long double budgetedVector::polyKernel(unsigned int t, budgetedData* inputData, parameters *param)
{       
	return (long double) pow((long double) (param->KERNEL_COEF_PARAM + linearKernel(t, inputData, param)), (long double) param->KERNEL_DEGREE_PARAM);
}

/* \fn virtual long double polyKernel(budgetedVector* otherVector, parameters *param)
	\brief Computes polynomial kernel between this budgetedVector vector and another vector stored in budgetedVector.
	\param [in] otherVector The second input vector to polynomial kernel.
	\param [in] param The parameters of the algorithm.
	\return Value of polynomial kernel between two vectors.
	
	Function computes the value of polynomial kernel between two vectors. The computation is very fast for sparse data, being only linear in a number of non-zero features.
*/
long double budgetedVector::polyKernel(budgetedVector* otherVector, parameters *param)
{
	return (long double) pow((long double) (param->KERNEL_COEF_PARAM + linearKernel(otherVector)), (long double) param->KERNEL_DEGREE_PARAM);
}

/* \fn long double budgetedVector::linearKernel(unsigned int t, budgetedData* inputData, parameters *param)
	\brief Computes linear kernel between vector and given input data point.
	\param [in] t Index of the input vector in the input data.
	\param [in] inputData Input data from which t-th vector is considered.
	\param [in] param The parameters of the algorithm.
	\return Value of linear kernel between two input vectors.
	
	Function computes the dot product of \link budgetedVectorAMM \endlink vector, and the input data point from \link budgetedData \endlink. 
*/
long double budgetedVector::linearKernel(unsigned int t, budgetedData* inputData, parameters *param)
{
	long double result = 0.0;
	long unsigned int pointIndexPointer = inputData->ai[t];
	long unsigned int maxPointIndex = ((unsigned int)(t + 1) == inputData->N) ? inputData->aj.size() : inputData->ai[t + 1];
	char text[256];
	unsigned int idx, vectorInd, arrayInd;
	
    for (long unsigned int i = pointIndexPointer; i < maxPointIndex; i++)
	{
		idx = inputData->aj[i] - 1;
		vectorInd = (int) (idx / chunkWeight);
		arrayInd = (int) (idx % chunkWeight);
		
		// if the input vector is longer than the budgeted vector, this can happen when the test data has
		//	vectors with dimensionality that is longer than previously seen during training, check your test data!
		if (vectorInd >= arrayLength)
		{
			sprintf(text, "Error, input vector is longer than the budgeted vector, detected dimension %d in function linearKernel(), check your input data.\n", idx + 1);
			svmPrintErrorString(text);
		}
		
		// this means that all elements of this chunk are 0
		if (array[vectorInd] == NULL)
			continue;
		else
			result += array[vectorInd][arrayInd] * inputData->an[i];
	}
	if ((*param).BIAS_TERM != 0)
	    result += (((*this)[(*param).DIMENSION - 1]) * (*param).BIAS_TERM);
	return result;
}

/* \fn virtual long double linearKernel(budgetedVector* otherVector)
	\brief Computes linear kernel between this budgetedVector vector and another vector stored in budgetedVector.
	\param [in] otherVector The second input vector to linear kernel.
	\return Value of linear kernel between two input vectors.
	
	Function computes the value of linear kernel between two vectors.
*/
long double budgetedVector::linearKernel(budgetedVector* otherVector)
{
	long double result = 0.0L;
	unsigned long chunkSize = chunkWeight;
	for (unsigned int i = 0; i < arrayLength; i++)
	{	
		// if either of them is NULL, meaning all-zeros vector chunk, move on to the next chunk
		if ((this->array[i] == NULL) || (otherVector->array[i] == NULL))
			continue;
		
		// now we know that i-th vector chunks of both vectors have non-zero elements, go one by one and compute linear kernel
		if ((i == (arrayLength - 1)) && (dimension != chunkWeight))
			chunkSize = dimension % chunkWeight;
		for (unsigned int j = 0; j < chunkSize; j++)
		{
			result += this->array[i][j] * otherVector->array[i][j];
		}
	}
	return result;
}

/* \fn virtual long double userDefinedKernel(unsigned int t, budgetedData* inputData, parameters *param)
	\brief Computes user-defined kernel between this budgetedVector vector and another vector stored in budgetedData.
	\param [in] t Index of the input vector in the input data.
	\param [in] inputData Input data from which t-th vector is considered.
	\param [in] param The parameters of the algorithm.
	\return Value of user-defined kernel between two input vectors.
	
	Function computes the value of user-defined kernel between budgetedVector vector, and the input data point stored in budgetedData.
*/
long double budgetedVector::userDefinedKernel(unsigned int t, budgetedData* inputData, parameters *param)
{
	// NOTE TO USER: here add your kernel function, be sure to modify BOTH userDefinedKernel() methods; after adding your function make sure to comment the below warnings
	svmPrintString("\nError, non-implemented user-defined kernel function!\n");
	svmPrintErrorString("To add your kernel function please open file 'src/budgetedSVM.cpp' and modify\ntwo userDefinedKernel() methods, you can take a look at implementations of\nother kernel functions for examples.\n");
	return -1.0;
}

/* \fn virtual long double userDefinedKernel(budgetedVector* otherVector, parameters *param)
	\brief Computes user-defined kernel between this budgetedVector vector and another vector stored in budgetedVector.
	\param [in] otherVector The second input vector to user-defined kernel.
	\param [in] param The parameters of the algorithm.
	\param [in] inputVectorSqrNorm If zero or not provided, the norm of t-th vector from inputData is computed on-the-fly.
	\return Value of user-defined kernel between two input vectors.
	
	Function computes the value of user-defined kernel between two vectors.
*/
long double budgetedVector::userDefinedKernel(budgetedVector* otherVector, parameters *param)
{
	// NOTE TO USER: here add your kernel function, be sure to modify BOTH userDefinedKernel() methods; after adding your function make sure to comment the below warnings
	svmPrintString("\nError, non-implemented user-defined kernel function invoked!\n");
	svmPrintErrorString("To add your kernel function please open file 'src/budgetedSVM.cpp' and modify\ntwo userDefinedKernel() methods, you can take a look at implementations of\nother kernel functions for examples.\n");
	return -1.0;
}

/* \fn long double budgetedVector::computeKernel(budgetedVector* otherVector, parameters *param)
	\brief An umbrella function for all different kernels. Computes kernel between this and some other vector.
	\param [in] otherVector The second input vector to RBF kernel.
	\param [in] param The parameters of the algorithm.
	\return Value of kernel between two vectors.
	
	This is an umbrella function for all different kernels. Function computes the value of kernel between two vectors. The computation is very fast for sparse data, being only linear in a number of non-zero features.
*/
long double budgetedVector::computeKernel(budgetedVector* otherVector, parameters *param)
{
	switch ((*param).KERNEL)
	{
		case KERNEL_FUNC_GAUSSIAN:
			return gaussianKernel(otherVector, param);
			break;
		
		case KERNEL_FUNC_EXPONENTIAL:
			return exponentialKernel(otherVector, param);
			break;
		
		case KERNEL_FUNC_SIGMOID:
			return sigmoidKernel(otherVector, param);
			break;
		
		case KERNEL_FUNC_POLYNOMIAL:
			return polyKernel(otherVector, param);
			break;
		
		case KERNEL_FUNC_LINEAR:
			return linearKernel(otherVector);
			break;
		
		case KERNEL_FUNC_USER_DEFINED:
			return userDefinedKernel(otherVector, param);
			break;
		
		default:
			svmPrintErrorString("Error, undefined kernel function found! Run 'budgetedsvm-train' for help.\n");
			return -1.0;
	}
}

/* \fn virtual long double budgetedVector::computeKernel(unsigned int t, budgetedData* inputData, parameters *param, inputVectorSqrNorm)
	\brief An umbrella function for all different kernels. Computes kernel between this and other vector from input data stored in \link budgetedData\endlink.
	\param [in] t Index of the input vector in the input data.
	\param [in] inputData Input data from which t-th vector is considered.
	\param [in] param The parameters of the algorithm.
	\param [in] inputVectorSqrNorm If equal to zero or not provided, the norm of the t-th vector from inputData is computed on-the-fly if necessary (i.e., if RBF kernel is computed).
	\return Value of kernel between two vectors.
	
	This is an umbrella function for all different kernels. Function computes the value of kernel between two vectors. The computation is very fast for sparse data, being only linear in a number of non-zero features.
*/
long double budgetedVector::computeKernel(unsigned int t, budgetedData* inputData, parameters *param, long double inputVectorSqrNorm)
{
	switch ((*param).KERNEL)
	{
		case KERNEL_FUNC_GAUSSIAN:
			return gaussianKernel(t, inputData, param, inputVectorSqrNorm);
			break;
		
		case KERNEL_FUNC_EXPONENTIAL:
			return exponentialKernel(t, inputData, param, inputVectorSqrNorm);
			break;
		
		case KERNEL_FUNC_SIGMOID:
			return sigmoidKernel(t, inputData, param);
			break;
		
		case KERNEL_FUNC_POLYNOMIAL:
			return polyKernel(t, inputData, param);
			break;
		
		case KERNEL_FUNC_LINEAR:
			return linearKernel(t, inputData, param);
			break;
		
		case KERNEL_FUNC_USER_DEFINED:
			return userDefinedKernel(t, inputData, param);
			break;
		
		default:
			svmPrintErrorString("Error, undefined kernel function found! Run 'budgetedsvm-train' for help.\n");
			return -1.0;
	}
}

/* \fn void printUsagePrompt(bool trainingPhase)
	\brief Prints the instructions on how to use the software to standard output.
	\param [in] trainingPhase Indicator if training or testing phase instructions.
*/
void printUsagePrompt(bool trainingPhase, parameters *param)
{
	char text[256];
	if (trainingPhase)
	{
		svmPrintString("\n Usage:\n");
		svmPrintString(" budgetedsvm-train [options] train_file [model_file]\n\n");
		svmPrintString(" Inputs:\n");
		svmPrintString(" options\t- parameters of the model\n");
		svmPrintString(" train_file\t- url of training file in LIBSVM format\n");
		svmPrintString(" model_file\t- file that will hold a learned model\n");
		svmPrintString(" --------------------------------------------\n");
		svmPrintString(" Options are specified in the following format:\n");
		svmPrintString(" '-OPTION1 VALUE1 -OPTION2 VALUE2 ...'\n\n");
		svmPrintString(" Following options are available; affected algorithm and default values are\n");
		svmPrintString("   given in parentheses (algorithm not specified if option affects all):\n\n");
		sprintf(text,  " A - algorithm, which large-scale SVM approximation to use (%d):\n", (*param).ALGORITHM);
		svmPrintString(text);
		svmPrintString("       0 - Pegasos\n");
		svmPrintString("       1 - AMM batch\n");
		svmPrintString("       2 - AMM online\n");
		svmPrintString("       3 - LLSVM\n");
		svmPrintString("       4 - BSGD\n");
		svmPrintString(" D - dimensionality (faster loading if set, if omitted inferred from the data)\n");
		svmPrintString(" B - limit on the number of weights per class in AMM, OR\n");
		sprintf(text, "       total SV set budget in BSGD, OR number of landmark points in LLSVM (%d)\n", (*param).BUDGET_SIZE);
		svmPrintString(text);
		sprintf(text, " L - lambda regularization parameter; high value -> less complex model (%.5f)\n", (*param).LAMBDA_PARAM);
		svmPrintString(text);
		sprintf(text, " b - bias term, if 0 no bias added (%.1f)\n", (*param).BIAS_TERM);
		svmPrintString(text);
		sprintf(text, " e - number of training epochs (AMM, BSGD; %d)\n", (*param).NUM_EPOCHS);
		svmPrintString(text);
		sprintf(text, " s - number of subepochs (AMM batch; %d)\n", (*param).NUM_SUBEPOCHS);
		svmPrintString(text);
		sprintf(text, " k - pruning frequency, after how many examples is pruning done (AMM; %d)\n", (*param).K_PARAM);
		svmPrintString(text);
		sprintf(text, " c - pruning threshold; high value -> less complex model (AMM; %.2f)\n", (*param).C_PARAM);
		svmPrintString(text);
		svmPrintString(" K - kernel function (0 - RBF; 1 - exponential, 2 - polynomial; 3 - linear, \n");
		sprintf(text, "       4 - sigmoid; 5 - user-defined) (LLSVM, BSGD; %d)\n", (*param).KERNEL);
		svmPrintString(text);
		sprintf(text, " g - RBF or exponential kernel width gamma (LLSVM, BSGD; 1/DIMENSIONALITY)\n");
		svmPrintString(text);
		sprintf(text, " d - polynomial kernel degree or sigmoid kernel slope (LLSVM, BSGD; %.2f)\n", (*param).KERNEL_DEGREE_PARAM);
		svmPrintString(text);
		sprintf(text, " i - polynomial or sigmoid kernel intercept (LLSVM, BSGD; %.2f)\n", (*param).KERNEL_COEF_PARAM);
		svmPrintString(text);
		svmPrintString(" m - budget maintenance in BSGD (0 - removal; 1 - merging, uses Gaussian kernel), OR\n");
		sprintf(text,  "       landmark selection in LLSVM (0 - random; 1 - k-means; 2 - k-medoids) (%d)\n\n", (*param).MAINTENANCE_SAMPLING_STRATEGY);
		svmPrintString(text);	
		
		svmPrintString(" z - training and test file are loaded in chunks so that the algorithms can\n");
		svmPrintString("       handle budget files on weaker computers; z specifies number of examples\n");
		sprintf(text,  "       loaded in a single chunk of data (%d)\n", (*param).CHUNK_SIZE);
		svmPrintString(text);
		svmPrintString(" w - model weights are split in chunks, so that the algorithm can handle\n");
		svmPrintString("       highly dimensional data on weaker computers; w specifies number of\n");
		sprintf(text,  "       dimensions stored in one chunk (%d)\n", (*param).CHUNK_WEIGHT);
		svmPrintString(text);
		svmPrintString(" S - if set to 1 data is assumed sparse, if 0 data assumed non-sparse; used to\n");
		svmPrintString("       speed up kernel computations (default is 1 when percentage of non-zero\n");
		svmPrintString("       features is less than 5%, and 0 when percentage is larger than 5%)\n");
		sprintf(text, " r - randomize the algorithms; 1 to randomize, 0 not to randomize (%d)\n", (*param).RANDOMIZE);
		svmPrintString(text);
		sprintf(text, " v - verbose output; 1 to show the algorithm steps, 0 for quiet mode (%d)\n\n", (*param).VERBOSE);
		svmPrintString(text);
	}
	else
	{
		svmPrintString("\n Usage:\n");
		svmPrintString(" budgetedsvm-predict [options] test_file model_file output_file\n\n");
		svmPrintString(" Inputs:\n");
		svmPrintString(" options\t- parameters of the model\n");
		svmPrintString(" test_file\t- url of test file in LIBSVM format\n");
		svmPrintString(" model_file\t- file that holds a learned model\n");
		svmPrintString(" output_file\t- url of file where output will be written\n");
		svmPrintString(" --------------------------------------------\n");
		svmPrintString(" Options are specified in the following format:\n");
		svmPrintString(" '-OPTION1 VALUE1 -OPTION2 VALUE2 ...'\n\n");
		svmPrintString(" The following options are available (default values in parentheses):\n\n");
		
		svmPrintString(" z - the training and test file are loaded in chunks so that the algorithm can\n");
		svmPrintString("       handle budget files on weaker computers; z specifies number of examples\n");
		sprintf(text,  "       loaded in a single chunk of data (%d)\n", (*param).CHUNK_SIZE);
		svmPrintString(text);
		svmPrintString(" w - the model weight is split in parts, so that the algorithm can handle\n");
		svmPrintString("       highly dimensional data on weaker computers; w specifies number of\n");
		sprintf(text,  "       dimensions stored in one chunk (%d)\n", (*param).CHUNK_WEIGHT);
		svmPrintString(text);
		svmPrintString(" S - if set to 1 data is assumed sparse, if 0 data assumed non-sparse, used to\n");
		svmPrintString("       speed up kernel computations (default is 1 when percentage of non-zero\n");
		svmPrintString("       features is less than 5%, and 0 when percentage is larger than 5%)\n");
		svmPrintString(" o - if set to 1, the output file will contain not only the class predictions,\n");
		sprintf(text,  "       but also tab-delimited scores of the winning class (%d)\n", (*param).OUTPUT_SCORES);
		svmPrintString(text);
		sprintf(text, " v - verbose output; 1 to show algorithm steps, 0 for quiet mode (%d)\n\n", (*param).VERBOSE);
		svmPrintString(text);
	}
}

/* \fn void parseInputPrompt(int argc, char **argv, bool trainingPhase, char *inputFile, char *modelFile, char *outputFile, parameters *param)
	\brief Parses the user input from command prompt and modifies parameter settings as necessary, taken from LIBLINEAR implementation.
	\param [in] argc Argument count.
	\param [in] argv Argument vector.
	\param [in] trainingPhase True for training phase parsing, false for testing phase.
	\param [out] inputFile Filename of input data file.
	\param [out] modelFile Filename of model file.
	\param [out] outputFile Filename of output file (only used during testing phase).
	\param [out] param Parameter object modified by user input.
*/
void parseInputPrompt(int argc, char **argv, bool trainingPhase, char *inputFile, char *modelFile, char *outputFile, parameters *param)
{
	vector <char> option;
	vector <float> value;
	int i;
	FILE *pFile = NULL;
	char text[1024];
	
	// parse options
	for (i = 1; i < argc; i++)
	{
		if (argv[i][0] != '-') 
			break;
		++i;
		option.push_back(argv[i - 1][1]);
		value.push_back((float) atof(argv[i]));
	}
	
	if (trainingPhase)
	{
		if (i >= argc)
		{
			svmPrintErrorString("Error, input format not recognized. Run 'budgetedsvm-train' for help.\n");
		}
		
		pFile = fopen(argv[i], "r");
		if (pFile == NULL)
		{
			sprintf(text, "Can't open input file %s!\n", argv[i]);
			svmPrintErrorString(text);
		}
		else
		{
			fclose(pFile);
			strcpy(inputFile, argv[i]);
		}
		
		// take model file if provided by a user
		if (i < argc - 1)
			strcpy(modelFile, argv[i + 1]);
		else
		{
			char *p = strrchr(argv[i], '/');
			if (p == NULL)
				p = argv[i];
			else
				++p;
			sprintf(modelFile, "%s.model", p);
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
						sprintf(text, "Input parameter '-A %d' out of bounds!\nRun 'budgetedsvm-train' for help.\n", (*param).ALGORITHM);
						svmPrintErrorString(text);
					}
					break;
				
				case 'e':
					(*param).NUM_EPOCHS = (unsigned int) value[i];
					break;
				
				case 'D':
					(*param).DIMENSION = (unsigned int) value[i];
					break;
				
				case 's':
					(*param).NUM_SUBEPOCHS = (unsigned int) value[i];
					break;
				
				case 'k':
					(*param).K_PARAM = (unsigned int) value[i];
					break;
				
				case 'c':
					(*param).C_PARAM = (double) value[i];
					if ((*param).C_PARAM < 0.0)
					{
						sprintf(text, "Input parameter '-c' should be a non-negative real number!\nRun 'budgetedsvm-train' for help.\n");
						svmPrintErrorString(text);
					}
					break;
				
				case 'L':
					(*param).LAMBDA_PARAM = (double) value[i];
					if ((*param).LAMBDA_PARAM <= 0.0)
					{
						sprintf(text, "Input parameter '-L' should be a positive real number!\nRun 'budgetedsvm-train' for help.\n");
						svmPrintErrorString(text);
					}
					break;
				
				case 'B':
					(*param).BUDGET_SIZE = (unsigned int) value[i];
					if ((*param).BUDGET_SIZE < 1)
					{
						sprintf(text, "Input parameter '-B' should be a positive integer!\nRun 'budgetedsvm-train' for help.\n");
						svmPrintErrorString(text);
					}
					break;
				
				case 'g':
					(*param).KERNEL_GAMMA_PARAM = (double) value[i];
					if ((*param).KERNEL_GAMMA_PARAM <= 0.0)
					{
						sprintf(text, "Input parameter '-g' should be a positive real number!\nRun 'budgetedsvm-train' for help.\n");
						svmPrintErrorString(text);
					}
					break;
				
				case 'd':
					(*param).KERNEL_DEGREE_PARAM = (double) value[i];
					if ((*param).KERNEL_DEGREE_PARAM <= 0.0)
					{
						sprintf(text, "Input parameter '-d' should be a positive real number!\nRun 'budgetedsvm-train' for help.\n");
						svmPrintErrorString(text);
					}
					break;
				
				case 'i':
					(*param).KERNEL_COEF_PARAM = (double) value[i];
					break;
				
				case 'K':
					(*param).KERNEL = (unsigned int) value[i];
					if ((*param).KERNEL > 5)
					{
						sprintf(text, "Input parameter '-K %d' out of bounds!\nRun 'budgetedsvm-train' for help.\n", (*param).KERNEL);
						svmPrintErrorString(text);
					}
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
				
				case 'z':
					(*param).CHUNK_SIZE = (unsigned int) value[i];
					if ((*param).CHUNK_SIZE < 1)
					{
						sprintf(text, "Input parameter '-z' should be a positive real number!\nRun 'budgetedsvm-train' for help.\n");
						svmPrintErrorString(text);
					}
					break;
				
				case 'w':
					(*param).CHUNK_WEIGHT = (unsigned int) value[i];
					if ((*param).CHUNK_WEIGHT < 1)
					{
						sprintf(text, "Input parameter '-w' should be a positive real number!\nRun 'budgetedsvm-train' for help.\n");
						svmPrintErrorString(text);
					}
					break;
				
				case 'S':
					(*param).VERY_SPARSE_DATA = (unsigned int) (value[i] != 0);
					break;
				
				case 'r':
					(*param).RANDOMIZE = (value[i] != 0);
					break;

				default:
					sprintf(text, "Error, unknown input parameter '-%c'!\nRun 'budgetedsvm-train' for help.\n", option[i]);
					svmPrintErrorString(text);
					break;
			}
		}
		
		// for BSGD, when we use merging budget maintenance strategy then only Gaussian kernel can be used,
		//	due to the nature of merging; here check if user specified some other kernel while merging
		if (((*param).ALGORITHM == BSGD) && ((*param).KERNEL != KERNEL_FUNC_GAUSSIAN) && ((*param).MAINTENANCE_SAMPLING_STRATEGY == BUDGET_MAINTAIN_MERGE))
		{
			svmPrintString("Warning, BSGD with merging strategy can only use Gaussian kernel!\nKernel function switched to Gaussian.\n");
			(*param).KERNEL = KERNEL_FUNC_GAUSSIAN;
		}
		
		// signal error if a user wants to use RBF kernel, but didn't specify either data dimension or kernel width
		if ((((*param).ALGORITHM == LLSVM) || ((*param).ALGORITHM == BSGD)) && (((*param).KERNEL == KERNEL_FUNC_GAUSSIAN) || ((*param).KERNEL == KERNEL_FUNC_EXPONENTIAL)))
		{
			if (((*param).KERNEL_GAMMA_PARAM == 0.0) && ((*param).DIMENSION == 0))
			{
				// this means that both RBF kernel width and dimension were not set by the user in the input string to the toolbox
				//	since in this case the default value of RBF kernel is 1/dimensionality, report error to the user
				svmPrintErrorString("Error, RBF kernel in use, please set either kernel width or dimensionality!\nRun 'budgetedsvm-train' for help.\n");
			}
		}
		
		// check the MAINTENANCE_SAMPLING_STRATEGY validity
		if ((*param).ALGORITHM == LLSVM)
		{
			if ((*param).MAINTENANCE_SAMPLING_STRATEGY > 2)
			{
				// 0 - random removal, 1 - k-means, 2 - k-medoids
				sprintf(text, "Error, unknown input parameter '-m %d'!\nRun 'budgetedsvm-train' for help.\n", (*param).MAINTENANCE_SAMPLING_STRATEGY);
				svmPrintErrorString(text);
			}
		}
		else if ((*param).ALGORITHM == BSGD)
		{
			if ((*param).MAINTENANCE_SAMPLING_STRATEGY > 1)
			{
				// 0 - smallest removal, 1 - merging
				sprintf(text, "Error, unknown input parameter '-m %d'!\nRun 'budgetedsvm-train' for help.\n", (*param).MAINTENANCE_SAMPLING_STRATEGY);
				svmPrintErrorString(text);
			}
		}
		
		// shut down printing to screen if user specified so
		if (!(*param).VERBOSE)
			setPrintStringFunction(NULL);
		
		// no bias term for LLSVM and BSGD functions
		if (((*param).ALGORITHM == LLSVM) || ((*param).ALGORITHM == BSGD))
		{
			(*param).BIAS_TERM = 0.0;
		}
		
		if ((*param).VERBOSE)
		{
			svmPrintString("\n*** Training started with the following parameters:\n");
			switch ((*param).ALGORITHM)
			{
				case PEGASOS:
					svmPrintString("Algorithm \t\t\t: Pegasos\n");
					break;
				case AMM_ONLINE:
					svmPrintString("Algorithm \t\t\t: AMM online\n");
					break;
				case AMM_BATCH:
					svmPrintString("Algorithm \t\t\t: AMM batch\n");
					break;
				case BSGD:
					svmPrintString("Algorithm \t\t\t: BSGD\n");
					break;
				case LLSVM:
					svmPrintString("Algorithm \t\t\t: LLSVM\n");
					break;
			}
			
			if (((*param).ALGORITHM == PEGASOS) || ((*param).ALGORITHM == AMM_BATCH) || ((*param).ALGORITHM == AMM_ONLINE))
			{
				sprintf(text, "Lambda parameter\t\t: %f\n", (*param).LAMBDA_PARAM);
				svmPrintString(text);
				sprintf(text, "Bias term \t\t\t: %f\n", (*param).BIAS_TERM);
				svmPrintString(text);
				if ((*param).ALGORITHM != PEGASOS)
				{
					sprintf(text, "Pruning frequency k \t\t: %d\n", (*param).K_PARAM);
					svmPrintString(text);
					sprintf(text, "Pruning parameter c \t\t: %.2f\n", (*param).C_PARAM);
					svmPrintString(text);
					sprintf(text, "Max num. of weights per class \t: %d\n", (*param).BUDGET_SIZE);
					svmPrintString(text);
					sprintf(text, "Number of epochs \t\t: %d\n\n", (*param).NUM_EPOCHS);
					svmPrintString(text);
				}
				else
					svmPrintString("\n");
			}
			else if (((*param).ALGORITHM == BSGD) || ((*param).ALGORITHM == LLSVM))
			{
				if ((*param).ALGORITHM == BSGD)
				{
					sprintf(text, "Number of epochs \t\t: %d\n", (*param).NUM_EPOCHS);
					svmPrintString(text);
					if ((*param).MAINTENANCE_SAMPLING_STRATEGY == BUDGET_MAINTAIN_REMOVE)
						svmPrintString("Maintenance strategy \t\t: 0 (smallest removal)\n");
					else if ((*param).MAINTENANCE_SAMPLING_STRATEGY == BUDGET_MAINTAIN_MERGE)
						svmPrintString("Maintenance strategy \t\t: 1 (merging)\n");
					else
						svmPrintErrorString("Error, unknown budget maintenance set. Run 'budgetedsvm-train' for help.\n");
					
					svmPrintString(text);
					sprintf(text, "Size of the budget \t\t: %d\n", (*param).BUDGET_SIZE);
					svmPrintString(text);
				}
				else if ((*param).ALGORITHM == LLSVM)
				{		
					switch ((*param).MAINTENANCE_SAMPLING_STRATEGY)
					{
						case LANDMARK_SAMPLE_RANDOM:
							svmPrintString("Landmark sampling \t\t: 0 (random sampling)\n");
							break;
						case LANDMARK_SAMPLE_KMEANS:
							svmPrintString("Landmark sampling \t\t: k-means initialization\n");
							break;
						case LANDMARK_SAMPLE_KMEDOIDS:
							svmPrintString("Landmark sampling \t\t: 1 (k-medoids initialization)\n");
							break;
						default:
							svmPrintErrorString("Error, unknown landmark sampling set. Run 'budgetedsvm-train' for help.\n");
							break;
					}
					sprintf(text, "Number of landmark points \t: %d\n", (*param).BUDGET_SIZE);
					svmPrintString(text);
				}
				
				// now print the common parameters
				sprintf(text, "Lambda regularization param. \t: %f\n", (*param).LAMBDA_PARAM);
				svmPrintString(text);
				switch ((*param).KERNEL)
				{
					case KERNEL_FUNC_GAUSSIAN:
						svmPrintString("Gaussian kernel used \t\t: K(x, y) = exp(-0.5 * gamma * ||x - y||^2)\n");
						if ((*param).KERNEL_GAMMA_PARAM != 0.0)
						{
							sprintf(text, "Kernel width gamma \t\t: %f\n\n", (*param).KERNEL_GAMMA_PARAM);
							svmPrintString(text);
						}
						else
							svmPrintString("Kernel width gamma \t\t: 1 / DIMENSIONALITY\n\n");
						break;
					
					case KERNEL_FUNC_EXPONENTIAL:
						svmPrintString("Exponential kernel used \t: K(x, y) = exp(-0.5 * gamma * ||x - y||)\n");
						if ((*param).KERNEL_GAMMA_PARAM != 0.0)
						{
							sprintf(text, "Kernel width gamma \t\t: %f\n\n", (*param).KERNEL_GAMMA_PARAM);
							svmPrintString(text);
						}
						else
							svmPrintString("Kernel width gamma \t\t: 1 / DIMENSIONALITY\n\n");
						break;
					
					case KERNEL_FUNC_POLYNOMIAL:
						sprintf(text, "Polynomial kernel used \t\t: K(x, y) = (x^T * y + %.2f)^%.2f\n\n", (*param).KERNEL_COEF_PARAM, (*param).KERNEL_DEGREE_PARAM);
						svmPrintString(text);
						break;
						
					case KERNEL_FUNC_SIGMOID:
						sprintf(text, "Sigmoid kernel used \t\t: K(x, y) = tanh(%.2f * x^T * y + %.2f)\n\n", (*param).KERNEL_DEGREE_PARAM, (*param).KERNEL_COEF_PARAM);
						svmPrintString(text);
						break;
					
					case KERNEL_FUNC_LINEAR:
						svmPrintString("Linear kernel used \t\t: K(x, y) = (x^T * y)\n\n");
						break;
					
					case KERNEL_FUNC_USER_DEFINED:
						svmPrintString("User-defined kernel function used.\n\n");
						break;
				}
			}
		}
		
		// increase dimensionality if bias term included
		if ((*param).BIAS_TERM != 0.0)
			(*param).DIMENSION++;
		
		// set gamma to default value of inverse dimensionality if not specified by a user
		if ((*param).KERNEL_GAMMA_PARAM == 0.0)
			(*param).KERNEL_GAMMA_PARAM = 1.0 / (*param).DIMENSION;
	}
	else
	{
		if (i >= argc - 2)
		{
			svmPrintErrorString("Error, input format not recognized. Run 'budgetedsvm-predict' for help.\n");
		}
		
		pFile = fopen(argv[i], "r");
		if (pFile == NULL)
		{
			sprintf(text, "Can't open input file %s!\n", argv[i]);
			svmPrintErrorString(text);
		}
		else
		{
			fclose(pFile);
			strcpy(inputFile, argv[i]);
		}
		
		pFile = fopen(argv[i + 1], "r");
		if (pFile == NULL)
		{
			sprintf(text, "Can't open model file %s!\n", argv[i + 1]);
			svmPrintErrorString(text);
		}
		else
		{
			fclose(pFile);
			strcpy(modelFile, argv[i + 1]);
		}
		
		pFile = fopen(argv[i + 2], "w");
		if (pFile == NULL)
		{
			sprintf(text, "Can't create output file %s!\n", argv[i + 2]);
			svmPrintErrorString(text);
		}
		else
		{
			fclose(pFile);
			strcpy(outputFile, argv[i + 2]);
		}
		
		// modify parameters
		for (unsigned int i = 0; i < option.size(); i++)
		{
			switch (option[i])
			{
				case 'v':
					(*param).VERBOSE = (value[i] != 0);
					break;
					
				case 'z':
					(*param).CHUNK_SIZE = (unsigned int) value[i];
					if ((*param).CHUNK_SIZE < 1)
					{
						sprintf(text, "Input parameter '-z' should be a positive real number!\nRun 'budgetedsvm-predict' for help.\n");
						svmPrintErrorString(text);
					}
					break;
					
				case 'w':
					(*param).CHUNK_WEIGHT = (unsigned int) value[i];
					if ((*param).CHUNK_WEIGHT < 1)
					{
						sprintf(text, "Input parameter '-w' should be a positive real number!\nRun 'budgetedsvm-predict' for help.\n");
						svmPrintErrorString(text);
					}
					break;
					
				case 'S':
					(*param).VERY_SPARSE_DATA = (unsigned int) (value[i] != 0);
					break;
				
				case 'o':
					(*param).OUTPUT_SCORES = (value[i] != 0);
					break;

				default:
					sprintf(text, "Error, unknown input parameter '-%c'!\nRun 'budgetedsvm-predict' for help.\n", option[i]);
					svmPrintErrorString(text);
					break;
			}
		}
		
		// shut down printing to screen if user specified so
		if (!(*param).VERBOSE)
			setPrintStringFunction(NULL);
	}
}
