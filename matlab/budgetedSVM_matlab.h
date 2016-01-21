/*!
	\file budgetedSVM_matlab.h
	\brief Implements classes and functions used for training and testing of budgetedSVM algorithms in Matlab.
*/
/* 
	Copyright (c) 2013-2014 Nemanja Djuric, Liang Lan, Slobodan Vucetic, and Zhuang Wang
	All rights reserved.
	
	Author	:	Nemanja Djuric
	Name	:	budgetedSVM_matlab.h
	Date	:	December 10th, 2012
	Desc.	:	Implements classes and functions used for training and testing of budgetedSVM algorithms in Matlab.
	Version	:	v1.01
*/

#ifndef _BUDGETEDSVM_MAT_H
#define _BUDGETEDSVM_MAT_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \class budgetedDataMatlab
    \brief Class which manipulates sparse array of vectors (similarly to Matlab sparse matrix structure), with added functionality to load data directly from Matlab.
	
	Class which manipulates sparse array of vectors (similarly to Matlab sparse matrix structure), with added functionality to load data directly from Matlab. Unlike \link budgetedData\endlink, where we load the data in smaller chunks, in this class we assume that the entire data can be loaded into memory, as it is already loaded in Matlab.
*/
class budgetedDataMatlab : public budgetedData
{
	protected:		
		/*! \fn void readDataFromMatlab(const mxArray *labelVec, const mxArray *instanceMat, parameters *param)
			\brief Loads the data from Matlab.
			\param [in] labelVec Vector of labels.
			\param [in] instanceMat Matrix of data points, each row is a single data point.
			\param [in] param The parameters of the algorithm.
		*/	
		void readDataFromMatlab(const mxArray *labelVec, const mxArray *instanceMat, parameters *param);
		
	public:	
		/*! \fn bool readChunk(unsigned int size, bool assign = false)
			\brief Overrides virtual function from \link budgetedData \endlink, simply returns false regardless of inputs as the data is fully loaded from Matlab.
			\param [in] size Size of the chunk to be loaded.
			\param [in] assign True if assignment should be saved, false otherwise.
			\return False regardless of inputs, since the data is fully loaded from Matlab.
		*/	
		bool readChunk(unsigned int size, bool assign = false)
		{
			return false;
		};
		
		/*! \fn budgetedDataMatlab(const mxArray *labelVec, const mxArray *instanceMat, parameters *param, bool keepAssignments = false, vector <int> *yLabels = NULL) : budgetedData(keepAssignments, NULL)
			\brief Constructor, invokes \link readDataFromMatlab \endlink that loads Matlab data.
			\param [in] labelVec Vector of labels.
			\param [in] instanceMat Matrix of data points, each row is a single data point.
			\param [in] param The parameters of the algorithm.
			\param [in] keepAssignments True for AMM batch, otherwise false. Unlike in budgetedData case, no file is created to store the assignments as it is assumed that the memory to hold the assignments can be allocated in whole.
			\param [in] yLabels Possible labels in the classification problem, for training data is NULL since inferred from data.
		*/	
		budgetedDataMatlab(const mxArray *labelVec, const mxArray *instanceMat, parameters *param, bool keepAssignments = false, vector <int> *yLabels = NULL) : budgetedData(keepAssignments, yLabels)
		{
			readDataFromMatlab(labelVec, instanceMat, param);
		};
		
		/*! \fn ~budgetedDataMatlab(void)
			\brief Destructor, cleans up the memory.
		*/		
		~budgetedDataMatlab(void) {};
};

/*! \class budgetedModelMatlab
    \brief Interface which defines methods to load model from and save model to Matlab environment.
*/
class budgetedModelMatlab
{
	public:
		/*! \fn static int getAlgorithm(const mxArray *matlabStruct)
			\brief Get algorithm from the trained model stored in Matlab structure.
			\param [in] matlabStruct Pointer to Matlab structure.
			\return -1 if error, otherwise returns algorithm code from the model file.
		*/
		static int getAlgorithm(const mxArray *matlabStruct);
		
		/* \fn virtual ~budgetedModelMatlab()
			\brief Destructor, cleans up the memory.
		*/	
		//virtual ~budgetedModelMatlab(void) {};
		/*! \fn virtual void saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param) = 0
			\brief Save the trained model to Matlab, by creating Matlab structure.
			\param [out] plhs Pointer to Matlab output.
			\param [in] yLabels Vector of possible labels.
			\param [in] param The parameters of the algorithm.
			
			The Matlab structure is organized as [\a algorithm, \a dimension, \a numClasses, \a labels, \a numWeights, \a paramBias, \a kernelWidth, \a model]. In order to compress memory and to use the 
			memory efficiently, we coded the model in the following way:
			
			- AMM online, AMM batch, and Pegasos: The model is stored as ((\a dimension + 1) x \a numWeights) matrix. The first element of each weight is the degradation of the weight, followed
			by values of the weight for each feature of the data set. If bias term is non-zero, then the final element of each weight corresponds to bias term, and the matrix is of 
			size ((\a dimension + 2) x \a numWeights). By looking at \a labels and \a numWeights members of Matlab structure we can find out which weights belong to which class. For example, first
			numWeights[0] weights belong to labels[0] class, next numWeights[1] weights belong to labels[1] class, and so on.
			
			- BSGD: The model is stored as ((\a numClasses + \a dimension) x \a numWeights) matrix. The first \a numClasses elements of each weight correspond to alpha parameters for each class, 
			given in order of \a labels member of the Matlab structure. This is followed by elements of the weights (or support vectors) for each feature of the data set.
			
			- LLSVM: The model is stored as ((1 + \a dimension) x \a numWeights) matrix. Each row corresponds to one landmark point. The first element of each row corresponds to 
			element of linear SVM hyperplane for that particular landmark point. This is followed by features of the landmark point in the original feature space.
		*/
		virtual void saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param) = 0;
		
		/*! \fn virtual bool loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg) = 0
			\brief Loads the trained model from Matlab structure.
			\param [in] matlabStruct Pointer to Matlab structure.
			\param [in] yLabels Vector of possible labels.
			\param [in] param The parameters of the algorithm.
			\param [out] msg Error message, if error encountered.
			\return False if error encountered, otherwise true.
			
			The Matlab structure is organized as [\a algorithm, \a dimension, \a numClasses, \a labels, \a numWeights, \a paramBias, \a kernelWidth, \a model]. In order to compress memory and to use the 
			memory efficiently, we coded the model in the following way:
			
			- AMM online, AMM batch, and Pegasos: The model is stored as ((\a dimension + 1) x \a numWeights) matrix. The first element of each weight is the degradation of the weight, followed
			by values of the weight for each feature of the data set. If bias term is non-zero, then the final element of each weight corresponds to bias term, and the matrix is of 
			size ((\a dimension + 2) x \a numWeights). By looking at \a labels and \a numWeights members of Matlab structure we can find out which weights belong to which class. For example, first
			numWeights[0] weights belong to labels[0] class, next numWeights[1] weights belong to labels[1] class, and so on.
			
			- BSGD: The model is stored as ((\a numClasses + \a dimension) x \a numWeights) matrix. The first \a numClasses elements of each weight correspond to alpha parameters for each class, 
			given in order of "labels" member of the Matlab structure. This is followed by elements of the weights (or support vectors) for each feature of the data set.
			
			- LLSVM: The model is stored as ((1 + \a dimension) x \a numWeights) matrix. Each row corresponds to one landmark point. The first element of each row corresponds to 
			element of linear SVM hyperplane for that particular landmark point. This is followed by features of the landmark point in the original feature space.
		*/
		virtual bool loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg) = 0;
};

/*! \class budgetedModelMatlabAMM
    \brief Class which holds the AMM model, and implements methods to load AMM model from and save AMM model to Matlab environment.
*/
class budgetedModelMatlabAMM : public budgetedModelMatlab, public budgetedModelAMM
{	
	public:
		/*! \fn void saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param)
			\brief Save the trained model to Matlab, by creating Matlab structure.
			\param [out] plhs Pointer to Matlab output.
			\param [in] yLabels Vector of possible labels.
			\param [in] param The parameters of the algorithm.
			
			The Matlab structure is organized as ["algorithm", "dimension", "numClasses", "labels", "numWeights", "paramBias", "kernelWidth", "model"]. In order to compress memory and to use the 
			memory efficiently, we coded the model in the following way:
			
			The model is stored as (("dimension" + 1) by "numWeights") matrix. The first element of each weight is the degradation of the weight, followed
			by values of the weight for each feature of the data set. If bias term is non-zero, then the final element of each weight corresponds to bias term, and the matrix is of 
			size (("dimension" + 2) by "numWeights"). By looking at "labels" and "numWeights" members of Matlab structure we can find out which weights belong to which class. For example, first
			numWeights[0] weights belong to labels[0] class, next numWeights[1] weights belong to labels[1] class, and so on.
		*/
		void saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param);
		
		/*! \fn bool loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg)
			\brief Loads the trained model from Matlab structure.
			\param [in] matlabStruct Pointer to Matlab structure.
			\param [in] yLabels Vector of possible labels.
			\param [in] param The parameters of the algorithm.
			\param [out] msg Error message, if error encountered.
			\return False if error encountered, otherwise true.
			
			The Matlab structure is organized as ["algorithm", "dimension", "numClasses", "labels", "numWeights", "paramBias", "kernelWidth", "model"]. In order to compress memory and to use the 
			memory efficiently, we coded the model in the following way:
			
			The model is stored as (("dimension" + 1) by "numWeights") matrix. The first element of each weight is the degradation of the weight, followed
			by values of the weight for each feature of the data set. If bias term is non-zero, then the final element of each weight corresponds to bias term, and the matrix is of 
			size (("dimension" + 2) by "numWeights"). By looking at "labels" and "numWeights" members of Matlab structure we can find out which weights belong to which class. For example, first
			numWeights[0] weights belong to labels[0] class, next numWeights[1] weights belong to labels[1] class, and so on.
		*/
		bool loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg);
};

/*! \class budgetedModelMatlabBSGD
    \brief Class which holds the BSGD model, and implements methods to load BSGD model from and save BSGD model to Matlab environment.
*/
class budgetedModelMatlabBSGD : public budgetedModelMatlab, public budgetedModelBSGD
{	
	public:
		/*! \fn void saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param)
			\brief Save the trained model to Matlab, by creating Matlab structure.
			\param [out] plhs Pointer to Matlab output.
			\param [in] yLabels Vector of possible labels.
			\param [in] param The parameters of the algorithm.
			
			The Matlab structure is organized as ["algorithm", "dimension", "numClasses", "labels", "numWeights", "paramBias", "kernelWidth", "model"]. In order to compress memory and to use the 
			memory efficiently, we coded the model in the following way:
			
			The model is stored as (("numClasses" + "dimension") by "numWeights") matrix. The first "numClasses" elements of each weight correspond to alpha parameters for each class, 
			given in order of "labels" member of the Matlab structure. This is followed by elements of the weights (or support vectors) for each feature of the data set.
		*/
		void saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param);
		
		/*! \fn bool loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg)
			\brief Loads the trained model from Matlab structure.
			\param [in] matlabStruct Pointer to Matlab structure.
			\param [in] yLabels Vector of possible labels.
			\param [in] param The parameters of the algorithm.
			\param [out] msg Error message, if error encountered.
			\return False if error encountered, otherwise true.
			
			The Matlab structure is organized as ["algorithm", "dimension", "numClasses", "labels", "numWeights", "paramBias", "kernelWidth", "model"]. In order to compress memory and to use the 
			memory efficiently, we coded the model in the following way:
			
			The model is stored as (("numClasses" + "dimension") by "numWeights") matrix. The first "numClasses" elements of each weight correspond to alpha parameters for each class, 
			given in order of "labels" member of the Matlab structure. This is followed by elements of the weights (or support vectors) for each feature of the data set.
		*/
		bool loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg);
};

/*! \class budgetedModelMatlabLLSVM
    \brief Class which holds the LLSVM model, and implements methods to load LLSVM model from and save LLSVM model to Matlab environment.
*/
class budgetedModelMatlabLLSVM : public budgetedModelMatlab, public budgetedModelLLSVM
{	
	public:
		/*! \fn void saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param)
			\brief Save the trained model to Matlab, by creating Matlab structure.
			\param [out] plhs Pointer to Matlab output.
			\param [in] yLabels Vector of possible labels.
			\param [in] param The parameters of the algorithm.
			
			The Matlab structure is organized as ["algorithm", "dimension", "numClasses", "labels", "numWeights", "paramBias", "kernelWidth", "model"]. In order to compress memory and to use the 
			memory efficiently, we coded the model in the following way:
			
			The model is stored as ((1 + "dimension") by "numWeights") matrix. Each row corresponds to one landmark point. The first element of each row corresponds to 
			element of linear SVM hyperplane for that particular landmark point. This is followed by features of the landmark point in the original feature space.
		*/
		void saveToMatlabStruct(mxArray *plhs[], vector <int>* yLabels, parameters *param);
		
		/*! \fn bool loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg)
			\brief Loads the trained model from Matlab structure.
			\param [in] matlabStruct Pointer to Matlab structure.
			\param [in] yLabels Vector of possible labels.
			\param [in] param The parameters of the algorithm.
			\param [out] msg Error message, if error encountered.
			\return False if error encountered, otherwise true.
			
			The Matlab structure is organized as ["algorithm", "dimension", "numClasses", "labels", "numWeights", "paramBias", "kernelWidth", "model"]. In order to compress memory and to use the 
			memory efficiently, we coded the model in the following way:
			
			The model is stored as ((1 + "dimension") by "numWeights") matrix. Each row corresponds to one landmark point. The first element of each row corresponds to 
			element of linear SVM hyperplane for that particular landmark point. This is followed by features of the landmark point in the original feature space.
		*/
		bool loadFromMatlabStruct(const mxArray *matlabStruct, vector <int>* yLabels, parameters *param, const char **msg);
};

/*! \fn void printStringMatlab(const char *s) 
	\brief Prints string to Matlab, used to modify callback in budgetedSVM.cpp
	\param [in] s Text to be printed.
*/
void printStringMatlab(const char *s);

/*! \fn void printErrorStringMatlab(const char *s) 
	\brief Prints error string to Matlab, used to modify callback found in budgetedSVM.cpp
	\param [in] s Text to be printed.
*/
void printErrorStringMatlab(const char *s);

/*! \fn void fakeAnswer(mxArray *plhs[])
	\brief Returns empty matrix to Matlab.
	\param [out] plhs Pointer to Matlab output.
*/
void fakeAnswer(mxArray *plhs[]);

/*! \fn void printUsageMatlab(bool trainingPhase, parameters *param)
	\brief Prints to standard output the instructions on how to use the software.
	\param [in] trainingPhase Indicator if training or testing phase.
	\param [in] param Parameter object modified by user input.
*/
void printUsageMatlab(bool trainingPhase, parameters *param);

/*! \fn void parseInputMatlab(parameters *param, const char *paramString, bool trainingPhase, const char *inputFileName = NULL, const char *modelFileName = NULL)
	\brief Parses the user input and modifies parameter settings as necessary.
	\param [out] param Parameter object modified by user input.
	\param [in] paramString User-provided parameter string, can be NULL in which case default parameters are used..
	\param [in] trainingPhase Indicator if training or testing phase.
	\param [in] inputFileName User-provided filename with input data (if NULL no check of filename validity).
	\param [in] modelFileName User-provided filename with learned model (if NULL no check of filename validity).
*/
void parseInputMatlab(parameters *param, const char *paramString, bool trainingPhase, const char *inputFileName = NULL, const char *modelFileName = NULL);

#ifdef __cplusplus
}
#endif

#endif /* _BUDGETEDSVM_MAT_H */
