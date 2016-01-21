/*!
	\file llsvm.h
	\brief Defines classes and functions used for training and testing of LLSVM algorithm.
*/
/* 
	Copyright (c) 2013-2014 Nemanja Djuric, Liang Lan, Slobodan Vucetic, and Zhuang Wang
	All rights reserved.
	
	Author	:	Nemanja Djuric
	Name	:	llsvm.h
	Date	:	November 20th, 2012
	Desc.	:	Defines classes and functions used for training and testing of LLSVM algorithm.
	Version	:	v1.01
*/

#ifndef _LLSVM_H
#define _LLSVM_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \class budgetedVectorLLSVM
    \brief Class which holds sparse vector, which is split into a number of arrays to trade-off between speed of access and memory usage of sparse data, with added methods for LLSVM algorithm.
*/
class budgetedVectorLLSVM : public budgetedVector
{
	// friends so that they can set sqrL2norm property during model loading
	friend class budgetedModelLLSVM;
	friend class budgetedModelMatlabLLSVM;
	
	public:
		/*! \fn void createVectorUsingDataPointMatrix(VectorXd &dataVector)
			\brief Initialize the vector using a data point represented as a (1 x DIMENSION) matrix
			\param [in] dataVector Row vector holding a data point.
			
			Used during the initialization stage of the LLSVM algorithm to store the found landmark point in an instance of budgetedVectorLLSVM class.
		*/
		void createVectorUsingDataPointMatrix(VectorXd &dataVector)
		{



			for (unsigned int i = 0; i < (unsigned int) dataVector.size(); i++)
			{
				if (dataVector[i] != 0.0)
				{


					(*this)[i] = (float) dataVector[i];
					sqrL2norm += (dataVector[i] * dataVector[i]);
				}
			}
		};
		
		/*! \fn budgetedVectorLLSVM(unsigned int dim = 0, unsigned int chnkWght = 0) : budgetedVector(dim, chnkWght)
			\brief Constructor, initializes the LLSVM vector to zero weights.
		*/
		budgetedVectorLLSVM(unsigned int dim = 0, unsigned int chnkWght = 0) : budgetedVector(dim, chnkWght) {};
};

/*! \class budgetedModelLLSVM
    \brief Class which holds the LLSVM model, and implements methods to load LLSVM model from and save LLSVM model to text file.
*/
class budgetedModelLLSVM : public budgetedModel
{
	/*! \var vector <budgetedVector*> *modelLLSVMlandmarks
		\brief Holds landmark points, used to compute the transformation matrix \link modelLLSVMmatrixW\endlink.
	*/
	/*! \var MatrixXd modelLLSVMmatrixW
		\brief Holds transformation matrix, used to compute the mapping from original feature space into low-D space.
	*/
	/*! \var VectorXd modelLLSVMweightVector
		\brief Holds weight vector, the solution of linear SVM on transformed points.
	*/
	public:
		vector <budgetedVectorLLSVM*> *modelLLSVMlandmarks;
		VectorXd modelLLSVMweightVector;
		MatrixXd modelLLSVMmatrixW;
		
		/*! \fn void extendDimensionalityOfModel(unsigned int newDim, parameters* param)
			\brief Extends the dimensionality of each landmark point in the LLSVM model.
			
			Extends the dimensionality of each landmark point in the LLSVM model. Called after new data chunk has been loaded, could be needed when user set the dimensionality of the data incorrectly, and we infer this important parameter during loading of the data.
		*/
		void extendDimensionalityOfModel(unsigned int newDim, parameters* param)
		{
			// extend the dimensionality of each weight vector
			for (unsigned int i = 0; i < (*modelLLSVMlandmarks).size(); i++)
				(*modelLLSVMlandmarks)[i]->extendDimensionality(newDim, param);
		};
		
		/*! \fn budgetedModelLLSVM(void)
			\brief Constructor, initializes the LLSVM model. Simply allocates memory for a vector of landmark points, where each is stored in budgetedVectorLLSVM.
		*/
		budgetedModelLLSVM(void)
		{
			modelLLSVMlandmarks = new vector <budgetedVectorLLSVM*>;
		};
		
		/*! \fn ~budgetedModelLLSVM(void)
			\brief Destructor, cleans up memory taken by LLSVM.
		*/	
		~budgetedModelLLSVM(void)
		{
			modelLLSVMweightVector.resize(0, 0);
			modelLLSVMmatrixW.resize(0, 0);
			if (modelLLSVMlandmarks)
			{
				for (unsigned int i = 0; i < (*modelLLSVMlandmarks).size(); i++)
					delete (*modelLLSVMlandmarks)[i];
				(*modelLLSVMlandmarks).clear();
				
				delete modelLLSVMlandmarks;
				modelLLSVMlandmarks = NULL;
			}
		};
		
		/*! \fn bool saveToTextFile(const char *filename, vector <int>* yLabels, parameters *param)
			\brief Saves the trained LLSVM model to .txt file.
			\param [in] filename Filename of the .txt file where the model is saved.
			\param [in] yLabels Vector of possible labels.
			\param [in] param The parameters of the algorithm.
			\return Returns false if error encountered, otherwise true.
			
			The text file has the following rows: [\a ALGORITHM, \a DIMENSION, \a NUMBER_OF_CLASSES, \a LABELS, \a NUMBER_OF_WEIGHTS, \a BIAS_TERM, \a KERNEL_WIDTH, \a MODEL]. In order to compress memory and to use the 
			memory efficiently, we coded the model in the following way:
			
			Each row corresponds to one landmark point. The first element of each row corresponds to element of linear SVM hyperplane for that particular
			landmark point. This is followed by features of the landmark point in the original feature space of the data set, stored in LIBSVM format.
		*/
		bool saveToTextFile(const char *filename, vector <int>* yLabels, parameters *param);
		
		/*! \fn bool loadFromTextFile(const char *filename, vector <int>* yLabels, parameters *param)
			\brief Loads the trained LLSVM model from .txt file.
			\param [in] filename Filename of the .txt file where the model is saved.
			\param [out] yLabels Vector of possible labels.
			\param [out] param The parameters of the algorithm.
			\return Returns false if error encountered, otherwise true.
			
			The text file has the following rows: [\a ALGORITHM, \a DIMENSION, \a NUMBER_OF_CLASSES, \a LABELS, \a NUMBER_OF_WEIGHTS, \a BIAS_TERM, \a KERNEL_WIDTH, \a MODEL]. In order to compress memory and to use the 
			memory efficiently, we coded the model in the following way:
			
			Each row corresponds to one landmark point. The first element of each row corresponds to element of linear SVM hyperplane for that particular
			landmark point. This is followed by features of the landmark point in the original feature space of the data set, stored in LIBSVM format.
		*/
		bool loadFromTextFile(const char *filename, vector <int>* yLabels, parameters *param);
};


/*! \fn void trainLLSVM(budgetedData *trainData, parameters *param, budgetedModelLLSVM *model)
	\brief Train LLSVM online.
	\param [in] trainData Input training data.
	\param [in] param The parameters of the algorithm.
	\param [in,out] model Initial LLSVM model.
	
	The function trains LLSVM model, given input data, the initial model (most often zero-weight model), and the parameters of the model.
*/
void trainLLSVM(budgetedData *trainData, parameters *param, budgetedModelLLSVM *model);


/*! \fn float predictLLSVM(budgetedData *testData, parameters *param, budgetedModelLLSVM *model, vector <int> *labels = NULL, vector <float> *scores = NULL)
	\brief Given an LLSVM model, predict the labels of testing data.
	\param [in] testData Input test data.
	\param [in] param The parameters of the algorithm.
	\param [in] model Trained LLSVM model.
	\param [out] labels Vector of predicted labels.
	\param [out] scores Vector of scores of the winning labels.
	\return Testing set error rate.
	
	Given the learned BSGD model, the function computes the predictions on the testing data, outputing the predicted labels and the error rate.
*/
float predictLLSVM(budgetedData *testData, parameters *param, budgetedModelLLSVM *model, vector <int> *labels = NULL, vector <float> *scores = NULL);

#ifdef __cplusplus
}
#endif

#endif /* _LLSVM_H */
