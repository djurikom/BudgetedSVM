/*!
	\file bsgd.h
	\brief Defines classes and functions used for training and testing of BSGD (Budgeted Stochastic Gradient Descent) algorithm.
*/
/*
	Copyright (c) 2013-2014 Nemanja Djuric, Liang Lan, Slobodan Vucetic, and Zhuang Wang
	All rights reserved.
	
	Author	:	Nemanja Djuric
	Name	:	bsgd.h
	Date	:	November 20th, 2012
	Desc.	:	Defines classes and functions used for training and testing of BSGD (Budgeted Stochastic Gradient Descent) algorithm.
	Version	:	v1.01
*/

#ifndef _BSGD_H
#define _BSGD_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \class budgetedVectorBSGD
    \brief Class which holds sparse vector, which is split into a number of arrays to trade-off between speed of access and memory usage of sparse data, with added methods for BSGD algorithm.
*/
class budgetedVectorBSGD : public budgetedVector
{
	// friends so that they can set sqrL2norm property during model loading
	friend class budgetedModelBSGD;
	friend class budgetedModelMatlabBSGD;
	
	/*! \var static unsigned int numClasses
		\brief Number of classes of the classification problem, specifies the size of \link alphas\endlink vector.
	*/
	/*! \var vector <double> alphas
		\brief Array of class-specific alpha parameters, used in BSGD algorithm.
		
		This vector is of the size that equals number of classes in the data set. Each element specifies the influence a \link budgetedVector \endlink has on a specific class.
	*/
	protected:
		static unsigned int numClasses;
		
	public:
    	vector <long double> alphas; 
		
		/*! \fn unsigned int getNumClasses(void)
			\brief Get the number of classes in the classification problem.
			\return Number of classes that are covered by this vector, also the length of \link alphas\endlink.
		*/	
		static unsigned int getNumClasses(void)
		{
			return numClasses;
		}
		
		/*! \fn void updateSV(budgetedVectorBSGD* v, long double kMax)
			\brief Updates the vector to obtain a merged vector, used during merging budget maintenance.
			\param [in] v Vector that is merged with this vector.
			\param [in] kMax Parameter that specifies how to combine them (currentVector <- kMax * currentVector + (1 - kMax) * v).
			
			When we find which two support vectors to merge, together with the value of the merging parameter kMax, this function updates one of the two vectors to obtain the
			merged support vector. After the merging, the other vector is no longer needed and can be deleted. \sa computeKmax
		*/	
		void updateSV(budgetedVectorBSGD* v, long double kMax);
 
    	/*! \fn budgetedVectorBSGD(unsigned long dim = 0, unsigned long chnkWght = 0, unsigned int numCls = 0) : budgetedVector(dim, chnkWght)
			\brief Constructor, initializes the vector to all zeros, and also initializes class-specific alpha parameters.
			\param [in] dim Dimensionality of the vector.
			\param [in] chnkWght Size of each vector chunk.
			\param [in] numCls Number of classes in the classification problem, specifies the size of \link alphas\endlink vector.
		*/
		budgetedVectorBSGD(unsigned int dim = 0, unsigned int chnkWght = 0, unsigned int numCls = 0) : budgetedVector(dim, chnkWght)
		{
			if (numClasses == 0)
				numClasses = numCls;
			
			for (unsigned int i = 0; i < numClasses; i++)
				this->alphas.push_back(0.0);
		}
		
		/*! \fn long double alphaNorm(void)
			\brief Computes the norm of alpha vector.
			\return Norm of the alpha vector.
			
			Computes the l2-norm of the alpha vector. \sa budgetedVector::alphas
		*/	
		long double alphaNorm(void);
		
		/*! \fn void downgrade(unsigned long oto)
			\brief Downgrade the alpha-parameters.
			\param [in] oto Total number of iterations so far.
			
			Each training iteration the alpha parameters are pushed towards 0 to ensure the convergence of the algorithm to the optimal solution.
		*/
		void downgrade(unsigned long oto)
		{
			for (unsigned int i = 0; i < alphas.size(); i++)
				if (alphas[i] != 0)
					alphas[i] *= (1.0 - 1.0 / (long double) oto);
		};
};

/*! \class budgetedModelBSGD
    \brief Class which holds the BSGD model (comprising the support vectors stored as \link budgetedVectorBSGD\endlink), and implements methods to load BSGD model from and save BSGD model to text file.
*/
class budgetedModelBSGD : public budgetedModel
{
	/*! \var vector <budgetedVectorBSGD*> *modelBSGD
		\brief Holds BSGD model.
	*/	
	public:
		vector <budgetedVectorBSGD*> *modelBSGD;
		
		/*! \fn void extendDimensionalityOfModel(unsigned int newDim, parameters* param)
			\brief Extends the dimensionality of each support vector in the BSGD model.
			
			Extends the dimensionality of each support vector in the BSGD model. Called after new data chunk has been loaded, could be needed when user set the dimensionality of the data incorrectly, and we infer this important parameter during loading of the data.
		*/
		void extendDimensionalityOfModel(unsigned int newDim, parameters* param)
		{
			// extend the dimensionality of each weight vector
			for (unsigned int i = 0; i < (*modelBSGD).size(); i++)
				(*modelBSGD)[i]->extendDimensionality(newDim, param);
		};
		
		/*! \fn budgetedModelBSGD(void)
			\brief Constructor, initializes the BSGD model to zero-vectors.
		*/
		budgetedModelBSGD(void)
		{
			modelBSGD = new vector <budgetedVectorBSGD*>;
		};
		
		/*! \fn ~budgetedModelBSGD(void)
			\brief Destructor, cleans up memory taken by BSGD.
		*/	
		~budgetedModelBSGD(void);
		
		/*! \fn bool saveToTextFile(const char *filename, vector <int>* yLabels, parameters *param)
			\brief Saves the trained BSGD model to .txt file.
			\param [in] filename Filename of the .txt file where the model is saved.
			\param [in] yLabels Vector of possible labels.
			\param [in] param The parameters of the algorithm.
			\return Returns false if error encountered, otherwise true.
			
			The text file has the following rows: [\a ALGORITHM, \a DIMENSION, \a NUMBER_OF_CLASSES, \a LABELS, \a NUMBER_OF_WEIGHTS, \a BIAS_TERM, \a KERNEL_WIDTH, \a MODEL]. In order to compress memory and to use the 
			memory efficiently, we coded the model in the following way:
			
			Each row corresponds to one support vector (or weight). The first elements of each weight correspond to alpha parameters for each class, 
			given in order of "labels" member of the Matlab structure. However, since alpha can be equal to 0, we use LIBSVM format to store alphas, as -class_index:class-specific_alpha, where we
			added '-' (minus sign) in front of the class index to differentiate between class indices and feature indices that follow. After the alphas, in the same row the elements of the 
			weights (or support vectors) for each feature are given in LIBSVM format.
		*/
		bool saveToTextFile(const char *filename, vector <int>* yLabels, parameters *param);
		
		/*! \fn bool loadFromTextFile(const char *filename, vector <int>* yLabels, parameters *param)
			\brief Loads the trained BSGD model from .txt file.
			\param [in] filename Filename of the .txt file where the model is saved.
			\param [out] yLabels Vector of possible labels.
			\param [out] param The parameters of the algorithm.
			\return Returns false if error encountered, otherwise true.
			
			The text file has the following rows: [\a ALGORITHM, \a DIMENSION, \a NUMBER_OF_CLASSES, \a LABELS, \a NUMBER_OF_WEIGHTS, \a BIAS_TERM, \a KERNEL_WIDTH, \a MODEL]. In order to compress memory and to use the 
			memory efficiently, we coded the model in the following way:
			
			Each row corresponds to one support vector (or weight). The first elements of each weight correspond to alpha parameters for each class, 
			given in order of "labels" member of the Matlab structure. However, since alpha can be equal to 0, we use LIBSVM format to store alphas, as -class_index:class-specific_alpha, where we
			added '-' (minus sign) in front of the class index to differentiate between class indices and feature indices that follow. After the alphas, in the same row the elements of the 
			weights (or support vectors) for each feature are given in LIBSVM format.
		*/
		bool loadFromTextFile(const char *filename, vector <int>* yLabels, parameters *param);
};

/*! \fn void trainBSGD(budgetedData *trainData, parameters *param, budgetedModelBSGD *model)
	\brief Train BSGD.
	\param [in] trainData Input training data.
	\param [in] param The parameters of the algorithm.
	\param [in,out] model Initial BSGD model.
	
	The function trains BSGD model, given input data, the initial model (most often zero-weight model), and the parameters of the model.
*/
void trainBSGD(budgetedData *trainData, parameters *param, budgetedModelBSGD *model);

/*! \fn float predictBSGD(budgetedData *testData, parameters *param, budgetedModelBSGD *model, vector <int> *labels = NULL, vector <float> *scores = NULL)
	\brief Given a BSGD model, predict the labels of testing data.
	\param [in] testData Input test data.
	\param [in] param The parameters of the algorithm.
	\param [in] model Trained BSGD model.
	\param [out] labels Vector of predicted labels.
	\param [out] scores Vector of scores of the winning labels.
	\return Testing set error rate.
	
	Given the learned BSGD model, the function computes the predictions on the testing data, outputing the predicted labels and the error rate.
*/
float predictBSGD(budgetedData *testData, parameters *param, budgetedModelBSGD *model, vector <int> *labels = NULL, vector <float> *scores = NULL);

#ifdef __cplusplus
}
#endif

#endif /* _BSGD_H */
