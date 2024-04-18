<h1>Dataiku ML Project: Prediction of heart attack failures using the Heart Failure Prediction Dataset</h1>

Project Objective: 1 - Build a precise machine learning predictive model using the Dataiku DSS (Dataiku Data Science Studio) to forecast heart failure incidents accurately.  
The model was trained and tested using Python and the Heart Failure Prediction Dataset.

Dataiku DSS(Dataiku Data Science Studio) is a Big Data solution and predictive analysis software developed by the French publisher Dataiku. It offers pre-built capabilities to evaluate, deploy & monitor Machine Learning models.

<h2>Project Steps: </h2>

Using Python notebooks and Dataiku Machine Learning experiment tracking capabilities, I went through:
<b>- Configuration of the Dataiku DSS environment and project,</b>

- Data preparation and EDA,
  
- Configuration of a Dataiku Flow
  
- Machine learning experimentation using classic scikit-learn models:
- test different Machine Learning approaches to predict heart failures using scikit-learn models (logistic regression, SVM, decision tree, and random forest).
-  For each model, we will first perform a grid search to find the best parameters, then train the model on the train set using these best parameters and finally log everything (parameters, performance metrics, and models) to keep track of the results of our different experiments and be able to compare afterward. 
   - Logistic Regression
   - Decision Tree
  
- Model evaluation.


<h2>Dataset </h2>

![EDA](https://github.com/Pollybs/dataiku_ML_heart_attack_prediction/blob/main/EDA-Heart-Failure-Prediction-Dataset.png)

Dataset Source: Heart Failure Prediction Dataset. Retrieved from <a href="https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction"> Kaggle</a>
