<h1>Dataiku Machine Learning: Heart Failures Prediction</h1>

Project Objective: 1 - Build a precise machine learning predictive model using the Dataiku DSS (Dataiku Data Science Studio) to forecast heart failure incidents accurately.  
The model was trained and tested using Python and the Heart Failure Prediction Dataset.

Dataiku DSS(Dataiku Data Science Studio) is a Big Data solution and predictive analysis software developed by the French publisher Dataiku. It offers pre-built capabilities to evaluate, deploy & monitor Machine Learning models.

<h2>Project Steps: </h2>

Using Python notebooks and Dataiku Machine Learning experiment tracking capabilities, I went through:

<h3>1 - Configuration of the Dataiku DSS environment and project,</h3>

<h3>2 - Data preparation and EDA</h3>
  
<h3>3 - Configuration of the Dataiku Flow</h3>
  
<h3>4 - Machine learning experimentation: the test of different Machine Learning approaches to predict heart failures using scikit-learn models</h3>

Scikit-learn models models tested: 
  - Logistic regression,
  - SVM
  - Decision Tree
  - Random Forest
  
a) For each model, a grid search was performed to find the best hyper parameters

b) Then the model was trained on the train set using these best parameters and cross-validation

c) Everything (parameters, performance metrics, and models) was logged in the Daitaku Experiment Tracking (MLFlow framework) to keep track of the results of the different experiments and be able to compare afterward. 

<h3>Model evaluation</h3>
![MOD](https://github.com/Pollybs/dataiku_ML_heart_attack_prediction/blob/main/dataiku_experiment_tracking.png)


<h2>Dataset </h2>

![EDA](https://github.com/Pollybs/dataiku_ML_heart_attack_prediction/blob/main/EDA-Heart-Failure-Prediction-Dataset.png)

Dataset Source: Heart Failure Prediction Dataset. Retrieved from <a href="https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction"> Kaggle</a>
