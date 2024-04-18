# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Step 1: Preparing the input dataset for ML modeling

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# The project is based on the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).
# 
# This first notebook:
# 
# - Performs a quick exploratory analysis of the input dataset: it looks at the structure of the dataset and the distribution of the values in the different categorical and continuous columns.
# 
# - Uses the functions from the <a href="https://doc.dataiku.com/dss/latest/python/reusing-code.html#sharing-python-code-within-a-project">project Python library</a> to clean & prepare the input dataset before Machine Learning modeling. We will first clean categorical and continuous columns, then split the dataset into a train set and a test set.
# 
# Finally, we will transform this notebook into a Python Dataiku recipe in the project Flow that will output the new train and test datasets.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# <div class="alert alert-block alert-info">
# <b>Tip:</b> The <a href="https://doc.dataiku.com/dss/latest/python/reusing-code.html#sharing-python-code-within-a-project">project libraries</a> allow you to build shared code repositories. They can be synchronized with an external Git repository.
# </div>

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 0. Import packages

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# **Making sure that the `heart-attack-project` code environment** is being used(see prerequisites)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE_MAGIC_CELL
# Automatically replaced inline charts by "no-op" charts
# %pylab inline
import matplotlib
matplotlib.use("Agg")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from heart_attack_library import data_processing
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import seaborn as sns

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 1. Import the data

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Let’s use the Dataiku Python API to import the input dataset. This piece of code allows retrieving data in the same manner, no matter where the dataset is stored (local filesystem, SQL database, Cloud data lakes, etc.)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
dataset_heart_measures = dataiku.Dataset("heart_measures")
df = dataset_heart_measures.get_dataframe(limit=100000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 2. A quick audit of the dataset

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### 2.1 Compute the shape of the dataset

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(f'The shape of the dataset is {df.shape}')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### 2.2 Look at a preview of the first rows of the dataset

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### 2.3 Inspect missing values & number of distinct values (cardinality) for each column

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pdu.audit(df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 3. Exploratory data analysis

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### 3.1 Define categorical & continuous columns

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
categorical_cols = ['Sex','ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
continuous_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### 3.2 Look at the distibution of continuous features

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
nb_cols=2
fig = plt.figure(figsize=(8,6))
fig.suptitle('Distribution of continuous features', fontsize=11)
gs = fig.add_gridspec(math.ceil(len(continuous_cols)/nb_cols),nb_cols)
gs.update(wspace=0.3, hspace=0.4)
for i, col in enumerate(continuous_cols):
    ax = fig.add_subplot(gs[math.floor(i/nb_cols),i%nb_cols])
    sns.histplot(df[col], ax=ax)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### 3.3 Look at the distribution of categorical columns

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
nb_cols=2
fig = plt.figure(figsize=(8,6))
fig.suptitle('Distribution of categorical features', fontsize=11)
gs = fig.add_gridspec(math.ceil(len(categorical_cols)/nb_cols),nb_cols)
gs.update(wspace=0.3, hspace=0.4)
for i, col in enumerate(categorical_cols):
    ax = fig.add_subplot(gs[math.floor(i/nb_cols),i%nb_cols])
    plot = sns.countplot(df[col])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### 3.4 Look at the distribution of target variable

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
target = "HeartDisease"
fig = plt.figure(figsize=(4,2.5))
fig.suptitle('Distribution of heart attack diseases', fontsize=11, y=1.11)
plot = sns.countplot(df[target])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# <div class="alert alert-block alert-info">
# <b>Tip:</b> To ease collaboration, all the insights you create from Jupyter Notebooks can be
# shared with other users by publishing them on dashboards. See <a href="https://doc.dataiku.com/dss/latest/dashboards/insights/jupyter-notebook.html">documentation</a> for more information.
# </div>

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 4. Prepare data

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### 4.1 Clean categorical columns

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Transform string values from categorical columns into int, using the functions from the project libraries
df_cleaned = data_processing.transform_heart_categorical_measures(df, "ChestPainType", "RestingECG", 
                                                                  "ExerciseAngina", "ST_Slope", "Sex")

df_cleaned.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### 4.2 Transform categorical columns into dummies

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_cleaned = pd.get_dummies(df_cleaned, columns = categorical_cols, drop_first = True)

print("Shape after dummies transformation: " + str(df_cleaned.shape))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### 4.3 Scale continuous columns

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Let's use the Scikit-Learn Robust Scaler to scale continuous features

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
scaler = RobustScaler()
df_cleaned[continuous_cols] = scaler.fit_transform(df_cleaned[continuous_cols])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 5. Split the dataset into train and test

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Let's now split the dataset into a train set that will be used for experimenting and training the Machine Learning models and test set that will be used to evaluate the deployed model.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
heart_measures_train_df, heart_measures_test_df = train_test_split(df_cleaned, test_size=0.2, stratify=df_cleaned.HeartDisease)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 6. Next: use this notebook to create a new step in the project workflow

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Now that our notebook is up and running, we can use it to create the first step of our pipeline in the Flow:
# 
# - Click on the **+ Create Recipe** button at the top right of the screen.
# 
# - Select the **Python recipe** option.
# 
# - Choose the ```heart_measures``` dataset as the input dataset and create two output datasets: ```heart_measures_train``` and ```heart_measures_test```.
# 
# - Click on the **Create recipe** button.
# 
# - At the end of the recipe script, replace the last four rows of code with:
# 
# 
# ```python
# heart_measures_train = dataiku.Dataset("heart_measures_train")
# heart_measures_train.write_with_schema(heart_measures_train_df)
# heart_measured_test = dataiku.Dataset("heart_measures_test")
# heart_measured_test.write_with_schema(heart_measures_test_df)
# ```
# 
# - Run the recipe

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# <div class="alert alert-block alert-success">
# <b>Success:</b> We can now go on the Flow, we'll see an orange circle that represents your first step (we call it a 'Recipe'), and two output datasets.
# </div>

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
heart_measures_train = dataiku.Dataset("heart_measures_train")
heart_measures_train.write_with_schema(heart_measures_train_df)
heart_measured_test = dataiku.Dataset("heart_measures_test")
heart_measured_test.write_with_schema(heart_measures_test_df)