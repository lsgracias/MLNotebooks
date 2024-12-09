{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Lab 6:  Train Various Regression Models and Compare Their Performances"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os \n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV\n",
    "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error, r2_score"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this lab assignment, you will train various regression models (regressors) and compare their performances. You will train, test and evaluate individual models as well as ensemble models. You will:\n",
    "\n",
    "1. Build your DataFrame and define your ML problem:\n",
    "    * Load the Airbnb \"listings\" data set\n",
    "    * Define the label - what are you predicting?\n",
    "    * Identify the features\n",
    "2. Create labeled examples from the data set.\n",
    "3. Split the data into training and test data sets.\n",
    "4. Train, test and evaluate two individual regressors.\n",
    "5. Use the stacking ensemble method to train the same regressors.\n",
    "6. Train, test and evaluate Gradient Boosted Decision Trees.\n",
    "7. Train, test and evaluate Random Forest.\n",
    "8. Visualize and compare the performance of all of the models.\n",
    "\n",
    "<font color='red'><b>Note:</font><br> \n",
    "<font color='red'><b>1. Some of the code cells in this notebook may take a while to run.</font><br>\n",
    "<font color='red'><b>2. Ignore warning messages that pertain to deprecated packages.</font>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 1. Build Your DataFrame and Define Your ML Problem\n",
    "\n",
    "#### Load a Data Set and Save it as a Pandas DataFrame\n",
    "\n",
    "We will work with the data set ``airbnbData_train``. This data set already has all the necessary preprocessing steps implemented, including one-hot encoding of the categorical variables, scaling of all numerical variable values, and imputing missing values. It is ready for modeling.\n",
    "\n",
    "<b>Task</b>: In the code cell below, use the same method you have been using to load the data using `pd.read_csv()` and save it to DataFrame `df`.\n",
    "\n",
    "You will be working with the file named \"airbnbData_train.csv\" that is located in a folder named \"data_regressors\"."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "filename = os.path.join(os.getcwd(), \"data_regressors\",\"airbnbData_train.csv\")\n",
    "df = pd.read_csv(filename, header=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Define the Label\n",
    "\n",
    "Your goal is to train a machine learning model that predicts the price of an Airbnb listing. This is an example of supervised learning and is a regression problem. In our dataset, our label will be the `price` column and the label contains continuous values.\n",
    "\n",
    "#### Evaluation Metrics for Regressors\n",
    "\n",
    "So far, we have mostly focused on classification problems. For this assignment, we will focus on a regression problem and predict a continuous outcome. There are different evaluation metrics that are used to determine the performance of a regressor. We will use two metrics to evaluate our regressors: RMSE (root mean square error) and $R^2$ (coefficient of determination).\n",
    "\n",
    "RMSE:<br>\n",
    "RMSE finds the average difference between the predicted values and the actual values. We will compute the RMSE on the test set.  To compute the RMSE, we will use the scikit-learn ```mean_squared_error()``` function. Since RMSE finds the difference between the predicted and actual values, lower RMSE values indicate good performance - the model fits the data well and makes more accurate predictions. On the other hand, higher RSME values indicate that the model is not performing well.\n",
    "\n",
    "$R^2$:<br>\n",
    "$R^2$ is a measure of the proportion of variability in the prediction that the model was able to make using the test data. An $R^2$ value of 1 is perfect and 0 implies no explanatory value. We can use scikit-learn's ```r2_score()``` function to compute it. Since $R^2$ measures how well the model fits the data, a higher $R^2$ value indicates that good performance and a lower $R^2$ indicates that poor performance.\n",
    "\n",
    "#### Identify Features\n",
    "\n",
    "Our features will be all of the remaining columns in the dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 2. Create Labeled Examples from the Data Set \n",
    "\n",
    "<b>Task</b>: In the code cell below, create labeled examples from DataFrame `df`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = df['price'] \n",
    "X = df.drop(columns = 'price', axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 3. Create Training and Test Data Sets\n",
    "\n",
    "<b>Task</b>: In the code cell below, create training and test sets out of the labeled examples. Create a test set that is 30 percent of the size of the data set. Save the results to variables `X_train, X_test, y_train, y_test`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.30, random_state=1234)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 4: Train, Test and Evaluate Two Regression Models: Linear Regression and Decision Tree\n",
    "\n",
    "### a. Train, Test and Evaluate a Linear Regression\n",
    "\n",
    "You will use the scikit-learn `LinearRegression` class to create a linear regression model. For more information, consult the online [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).\n",
    "\n",
    "First let's import `LinearRegression`:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: Initialize a scikit-learn `LinearRegression` model object with no arguments, and fit the model to the training data. The model object should be named `lr_model`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: black;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-1 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-1 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: block;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-1 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-1 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 1ex;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-1 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;LinearRegression<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LinearRegression.html\">?<span>Documentation for LinearRegression</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>LinearRegression()</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr_model = LinearRegression()\n",
    "\n",
    "lr_model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> Test your model on the test set (`X_test`). Call the ``predict()`` method  to use the fitted model to generate a vector of predictions on the test set. Save the result to the variable ``y_lr_pred``."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Call predict() to use the fitted model to make predictions on the test data\n",
    "y_lr_pred = lr_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To compute the RMSE, we will use the scikit-learn ```mean_squared_error()``` function, which computes the mean squared error between the predicted values and the actual values: ```y_lr_pred``` and```y_test```. In order to obtain the root mean squared error, we will specify the parameter `squared=False`. \n",
    "\n",
    "To compute the $R^2$, we will use the scikit-learn ```r2_score()``` function. \n",
    "\n",
    "<b>Task</b>: In the code cell below, do the following:\n",
    "\n",
    "1. Call the `mean_squared_error()` function with arguments `y_test` and `y_lr_pred` and the parameter `squared=False` to find the RMSE. Save your result to the variable `lr_rmse`.\n",
    "\n",
    "2. Call the `r2_score()` function with the arguments `y_test` and `y_lr_pred`.  Save the result to the variable `lr_r2`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[LR] Root Mean Squared Error: 0.7449290413154662\n",
      "[LR] R2: 0.4743953999284285\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/ubuntu/.pyenv/versions/3.9.19/lib/python3.9/site-packages/sklearn/metrics/_regression.py:483: FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "# 1. Compute the RMSE using mean_squared_error()\n",
    "lr_rmse = mean_squared_error(y_test,y_lr_pred, squared=False)\n",
    "\n",
    "\n",
    "# 2. Compute the R2 score using r2_score()\n",
    "lr_r2 = r2_score(y_test,y_lr_pred)\n",
    "\n",
    "print('[LR] Root Mean Squared Error: {0}'.format(lr_rmse))\n",
    "print('[LR] R2: {0}'.format(lr_r2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### b. Train, Test and Evaluate a Decision Tree Using GridSearch"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You will use the scikit-learn `DecisionTreeRegressor` class to create a decision tree regressor. For more information, consult the online [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html).\n",
    "\n",
    "First let's import `DecisionTreeRegressor`:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeRegressor"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Set Up a Parameter Grid \n",
    "\n",
    "<b>Task</b>: Create a dictionary called `param_grid` that contains possible hyperparameter values for `max_depth` and `min_samples_leaf`. The dictionary should contain the following key/value pairs:\n",
    "\n",
    "* a key called 'max_depth' with a value which is a list consisting of the integers 4 and 8\n",
    "* a key called 'min_samples_leaf' with a value which is a list consisting of the integers 25 and 50"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "param_grid = {'max_depth': [4, 8],'min_samples_leaf': [25, 50]}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> Use `GridSearchCV` to fit a grid of decision tree regressors and search over the different values of hyperparameters `max_depth` and `min_samples_leaf` to find the ones that results in the best 3-fold cross-validation (CV) score.\n",
    "\n",
    "\n",
    "You will pass the following arguments to `GridSearchCV()`:\n",
    "\n",
    "1. A decision tree **regressor** model object.\n",
    "2. The `param_grid` variable.\n",
    "3. The number of folds (`cv=3`).\n",
    "4. The scoring method `scoring='neg_root_mean_squared_error'`. Note that `neg_root_mean_squared_error` returns the negative RMSE.\n",
    "\n",
    "\n",
    "Complete the code in the cell below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running Grid Search...\n",
      "Done\n"
     ]
    }
   ],
   "source": [
    "print('Running Grid Search...')\n",
    "\n",
    "# 1. Create a DecisionTreeRegressor model object without supplying arguments. \n",
    "#    Save the model object to the variable 'dt_regressor'\n",
    "\n",
    "dt_regressor = DecisionTreeRegressor()\n",
    "\n",
    "\n",
    "# 2. Run a Grid Search with 3-fold cross-validation and assign the output to the object 'dt_grid'.\n",
    "#    * Pass the model and the parameter grid to GridSearchCV()\n",
    "#    * Set the number of folds to 3\n",
    "#    * Specify the scoring method\n",
    "\n",
    "dt_grid = GridSearchCV(dt_regressor, param_grid, cv=3, scoring='neg_root_mean_squared_error')\n",
    "\n",
    "\n",
    "# 3. Fit the model (use the 'grid' variable) on the training data and assign the fitted model to the \n",
    "#    variable 'dt_grid_search'\n",
    "\n",
    "dt_grid_search = dt_grid.fit(X_train, y_train)\n",
    "\n",
    "print('Done')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The code cell below prints the RMSE score of the best model using the `best_score_` attribute of the fitted grid search object `dt_grid_search`. Note that specifying a scoring method of `neg_root_mean_squared_error` will result in the negative RMSE, so we will multiply `dt_grid_search.best_score` by -1 to obtain the RMSE."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[DT] RMSE for the best model is : 0.72\n"
     ]
    }
   ],
   "source": [
    "rmse_DT = -1 * dt_grid_search.best_score_\n",
    "print(\"[DT] RMSE for the best model is : {:.2f}\".format(rmse_DT) )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: In the code cell below, obtain the best model hyperparameters identified by the grid search and save them to the variable `dt_best_params`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'max_depth': 8, 'min_samples_leaf': 25}"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dt_best_params = dt_grid_search.best_params_\n",
    "\n",
    "dt_best_params"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: In the code cell below, initialize a `DecisionTreeRegressor` model object, supplying the best values of hyperparameters `max_depth` and `min_samples_leaf` as arguments.  Name the model object `dt_model`. Then fit the model `dt_model` to the training data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: black;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-2 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-2 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-2 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: block;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-2 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-2 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-2 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-2 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 1ex;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-2 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-2 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeRegressor(max_depth=8, min_samples_leaf=25)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;DecisionTreeRegressor<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.4/modules/generated/sklearn.tree.DecisionTreeRegressor.html\">?<span>Documentation for DecisionTreeRegressor</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>DecisionTreeRegressor(max_depth=8, min_samples_leaf=25)</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "DecisionTreeRegressor(max_depth=8, min_samples_leaf=25)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dt_model = DecisionTreeRegressor(max_depth=dt_best_params['max_depth'], min_samples_leaf=dt_best_params['min_samples_leaf'])\n",
    "\n",
    "dt_model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> Test your model `dt_model` on the test set `X_test`. Call the ``predict()`` method  to use the fitted model to generate a vector of predictions on the test set. Save the result to the variable ``y_dt_pred``. Evaluate the results by computing the RMSE and R2 score in the same manner as you did above. Save the results to the variables `dt_rmse` and `dt_r2`.\n",
    "\n",
    "Complete the code in the cell below to accomplish this."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[DT] Root Mean Squared Error: 0.716760961165418\n",
      "[DT] R2: 0.5133933582135196\n"
     ]
    }
   ],
   "source": [
    "# 1. Use the fitted model to make predictions on the test data\n",
    "y_dt_pred = dt_model.predict(X_test)\n",
    "\n",
    "\n",
    "# 2. Compute the RMSE using mean_squared_error()\n",
    "dt_rmse = np.sqrt(mean_squared_error(y_test, y_dt_pred))\n",
    "\n",
    "\n",
    "# 3. Compute the R2 score using r2_score()\n",
    "dt_r2 = r2_score(y_test, y_dt_pred)\n",
    "\n",
    "\n",
    "print('[DT] Root Mean Squared Error: {0}'.format(dt_rmse))\n",
    "print('[DT] R2: {0}'.format(dt_r2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 5: Train, Test and Evaluate Ensemble Models: Stacking "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You will use the stacking ensemble method to train two regression models. You will use the scikit-learn `StackingRegressor` class. For more information, consult the online [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html).\n",
    "\n",
    "First let's import `StackingRegressor`:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import StackingRegressor"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this part of the assignment, we will use two models jointly. In the code cell below, we creates a list of tuples, each consisting of a scikit-learn model function and the corresponding shorthand name that we choose. We will specify the hyperparameters for the decision tree that we determined through the grid search above."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "estimators = [(\"DT\", DecisionTreeRegressor(max_depth=8, min_samples_leaf=25)),\n",
    "              (\"LR\", LinearRegression())\n",
    "             ]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: \n",
    "\n",
    "\n",
    "1. Create a `StackingRegressor` model object. Call `StackingRegressor()` with the following parameters:\n",
    "    * Assign the list `estimators` to the parameter `estimators`.\n",
    "    * Use the parameter 'passthrough=False'. \n",
    "Assign the results to the variable `stacking_model`.\n",
    "\n",
    "2. Fit `stacking_model` to the training data.\n",
    "\n",
    "As you read up on the definition of the `StackingRegressor` class, you will notice that by default, the results of each model are combined using a ridge regression (a \"final regressor\")."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Implement Stacking...\n",
      "End\n"
     ]
    }
   ],
   "source": [
    "print('Implement Stacking...')\n",
    "\n",
    "stacking_model = StackingRegressor(estimators=estimators, passthrough=False)\n",
    "\n",
    "stacking_model.fit(X_train, y_train)\n",
    "\n",
    "print('End')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> Use the `predict()` method to test your ensemble model `stacking_model` on the test set (`X_test`). Save the result to the variable `stacking_pred`. Evaluate the results by computing the RMSE and R2 score. Save the results to the variables `stack_rmse` and `stack_r2`.\n",
    "\n",
    "Complete the code in the cell below to accomplish this."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Root Mean Squared Error: 0.6932026268071889\n",
      "R2: 0.5448550320649771\n"
     ]
    }
   ],
   "source": [
    "from math import sqrt\n",
    "# 1. Use the fitted model to make predictions on the test data\n",
    "stacking_pred = stacking_model.predict(X_test)\n",
    "\n",
    "\n",
    "# 2. Compute the RMSE \n",
    "stack_rmse = sqrt(mean_squared_error(y_test, stacking_pred))\n",
    "\n",
    "\n",
    "# 3. Compute the R2 score\n",
    "stack_r2 = r2_score(y_test, stacking_pred)\n",
    "\n",
    "\n",
    "   \n",
    "print('Root Mean Squared Error: {0}'.format(stack_rmse))\n",
    "print('R2: {0}'.format(stack_r2))                       "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 6: Train, Test and Evaluate  Evaluate Ensemble Models: Gradient Boosted Decision Trees \n",
    "\n",
    "You will use the scikit-learn `GradientBoostingRegressor` class to create a gradient boosted decision tree. For more information, consult the online [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html).\n",
    "\n",
    "First let's import `GradientBoostingRegressor`:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import GradientBoostingRegressor"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's assume you already performed a grid search to find the best model hyperparameters for your gradient boosted decision tree. (We are omitting this step to save computation time.) The best values are: `max_depth=2`, and `n_estimators = 300`. \n",
    "\n",
    "<b>Task</b>: Initialize a `GradientBoostingRegressor` model object with the above values as arguments. Save the result to the variable `gbdt_model`. Fit the `gbdt_model` model to the training data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Begin GBDT Implementation...\n",
      "End\n"
     ]
    }
   ],
   "source": [
    "print('Begin GBDT Implementation...')\n",
    "\n",
    "# Initialize a GradientBoostingRegressor model object with the best hyperparameters\n",
    "gbdt_model = GradientBoostingRegressor(max_depth=2, n_estimators=300)\n",
    "\n",
    "# Fit the model to the training data\n",
    "gbdt_model.fit(X_train, y_train)\n",
    "\n",
    "print('End')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> Use the `predict()` method to test your model `gbdt_model` on the test set `X_test`. Save the result to the variable ``y_gbdt_pred``. Evaluate the results by computing the RMSE and R2 score in the same manner as you did above. Save the results to the variables `gbdt_rmse` and `gbdt_r2`.\n",
    "\n",
    "Complete the code in the cell below to accomplish this."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[GBDT] Root Mean Squared Error: 0.6607490625613496\n",
      "[GBDT] R2: 0.5864743458290619\n"
     ]
    }
   ],
   "source": [
    "# 1. Use the fitted model to make predictions on the test data\n",
    "y_gbdt_pred = gbdt_model.predict(X_test)\n",
    "\n",
    "# 2. Compute the RMSE \n",
    "gbdt_rmse = sqrt(mean_squared_error(y_test, y_gbdt_pred))\n",
    "\n",
    "\n",
    "# 3. Compute the R2 score \n",
    "gbdt_r2 = r2_score(y_test, y_gbdt_pred)\n",
    "\n",
    "\n",
    "print('[GBDT] Root Mean Squared Error: {0}'.format(gbdt_rmse))\n",
    "print('[GBDT] R2: {0}'.format(gbdt_r2))                 "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 7: Train, Test and Evaluate  Ensemble Models: Random Forest"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You will use the scikit-learn `RandomForestRegressor` class to create a gradient boosted decision tree. For more information, consult the online [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).\n",
    "\n",
    "First let's import `RandomForestRegressor`:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's assume you already performed a grid search to find the best model hyperparameters for your random forest model. (We are omitting this step to save computation time.) The best values are: `max_depth=32`, and `n_estimators = 300`. \n",
    "\n",
    "<b>Task</b>: Initialize a `RandomForestRegressor` model object with the above values as arguments. Save the result to the variable `rf_model`. Fit the `rf_model` model to the training data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Begin RF Implementation...\n",
      "End\n"
     ]
    }
   ],
   "source": [
    "print('Begin RF Implementation...')\n",
    "\n",
    "# Initialize a RandomForestRegressor model object with the best hyperparameters\n",
    "rf_model = RandomForestRegressor(max_depth=32, n_estimators=300)\n",
    "\n",
    "# Fit the model to the training data\n",
    "rf_model.fit(X_train, y_train)\n",
    "\n",
    "print('End')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> Use the `predict()` method to test your model `rf_model` on the test set `X_test`. Save the result to the variable ``y_rf_pred``. Evaluate the results by computing the RMSE and R2 score in the same manner as you did above. Save the results to the variables `rf_rmse` and `rf_r2`.\n",
    "\n",
    "Complete the code in the cell below to accomplish this."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[RF] Root Mean Squared Error: 0.6291608269691636\n",
      "[RF] R2: 0.6250678387979424\n"
     ]
    }
   ],
   "source": [
    "# 1. Use the fitted model to make predictions on the test data\n",
    "y_rf_pred = rf_model.predict(X_test)\n",
    "\n",
    "\n",
    "# 2. Compute the RMSE \n",
    "rf_rmse = sqrt(mean_squared_error(y_test, y_rf_pred))\n",
    "\n",
    "\n",
    "# 3. Compute the R2 score \n",
    "rf_r2 = r2_score(y_test, y_rf_pred)\n",
    "\n",
    "\n",
    "print('[RF] Root Mean Squared Error: {0}'.format(rf_rmse))\n",
    "print('[RF] R2: {0}'.format(rf_r2))                 "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 8: Visualize and Compare Model Performance\n",
    "\n",
    "The code cell below will plot the RMSE and R2 score for each regressor. \n",
    "\n",
    "<b>Task:</b> Complete the code in the cell below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjcAAAHHCAYAAABDUnkqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/P9b71AAAACXBIWXMAAA9hAAAPYQGoP6dpAAA9PklEQVR4nO3deVwVZf//8ffhKAcRwQVkURSVMi13k9wyvTFQw7rzNsvbRDQqizRpMcvQtNs9tXJLE23ROzOXu9I0JWnTsjQqS0lNv5oJYiooCCjM7w9/njwBBgocGF/Px+M8aq65ZuYzc47w5pqZMxbDMAwBAACYhIuzCwAAAChNhBsAAGAqhBsAAGAqhBsAAGAqhBsAAGAqhBsAAGAqhBsAAGAqhBsAAGAqhBsAAGAqhBsADiwWi8aPH1/i5Q4ePCiLxaKlS5eWek1X46233tINN9ygqlWrqmbNms4uB0A5INwAFdDSpUtlsVhksVj0xRdfFJhvGIYCAwNlsVh0xx13OKHCK5eYmGjfN4vFoqpVq6px48YaPHiwfv3111Ld1p49ezRkyBA1adJEixYt0sKFC0t1/QAqpirOLgBA0dzc3LR8+XJ16dLFof3TTz/Vb7/9JpvN5qTKrt6IESN0880369y5c9q5c6cWLlyodevW6ccff1RAQECpbCMxMVH5+fl6+eWXFRwcXCrrBFDxMXIDVGC9e/fWypUrdf78eYf25cuXq127dvLz83NSZVeva9euGjRokKKiovTqq69qxowZOnHihN54442rXndmZqYk6dixY5JUqqejsrKySm1dAMoG4QaowO677z798ccf2rRpk70tNzdX7733ngYOHFjoMpmZmXriiScUGBgom82mpk2basaMGTIMw6FfTk6ORo0aJR8fH9WoUUN9+/bVb7/9Vug6jxw5oqFDh8rX11c2m0033nij4uPjS29HJfXo0UOSdODAAXvbRx99pK5du6p69eqqUaOG+vTpo59++slhuSFDhsjDw0P79+9X7969VaNGDf373/9WUFCQxo0bJ0ny8fEpcC3RvHnzdOONN8pmsykgIECPPvqoTp065bDu2267TTfddJN27NihW2+9Ve7u7nr22Wft1xfNmDFDc+fOVePGjeXu7q7bb79dhw8flmEYmjhxourXr69q1arpzjvv1IkTJxzW/b///U99+vRRQECAbDab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Ro0auueee7Rs2TKlpaUpPj5eVqtV27ZtK43VAwAAFFuJ75Y6duyY6tate9k+NWrU0Msvv3zFRQHAFePZasA1r8QjN/7+/jp27Jh9ukWLFjp8+LB9+vjx4+rYsWPpVAcAAFBCJQ43hmE4TB88eFDnzp27bB8AAIDyUirX3PyVxWIpi9UCAAD8rTIJNwAAAM5S4guKLRaLTp8+LTc3NxmGIYvFojNnzigjI0OS7P8FAABwhhKHG8MwdP311ztMt2nTxmGa01IAAMBZShxutmzZUhZ1AAAAlIoSh5tu3bqVRR0AAAClosTh5vz588rLy5PNZrO3paamasGCBcrMzFTfvn3VpUuXUi0SAACguEocbqKjo+Xq6qrXXntNknT69GndfPPNys7Olr+/v2bNmqX//e9/6t27d6kXCwAA8HdKfCv4l19+qX79+tmn33zzTeXl5Wnv3r36/vvvFRsbq+nTp5dqkQAAAMVV4nBz5MgRXXfddfbphIQE9evXT15eF57nEhkZydPAAQCA05Q43Li5uens2bP26a+++kohISEO88+cOVM61QEAAJRQia+5ad26td566y1NnjxZn3/+uVJTU9WjRw/7/P379ysgIKBUiwQqFZ5KDQBOVeJwExcXp169eundd9/V0aNHNWTIEPn7+9vnr1mzRp07dy7VIgEAAIrrir7nZseOHfr444/l5+en/v37O8xv3bq1OnToUGoFAgAAlESJw40kNWvWTM2aNSt03oMPPnhVBQEAAFyNEoebzz77rFj9br311hIXAwAAcLVKHG5uu+02+4MxDcMotI/FYlFeXt7VVQYAAHAFShxuatWqpRo1amjIkCG6//775e3tXRZ1AQAAXJESf8/N0aNHNXXqVG3btk0tWrTQsGHDtHXrVnl6esrLy8v+AgAAcIYShxtXV1cNGDBAGzdu1J49e9SyZUvFxMQoMDBQzz33nM6fP18WdQIAABRLicPNpRo0aKC4uDht3rxZ119/vaZMmaKMjIzSqg0AAKDErjjc5OTkaPny5QoNDdVNN90kb29vrVu3TrVr1y7N+gAAAEqkxBcUb9++XUuWLNE777yjoKAgRUVF6d133yXUAACACqHE4eaWW25RgwYNNGLECLVr106S9MUXXxTo17dv36uvDgAAoISu6BuKDx06pIkTJxY5n++5AQAAzlLicJOfn/+3fbKysq6oGAAAgKt1VXdL/VVOTo5mzpypxo0bl+ZqAQAAiq3E4SYnJ0djxoxR+/bt1alTJ61du1aSFB8fr0aNGmnWrFkaNWpUadcJAABQLCU+LRUXF6fXXntNoaGh2rp1q/r376+oqCh99dVXmjlzpvr37y+r1VoWtQIAAPytEoeblStX6s0331Tfvn21a9cutWzZUufPn9f3339vf6AmAACAs5T4tNRvv/1mvwX8pptuks1m06hRowg2AACgQihxuMnLy5Orq6t9ukqVKvLw8CjVogAAAK5UiU9LGYahIUOGyGazSZKys7P18MMPq3r16g79Vq9eXToVAgAAlECJw01kZKTD9KBBg0qtGAAAgKtV4nCzZMmSsqgDAACgVJTql/hdqblz5yooKEhubm4KCQnR9u3bi7XcO++8I4vForvuuqtsCwQAAJWG08PNihUrFBsbq3Hjxmnnzp1q1aqVwsLCdOzYscsud/DgQT355JPq2rVrOVUKAAAqA6eHm5kzZyo6OlpRUVFq3ry5FixYIHd3d8XHxxe5TF5env7973/rhRde4FEPAADAgVPDTW5urnbs2KHQ0FB7m4uLi0JDQ7Vt27Yil5swYYLq1q2rYcOG/e02cnJylJGR4fACAADm5dRwc/z4ceXl5cnX19eh3dfXVykpKYUu88UXX2jx4sVatGhRsbYxefJkeXl52V+BgYFXXTcAAKi4nH5aqiROnz6t+++/X4sWLZK3t3exlhkzZozS09Ptr8OHD5dxlQAAwJlKfCt4afL29pbValVqaqpDe2pqqvz8/Ar0379/vw4ePKiIiAh7W35+vqQL35ScnJysJk2aOCxjs9nsXzgIAADMz6kjN66urmrXrp0SEhLsbfn5+UpISFDHjh0L9L/hhhv0448/Kikpyf7q27evunfvrqSkJE45AQAA547cSFJsbKwiIyPVvn17dejQQbNnz1ZmZqaioqIkSYMHD1a9evU0efJkubm56aabbnJYvmbNmpJUoB0AAFybnB5uBgwYoLS0NMXFxSklJUWtW7fWhg0b7BcZHzp0SC4ulerSIAAA4ERODzeSFBMTo5iYmELnJSYmXnbZpUuXln5BAACg0mJIBAAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmArhBgAAmEqFCDdz585VUFCQ3NzcFBISou3btxfZd9GiReratatq1aqlWrVqKTQ09LL9AQDAtcXp4WbFihWKjY3VuHHjtHPnTrVq1UphYWE6duxYof0TExN13333acuWLdq2bZsCAwN1++2368iRI+VcOQAAqIicHm5mzpyp6OhoRUVFqXnz5lqwYIHc3d0VHx9faP9ly5bpkUceUevWrXXDDTfo9ddfV35+vhISEsq5cgAAUBE5Ndzk5uZqx44dCg0Ntbe5uLgoNDRU27ZtK9Y6srKydO7cOdWuXbvQ+Tk5OcrIyHB4AQAA83JquDl+/Ljy8vLk6+vr0O7r66uUlJRirWP06NEKCAhwCEiXmjx5sry8vOyvwMDAq64bAABUXE4/LXU1pkyZonfeeUdr1qyRm5tboX3GjBmj9PR0++vw4cPlXCUAAChPVZy5cW9vb1mtVqWmpjq0p6amys/P77LLzpgxQ1OmTNHmzZvVsmXLIvvZbDbZbLZSqRcAAFR8Th25cXV1Vbt27RwuBr54cXDHjh2LXG7atGmaOHGiNmzYoPbt25dHqQAAoJJw6siNJMXGxioyMlLt27dXhw4dNHv2bGVmZioqKkqSNHjwYNWrV0+TJ0+WJE2dOlVxcXFavny5goKC7NfmeHh4yMPDw2n7AQAAKganh5sBAwYoLS1NcXFxSklJUevWrbVhwwb7RcaHDh2Si8ufA0zz589Xbm6u/vWvfzmsZ9y4cRo/fnx5lg4AACogp4cbSYqJiVFMTEyh8xITEx2mDx48WPYFAQCASqtS3y0FAADwV4QbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKoQbAABgKhUi3MydO1dBQUFyc3NTSEiItm/fftn+K1eu1A033CA3Nze1aNFC69evL6dKAQBARef0cLNixQrFxsZq3Lhx2rlzp1q1aqWwsDAdO3as0P5bt27Vfffdp2HDhum7777TXXfdpbvuuku7du0q58oBAEBF5PRwM3PmTEVHRysqKkrNmzfXggUL5O7urvj4+EL7v/zyywoPD9dTTz2lZs2aaeLEiWrbtq3mzJlTzpUDAICKyKnhJjc3Vzt27FBoaKi9zcXFRaGhodq2bVuhy2zbts2hvySFhYUV2R8AAFxbqjhz48ePH1deXp58fX0d2n19fbVnz55Cl0lJSSm0f0pKSqH9c3JylJOTY59OT0+XJGVkZFxN6UXKz8kqk/WWRIbFcHYJF5TRMb4cjv8lOP7OxfF3Lo6/c5XB8b/4e9sw/n4fnRpuysPkyZP1wgsvFGgPDAx0QjXlw8vZBVw0pcJUUq4qzF5z/J2L4+9cHH/nKsPjf/r0aXl5XX79Tg033t7eslqtSk1NdWhPTU2Vn59focv4+fmVqP+YMWMUGxtrn87Pz9eJEydUp04dWSyWq9yDiicjI0OBgYE6fPiwPD09nV3ONYfj71wcf+fi+DuX2Y+/YRg6ffq0AgIC/ravU8ONq6ur2rVrp4SEBN11112SLoSPhIQExcTEFLpMx44dlZCQoMcff9zetmnTJnXs2LHQ/jab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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "RMSE_Results = [stack_rmse, lr_rmse, dt_rmse, gbdt_rmse, rf_rmse]\n",
    "R2_Results = [stack_r2, lr_r2, dt_r2, gbdt_r2, rf_r2]\n",
    "\n",
    "rg= np.arange(5)\n",
    "width = 0.35\n",
    "\n",
    "# 1. Create bar plot with RMSE results\n",
    "plt.bar(rg, RMSE_Results, width, label='RMSE')\n",
    "\n",
    "\n",
    "# 2. Create bar plot with R2 results\n",
    "plt.bar(rg + width, R2_Results, width, label='R2')\n",
    "\n",
    "\n",
    "\n",
    "labels = ['Stacking','LR', 'DT', 'GBDT', 'RF']\n",
    "plt.xticks(rg + width/2, labels)\n",
    "\n",
    "plt.xlabel(\"Models\")\n",
    "plt.ylabel(\"RMSE/R2\")\n",
    "\n",
    "\n",
    "plt.ylim([0,1])\n",
    "plt.title('Model Performance')\n",
    "plt.legend(loc='upper left', ncol=2)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Analysis</b>: Compare and contrast the resulting $R^2$ and RSME scores of the ensemble models and the individual models. Are the ensemble models performing better? Which is the best performing model? Explain."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The lower the RSME value and the closer that R2 is to 1 means that it is the best model. This shows that the model created by Random Forest is the best performing model. The ensemble models are not performing better since they are all higher."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.19"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": false,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
