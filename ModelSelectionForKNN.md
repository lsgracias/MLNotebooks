{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Assignment 5: Model Selection for KNN"
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
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this assignment, you will continue practicing the fifth step of the machine learning life cycle and perform model selection to find the best performing KNN model for a classification problem.\n",
    "\n",
    "You will complete the following tasks:\n",
    "\n",
    "1. Build your DataFrame and define your ML problem\n",
    "3. Create labeled examples from the data set\n",
    "4. Split the data into training and test data sets\n",
    "5. Perform a grid search to identify the optimal value of $K$ for a KNN classifier\n",
    "6. Fit the optimal KNN classifier to the training data and make predictions on the test data\n",
    "7. Evaluate the accuracy of the model\n",
    "8. Plot a precision-recall curve for the model\n",
    "\n",
    "\n",
    "<b>Note</b>: Some of the evaluation metrics we will be using are suited for binary classification models that produce probabilities. For this reason, we will be using the `predict_proba()` method to produce class label probability predictions. Recall that KNN is *not* a probabilistic method. Because of this, `predict_proba()` does not output true probabilities. What it does is the following: For n_neighbors=$k$, it identifies the closest $k$ points to a given input point. It then counts up the likelihood, among these $k$ points, of belonging to one of the classes and uses that as the class \"probabilities.\" We will be using KNN for the sake of demonstrating how to use these evaluation metrics.\n",
    "\n",
    "**<font color='red'>Note: Some of the code cells in this notebook may take a while to run.</font>**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 1. Build Your DataFrame and Define Your ML Problem"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Load a Data Set and Save it as a Pandas DataFrame\n",
    "\n",
    "We will work with the \"cell2celltrain\" data set. This version of the data set has been preprocessed and is ready for modeling."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Do not remove or edit the line below:\n",
    "filename = os.path.join(os.getcwd(), \"data_KNN\", \"cell2celltrain.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Task**: Load the data and save it to DataFrame `df`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(filename)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Define the Label\n",
    "\n",
    "This is a binary classification problem in which we will predict customer churn. The label is the `Churn` column.\n",
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
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = df['Churn']\n",
    "X = df.drop(columns = 'Churn', axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 3. Create Training and Test Data Sets\n",
    "<b>Task</b>: In the code cell below, create training and test sets out of the labeled examples. Create a test set that is 10 percent of the size of the data set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1234)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 4. Perform KNN Model Selection Using `GridSearchSV()`\n",
    "\n",
    "Our goal is to find the optimal choice of hyperparameter $K$. We will then train a KNN model using that value of $K$."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Set Up a Parameter Grid \n",
    "\n",
    "<b>Task</b>: Create a dictionary called `param_grid` that contains 10 possible hyperparameter values for $K$. The dictionary should contain the following key/value pair:\n",
    "\n",
    "* A key called 'n_neighbors' \n",
    "* A value which is a list consisting of 10 values for the hyperparameter $K$ \n",
    "\n",
    "For example, your dictionary would look like this: `{'n_neighbors': [1, 2, 3,..]}`\n",
    "\n",
    "The values for hyperparameter $K$  will be in a range that starts at $2$ and ends with $\\sqrt{num\\_examples}$, where `num_examples` is the number of examples in our training set `X_train`. Use the NumPy [np.linspace()](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) function to generate these values, then convert each value to an `int`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'n_neighbors': [1, 24, 48, 72, 95, 119, 143, 166, 190, 214]}"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "num_examples =  np.linspace(1, 214, 10).astype(int)\n",
    "param_grid = {'n_neighbors': num_examples.tolist()}\n",
    "\n",
    "param_grid"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Perform Grid Search Cross-Validation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> Use `GridSearchCV` to search over the different values of hyperparameter $K$ to find the one that results in the best cross-validation (CV) score.\n",
    "\n",
    "Complete the code in the cell below. <b>Note</b>: This will take a few minutes to run."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
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
    "# 1. Create a KNeighborsClassifier model object without supplying arguments. \n",
    "#    Save the model object to the variable 'model'\n",
    "\n",
    "model = KNeighborsClassifier()\n",
    "\n",
    "\n",
    "# 2. Run a grid search with 5-fold cross-validation and assign the output to the object 'grid'.\n",
    "#    * Pass the model and the parameter grid to GridSearchCV()\n",
    "#    * Set the number of folds to 5\n",
    "\n",
    "grid = GridSearchCV(model, param_grid, cv=5)\n",
    "\n",
    "\n",
    "# 3. Fit the model (use the 'grid' variable) on the training data and assign the fitted model to the \n",
    "#    variable 'grid_search'\n",
    "\n",
    "grid_search = grid.fit(X_train, y_train)\n",
    "\n",
    "\n",
    "print('Done')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: Retrieve the value of the hyperparameter $K$ for which the best score was attained. Save the result to the variable `best_k`. Print the result."
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
      "95\n"
     ]
    }
   ],
   "source": [
    "best_k = grid_search.best_params_['n_neighbors']\n",
    "print(best_k)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 5. Train the Optimal KNN Model and Make Predictions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: Initialize a `KNeighborsClassifier` model object with the best value of hyperparameter `K` and fit the model to the training data. The model object should be named `model_best`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
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
       "</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>KNeighborsClassifier(n_neighbors=95)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;KNeighborsClassifier<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.4/modules/generated/sklearn.neighbors.KNeighborsClassifier.html\">?<span>Documentation for KNeighborsClassifier</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>KNeighborsClassifier(n_neighbors=95)</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "KNeighborsClassifier(n_neighbors=95)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_best = KNeighborsClassifier(n_neighbors=best_k)\n",
    "model_best.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> Test your model on the test set (`X_test`).\n",
    "\n",
    "1. Use the ``predict_proba()`` method  to use the fitted model `model_best` to predict class probabilities for the test set. Note that the `predict_proba()` method returns two columns, one column per class label. The first column contains the probability that an unlabeled example belongs to class `False` (Churn is \"False\") and the second column contains the probability that an unlabeled example belongs to class `True` (Churn is \"True\"). Save the values of the *second* column to a list called ``probability_predictions``.\n",
    "\n",
    "2. Use the ```predict()``` method to use the fitted model `model_best` to predict the class labels for the test set. Store the outcome in the variable ```class_label_predictions```. Note that the `predict()` method returns the class label (True or False) per unlabeled example."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Make predictions on the test data using the predict_proba() method\n",
    "probabilities = model_best.predict_proba(X_test)\n",
    "probability_predictions = probabilities[:, 1]\n",
    "    \n",
    "# 2. Make predictions on the test data using the predict() method \n",
    "class_label_predictions = model_best.predict(X_test)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 6. Evaluate the Accuracy of the Model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: Compute and print the model's accuracy score using `accuracy_score()`."
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
      "Model's accuracy score: 0.7134182174338883\n"
     ]
    }
   ],
   "source": [
    "accuracy = accuracy_score(y_test, class_label_predictions)\n",
    "\n",
    "print(f\"Model's accuracy score: {accuracy}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> Create a confusion matrix to evaluate your model. Use the Confusion Matrix Demo as a reference."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix for the model: \n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Predicted: Customer Will Leave</th>\n",
       "      <th>Predicted: Customer Will Stay</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Actual: Customer Will Leave</th>\n",
       "      <td>0</td>\n",
       "      <td>1463</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Actual: Customer Will Stay</th>\n",
       "      <td>0</td>\n",
       "      <td>3642</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                             Predicted: Customer Will Leave  \\\n",
       "Actual: Customer Will Leave                               0   \n",
       "Actual: Customer Will Stay                                0   \n",
       "\n",
       "                             Predicted: Customer Will Stay  \n",
       "Actual: Customer Will Leave                           1463  \n",
       "Actual: Customer Will Stay                            3642  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Display a confusion matrix\n",
    "print('Confusion Matrix for the model: ')\n",
    "\n",
    "# Computes the confusion matrix\n",
    "c_m = confusion_matrix(y_test, class_label_predictions, labels=[True, False])\n",
    "\n",
    "# Create a Pandas DataFrame out of the confusion matrix for display purposes\n",
    "pd.DataFrame(\n",
    "c_m,\n",
    "columns=['Predicted: Customer Will Leave', 'Predicted: Customer Will Stay'],\n",
    "index=['Actual: Customer Will Leave', 'Actual: Customer Will Stay']\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 7.  Plot the Precision-Recall Curve "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Recall that scikit-learn defaults to a 0.5 classification threshold. Sometimes we may want a different threshold. We can use the precision-recall curve to show the trade-off between precision and recall for different classification thresholds. Scikit-learn's `precision_recall_curve()` function computes precision-recall pairs for different probability thresholds. For more information, consult the [Scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html).\n",
    "\n",
    "Let's first import the function."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import precision_recall_curve"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> You will use `precision_recall_curve()` to compute precision-recall pairs. In the code cell below, call the function with the arguments `y_test` and `probability_predictions`. The function returns three outputs. Save the three items to the variables `precision`, `recall`, and `thresholds`, respectively. \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "precision, recall, thresholds = precision_recall_curve(y_test, probability_predictions)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The code cell below uses seaborn's `lineplot()` function to visualize the precision-recall curve. Variable `recall` will be on the $x$ axis and `precision` will be on the $y$-axis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjcAAAHHCAYAAABDUnkqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/P9b71AAAACXBIWXMAAA9hAAAPYQGoP6dpAABIHklEQVR4nO3deXhU1f3H8c/MJDMJ2VhCwmIk7KhsGiAGpFEMICj9UVulghJxQ8EWoS7ghkprtEWFKopaBdti2arWKqIYRUVoVQSLyqaAIJCwJyHbZGbu74+QISNJSMIkN7l5v55nnmRuzp35zs0yn5x7zrk2wzAMAQAAWITd7AIAAACCiXADAAAshXADAAAshXADAAAshXADAAAshXADAAAshXADAAAshXADAAAshXADAAAshXADQNdff70SExNrtM/q1atls9m0evXqOqmpMdm1a5dsNpsWLlzo3/bQQw/JZrOZVxTQhBFuABMsXLhQNpvNfwsLC1O3bt10++23Kzs72+zyAKBRCzG7AKApe+SRR9SxY0cVFRVpzZo1eu6557RixQp9/fXXatasWb3V8eKLL8rn89Von5/97GcqLCyU0+mso6oAoHYIN4CJRowYoX79+kmSbrrpJrVq1UpPPvmk/vWvf+maa66pcJ/8/HxFREQEtY7Q0NAa72O32xUWFhbUOs5EUVGRnE6n7HY6pMvzeDzy+XyEUDQp/BUAGpAhQ4ZIknbu3CmpdCxMZGSkvv/+e40cOVJRUVEaN26cJMnn82nOnDk677zzFBYWpvj4eE2cOFFHjx495XHfeecdpaamKioqStHR0erfv79effVV/9crGnOzePFiJSUl+ffp1auX5s6d6/96ZWNuli1bpqSkJIWHhys2NlbXXnut9u7dG9Cm7HXt3btXo0ePVmRkpFq3bq0777xTXq/3tMep7LkXL16s+++/X+3bt1ezZs2Um5srSfrvf/+ryy67TDExMWrWrJlSU1P16aefnvI4e/fu1Y033qh27drJ5XKpY8eOuu222+R2uyVJR44c0Z133qlevXopMjJS0dHRGjFihL766qvT1lgT//3vfzVy5Ei1aNFCERER6t27d8Cxvvjii3XxxRefst9Pv29lY39mz56tOXPmqHPnznK5XNqwYYNCQkL08MMPn/IYW7dulc1m0zPPPOPfduzYMd1xxx1KSEiQy+VSly5d9Pjjj9e4dw8wCz03QAPy/fffS5JatWrl3+bxeDR8+HBddNFFmj17tv901cSJE7Vw4UJNmDBBv/3tb7Vz504988wz2rBhgz799FN/b8zChQt1ww036LzzztOMGTPUvHlzbdiwQStXrtTYsWMrrGPVqlW65pprdOmll+rxxx+XJG3evFmffvqppkyZUmn9ZfX0799fGRkZys7O1ty5c/Xpp59qw4YNat68ub+t1+vV8OHDlZycrNmzZ+v999/XE088oc6dO+u2226r1vGaNWuWnE6n7rzzThUXF8vpdOqDDz7QiBEjlJSUpJkzZ8put2vBggUaMmSIPvnkEw0YMECStG/fPg0YMEDHjh3TLbfcoh49emjv3r1avny5CgoK5HQ6tWPHDr3xxhu66qqr1LFjR2VnZ+v5559Xamqqvv32W7Vr165adVZl1apVuuKKK9S2bVtNmTJFbdq00ebNm/XWW29VeayrsmDBAhUVFemWW26Ry+VS27ZtlZqaqqVLl2rmzJkBbZcsWSKHw6GrrrpKklRQUKDU1FTt3btXEydO1Nlnn621a9dqxowZ2r9/v+bMmXOmLxmoewaAerdgwQJDkvH+++8bBw8eNPbs2WMsXrzYaNWqlREeHm78+OOPhmEYRnp6uiHJmD59esD+n3zyiSHJWLRoUcD2lStXBmw/duyYERUVZSQnJxuFhYUBbX0+n//z9PR0o0OHDv77U6ZMMaKjow2Px1Ppa/jwww8NScaHH35oGIZhuN1uIy4uzujZs2fAc7311luGJOPBBx8MeD5JxiOPPBLwmOeff76RlJRU6XP+9Lk7depkFBQUBLymrl27GsOHDw94fQUFBUbHjh2NoUOH+reNHz/esNvtxueff37K45ftW1RUZHi93oCv7dy503C5XAG179y505BkLFiwwL9t5syZxun+xHo8HqNjx45Ghw4djKNHj1ZYg2EYRmpqqpGamnrK/j/9vpXVER0dbRw4cCCg7fPPP29IMjZt2hSw/dxzzzWGDBnivz9r1iwjIiLC2LZtW0C76dOnGw6Hw9i9e3eVrwloCDgtBZgoLS1NrVu3VkJCgn79618rMjJSr7/+utq3bx/Q7qc9GcuWLVNMTIyGDh2qQ4cO+W9JSUmKjIzUhx9+KKm0VyAvL0/Tp08/ZXxMVdOUmzdvrvz8fK1atarar+WLL77QgQMHNGnSpIDnuvzyy9WjRw+9/fbbp+xz6623BtwfPHiwduzYUe3nTE9PV3h4uP/+xo0btX37do0dO1aHDx/2H5f8/Hxdeuml+vjjj+Xz+eTz+fTGG29o1KhR/jFP5ZUdG5fL5R/D4/V6dfjwYUVGRqp79+768ssvq11nZTZs2KCdO3fqjjvuCOjVKl9Dbfzyl79U69atA7ZdeeWVCgkJ0ZIlS/zbvv76a3377bcaM2aMf9uyZcs0ePBgtWjRIuBnKy0tTV6vVx9//HGt6wLqC6elABPNmzdP3bp1U0hIiOLj49W9e/dTBsSGhITorLPOCti2fft25eTkKC4ursLHPXDggKSTp7l69uxZo7omTZqkpUuXasSIEWrfvr2GDRumq6++Wpdddlml+/zwww+SpO7du5/ytR49emjNmjUB28LCwk55A27RokXAmKGDBw8GjMGJjIxUZGSk/37Hjh0D9t++fbuk0tBTmZycHLndbuXm5p72uPh8Ps2dO1fPPvusdu7cGVBL+VOHtVXb78/p/PS4SFJsbKwuvfRSLV26VLNmzZJUekoqJCREV155pb/d9u3b9b///e+U702Zsp8toCEj3AAmGjBgQIU9B+WV7z0o4/P5FBcXp0WLFlW4T2VvTNUVFxenjRs36t1339U777yjd955RwsWLND48eP1yiuvnNFjl3E4HKdt079/f39okqSZM2fqoYce8t8v32sjyT/g9U9/+pP69u1b4WNGRkbqyJEj1arx0Ucf1QMPPKAbbrhBs2bNUsuWLWW323XHHXfU6+Bam80mwzBO2V7Z4OufHpcyv/71rzVhwgRt3LhRffv21dKlS3XppZcqNjbW38bn82no0KG6++67K3yMbt261eIVAPWLcAM0Qp07d9b777+vQYMGVfpGVtZOKj390KVLlxo9h9Pp1KhRozRq1Cj5fD5NmjRJzz//vB544IEKH6tDhw6SSmfflM36KrN161b/12ti0aJFKiws9N/v1KlTle3LXm90dLTS0tIqbde6dWtFR0fr66+/rvLxli9frksuuUQvvfRSwPZjx44FBILaKv/9qareFi1aVHi6rnzwq47Ro0dr4sSJ/lNT27Zt04wZM06p6fjx41XWAzR0jLkBGqGrr75aXq/Xf3qhPI/Ho2PHjkmShg0bpqioKGVkZKioqCigXUU9AWUOHz4ccN9ut6t3796SpOLi4gr36devn+Li4jR//vyANu+88442b96syy+/vFqvrbxBgwYpLS3NfztduElKSlLnzp01e/ZsHT9+/JSvHzx40P96Ro8erX//+9/64osvTmlXdmwcDscpx2nZsmWnTG2vrQsuuEAdO3bUnDlz/N+zn9YglQaOLVu2+OuXpK+++qrC6e1Vad68uYYPH66lS5dq8eLFcjqdGj16dECbq6++WuvWrdO77757yv7Hjh2Tx+Op0XMCZqDnBmiEUlNTNXHiRGVkZGjjxo0aNmyYQkNDtX37di1btkxz587Vr371K0VHR+upp57STTfdpP79+2vs2LFq0aKFvvrqKxUUFFR6iummm27SkSNHNGTIEJ111ln64Ycf9PTTT6tv374655xzKtwnNDRUjz/+uCZMmKDU1FRdc801/qngiYmJmjp1al0eEkmloeUvf/mLRowYofPOO08TJkxQ+/bttXfvXn344YeKjo7Wv//9b0mlp5zee+89paam6pZbbtE555yj/fv3a9myZVqzZo2aN2+uK664Qo888ogmTJiggQMHatOmTVq0aNFpQ1ZN6n3uuec0atQo9e3bVxMmTFDbtm21ZcsWffPNN/6AccMNN+jJJ5/U8OHDdeONN+rAgQOaP3++zjvvPP/aPtU1ZswYXXvttXr22Wc1fPjwUwYy33XXXXrzzTd1xRVX6Prrr1dSUpLy8/O1adMmLV++XLt27QpKrxVQp0ydqwU0UWVTwSuahlxeenq6ERERUenXX3jhBSMpKckIDw83oqKijF69ehl33323sW/fvoB2b775pjFw4EAjPDzciI6ONgYMGGD84x//CHie8lOKly9fbgwbNsyIi4sznE6ncfbZZxsTJ0409u/f72/z06ngZZYsWWKcf/75hsvlMlq2bGmMGzfOP7X9dK+rOtOnyz/3smXLKvz6hg0bjCuvvNJo1aqV4XK5jA4dOhhXX321kZmZGdDuhx9+MMaPH2+0bt3acLlcRqdOnYzJkycbxcXFhmGUTgX/3e9+Z7Rt29YIDw83Bg0aZKxbt+6Uqdm1nQpeZs2aNcbQoUONqKgoIyIiwujdu7fx9NNPB7T5+9//bnTq1MlwOp1G3759jXfffbfSqeB/+tOfKn2u3NxcIzw83JBk/P3vf6+wTV5enjFjxgyjS5cuhtPpNGJjY42BAwcas2fPNtxud7VeE2Am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f/55paamat68eYqJidHixYsVGhoqSerWrZv/sYcMGRLwXC+88IKaN2+ujz76SFdccUX9vUgA9YJwA6BBuOSSS/Tcc88pPz9fTz31lEJCQvTLX/5S33zzjQoKCjR06NCA9m63W+eff74kaePGjRo8eLA/2PxUdna27r//fq1evVoHDhyQ1+tVQUGBdu/eXeevC0D9I9wAaBAiIiLUpUsXSdLLL7+sPn366KWXXlLPnj0lSW+//bbat28fsI/L5ZIkhYeHV/nY6enpOnz4sObOnasOHTrI5XIpJSVFbre7Dl4JALMRbgA0OHa7Xffee6+mTZumbdu2yeVyaffu3UpNTa2wfe/evfXKK6+opKSkwt6bTz/9VM8++6xGjhwpSdqzZ48OHTpUp68BgHkYUAygQbrqqqvkcDj0/PPP684779TUqVP1yiuv6Pvvv9eXX36pp59+Wq+88ook6fbbb1dubq5+/etf64svvtD27dv1t7/9TVu3bpUkde3aVX/729+0efNm/fe//9W4ceNO29sDoPGi5wZAgxQSEqLbb79df/zjH7Vz5061bt1aGRkZ2rFjh5o3b64LLrhA9957rySpVatW+uCDD3TXXXcpNTVVDodDffv21aBBgyRJL730km655RZdcMEFSkhI0KOPPqo777zTzJcHoA7ZDMMwzC4CAAAgWDgtBQAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALIVwAwAALOX/AZFhIeZ6M89hAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure()\n",
    "ax = fig.add_subplot(111)\n",
    "\n",
    "sns.lineplot(x=recall, y=precision, marker = 'o')\n",
    "\n",
    "plt.title(\"Precision-recall curve\")\n",
    "plt.xlabel(\"Recall\")\n",
    "plt.ylabel(\"Precision\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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