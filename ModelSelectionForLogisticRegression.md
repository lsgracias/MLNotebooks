{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Lab 5: ML Life Cycle: Evaluation and Deployment"
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
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this lab, you will continue practicing the evaluation phase of the machine learning life cycle. You will perform model selection for logistic regression to solve a classification problem. You will complete the following tasks:\n",
    "    \n",
    "\n",
    "1. Build your DataFrame and define your ML problem:\n",
    "    * Load the Airbnb \"listings\" data set\n",
    "    * Define the label - what are you predicting?\n",
    "    * Identify the features\n",
    "2. Create labeled examples from the data set\n",
    "3. Split the data into training and test data sets\n",
    "4. Train, test and evaluate a logistic regression (LR) model using the scikit-learn default value for hyperparameter $C$\n",
    "5. Perform a grid search to identify the optimal value of $C$ for a logistic regression model\n",
    "6. Train, test and evaluate a logisitic regression model using the optimal value of $C$\n",
    "7. Plot a precision-recall curve for both models\n",
    "8. Plot the ROC and compute the AUC for both models\n",
    "9. Perform feature selection\n",
    "10. Make your model persistent for future use\n",
    "\n",
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
    "We will work with the data set ``airbnbData_train``. This data set already has all the necessary preprocessing steps implemented, including one-hot encoding of the categorical variables, scaling of all numerical variable values, and imputing missing values. It is ready for modeling.\n",
    "\n",
    "<b>Task</b>: In the code cell below, use the same method you have been using to load the data using `pd.read_csv()` and save it to DataFrame `df`.\n",
    "\n",
    "You will be working with the file named \"airbnbData_train.csv\" that is located in a folder named \"data_LR\"."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "filename = os.path.join(os.getcwd(), \"data_LR\", \"airbnbData_train.csv\")\n",
    "df = pd.read_csv(filename, header=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Define the Label\n",
    "\n",
    "Your goal is to train a machine learning model that predicts whether an Airbnb host is a 'super host'. This is an example of supervised learning and is a binary classification problem. In our dataset, our label will be the `host_is_superhost` column and the label will either contain the value `True` or `False`.\n",
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
    "<b>Task</b>: In the code cell below, create labeled examples from DataFrame `df`. Assign the label to variable `y` and the features to variable `X`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = df['host_is_superhost']\n",
    "X = df.drop(columns = 'host_is_superhost', axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 3. Create Training and Test Data Sets\n",
    "<b>Task</b>: In the code cell below, create training and test sets out of the labeled examples. Create a test set that is 10 percent of the size of the data set. Save the results to variables `X_train, X_test, y_train, y_test`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.10, random_state=1234)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 4. Train, Test and Evaluate a Logistic Regression Model With Default Hyperparameter Values\n",
    "\n",
    "You will fit a logisitic regression model to the training data using scikit-learn's default value for hyperparameter $C$. You will then make predictions on the test data and evaluate the model's performance. The goal is to later find a value for hyperparameter $C$ that can improve this performance of the model on the test data.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: In the code cell below:\n",
    "\n",
    "1. Using the scikit-learn `LogisticRegression` class, create a logistic regression model object with the following arguments: `max_iter=1000`. You will use the scikit-learn default value for hyperparameter $C$, which is 1.0. Assign the model object to the variable `model_default`.\n",
    "\n",
    "2. Fit the model to the training data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
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
       "</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression(max_iter=1000)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;LogisticRegression<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html\">?<span>Documentation for LogisticRegression</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>LogisticRegression(max_iter=1000)</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression(max_iter=1000)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = LogisticRegression(max_iter=1000)\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> Test your model on the test set (`X_test`). \n",
    "\n",
    "1. Use the ``predict_proba()`` method  to use the fitted model to predict class probabilities for the test set. Note that the `predict_proba()` method returns two columns, one column per class label. The first column contains the probability that an unlabeled example belongs to class `False` (`great_quality` is \"False\") and the second column contains the probability that an unlabeled example belongs to class `True` (`great_quality` is \"True\"). Save the values of the *second* column to a list called ``proba_predictions_default``.\n",
    "\n",
    "2. Use the ```predict()``` method to use the fitted model `model_default` to predict the class labels for the test set. Store the outcome in the variable ```class_label_predictions_default```. Note that the `predict()` method returns the class label (True or False) per unlabeled example."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Make predictions on the test data using the predict_proba() method\n",
    "proba_predictions = model.predict_proba(X_test)\n",
    "\n",
    "# Save the values of the second column to a list called proba_predictions_default\n",
    "proba_predictions_default = proba_predictions[:, 1]\n",
    "\n",
    "# 2. Make predictions on the test data using the predict() method\n",
    "class_label_predictions_default = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: Evaluate the accuracy of the model using a confusion matrix. In the cell below, create a confusion matrix out of `y_test` and `class_label_predictions_default`."
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
      "[[1997   91]\n",
      " [ 451  264]]\n"
     ]
    }
   ],
   "source": [
    "conf_matrix = confusion_matrix(y_test, class_label_predictions_default)\n",
    "print(conf_matrix)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 5. Perform Logistic Regression Model Selection Using `GridSearchSV()`\n",
    "\n",
    "Our goal is to find the optimal choice of hyperparameter $C$. We will then fit a logistic regression model to the training data using this value of $C$. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Set Up a Parameter Grid \n",
    "\n",
    "<b>Task</b>: Create a dictionary called `param_grid` that contains 10 possible hyperparameter values for $C$. The dictionary should contain the following key/value pair:\n",
    "\n",
    "* a key called `C` \n",
    "* a value which is a list consisting of 10 values for the hyperparameter $C$. A smaller value for “C” (e.g. C=0.01) leads to stronger regularization and a simpler model, while a larger value (e.g. C=1.0) leads to weaker regularization and a more complex model. Use the following values for $C$: `cs=[10**i for i in range(-5,5)]`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'C': [1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create a list of 10 values for the hyperparameter 'C'\n",
    "cs = [10**i for i in range(-5,5)]\n",
    "\n",
    "# Create a dictionary for the parameter grid\n",
    "param_grid = {'C': cs}\n",
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
    "<b>Task:</b> Use `GridSearchCV` to search over the different values of hyperparameter $C$ to find the one that results in the best cross-validation (CV) score.\n",
    "\n",
    "Complete the code in the cell below. <b>Note</b>: This will take a few minutes to run."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
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
    "# 1. Create a LogisticRegression model object with the argument max_iter=1000. \n",
    "#    Save the model object to the variable 'model'\n",
    "model = LogisticRegression(max_iter=1000)\n",
    "\n",
    "\n",
    "# 2. Run a grid search with 5-fold cross-validation and assign the output to the \n",
    "# object 'grid'.\n",
    "grid = GridSearchCV(model, param_grid, cv=5)\n",
    "\n",
    "\n",
    "# 3. Fit the model on the training data and assign the fitted model to the \n",
    "#    variable 'grid_search'\n",
    "grid_search = grid.fit(X_train, y_train)\n",
    "\n",
    "print('Done')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: Retrieve the value of the hyperparameter $C$ for which the best score was attained. Save the result to the variable `best_c`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "100"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "best_C = grid_search.best_params_['C']\n",
    "\n",
    "best_C"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 6. Train, Test and Evaluate the Optimal Logistic Regression Model \n",
    "\n",
    "Now that we have the optimal value for hyperparameter $C$, let's train a logistic regression model using that value, test the model on our test data, and evaluate the model's performance. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: Initialize a `LogisticRegression` model object with the best value of hyperparameter `C` model and fit the model to the training data. The model object should be named `model_best`. Note: Supply `max_iter=1000` as an argument when creating the model object."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
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
       "</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression(C=100, max_iter=1000)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;LogisticRegression<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html\">?<span>Documentation for LogisticRegression</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>LogisticRegression(C=100, max_iter=1000)</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression(C=100, max_iter=1000)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Initialize the LogisticRegression model with the best value of C\n",
    "model_best = LogisticRegression(C=best_C, max_iter=1000)\n",
    "\n",
    "# Fit the model to the training data\n",
    "model_best.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> Test your model on the test set (`X_test`).\n",
    "\n",
    "1. Use the ``predict_proba()`` method  to use the fitted model `model_best` to predict class probabilities for the test set. Save the values of the *second* column to a list called ``proba_predictions_best``.\n",
    "\n",
    "2. Use the ```predict()``` method to use the fitted model `model_best` to predict the class labels for the test set. Store the outcome in the variable ```class_label_predictions_best```. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Make predictions on the test data using the predict_proba() method\n",
    "proba_predictions = model_best.predict_proba(X_test)\n",
    "\n",
    "# Save the values of the second column to a list \n",
    "proba_predictions_best = proba_predictions[:, 1]\n",
    "\n",
    "\n",
    "# 2. Make predictions on the test data using the predict() method\n",
    "class_label_predictions_best = model_best.predict(X_test)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: Evaluate the accuracy of the model using a confusion matrix. In the cell below, create a confusion matrix out of `y_test` and `class_label_predictions_best`."
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
      "[[1997   91]\n",
      " [ 447  268]]\n"
     ]
    }
   ],
   "source": [
    "cm = confusion_matrix(y_test, class_label_predictions_best)\n",
    "print(cm)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 7.  Plot Precision-Recall Curves for Both Models"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> In the code cell below, use `precision_recall_curve()` to compute precision-recall pairs for both models.\n",
    "\n",
    "For `model_default`:\n",
    "* call `precision_recall_curve()` with `y_test` and `proba_predictions_default`\n",
    "* save the output to the variables `precision_default`, `recall_default` and `thresholds_default`, respectively\n",
    "\n",
    "For `model_best`:\n",
    "* call `precision_recall_curve()` with `y_test` and `proba_predictions_best`\n",
    "* save the output to the variables `precision_best`, `recall_best` and `thresholds_best`, respectively\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "precision_default, recall_default, thresholds_default = precision_recall_curve(y_test, proba_predictions_default)\n",
    "precision_best, recall_best, thresholds_best = precision_recall_curve(y_test, proba_predictions_best)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the code cell below, create two `seaborn` lineplots to visualize the precision-recall curve for both models. \"Recall\" will be on the $x$-axis and \"Precision\" will be on the $y$-axis. \n",
    "\n",
    "The plot for \"default\" should be green. The plot for the \"best\" should be red.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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D8uXL2f06NHMnj37gsetWLEiv/32G0uXLmXLli0MGDCAY8eO5enTunVrvvrqK5YsWcKGDRvo06fPOaNa/xUeHo63tzdz587l2LFjJCUlFcyHcAGFKjhFRkae8wEfO3aMgIAAvL29z7uPp6cnAQEBeV4iIiIiInJ+Y8aMoXnz5nTu3Jk2bdrQrFkz4uLicrdPmjSJ3r17M2zYMCpXrkzXrl1ZsWIFpUuXvuAxn3nmGW644QbatWtHy5YtiYyMPGc17ZEjR9KiRQs6derELbfcQteuXSlfvvwFj+nm5sa7777LRx99RFRUFLfeeutVX/vFWIzLWdT9GrJYLPkuR/7UU08xZ84cNmzYkNt27733cvLkSebOnXtJ50lOTiYwMJCkpCSXCFFpScfZt2wuZRq3xycw1OxyRERERKQAZGRksGfPHsqWLYuXl5fZ5RRrF/tdXE42MHXEKSUlhbVr17J27VrAudz42rVr2b9/P+BMnb17987t/9BDD7F7926efPJJtm7dyocffsjUqVN5/PHHzShfRERERESKCVOD08qVK6lbty5169YFYOjQodStW5fnnnsOgCNHjuSGKHA+HXj27Nn89ttv1K5dmzfffJNPPvlES5GLiIiIiMg1Zepy5C1btjxnycH/+vzzz8+7z5o1a65hVSIiIiIiInkVqsUhREREREREzKDgJCIiIiIikg8FJxERERERkXwoOImIiIiIiORDwUlERERERCQfCk4iIiIiIiL5UHASEREREZFcLVu25LHHHjO7DJej4CQiIiIiItfFokWLsFgsJCYmml3KZVNwEhERERERyYeCk4iIiIjI9WAYkJpqzsswLqvUnJwchgwZQmBgIKGhoTz77LMYZ46RmZnJ8OHDiY6OxtfXl4YNG7Jo0aLcffft20fnzp0JDg7G19eX6tWrM2fOHPbu3UurVq0ACA4OxmKx0Ldv34L6dK85N7MLEBEREREpFtLSwM/PnHOnpICv7yV3/+KLL7j//vtZvnw5K1eu5MEHH6R06dI88MADDBkyhM2bNzN58mSioqKYPn067du3Z8OGDVSsWJHBgweTlZXFH3/8ga+vL5s3b8bPz4+YmBh+/PFH7rjjDrZt20ZAQADe3t7X8KILloKTiIiIiIjkERMTw9tvv43FYqFy5cps2LCBt99+m3bt2jFp0iT2799PVFQUAMOHD2fu3LlMmjSJV199lf3793PHHXdQs2ZNAMqVK5d73JCQEADCw8MJCgq67td1NRScRERERESuBx8f58iPWee+DI0aNcJiseS+b9y4MW+++SYbNmzAbrdTqVKlPP0zMzMpUaIEAI888ggDBw7k119/pU2bNtxxxx3UqlXr6q/BZApOIiIiIiLXg8VyWdPlXFFKSgo2m41Vq1Zhs9nybPM7Mw2xf//+tGvXjtmzZ/Prr78yevRo3nzzTR5++GEzSi4wWhxCRERERETy+Oeff/K8//vvv6lYsSJ169bFbrcTHx9PhQoV8rwiIyNz+8fExPDQQw8xbdo0hg0bxsSJEwHw8PAAwG63X7+LKSAKTiIiIiIiksf+/fsZOnQo27Zt47vvvuO9997j0UcfpVKlSvTo0YPevXszbdo09uzZw/Llyxk9ejSzZ88G4LHHHmPevHns2bOH1atXs3DhQqpWrQpAmTJlsFgszJo1i4SEBFLMmrp4BRScREREREQkj969e5Oenk6DBg0YPHgwjz76KA8++CAAkyZNonfv3gwbNozKlSvTtWtXVqxYQenSpQHnaNLgwYOpWrUq7du3p1KlSnz44YcAREdH88ILLzBixAgiIiIYMmSIadd4uSyGcZmLuhdyycnJBAYGkpSUREBAgNnlkJZ0nH3L5lKmcXt8AkPNLkdERERECkBGRgZ79uyhbNmyeHl5mV1OsXax38XlZAONOImIiIiIiORDwUlERERERCQfCk4iIiIiIiL5UHASERERERHJh4KTiIiIiMg1UszWYXNJBfU7UHASERERESlg7u7uAKSlpZlciWRlZQFgs9mu6jhuBVGMiIiIiIj8y2azERQURHx8PAA+Pj5YLBaTqyp+HA4HCQkJ+Pj44OZ2ddFHwUlERERE5BqIjIwEyA1PYg6r1Urp0qWvOrgqOImIiIiIXAMWi4WSJUsSHh5Odna22eUUWx4eHlitV3+HkoKTiIiIiMg1ZLPZrvr+GjGfFocQERERERHJh4KTiIiIiIhIPhScRERERERE8qHgJCIiIiIikg8FJxERERERkXwoOImIiIiIiORDwUlERERERCQfCk4iIiIiIiL5UHASTqSdICE1wewyRERERERcloJTMZaZmsyyRqXY1KIq+zcthexss0sSEREREXFJCk7F2Iq7m9L4n0PcuDKB+N9nKDiJiIiIiFyAglMxtfTtYTSbszH3fc7yZRiGYWJFIiIiIiKuS8GpGDq06W+qPf0WAKd8bQDErNtDlj3LzLJERERERFyWglMx47DncPzODgRlwKayfpycPwuAGvszOJ1w0OTqRERERERck4JTMbPkf72ovTWRFA/wnfoT5Rq0Y3uoBTcHnPhlutnliYiIiIi4JAWnYuTw1hXUGTcZgFWDbye23k1YLBb+rhoAgG3BAjPLExERERFxWQpOxciBvrcTmAkby/nR/I3Jue1b6pYCIGzpWjh0KO/qesnJ0KMHvPoqHD6slfdEREREpFhScComVn81hob/HCTHCm6ffIbVzT13W0JcVTJsEHgsCf78899wlJMD3brBt9/C88/DgQMKTiIiIiJSLCk4FQM52Zn4jHwWgD9vqUWVVnfl2R4WVoZ5Fc68mT/f+adhwMMPw9y5zvfZ2XBQi0eIiIiISPGk4FQM/P3yQ1Q5lEmit4U6H047Z3vp4FimVT3zZuFCAI6NewUmTACLBUJCnNv27r0+BYuIiIiIuBgFpyIuMyWJiuO+AmBN/04ElSp/Tp+yoRWYURmyrcDOnez4+DWCnnCOUJ14djiOjh2cHXfvvl5li4iIiIi4FAWnIm7pqPuJSLZzKMhGo1e/PG+fCiEVSPSGBWWd78sPfQlPO/xUGSbeFExOlUrODXv2XKeqRURERERci4JTEZaRmkTlT53PZtr9UDe8/YLO26+kX0lKBZTKna5nNWBnMPTtCjN3zsKoUsW5QcFJRERERIopBacibMXLg4hKcnAk0EajZz+6YD8vNy9uLHMjP1WBNDdId4NdE18nyRv+Pvg3S/0TnR337AGH498d16+H48ev7UWIiIiIiLgABaciyp6dRezHUwHYev+tuPv4XbCvzWrjptibiPeDhg/A1M+G0e6OJ2lSqgkOw8HNfw4g0wZkZpK9Z5dzpw8+gNq1oWtXPd9JRERERIo8N7MLkGtjxYTnaHQyhxM+Fuo/NyHf/tXCqtGvVh9KWgPo3eFVAHrW6snSg0ux22BbCagVD/atm2H3btyHDHHu+NdfzpGooCBwd7/wCURERERECjGNOBVR7h+OB2Bjl0b4BYbl2z8qIIqXbnqFV259F4uHBwC3V72dCiHOBzxtOXOI5G8nkXH37Xl33rnz/AdNSICRI+GffzQqJSIiIiKFmkaciqDti6YRtzUZuwWqPPvOJe1TOrD0OW0h3iEs7rsYh8PBZ4tjYZOd8J9+A2BhLFQIq0zMim2wdeu5Bzx2DKpXhxMnYMkSeOcdjUqJiIiISKGlEaciKP6NUQAsbxBFRLX6V3wcd5s7Uf5RlAosRWDtBrntO0Lgjrvh97DTzoYtW/LumJAAN93kDE3gnM6XmXnFdYiIiIiImE3BqYhJTzpB7fkbAbA9/GiBHbdpp0HYLZDoBfu+fp9TPjA3IN658T8jTsf3b+Nww2qwaRP2kpH/HmDz5gKrRURERETkelNwKmJWfvgM/lmwr4Qb9boPK7DjVqvXnvkv3Uf8/Bm0bj8QH3cf/o7McW7ctQsyM0mOP0BC87pE7TnOUX8Lyz5/mZzbb3P2WbOmwGoREREREbneFJyKGI9vpwCwr3NzrFZbgR3X3dOHWncNpuwNN2G1WKkYUpH9gZAR4AM5OWSu/IedLWpRdX868T7QupfB6IRpOJo2dh5g7drcYx0+fRjDMAqsNhERERGRa03BqQg5vH0V9TadAqDCoy8U6LHdvXyIrHQD7l4+gHP5ciywu2wQACn33M4NWxM57QFLPv4fW8MtzNk5h5XlvZ0HWLcOR042j819jOi3oun7U18OJR8i266V9kRERETE9Sk4FSE7338RmwHrKwQQVaf5NT1X3ci6AKyItANQ4mQ62VbYOfF17ujxMvfVvQ+AB/a9j8PPF1JT+WbELXz4l3OVvy/Xf8mnaz4l26HgJCIiIiKuT8GpCIn4eT4Ax+/ocM3P1SK2BRYszAo4ltv29/P9qdv7SQCeb/k8QV5BbD61jW0VggHoNf4vZn8LFX1iAHj9r9fZfWr3Na9VRERERORqKTgVEYfW/0Xl/anYLVDtoWeu+fmqhVWjbfm2zKwEX9aC6Y+3p/mzE3O3R/hG8FyL5wD4OuRgbvvNu2HrsbtoFtOMtOw0+v7Ul32J+654yl56djqPzX2MJp82YeXhlZr6JyIiIiLXhIJTEbFt0hgANlQKJDK2xjU/n4+7D4PqDaJ0eAX+eaYPXV/7Oc92d5s7jzV8jJaxLfnkBphcHY7GVQbAOms2n3b6GH8Pf1YdWcWzC5+9oil7O07soNGnjXjnn3dYdnAZ7/7zrqb+iYiIiMg1oeBURATMXQhASoc21+V8VouVGhE1WDlgFR/c9TkWD49z+lgsFiZ2nki92h2wfDieyMmznBu2b6dsijtvtXsLgK/Wf8XkDZM5fPow2TlZcPhwvueftmUa9SbWY/2x9bltc3bMIceRUzAXKCIiIiLyHwpORcCJgzuosy0ZgAp9h16385YLLkeAZ8BF+5QJLMPEWz/h9qb3Q9myUKYMGAbufy6l/9Eo9k6O5PbNMGTOYBZunM2pW9tBdDQ88ABkZZ1zvOzMdBZ0a0C15nfQYHMyzUo1Zc+jeyjhXYIT6SdYsGfBtbpcERERESnGFJyKgG1fvoWbATuivIis3cTscvJwt7kT5R+Fu80dbDZo1Mi5YcwY6NqVMluP8t00C023ZxDZ40HC5yxybv/kE7jxRmjfHmbNguxsTh3ezbq4aFpPXUGVEzBnspWFQY8RGxTLXdXuAuCbDd+Yc6EiIiIiUqQpOBUBbjPnALC/RV2TK7kEvXqBmxts3AjZ2RAWhkeOwW9fwU17IMUdPjl7Gf/8A/PmYXTvzuFPx5FUpwr1Np0i1R2O162Me7YDt3t7wOef06NaNwBmbZ/FyfST5l2fiIiIiBRJCk6FnCMnm8prDwAQfFdPk6u5BA0bwrBh4O4O/frBjh1QvToAyf6eTBzbnQe6wNjG/+5iSUkhauCTxCZkczDYxuFfphK68B9o3do5ne/++2nc9WH+/NabCocyaDmpBfsT95t0gSIiIiJSFCk4FXJbf/uOwAyDJE+o0bGv2eXkLyQEHnwQ4uPhs88gMBB+/hn7gw+S8ets7ntwPE82Gs6YW0vg+zQEjoDVkc5d11b0x3PVOiredBf4+MCkSXDHHeBwYNuwkabb0/lzEpRYvpFmnzVhc8JmAByGQ4tGiIiIiMhVUXAq5I787LynZ1uNSDw8fUyu5hJYrVCuHAQF/dtWujS2UaMIj7sRH3cfHm36OAeGHebws4nUrXgjN/aD159pRZW1Bwkr6xydwt0dSpd2hq/bbwdvbwACM2De19Dsr0PUmVCHWuNrceOkG5m2ZRoOw3Htr88wYOVKOHXq2p9LRERERK4bN7MLkKvj99dyAHJubG5yJVfB3R2iopw/AlH+zp8tFgvf3PktadlpVAipgMViOXdfb2947z0IC4PTp3HcfTce8+fz7TTYFZLNcscGAPbtXkPtW3yo3LgTANn2bDLtmfh5+BXYZRxeMofQkS/h8dffUKsWzJwJJUs6r09ERERECjXTR5w++OADYmNj8fLyomHDhixfvvyi/ceNG0flypXx9vYmJiaGxx9/nIyMjOtUrWvJSEumxrZEAKJv621uMdeAu82d6IBoKpaoeP7QBP+GLnd38PfH+vnn0LEjAGPWhhHqE0qnbbD2rTQqNu0Mjz3Gyh/eo+eQKCLHRDDu73GsObKG5Ycu/vfuYo7uXs+CmysSeeMtztAEsH49jB/vXAAjH6lZqfyy4xfSs9OvuAYRERERubZMDU5Tpkxh6NChjBo1itWrV1O7dm3atWtHfHz8eft/++23jBgxglGjRrFlyxY+/fRTpkyZwtNPP32dK3cN23/5Gt9sOO5roXSTDmaXYz53dyhVCoY6n2V148oEjs2txczvoEQ6WA3gnXeod9cjTJlwnFq703h83uPc8PENNPykIX1+6kNSRtIlny47PZXFD3fBu1ptWv++EyvwbQ0Y3erMQO748ZCYmNs/IyeD6VumczD5IACZqcksHnonWysGM+G5jtz5/Z0cSj5Etj3/sCUiIiIi15epwemtt97igQceoF+/flSrVo0JEybg4+PDZ599dt7+S5cupWnTptx7773ExsbStm1bunfvnu8oVVGVNGc6ANtqRWGx2UyuxoW0aAG1awNgXeB8IO4nN/rRuyuk/mfW3KerS2G1/Ps/gS/XfUmzSc3Yc2rPOeFlw7EN7D61O/f9um/e4mBsCC3en0lgJmwp48P6Hz7kpQFVeLZZDnujfCApCR56CMehg/w26Vkav1aR26feTqVx5ZnwcGPiY0Jo8faPxO3P5ovpsGbVHD5f9RnZqcmwfDmkpjpPlpICf/4Jjiu7R+tg8kGWH1p+fe7xEhERESmiTAtOWVlZrFq1ijZt2vxbjNVKmzZtWLZs2Xn3adKkCatWrcoNSrt372bOnDl0PDM163wyMzNJTk7O8yoqAv5eA0BGs8b59Cxm3Nzg7behWTOIjoaffyb8o29Yd3NNmg8vwYrXHwGLhaprD/JHw49Y2HshM7vPJMgriI3xG5mwcgLZDmdwSk47xcCZD1FrQi0qv1+ZJ7/py583xlK75zDKxmdxzN/CH6P6UnlXErXuGMjoNqMx3Kz0vykNhwWYORNrqRhuvu9lFr90kHG/ubH8/Sweev9vYk7ZORJg5VSZCIIyYfxsSHp7NO6VqjiXbS9b1vncq9hYaN4cHnggz9S//Un7mbV9FsmZzr/TWxK28OPmH3NHzVYcWkH3H7sTOy6Whp80pNw75Rjz1xhOpWvhChEREZHLZTEMwzDjxIcPHyY6OpqlS5fSuPG/X/yffPJJFi9ezD///HPe/d59912GDx+OYRjk5OTw0EMPMX78+Aue5/nnn+eFF144pz0pKYmAgICrv5CrlJZ0nH3L5lKmcXt8AkMv2jfbnk1CWgJhPmHYDEj19cA/CzbM/46are+5ThUXEjk5kJEBvr5gsZBtzybLnoW3u7dzlKlmTedDeCdMgAEDyLZnM3bZWJ6e/zRhPmFsHbKVPV++S8kRL5NmsfNYeyiZAq//BiEZ4AAWd6pB7U9mEhIRm3tah+Hguw3f8f6K97Es/Zu35kGjQ+eWl+zrxpo+7Wgw+ku89x7EiIvDkvPvkul2Nxu2HPu5O06fzt5IL+Z/9yqPBi4l1WbH38OfG0rewOJ9iwHw8/CjconKrDqyCo8c6L4BKpx257sq2RwIgHC7J51bD2RM2zG4WbU+jIiIiBRfycnJBAYGXlI2KFTBadGiRdxzzz28/PLLNGzYkJ07d/Loo4/ywAMP8Oyzz573PJmZmWRmZua+T05OJiYmplAGp7TsNLYkbKFqWFUOL/2VCi1v47QHeJ1Ox93D6zpVXET06AHffgv9+8PEiQBk5mRS4b0KpB49yJeLgum0/PwjM9tK+2If/wHVOva54OENw2Dqpqm88edrdE8oSZ97Xyfs7/Xw/vvOqYQjRuRdkv2ZZ+CVVzgSaOPFZnZ+bhjIEp8hnJw5lWll07klPphmczaQ4emGR1YOVgO+rAVP9AwnPs15T6AFC6UCSnEg+QABGTB4lZUnV3oSdOrcRSeWlgLH/f1o9uT7znvDfvoJPvoIUlLIvvN2Mu++A++SMdjcPa74IwZI3bsD77m/Y23WHGrUuHjn9HTnSoQLFkBMDAwc6Hzul4iIiMg1UiiCU1ZWFj4+Pvzwww907do1t71Pnz4kJiby888/n7NP8+bNadSoEWPGjMlt+/rrr3nwwQdJSUnBas1/5uHlfDjXw5UGp+UvPUTLl75ibZUg6mzR1KvL9sYb8NRTzilxf/+d2/zze0Oo//QHRKWA3QJ/3B5H05IN8Rj/EYaXF8n/G47P0Kdw9/Qu2HqysmDxYlJvqEnNr5uwJ3FPns3eWbDqY6h63PneYXEudpE94UMmN/LjYPJB7qh2B7GpHuwa9TDlp/6GR9qZfzCIinI+82r58nPukzL8/bD4B8Dhw3nas62wpEEkTeduwjMwxDmCB85pkAD79zufofXNN86HEd99N9xzD5QvT3Z6Kmsmvohl0iRuWJeAzQDD3x/L5Mlw881s27aU2QlLKR1RkbaxN7H9x48xvv6a2n/txCP13xUyV0ZbeHp4XW6qdzcPN3wYH/d8nlMWH+8MgN9/j7FtG0bjxlhLlYLjx8lp0phltYKZcmoJJ9JPMObmMZQKKHVZvyIREREpei4nG5g2T8fDw4O4uDjmz5+fG5wcDgfz589nyJAh590nLS3tnHBkO7Mogkn5zzT2f5xf9lPqVDO5kkKqQQPnn5s2OcNEVhaMHMmt4z4AYG+EJ6c/eo9Wt565r6hPPyylShEYGXlt6vHwgJtvxheYdOskBswawLYT22gY3ZCetXry5dovuK3XOl7cX56aA5+n6h+b4cUXcR/2BL3eegsadICn3oCvvqJqVpbzmNWrw5NPQvfuzvfr1wOQGuzLm4NuoMfydMqfSoHTKST6u/Ne3WyO+EHftdDgMLT++ygHWjcg5oZWMGWK8xi33QYnTsAvv+QNYevXwzPPcKhsKF7HTtAg7d//Pcb7QPjp02TcfTt7QqxUPZBOSQ9YWBZOH4Z6p/89zN5AZ/utW6HeIYMxr63m5l6r+efQP0zrNs35e/r1V5g8GVatIr1uDQ6WDib27624LfkLy5maLIDlwIHc47p9+SWNrPDlLTA5DhbtWcRLrV6ienh1GsfoHkERERHJn2kjTuBcjrxPnz589NFHNGjQgHHjxjF16lS2bt1KREQEvXv3Jjo6mtGjRwPO+5XeeustPv7449ypegMHDiQuLo4pZ7/Y5aOwjjilZKWwIX4DHlYPqoZVZU+5EKofzGTNe/+j7pCXr2PFRURyMoSHQ2am8wG6H38MG5wPy03t3wePt97B3T/QtPJSMlOwG3YCPAPIceTk3tvmbjuzLODp03DTTbBixbk7N2vmHE3r2BEuMAr7/KLneXHhC7TcB/6ZMLcCZLtZ6FK5C480fISt419mwNiF2C7y/w7ba5Ui+74+JJ44hPsP04jbmpzb/0iAla0d6pPZqzv9VvyPHz9NpcnB8x/nlLeFNc0r8k1Ng0m+O6gYVonHPVpw/1NTcE9MZmsojGkCT1mbU3HRBiz/WeL9/1sRBT9Ug1UloeVeKJMEx3yhxT6of2ZQ7ZF+kbxX5mjuPuNvGc9D9R668IWKiIhIkVUopuqd9f777zNmzBiOHj1KnTp1ePfdd2nYsCEALVu2JDY2ls8//xyAnJwcXnnlFb766isOHTpEWFgYnTt35pVXXiHov/eLXERhDU4DZw1kwqoJjGs3ju4VuhISHoubAQlbVhFW5YbrWHERYRhwxx0wffq/bWFhzulnnTqZV9elys52TpebPNm5wMXBg9ClizMwNWmS7+7HU49T8b2KJGYmEugZSP8b+jO4/mDKBpcF4OjpozzweHl6/p2G3QKf1YUMN7hjC6S7wed1YMf/++samWrhyZNVaVivK/X7PZM7nXHB7gU8+PVdPPuHlTI1m1PjkZcJPXiCnFkz2VM+hKjuD+Lr77yXKS07DW83b+cDj1evhvbtISEhz3kO+8PUarAoFpocgLqJXsyLzuCHanAszIuOFTtyZ9U7qRlRkzk75pCYkUi7cm1p/srXWD/9FIefHz82DmRczCGWlgZ/D382DtxI6aDSV/1rERERkcKlUAWn662wBqea42uyMX4jd1W7iyctzal31yMc87cSkZQDFst1rLgImT3bGZ4yM52jM599BhERZld1ebKz4dgxCAwEf//L2nXd0XVsTthMl8pd8PXwPWf7xNUTeXjOw2TaM2lUqhGD6g3C3ebO1E1T8XX3xcBgzo45lPAuQb+6/ehTuw/RAdEFdWVOmzeTM3IEO1fMZVFUNpNrwJIyULNkbe6pcQ93V7+bmIAY1h5dy8n0kzQt3RQ/D7/zHys5GRo1gi1bcpuG9inJ22WP0L9ufyZ2mViwtYuIiIjLU3C6iMIYnByGA79X/UjPSadSiUp8vLMaLd75iWU3hNN41bHrXHERcvTov1PdOnVSAP1/MnMy2XVqF4ZhUD28uqm1rDi0gmcXPkvD6IZ0q9GNamFXeG/fqVPOBSSmToW5c8ny9aJVtwxWxnqwdcjW3BE3ERERKR4KxeIQcukOnz5Meo5zSemdJ3diW+n8OaV2VTPLKvwiI6FzZ7OrcFmebp5XHlAKWP3o+sztOffqDxQcDP36wb33Qr16eGzcyF+fwZrILD7yGMBrT/169ecQERGRIin/9bvFdDtP7sz92WE4KLHdeZe9V4P872URkfPw9HSOPN1yC4bFQt2j8Oyzv7Hv5eGcPrLP7OpERETEBSk4FQL/DU7uOVDhhHN2ZUzTDmaVJFL4lS8Ps2Zh2bOHFZX98c2GMs++iX9ULP9U9GHNurkcTD7I0ZSj+R9LREREijwFp0Jgx4kdAHi7eVPpBLg7IMkTSlfXiJPIVStThpDFy3n9Zh+2lXA2NdyZzv5uHbjhpRjqja2U5x8vREREpHhScCoEdp5yfmnrXLkz1c+szLy/lD9Wq83EqkSKjvIRVeg3fQ8ZG9eyd9rnZNks3LoN4sfCyrGnefrHQWaXKCIiIiZTcCoE9ibuBaBLpS7UTnD+yk5XiDGxIpGiJ9w3nNqRtYm9rQ8eb43DEeBc3j0yFQJm/caGYxtMrlBERETMpOBUCOxLdN6sXqFEBRolOb/MudesY2JFIkXckCFYd+6CwYMB6LEe3l76pslFiYiIiJkUnFxcalYqJ9JPAFA6oDT1E30AuKFNLzPLEinarFYIC4MnngCgxV5Ysmwy+xP3m1uXiIiImEbBycXtS3KONgV6BhKIJ377nQ+8tdWqbWZZIsVDmTIYTZpgBbquzWTor0PNrkhERERMogfguriz0/TKBJXBsm07FocDIzgYS2SkyZWJFA+WHj1g6VIGroA33X9kg/VDara5F4KCnB0MAzZtgpkzweGAqlVhyxaYMwfsdujdGwYOBIvF1OsQERGRq6Pg5OLOjjiVCSyDddNmABzVqmLTlzCR6+Oee2DYMMolZvDBHGDOYGAwKeHBZFWtRNChE1h3XmS58n/+cQaohx++9HPa7bBrF1So4Jw2KCIiIqbTf5Fd3NkV9coElsG6eQvgDE4icp2EhMD8+Zwa0p95lWzsD3A2+8WfImTxP1h37sTh6QGdO0OPHlCrFnTsiP3DD4hv18zZ+amnYNs258/p6bB0KZw4kfc8qanw009w331QsiRUrgy33gpZWdftUkVEROTCNOLk4vYnOW9GLxNUBsuWxQAYCk4i11eTJgQ3aUKFkyP4Zffv7N27ltOrl+K2eRuHPDLZWT+GGyqFMXP7TIzaBhVDfNly/H8kN0xk2VY3GuxLh27doHx5mDsX0tIgMBCGDYMSJWD2bJg/HzIz85531iy47Tb45pt/pwaKiIiIKRScXNyh04cAiAmIwbZrFwDWKtXMLEmk2CofUp7yIeWhHnAnrDu6jlZftOJU2i7Wrt2V2+942nEArDYr/W7J4e9PLfivWwfr1gGQ7e2Je1ISPPdc3hOUKwddujhfaWlw++3Oe6WCgyEqyjl1LzAQdu+GN9+Edu2u16WLiIgUewpOLu5QsjM4RflGYtm1GwC3ylXMLElEzqgdWZsl/ZYwYeUEAG6reht+Hn6sOrKKqiWqUqFEBZp91oyaD+3jkX8g1QN+qgLrIjJ5eK07j/9tISPIj9031uR4mybMsO5g0b5vCNo4m/Ih5Wk/vDkPfbwaz+On4PBh5+use++FDRucgUpERESuOYthGIbZRVxPycnJBAYGkpSUREBAgNnlkJZ0nH3L5lKmcXt8AkPzbDMMA99XfUnPSWfv7UsoU6s5uLs7/yXaTZlXpDDYnLCZe364hxDvELpU7sLB5INMWDmB9Jz0S9rfgoW36j/DY6GdYOtW54p9Y8Y4F5AoVw5ee815f5WX1zW+EhERkaLncrKBgpPJLhacTqWfIuSNEAAymvyCZ9sOUKnSvzeZi0ihlJ6dzv6k/Ww9vpV1x9Zx5PQRdifupm5kXTpX6kxadhrrjq1jwZ4F/LLzF9ysbizss5Bmpc8sNrF6NTRp8u89UXFxznunQkMvfFIRERE5x+VkAw1buLDDp53TckK8Q/Dce8DZWKGCiRWJSEHwdvemcmhlKodW5tYqt563z83lb2ZY42G0+7odv+3+jd7Te7Nx4EZ8PHwwatRg3zNDiB79Pu5pmbBqlXPq3nffwa+/OheV2LMHOnWCJ55wjlSLiIjIVVFwcmFnF4aI8o+Cs8+JUXASKTYsFguf3foZtSfUZk/iHmpOqEnjUo1ZtHcRh+yH4Em4JzGGb987jOW3384dcVq2DOOPP7A89BDExkKdOmZchoiISJGg4OTCzo445QlO5cubWJGIXG+lAkrxZdcvueeHe9h9aje7TzkXifF198XN6sbkoAPUvDOKEdMSsGZnc6B0EDsaVgC7neY/rcF93jyYN895sLp1ndP6SpWChx6CiAgTr0xERKRwUXByYWdX1Iv2j4Y9a52N5cqZV5CImOKWSrew85GdfLHuC1KyUmhcqjGtyrZiY/xGmn3WjP9VPcxzI8AvC5K8E4GVALTwhyHLISYZ6h4FjzVrYM0a50Fffx3694fRo8HX17RrExERKSwUnFxYnhGnfT87G8uUMbEiETFLhF8ETzZ9Mk9bvah6fHjLh4z4fQTJmcnUr9icpjFNOZZyDAODVne0onyJSnT8tiPuB4/S83AJHo2+nYg5fzgXmXnvPViyBH75BSIjTboyERGRwkHByYUdSTkCQGlLMJw86WxUcBKR/7iv7n3cV/c+DMPAYrGct8+s7rO4bcptjA46wIeeU5k//3fi5m+GQYNg7Vpo0ABefRV69ry+xYuIiBQiVrMLkAuLT40HIDb5zJeh4GBwgSXURcT1XCg0AcRFxbHywZXUiahDUmYSLb5oyRMlN7Dp23cgPBwOHIBevZzPhxIREZHzUnByYWeDU9SJLGdDbKx5xYhIoRbuG87M7jOJ8I0gNTuVsUvHUnfdQH7+8VXo08fZaeRI+OsvcwsVERFxUQpOLuxscApLSHM2aJqeiFyFUoGlWPnASt5o8waNSzUm25FNjz8epdPNCSyuFQh2O9k973U+YPfYMcjONrtkERERl6Hg5KIyczJJykwCIPDoKWejRpxE5CqVCizFE02f4Ndev1I7ojap2anM3jmHu9smcdIL3Pfudy5ZHhkJHh4QGAjt2kFGhtmli4iImEqLQ7iohLQEANysbngeOuZs1IiTiBQQPw8//uj3B1+v/5qjKUcJ9w3nnqzneOSXUzQ7ZCUw3cBiGJCcDL/+Cl9/7Vy+XEREpJhScHIx2RlpnNi/lcMe6YDzvgTLwYPOjTExJlYmIkVNgGcAg+oPyn3fokwLmpdsTlJmEjYHtAisxavbS9Pwo1mkv/IC3r16gaeniRWLiIiYR1P1XEx2Zhqndm/mcJIzLIX7hsPZ4FSqlImViUhRVzOiJnN7zqV0YGnsVlhwej1tQmaR6Aneew/yxavdSDpxmCNbVpCdkWZ2uSIiIteVgpOLSkg7DkCEdxgccT7PiehoEysSkeKgUalG7Hp4F5sHbWbszWOJLFmBCfWc2zqN+Rnv8GhKVmtASogvM3s2ICUzxdyCRURErhMFJxeVkO4MThVyAiAnB6xW583aIiLXmJvNjaphVRnWZBg7Ht7BiKmHsLu7USIdPBzOPsHp0PmbFTz2WGU2JWwyt2AREZHrQMHJRZ0dcSqfeuZ+gshIcNMtaSJigqgobHPnkfHOWxxauRB7ViY72jcAYOgPh2kxoRF/7vvT5CJFRESuLX0Td1FnV9UrnWJzNmianoiYqXVrvFq35uz/E1X8ajY5FcpR7fhpfv0whTf33EK9j47h5e5lapkiIiLXikacXNTJ9BMARJ468wBKBScRcSWhobi99wEANxyFLz5PZsaUF0wuSkRE5NpRcHJRp9KdD70NPXFm5SqtqCcirqZXL5gzh9NB3rgZUPb5d3A47GZXJSIick0oOLmoU5mJAASeSHU2aMRJRFxRhw6kL55PuhvU35XOywOr89uu38yuSkREpMApOLmok5lJAPieOO1sKFnSxGpERC4svFZj1nRvCcDIT7Zh79COY8Megu++g+Rkc4sTEREpIApOLsgwDE5lJALgedIZoLQUuYi4ssYfzeFwXGXcHdB+h0HEWx/BvfdC5cowahTs2GF2iSIiIldFwckFpTkyyDFyAHBPOOlsjIgwsSIRkYuzeHsTtXIr2xf9yPOtrUyqA4mBnnD0KLz4IsTFwbZtZpcpIiJyxRScXFBijvO+Jm/c4YRzdT2NOIlIYVCpxe34vDia+7pCqUGZvHazD/bQEnD6NHTqBMuXQ2qq2WWKiIhcNgUnF5SUkwJAxSx/LA4HWCwQGmpyVSIil+aJJk/wZdcv8QkOY2TTNNo/FYVRsiTs3AkNG5JVKorDH7xmdpkiIiKXRcHJBSWdGXGqkOntbAgLAzc9q1hECgeLxUKv2r2Y13MePu4+/J66gf69g0j3dD7Q2yMxmcghIxnbuQRvLXqNbHv2pR04MxPmz4d334UNG67hFYiIiJxL38Zd0NkRp9g0L2eD7m8SkUKobsm6fNTpI3pP781n3luYMhQMC7y/wIt+f2cwfNZJEhaMZEfYS1SKqI5bXD1o2RISEpyr8cXFQXg4/PEHzJsHixZB2pln27m5wdNPw/PPO0flRURErjEFJxd0dsQpJu3Mr0f3N4lIIdWzVk8i/SKZu3MuYT5hdKzYkRovVSd57Ku4vfwKYaczCNuXBvtWwPIVMH78xQ8YGQmlSzvvlXrxRdi7Fz75BNzdr8v1iIhI8aXg5GKObFlBzuRvCK4FUWlnZlJqxElECrE25drQplybPG0BTz4DQ4ay4ZcveX3OSLKSE2m9B27e70ZmVDgWHx/CNu/DmpXN6khYWzucjXWjmWrdQpY9nqFBVl77zYH1yy8x/PywfPBBwRRrt4NhaHq0iIicw2IYhmF2EddTcnIygYGBJCUlERAQYHY5pCUdZ9+yuZRp3J7s9FQSa1agzPGcvJ2GDYOxY80pUETkGjudeZqn5z/Nx6s+JsuRdcn79VgHX0+HHKuFn79+htlsZ+XhVeDjzdi2Y2lbvm2e/vtWLeDkjCkEHkukTPt7sJWv4JwGuGWLs8PBg87pgJmZ0KOH814qH58CvFIREXE1l5MNFJxM9t/gtHzIbbT8+s9zO40d6wxPIiJFWGJ6IjO2z2D+7vk4cFCvZD3alm+Lp5snS/YtwWa1UTeyLsHewRiGwRfrvqBm///ReXve46yJhEUV3Gh2z5PYMzPI+nUOsSt3UvpEzvlPfCFBQdCkCQwfDq1aFdh1ioiI61BwughXDk6rb2tEs4W7+LuKPxZfX2oEV8bXPwTeeQdiYswuVUTE5cxZ+Amx9z1Gtb35Pxsq2wobyvmyMSCDRnvtxCTD0hhYHg0hvqHcULU1R+IqkbNnF53HzsQ9OSV3X3uZ0tiiouGOO2DoUC1IISJSRFxONtAkbhdizXBOUUmIq8JNY37A53ACVK2qqSIiIhfQsVV/2NoLli6F6tUhK4u0mdNY9s3rhO48jM3Di+M3VMGnYxeq3j6AG0KjiE45xst/vMwf+//AarGyJWELmfbjwFTY5Txu0EBodBA6bYf+q8Fz337Ytx+WLYOUFBg1ytTrFhGR60/ByYXYMs/M7ffwNLcQEZHCxNMzz1Q6n4GPcNPAR0jNSsXXw/ec7hF+EbzX8b3c93tP7eWlP17in0P/UMKnBIGegSw7uIxF/imk3lSfz/ftImTXYTrugEf/AccLz+MoE4Nb3/uuy+WJiIhrUHByIW6ZzodAWjw9TK5ERKTwO19oOp/Y4Fg+vfXTPG2GYWBgYLVYSc9O58nfnuSVjZNxtx9n0Erg/v5QpRo0anQNKhcREVdkNbsA+Zf72eDk5WVyJSIixZvFYsFqcf4n0tvdm/c6vsexJ+LZOmow39UAq8OA7t0hMdHcQkVE5LpRcHIh7lnOFZ8sngpOIiKuxmKxMLjRwwy8BQ4E4Hz47pNPml2WiIhcJwpOLsQjyw6A1dPb5EpEROR8KodWpnL5BvTteqZh4kQoqIfvioiIS1NwciEeWQ4ArJqqJyLish6Me5AF5eC1FjZnw+OPw4oV5hYlIiLXnIKTC/E8E5zcPLX8uIiIq+pRqwcRvhE83cLOz5WB7GxSb27J4uf7kZadZnZ5IiJyjSg4uRDPbGdwsnlpqp6IiKvycvNi6f1L6VmnF/27wJ4g8E1Ko8ULnzP83lDaf92eh+c8zPYT2/M/WEoKHDgAW7bA9Omwb981r19ERK6MliN3Id7OtSFw97y0JXRFRMQc5YLL8eVtX7KgTl+erTqOO79cQddFR3lnWjr9c+Yxs8w8Ji/9mLsb9yc2sAwRtkDap5UkPMWA1FRYtQoWLYI1a8Aw/j2wpye89BI88YRp1yYiIuen4OQi7NlZ+DjXhsBdI04iIoVC67KtaV22NfRx4Lj1VtxnzeKLn85uzSLd7cPcfxS7kGw3KznuNqxWG56pGc6V+laudAaoSpWu8RWIiMilUnByERmnT+F/5mc3L19wd4eSJZ1/ioiIa7NasX7/PfTtCwsWYCQnY8nMzBOaEnxgVzBkuMH2ErCwLCwuA0cCHIADjGwmLvCl/5JUmDoV/vwT1q+HEiXMuioREfkPBScXkZGalPuzh5ePMzCFhppYkYiIXBYvL5g8GQALwPHjzml5Xl7g5cWe01v5ZPVEVh5eicNwUCW0Cj2CyuBwODAw+Hr91zzQOoFl5UOYONuG9fBhGDjQGaJERMR0Ck4uIivFGZwy3MBq069FRKTQCw3N8w9gDQIb0qBUwwt2H9p4KDd8dAOfxSYQOag5r7yQANOmwZ49ULbs9ahYREQuQqvquYis1GTAGZxERKT4KRVQik+7fIrVYuVVyxKWVvQCux0efhgcDrPLExEp9hScXERWypng5KFfiYhIcdW5cmemd5tOqE8oTzbLwG4BZs+GW2+F06fNLk9EpFjTt3QXkZPm/A9ipruV4HLVcNdDcEVEiqUulbuwbcg2TtWrRs/bIccKzJpFernS7Oh1C3tffgLHrp1mlykiUuwoOLmInLQUALI8bERWugF3LwUnEZHiKsQ7hDn3zmHFjeVp3Rv2BYL38UQqfj2H2GfHQsWKJNzTBdLTzS5VRKTYUHByEfYzI07ZHjaTKxEREVdQJqgMawasoWH34dz+fFWe6RnNV61D+bOMBasBYVNm4qhbFz78EE6eNLtcEZEiT8HJReSkpQKQ7anVIURExMnf058xbcew6rHNvPzVQXrNTyB2/X669fPjhDdYt22DwYMhNtYZoBwOMAyzyxYRKZIUnFyEIzc46YG3IiJyYaUCStH5sfHUeQjebmwhPaakc+GIwYPBzQ38/ODll80uU0SkyFFwchFGehoAdgUnERHJR89aPal+QzuGtjOIeSiN3QPuBk9P52hTWho8+6zCk4hIAVNwchGOtLPBycPkSkREpDD46ravqBZWjRPZSVSK+pGnf34E+/598Pjjzg7PPw//+x/89BNkZppZqohIkWB6cPrggw+IjY3Fy8uLhg0bsnz58ov2T0xMZPDgwZQsWRJPT08qVarEnDlzrlO111CGc2Uku5enyYWIiEhhEOYbxrL7lnFr5VuxG3ZG/z2GO/56BMfYMXD77c6H5776Ktx2G7RtS2b8ETIyU2H1ahg3Dvr0gd694c8/dV+UiMglMHUlgilTpjB06FAmTJhAw4YNGTduHO3atWPbtm2Eh4ef0z8rK4ubb76Z8PBwfvjhB6Kjo9m3bx9BQUHXv/iClp4BgEPBSURELlGAVwA/3fMTE1ZOYMicIfy87Wei3y4F9XO4y+5B361e1NlxGusff3CifBS+2eD1/wefvvoKypd3PmS3aVMIDYXmzcFiMeWaRERclcUwzPtnpoYNG1K/fn3ef/99ABwOBzExMTz88MOMGDHinP4TJkxgzJgxbN26FXf3K7sXKDk5mcDAQJKSkggICLiq+gtCWtJx9i2by5GJb9F62hoW3RFHyx9Wml2WiIgUMp+t+YzBcwaTkZORp73GMZjzDcQkO98necL2qmHc0OUhbGvXwdy5kJWV92CtWzun+Pn7X5/iRURMcjnZwLQRp6ysLFatWsXIkSNz26xWK23atGHZsmXn3WfGjBk0btyYwYMH8/PPPxMWFsa9997LU089hc12/ucfZWZmkvmfud3JyckFeyEFxJJxpkYvL3MLERGRQum+uvfRqVInFu5ZSLngctgddhbtW0Radhob7y1HwPpTLA5O5q7tL5FFAs1LL+L7J74nAl9O/Pg1yd99ju+OvYTuice6YAHcfz9MnWr2ZYmIuAzTgtPx48ex2+1ERETkaY+IiGDr1q3n3Wf37t0sWLCAHj16MGfOHHbu3MmgQYPIzs5m1KhR591n9OjRvPDCCwVef0GzZGc7f1BwEhGRKxTuG063Gt1y3zeKafTvxmbQBfh6UzV6Te/Fkv1LKPtOWcJ8w9iftB8aA42hxR6Y/yXYvv/eeR/U+PHg43Pdr0VExNWYvjjE5XA4HISHh/Pxxx8TFxdHt27d+N///seECRMuuM/IkSNJSkrKfR04cOA6VnzprNk5zh88tKqeiIhcO3dVv4vFfRdTNqgs6Tnp7E/aj9ViJa5kHN1rdCepUR0eaw8OC/Dll1CpErz+unNKnxaREJFizLQRp9DQUGw2G8eOHcvTfuzYMSIjI8+7T8mSJXF3d88zLa9q1aocPXqUrKwsPM4TOjw9PfH0dP0FF84GJ4uH69cqIiKFW8NSDdk2ZBsrD68kIyeDuKg4Ajydc/vtDjs3e99Mp+CFfDbLSuShQ3D2vuPKlWHOHChXzsTqRUTMYdqIk4eHB3FxccyfPz+3zeFwMH/+fBo3bnzefZo2bcrOnTtxOBy5bdu3b6dkyZLnDU2FSe6IUyEIeSIiUvi529xpHNOYVmVb5YYmAJvVxvd3fc+GeqWoMNjBW3fH4OjQHry9Yds2aNIE8nl0iIhIUWTqVL2hQ4cyceJEvvjiC7Zs2cLAgQNJTU2lX79+APTu3TvP4hEDBw7k5MmTPProo2zfvp3Zs2fz6quvMnjwYLMuocBYs+3OPzXiJCIiJivhU4LZ3Wdj+PkwrNoB7uzjzdjP+nMyKhiOHYPGjaFbN8jJMbtUEZHrxtTnOHXr1o2EhASee+45jh49Sp06dZg7d27ughH79+/Hav0328XExDBv3jwef/xxatWqRXR0NI8++ihPPfWUWZdQYGxn/uNj0YiTiIi4gFqRtZh06yTu+eEepm+dznTglV7w+Swbt26yO1fca9AAhg0zu1QRkevC1Oc4mcFVn+OU/vgQbtiaxNI3HqbJE++aXZaIiAgA36z/hpnbZ+Lp5skf+/5gb+JexqwNZ/hP8RAQAFu2QFSU2WWKiFyRQvEcJ8nLdnaqnqeWIxcREdfRo1YPetTqAcD+pP3U+LAGI2rG0/EfN6odSYabboLevaFuXahdG0qWNLliEZFro1AtR16UueU4F7xQcBIREVdVOrA0v/b6ldIlytL5rhxOeANbt8LTT0OHDlC2rHPp8tRUs0sVESlwCk4uwnYmONk8vU2uRERE5MIalWrEmgFrKF/vZmoOhIc7wI9V4UCgBTIzYcQIEiODOT3uDbNLFREpUJqq5yLOjjjZvBScRETEtQV6BTLr3ln08urFtxG/80H6KXAYPLQShi2D8qey4fGnOBUSRHDvB80uV0SkQCg4uQh3u0acRESk8PCweTDlrikA5DhyWH90PUl9k9iVmc6ivl25/59s3B8YwC9HVtDhqYnOnU6fhqwsCAkBi8XE6kVELp+Ck4twy3EubqgRJxERKWzcrG7cEHVD7vvNPy5naavGNNmRQYcRn7B/wo/42rwI2X0Ui2E4V+MrW9YZnqxW56ISTzwBVauaeBUiIhen4OQi3M9M1XPz8jG5EhERkatTLboOWesSWHRXY5r8spHSe0/l7ZCcDOvW/ft+9Wr4/nuYNQtatLi+xYqIXCIFJxfh7lyNXMFJRESKBA9vP1rO2sCUuW9y9JfvSTLS+cp/N/utKdRNsPFMmV50qn0XnDwJb7wBGzbAPffA/v3g7m52+SIi57ii4GS32/n888+ZP38+8fHxOByOPNsXLFhQIMUVJ+5251Q9BScRESlKurUfBu2HATA47QS9pvfiF7df6Jz1Oc2P7aJnrZ70/WMhHuUrwdGjMGkSPKgFJUTE9VzRcuSPPvoojz76KHa7nRo1alC7du08L7l8HmdGnNy9fM0tRERE5Bop4VOC2ffOZmSzkViwsGT/EgbMGkD5SXWYcUtFALJHPQspKSZXKiJyrisacZo8eTJTp06lY8eOBV1PsWQ4HLnBSSNOIiJSlFksFl696VW6Ve/GnB1zGLt0LAeTD9Kj1EE2BkKZo/GsaV0N7y+/pUqVZmaXKyKS64qCk4eHBxUqVCjoWoqtnOys3J89vP1MrEREROT6qB1Zm9qRtRkQN4A3l71JclYyo1Pn8MF7u6m74gDxcc2Z2boSbZv3wTM0Etq2hVKlzC5bRIoxi2EYxuXu9Oabb7J7927ef/99LIXsOQzJyckEBgaSlJREQECA2eWQlnScrb9N5oa7Hna+T0zAJzDU5KpERESuP8MwWPTF85QbOYYyR9PzbMuywZobKxH3wsfYmjZnx6mdrDy8kkOnD+GbbaV9WhTlsnyci014eTlfJ09CfDwcPw4VKkCNGs5pgCdOOJdEr1sXYmJMuloRcQWXkw2uKDjddtttLFy4kJCQEKpXr477/1v9Ztq0aZd7yOvGFYPTpjlfUP/e4QDkZKbj5uFlclUiIiImyspiw2tDWfPLZ3gnp1MmERoc/nfz1nAryyMdBGVAqWSoGQ/ujgse7eJuuw2++AL8/QuichEpZC4nG1zRVL2goCBuu+22KypOzpWTlQmA3YJCk4iIiIcHNZ97nxrPvsemhE1YsDB9xmfY33uP9luyqRLvoEp83l2O+sG+QDjp7VxwyTvb+XO8L6S7Q9UEKJsIiV7OV4k0qBUPTJ+O4eOD5euvzbhSESlErmjEqTBzxRGn1d+/R7MHXiTdDbyzi9WvQ0RE5JKlZaexaM10qvy2ltK2YNxCQiEiAurUITkiiMX7/uBE+gn2Je1jwe4FHEk5QqUSlQj2CsZu2IkJiKFKaBVWH1nNjO0zqLJqP/PO5qUlS6CZFqMQKW6u+VS9sxISEti2bRsAlStXJiws7EoPdd24YnBa+e2b3DjoNZI8ITBDwUlEROR6GP3naCIffpp+a+F43SqErt5idkkicp1dTja4ouc4paamct9991GyZEluvPFGbrzxRqKiorj//vtJS0u7oqKLM/uZqXrZboVroQ0REZHCbETTEfxzf3uyrBC6ZisrH7+H9Kw0MnIyzC5NRFzQFQWnoUOHsnjxYmbOnEliYiKJiYn8/PPPLF68mGHDhhV0jUWe48xy5Nk2BScREZHrxWKx8O7An/m+bTQA9cZNYW15X6o+HcjI30fiMK50xQkRKYquKDj9+OOPfPrpp3To0IGAgAACAgLo2LEjEydO5IcffijoGos8R7ZzxCnH7Yp+HSIiInKFPGwe3DVzF9/3qU+KBzQ+CFO/yeKdha/xyh+vmF2eiLiQK/qmnpaWRkRExDnt4eHhmqp3BRw5zhGnHE3VExERue483Dy56/Pl+KzdhCMwgPqHYer3sODrF9k8aQwcOmR2iSLiAq4oODVu3JhRo0aRkfHvHOD09HReeOEFGjduXGDFFRdG1tngpBEnERERs1irVsP6/Q8Ybm502gELP8mh2n1PklWuDJnzfzW7PBEx2RU9x+mdd96hXbt2lCpVitq1awOwbt06vLy8mDdvXoEWWBw4snMABScRERHT3Xwzlm+/xf7SiyTu24ZXeja+WXYO9OhK1J7j2Lx9zK5QRExyRd/Ua9SowY4dOxg9ejR16tShTp06vPbaa+zYsYPq1asXdI1FnpE7Vc9mciUiIiLCXXdhW7+BoFPp/Pz7+xz1g5hj6Sx4pLPZlYmIia5oxAnAx8eHBx54oCBrKbaM7GwA7O4KTiIiIq7CZrVx742DmT/kDyJfm0r9rxew+L4ptGjczezSRMQElxycZsyYQYcOHXB3d2fGjBkX7dulS5erLqxYObMcuV0jTiIiIi6n9UvfsOubOZQ/kIKj572sWVyeuqXqmV2WiFxnlxycunbtytGjRwkPD6dr164X7GexWLDb7QVRW7Fh5DjvcXJoxElERMTlWNzciJ6xkPSGDWi128G4vi2Jnr6LcP9zVxgWkaLrku9xcjgchIeH5/58oZdC0xXInap3xTMnRURE5BryqlOPnLFjAHhsfir+oSXJLFeGve+9xB97FnEs5RiGYZhcpYhcSwX2TT0xMZGgoKCCOlzxcmZVPYebgpOIiIir8n94GMc2rSXok6/xzjJgz35iH3kO75GwOxh+jfTAI7gEAQHheFarSUS/h6kaWw+rRavmihQFV/S/5Ndff50pU6bkvr/rrrsICQkhOjqadevWFVhxxYUlxzni5PBQcBIREXFlERO+YuOG+fR4tjrPtLZw2gMiUqHxQei1Motuvx2hw4/raP3S18RUacgXNwayetVss8sWkQJwRd/UJ0yYwDfffAPAb7/9xu+//87cuXOZOnUqTzzxBL/+qofEXZYz9zgZGnESERFxeXFVW/PNixudb06ehM2bydq0gRPb13Hy5AFOHtlN2ZW7KHUim35/ppBTvxMbotxZVy2Eeb2akOiWzYm0E0T6RXJ/3fu5pdIt5l6QiFySK/qmfvToUWJiYgCYNWsWd999N23btiU2NpaGDRsWaIHFgeXMVD3Dw93kSkREROSyhIRAs2Z4NGtGSaDk2XaHg+TpUzj85ENU2Z1MzUPZ1Dx0jAarpvNlbSh/CmKSIMc6nT9LlaPBDZ3wKFkKQkOhdWsoU8bEixKR87mi4BQcHMyBAweIiYlh7ty5vPzyywAYhqHFIa6A5eyIk7uCk4iISJFgtRJwR3cC7ujOoTV/cHruDEqN/YhKJ1N4eeH/67trNyx+N/etw8OdjFdfxGfYiOtbs4hc1BUFp9tvv517772XihUrcuLECTp06ADAmjVrqFChQoEWWBxYcpxhUyNOIiIiRU903Ruh7o0w4GkYPRqOHIHKlSE2lvWHVvPzHx/jnZhGRCpUTYB6R7LxfGIkXzg20ueJr80uX0TOuKLg9PbbbxMbG8uBAwd444038PPzA+DIkSMMGjSoQAssDqxnRpzQiJOIiEjRFRICY8bkaapFL0o98ixDfx3Kc5umkpmTweczbPRanUOnUd8ww+JNl65PgdUKsbHOP0XEFBajmD10IDk5mcDAQJKSkggICDC7HNKSjvNnlzq0/eMQi3rfSMsvFptdkoiIiJggx5GDBQvW9Az21y5DmV0n8naoWhXGj4cWLcwpUKQIupxscMkjTjNmzKBDhw64u7szY8aMi/bt0qXLpR5WAIvjTHbVqnoiIiLFlpv1zPcAX19ilm5i/e3NCFu3E59s8MoBzy1boFUrKF8eSpeG8HDo0wfatze3cJFi4pK/qXft2pWjR48SHh5O165dL9jPYrFogYjLZHE4nD/YbOYWIiIiIi7BGh5BrT93cDTlKIN/Hc4v/3zDm/Og7zoDdu50vgAmT2ZB3SCm1nHncLUYAnyC8TmdQUt7DDdRjoiSFZwjVOXKmXtBIkXAJQcnx9kv9//vZ7l6FrtGnERERORckX6RfHnblzzmHUI/n/d4qQWUOwWRKdB8Hzy4GlqvSaT1GoCECx/o5pvhk0+cI1UickX0Td0FWDXiJCIiIhdgtVh5t8O7DKo/iPjUeLLt2exJ3MOBpAN8fSiDxtNXEL5mO357DwGQ6evFvhAbm31TCU+BRgfB9ttvGPXrY/npJ2jc2NwLEimkrig4PfLII1SoUIFHHnkkT/v777/Pzp07GTduXEHUVmxY7c7gZHHTqnoiIiJyflVCq1AltMq5G3qe+TM7G9zc8LJYqAx4ntrLvdPu5dSaZUybAlXj46FJE4zoaCy1asFbb0GV8xxPRM7rita0/PHHH2natOk57U2aNOGHH3646qKKGy0OISIiIlfN3R0slty3scGx/NHvD7rdOYq2A7z5uqaz3XLoEPzyCzRpAsuXO++XmjEDvvgCsrJMKl7E9V3RN/UTJ04QGBh4TntAQADHjx+/6qKKm7NT9SwKTiIiIlKA3KxuPN/yeR5t+Chvtn6TcgvfpcyB07w1D+oePQUNG+bd4fvv4eefdfuAyHlc0YhThQoVmDt37jntv/zyC+W0astlOzvipKl6IiIici0EewfzcuuX2fDsESI6daNVH/i5snNbuhtsi/HG4WaD2bMxHngAjhwxt2ARF3RFQxxDhw5lyJAhJCQk0Lp1awDmz5/Pm2++qfubroAWhxAREZHrwdfDl8l3TmZzi+f4vevvPH1wC+9t/YIUezp3bIIpP4Bt0iSYNAmio50P3O3c2eyyRVzCFQWn++67j8zMTF555RVeeuklAGJjYxk/fjy9e/cu0AKLA+uZ5cgt7h4mVyIiIiLFQbWwalQLqwZAn+OP8cyCZ1jiu4Q+HGfkIjtVj4P10CHo0sU5da9LF5MrFjHfFd9UM3DgQAYOHEhCQgLe3t74+fkVZF3FivXMVD2rpuqJiIjIdVY5tDLf3/09AKfST3HzVzezbe8qPpwNvdZDxgP98GqxG85zf7tIcXJF9zgB5OTk8PvvvzNt2jQMw/nF//Dhw6SkpBRYccVF7lQ9LQ4hIiIiJgr2DubP+/7kgRaP81hXT/YEgVf8STKbNoJt28wuT8RUVxSc9u3bR82aNbn11lsZPHgwCQnOJ1W//vrrDB8+vEALLA6sZ3KT1V0jTiIiImIuLzcv3mr3Fvv+d5zHH4jhhDd4btqKUa0adOgACxeaXaKIKa4oOD366KPUq1ePU6dO4e3tndt+2223MX/+/AIrrrjIXY7cphEnERERcQ1+Hn68NXIR7Qb5M7siWBwOmDsXWrfGGDQIzs6YESkmrig4LVmyhGeeeQYPj7yLGcTGxnLo0KECKaw4sZ1ZHMKqxSFERETEhZQLLsenw5fw5v9aUXMgfFrX2W4ZP54T748xtziR6+yKgpPD4cBut5/TfvDgQfz9/a+6qOLm7FQ9i6bqiYiIiIupHVmbBX0W8Npjs5g+/Baevcn59TH7uf+RmnTc5OpErp8rCk5t27bN87wmi8VCSkoKo0aNomPHjgVVW7Fh06p6IiIi4uJuqXQLs+6dRd8v13Mg0EJkkp2p9zfCYWjKnhQPVxScxo4dy19//UW1atXIyMjg3nvvzZ2m9/rrrxd0jUVe7nLkmqonIiIiLq58VHVS/udcDOyOmbt4c3SX3BWWRYoyi3GFf9NzcnKYMmUK69atIyUlhRtuuIEePXrkWSzCFSUnJxMYGEhSUhIBAQFml0Na0nGOVoik3HE766a+R+27hphdkoiIiMjFORwkVC9L2Nb9OIBJbUNZct/NVCtVhyG1+uNj9QRfX7OrFMnX5WSDyw5O2dnZVKlShVmzZlG1atWrKtQMrhicEspFUOakgw3TP6Jm1wfNLklEREQkf8nJbLqrJdV/XQNAgg+kuUNMsnPzgWg/9tUqg79XIP6n0rAH+uPRtgNlmnZ0zrZxOCAmBsLCTLwIKe4uJxtc9vrX7u7uZGRkXHFxcq6zi0PY3D3NLURERETkUgUEUH3eana9/xJRI18lLCXv98MyB1Moc3BT3n1+WgI8nfvWsFigTh0sXbtCjx5Qvvy1r1vkCl3RVL1XX32V7du388knn+DmVriePeSKI06JpcOJSjbYMvdrqrbrYXZJIiIiIpcnKQnWrCHHZmWlfzI7jm/D/9fFeOzeT1J2Mie9DErEn6b6lpOEpRg4LM7dolL+cwybDW6/Hbp1g/r1oXRpUy5FipdrOlUP/n3QrZ+fHzVr1sT3/81hnTZt2uUe8rpxxeB0ulQ4ESkG2xd8T6VWd5pdkoiIiMg1kZGTwRdrv+Ddf95ly/EtRCUZdNoO92y10XLXv4+6MSyQ0P5GfL+eim9IhIkVS1F3TafqAQQFBXHHHXdcUXFyLtuZqXpWm5YjFxERkaLLy82LAfUGMKDeAFKzUjmScoRbJ9/KR/U3U/8gDF0GVY5DnWMQ/ssfJMREsqDLDXSauBiLn5/Z5Usxd1nByeFwMGbMGLZv305WVhatW7fm+eefd/mV9Fyd29nlyD20HLmIiIgUD74evlQIqcDf9//NI788wsaojXzRIpRI/0j8lq/nyQ/XEpPooPPk1Zz8uwIh/6yH8HCzy5Zi7LKC0yuvvMLzzz9PmzZt8Pb25t133yUhIYHPPvvsWtVXLFjPTJbU4hAiIiJS3Ph7+jOp66S8jbcCL+Tw3Qt30XLcT5Tce4zMtjfhuWY9WCym1ClyWQ/A/fLLL/nwww+ZN28eP/30EzNnzuSbb77B4dATo6+GW+6qehpxEhEREQHAzY07n5/KA8MqkuIOnus28vdbQ/WwXTHNZQWn/fv307Fjx9z3bdq0wWKxcPjw4QIvrDixaTlyERERkXO429x5Y/BPfNjc+R0p8qVxNH+3Lh/NeYnji+dCZqbJFUpxcllT9XJycvDy8srT5u7uTnZ2doEWVdycHXGyumlxCBEREZH/qhZWjagpOzlZpTKxJ9L487F1wDoAUmKj8Fu8TEuXy3VxWcHJMAz69u2Lp+e/IyMZGRk89NBDeZYkd+XlyF2N4XBg0z1OIiIiIhcUFFoKvpmG0bEjFocDhwUybOC39zAbb21C+eU78HbXYmVybV3WVL0+ffoQHh5OYGBg7qtnz55ERUXlaZNLZ8/Jyv3ZzcPrIj1FREREirF27bDs3QubN5N16jivjO0CQOX1h+g9qYu5tUmxcEUPwC3MXO0BuCcP7SSkVEUAkuMPEBBWyuSKRERERAqH5DKRBOw/Ro/b4d7Rs7il0i1mlySFzOVkg8sacZKCZ8/59/4wjTiJiIiIXLqAu3oCcNcmeHrB0yZXI0WdgpPJ7Nn/TtWzuWk5chEREZFL1rcvAJ12QMKO9aw6vMrceqRIU3AymT0rI/dnjTiJiIiIXIYaNSAuDjcHPL8IXlvwgtkVSRGm4GQyu/3fqXpW22UtcigiIiIiTz4JwIOr4ct+M9lYLZSdMz43tyYpkhScTObIck7Vy7GCxapfh4iIiMhluftu+PRTEoN98M6BGltOUKJbP97/6mGzK5MixiW+qX/wwQfExsbi5eVFw4YNWb58+SXtN3nyZCwWC127dr22BV5DjjMjTjku8ZsQERERKYTuu4+A48nM/OkNNpf1IzgD7hz4PoNfaMCWhC1mVydFhOlf16dMmcLQoUMZNWoUq1evpnbt2rRr1474+PiL7rd3716GDx9O8+bNr1Ol14Y9K9P5p8XkQkREREQKMavVRudbn6DaX9s5VjacyFQY8/IKxgxvzP6k/WAYkJwMe/dCWprZ5UohZHpweuutt3jggQfo168f1apVY8KECfj4+PDZZ59dcB+73U6PHj144YUXKFeu3HWstuDZs53BSSNOIiIiIgWgZEki1u0ktc2N+OTAZ18m4R1TjhxvTwgMhLJlsfv7saZVFdYv/Yli9khTuQqmfl3Pyspi1apVtGnTJrfNarXSpk0bli1bdsH9XnzxRcLDw7n//vvzPUdmZibJycl5Xq7EYc8BwG7TkJOIiIhIgfD3x/eX+ST36gZA2Gk7bpnO2yMybWBzGNRdtI1aTW8j2cfGyhcHmlmtFBKmLuN2/Phx7HY7ERERedojIiLYunXreff5888/+fTTT1m7du0lnWP06NG88ILrLk3pOPMcJ03VExERESlAbm4EfDmZDQ/34v05z2MEB2GLjMLNz5/qe1Kp/+FPxG1OJDDDoN6oCXzPSe56borZVYsLK1TrX58+fZpevXoxceJEQkNDL2mfkSNHMnTo0Nz3ycnJxMTEXKsSL5sjx/mvHxpxEhERESl4Nevfwkf1bzl3w+BJpB49wNrurWi6aBdNxk6lZ+l03uv2BcHewde/UHF5pgan0NBQbDYbx44dy9N+7NgxIiMjz+m/a9cu9u7dS+fOnXPbHA4HAG5ubmzbto3y5cvn2cfT0xNPT89rUH3BcGQ7g5NDuUlERETkuvKNjKHpnA2cjI0gOv40zwyfyaN/VeSlMasoE1TG7PLExZh6j5OHhwdxcXHMnz8/t83hcDB//nwaN258Tv8qVaqwYcMG1q5dm/vq0qULrVq1Yu3atS41knSpcqfqacRJRERE5Prz9iZk2i9klgiiygn48pMT/PpIR7OrEhdk+lS9oUOH0qdPH+rVq0eDBg0YN24cqamp9OvXD4DevXsTHR3N6NGj8fLyokaNGnn2DwoKAjinvbA4+xwnu1XBSURERMQUTZviuWM3CUPuI+zbn+j97Wa29p5ClTbdzK5MXIjpwalbt24kJCTw3HPPcfToUerUqcPcuXNzF4zYv38/VmvRXavbODtVT8FJRERExDzBwYR9PY1/NoXTcN1x3Hv25ptP95Bz7AiWI0fJ9HKjdKnqxMbUJNnXjVRvNwJOpeLjsBFarR6hgSXNvgK5xixGMVu8Pjk5mcDAQJKSkggICDC7HP757CUa3v8cO0p6UPFwptnliIiIiBRrh3etw1qnDpEpl75Plg2Oh/vhE12WnIhQ0sJDSK9aHv/b7iGqQt1rV6xctcvJBqaPOBV3Z+9x0oiTiIiIiPmiytdm8WdjSBr+PJX3p5Ic4MmpEr54ZGRjTUvDO8NOwJl/686yQY4VfLIh6kgKHNmQ92BPvsHuskGE1WmKX5VaWF5+GYrwTKqiTsHJZGen6tn1PyIRERERl9DiruFw5zBISyPA15dzxiEyMuD0aTxKlMADWLn8Jyb//Arpe3YQc9pCdKKd2nszqXU4h3J7EmHPbGA2KV5W/J57+bpfjxQMBSeTGfYcABxaVU9ERETEdVgs4Ot7/m1eXs7XGfUa3U69Rref0+3vFdP5/cPhNF+4mxb7wOPFV8ns3hPPilWuVdVyDWmYw2RGjhaHEBERESmKGtW/jWcm7SLk73UsKmfFw26wqGczxvz1BnN2zGHd0XXsPrWbLHuW2aXKJVBwMtm/I076VYiIiIgURTUja+H5+lgA2i0/wf1tnqJxrVvIvqEOY3qVp+7/SjBy7hNk5mihMFemb+smO3uPk6ERJxEREZEiq/Gdj7PnyQfJ8rARkgHBGVDvCIyfDZteT+HpW8fyT/Ugdr8+EhwOs8uV81BwMpkjxzniZNeIk4iIiEiRVvb1j/CIPwEbNsCmTTBmDEb58jisVvyz4MbtGZQb8Ron69WAhASzy5X/R9/WzXZmqp6h4CQiIiJS9AUGQo0aUK0aDB+OZedOrNnZJP69iIldS5PiDiFrtuCIioLBg6F4PXLVpenbusn+XRxCvwoRERGRYslqJahhC3p9v43Bz93ApjCw5uTAhx9yetwbZlcnZ+jbutly7IBGnERERESKOy83Lz4a8Rej3rudUS2dbW5PjmDHommm1iVO+rZuMiNHU/VERERExMnLzYsfuv1Iq0/ns7iSJ945ULrNHfxVK4QfRnRmxvKvOJl+0uwyiyU9ANdsdk3VExEREZG8WpZrzYl5a9jaoh5V9qfRdMMp2DCLzLGz2B4Km8pE4t+6PdYGjbBHhlO5bht8vPzNLrtIU3AymXF2qp6bzeRKRERERMSVlIitSom9KexYPI34ieMovXANMUdSqXkMOHYUln8OfA5AsidsvecWqnw8DTw8TKy66NIwh9nOrqqnEScRERER+f8sFiq2vIOm3ywh5nAKxvbtbPviLabfXYsllb05Emgj0wYBmVDli9mcigggvVVz+OYbsysvcvRt3WxaHEJERERELpGlYkUq936c26aso/nWNEom5kBqKuOHteCoLwQnZuK96E/o2ROefdbscosUfVs3W+5znDRVT0REREQun6enDw+NWcimf2Yy8L4IPoo7s+Hll0mf/LWptRUlCk5myx1xUnASERERkStjsVi4qXonXnh3PW/2qcj79Z3ttp69WP6KHqRbEBSczGbX4hAiIiIiUjDCfcNZev9SNo3sz8yannjYocEzH7KySVmO7lpndnmFmoKT2bQ4hIiIiIgUoFCfUMbfNpF2K04y7a6a5Fig3t/7CKpSh121YrD/Os/sEgslfVs30Y7F0ym/YK3zjUacRERERKQAeXj6cNuUdfw99U02lPHGKwfKbzgI7duT2OcemDwZjh0zu8xCQ8HJRIc+fI3Y42dGnLTevoiIiIgUMIvFQrM7h1JjdwrTprzAt7Vt2AwI+nIKdO+OPaok6x7oTGrGabNLdXkKTiayxpZjXXk//q7iT8n+j5ldjoiIiIgUURarldvvfo7687cw8pEazKgE6yLA5jCo/cks/o6LYMWibyE93exSXZbFMIrXEhvJyckEBgaSlJREQECA2eWQlnScfcvmUqZxe3wCQ80uR0RERESKgcOnD7P84HJOjn+Te8f/iZdzEhQOCyyr6MWRmxrSru5d+N96F4SHm1vsNXQ52UAjTiIiIiIixUyUfxRdq3blvneXkD7/VzZWLYEDsBrQdHsGd45fjP+DQ7CXioYJE8wu1yUoOImIiIiIFGPBN95Mjc3HSTt9gp1/z2H9/Z1YWtGLTWFgy86BgQPJ+d9IcDjMLtVUCk4iIiIiIoKfXwgVGnag1iczuWHzKcZPGsQrzZ3b3F59jeSIIE7e1ISTY1/icPwuHEbxClIKTiIiIiIikoeXmxfv3/IBNSf+zGO3enLKCwKOnyZkwTJCnngOo0IFPm7qxZQBzdj4/QdQDJZN0OIQJtPiECIiIiLiyvYn7ueTJe+QunAuIZv20OufdEon5e2zrXwQ0RO+xa9NB3OKvEKXkw0UnEym4CQiIiIihUpGBjk/TWfvz59zYstqam46js+ZVfl2Na9BuV9XYPHyMrfGS6RV9URERERE5Nrw8sLtnu5U+G4eDdcmsP6vH/m2sR85Fii/ZCN/vT7Y7AqvCQUnERERERG5Yo0a3M5ti+OZ0a0OAJYvviQ1K9Xcoq4BBScREREREbkq3u7edHx1Kg4LNN2Tw99/TjG7pAKn4CQiIiIiIlfNq2xFdpQNBODEvOkmV1PwFJxERERERKRApNSqDID7qtUmV1LwFJxERERERKRA+N3YBoBSO46ZXEnBU3ASEREREZECUarNHQDUOGzn6KkDJldTsBScRERERESkQPhWr0OylwXvHNi+8EezyylQCk4iIiIiIlIwrFZ2VgwF4PQvP5lbSwFTcBIRERERkQKT3Lw+AGHL1plcScFScBIRERERkQIT1OVuAKrvSISsLHOLKUAKTiIiIiIiUmAqtbyDY77gmwUJkz4wu5wCo+AkIiIiIiIFxsfTj2mtIwGwP/M0Rny8yRUVDAUnEREREREpUPXGfsdhP4g8nsHxilEcnzHZ7JKumoKTiIiIiIgUqPqVWvL7+OFsCbMQlmwnpGt3jraoh/3tt8AwzC7viig4iYiIiIhIgevdcwxeq9bzXQMfrAZE/rEK29BhnP5tttmlXREFJxERERERuSbKxtSg4+LDjH2mNSnuzrbDTz9sblFXSMFJRERERESumUCvQIa/NJ/ff3gDB1B51V7S9+0yu6zLpuAkIiIiIiLXXOdOQ9kc6Ywfu+d8bXI1l0/BSURERERErjmb1cbByiUBODm/8N3npOAkIiIiIiLXhaVxEwAC1m0xuZLLp+AkIiIiIiLXRUz7uwGosC8Fe1amydVcHgUnERERERG5Lio17UKSJ/hmw64/Z5hdzmVxM7sAEREREREpHtzcPFj37v8IqlKbqk1uNbucy6LgJCIiIiIi182ND75sdglXRFP1RERERERE8qHgJCIiIiIikg8FJxERERERkXwoOImIiIiIiORDwUlERERERCQfCk4iIiIiIiL5UHASERERERHJh4KTiIiIiIhIPhScRERERERE8qHgJCIiIiIikg8FJxERERERkXwoOImIiIiIiORDwUlERERERCQfCk4iIiIiIiL5UHASERERERHJh4KTiIiIiIhIPhScRERERERE8qHgJCIiIiIikg8FJxERERERkXy4RHD64IMPiI2NxcvLi4YNG7J8+fIL9p04cSLNmzcnODiY4OBg2rRpc9H+IiIiIiIiV8v04DRlyhSGDh3KqFGjWL16NbVr16Zdu3bEx8eft/+iRYvo3r07CxcuZNmyZcTExNC2bVsOHTp0nSsXEREREZHiwmIYhmFmAQ0bNqR+/fq8//77ADgcDmJiYnj44YcZMWJEvvvb7XaCg4N5//336d27d779k5OTCQwMJCkpiYCAgKuu/2qlJR1n37K5lGncHp/AULPLEREREREpNi4nG5g64pSVlcWqVato06ZNbpvVaqVNmzYsW7bsko6RlpZGdnY2ISEh592emZlJcnJynpeIiIiIiMjlMDU4HT9+HLvdTkRERJ72iIgIjh49eknHeOqpp4iKisoTvv5r9OjRBAYG5r5iYmKuum4RERERESleTL/H6Wq89tprTJ48menTp+Pl5XXePiNHjiQpKSn3deDAgetcpYiIiIiIFHZuZp48NDQUm83GsWPH8rQfO3aMyMjIi+47duxYXnvtNX7//Xdq1ap1wX6enp54enoWSL0iIiIiIlI8mTri5OHhQVxcHPPnz89tczgczJ8/n8aNG19wvzfeeIOXXnqJuXPnUq9evetRqoiIiIiIFGOmjjgBDB06lD59+lCvXj0aNGjAuHHjSE1NpV+/fgD07t2b6OhoRo8eDcDrr7/Oc889x7fffktsbGzuvVB+fn74+fmZdh0iIiIiIlJ0mR6cunXrRkJCAs899xxHjx6lTp06zJ07N3fBiP3792O1/jswNn78eLKysrjzzjvzHGfUqFE8//zz17N0EREREREpJkx/jtP1puc4iYiIiIgIFKLnOImIiIiIiBQGCk4iIiIiIiL5UHASERERERHJh4KTiIiIiIhIPhScRERERERE8qHgJCIiIiIikg8FJxERERERkXwoOImIiIiIiORDwUlERERERCQfCk4iIiIiIiL5UHASERERERHJh4KTiIiIiIhIPhScRERERERE8qHgJCIiIiIikg8FJxERERERkXwoOImIiIiIiORDwUlERERERCQfCk4iIiIiIiL5UHASERERERHJh4KTiIiIiIhIPhScRERERERE8qHgJCIiIiIikg8FJxERERERkXwoOImIiIiIiORDwUlERERERCQfCk4iIiIiIiL5UHASERERERHJh4KTiIiIiIhIPhScRERERERE8qHgJCIiIiIikg8FJxERERERkXwoOImIiIiIiORDwUlERERERCQfCk4iIiIiIiL5UHASERERERHJh4KTiIiIiIhIPhScRERERERE8qHgJCIiIiIikg8FJxERERERkXwoOImIiIiIiORDwUlERERERCQfCk4iIiIiIiL5UHASERERERHJh4KTiIiIiIhIPhScRERERERE8qHgJCIiIiIikg8FJxERERERkXwoOImIiIiIiORDwUlERERERCQfCk4iIiIiIiL5UHASERERERHJh4KTiIiIiIhIPhScRERERERE8qHgJCIiIiIikg8FJxERERERkXwoOImIiIiIiORDwUlERERERCQfCk4iIiIiIiL5UHASERERERHJh4KTiIiIiIhIPhScRERERERE8qHgJCIiIiIikg8FJxERERERkXwoOImIiIiIiORDwUlERERERCQfCk4iIiIiIiL5UHASERERERHJh4KTiIiIiIhIPhScRERERERE8qHgJCIiIiIikg8FJxERERERkXwoOImIiIiIiORDwUlERERERCQfLhGcPvi/9u4+pury/+P463B3wAK0SFA7qXiT/byB1CQ0YzqM5k2xtTQzo9JZE1vJ1Ait07LEnJVN8TZSczNMl66psxRvmoqZ3GxqhimazgKjlTBIubt+f8nvR6Lne/h2zueAz8d2/vDj9YHXtb13PC+vwyErS926dVNwcLDi4uJ09OjRW67fvHmz+vTpo+DgYPXv3187d+70UlIAAAAAtyPLi9OmTZuUlpYmp9OpgoICxcTEKCkpSZcvX252/eHDhzVx4kRNmTJFhYWFSk5OVnJysk6cOOHl5AAAAABuFzZjjLEyQFxcnB566CEtW7ZMktTQ0CCHw6FXX31V6enpN6yfMGGCqqqqtH379sZrDz/8sGJjY7Vy5UqX36+iokLh4eG6cuWKwsLC/r2NtFD1lXL9krdLXeMfV7vwCKvjAAAAALcNd7qBpSdONTU1ys/P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",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Create a new figure\n",
    "plt.figure(figsize=(10, 6))\n",
    "\n",
    "# Plot the precision-recall curve for the default model\n",
    "sns.lineplot(x=recall_default, y=precision_default, label='default', color='green')\n",
    "\n",
    "# Plot the precision-recall curve for the best model\n",
    "sns.lineplot(x=recall_best, y=precision_best, label='best', color='red')\n",
    "\n",
    "# Set the title and labels\n",
    "plt.title('Precision-Recall Curve')\n",
    "plt.xlabel('Recall')\n",
    "plt.ylabel('Precision')\n",
    "\n",
    "# Show the legend\n",
    "plt.legend()\n",
    "\n",
    "# Display the plot\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 8. Plot ROC Curves and Compute the AUC for Both Models"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You will next use scikit-learn's `roc_curve()` function to plot the receiver operating characteristic (ROC) curve and the `auc()` function to compute the area under the curve (AUC) for both models.\n",
    "\n",
    "* An ROC curve plots the performance of a binary classifier for varying classification thresholds. It plots the fraction of true positives out of the positives vs. the fraction of false positives out of the negatives. For more information on how to use the `roc_curve()` function, consult the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html).\n",
    "\n",
    "* The AUC measures the trade-off between the true positive rate and false positive rate. It provides a broad view of the performance of a classifier since it evaluates the performance for all the possible threshold values; it essentially provides a value that summarizes the the ROC curve. For more information on how to use the `auc()` function, consult the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html).\n",
    "\n",
    "Let's first import the functions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import roc_curve\n",
    "from sklearn.metrics import auc"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> Using the `roc_curve()` function, record the true positive and false positive rates for both models. \n",
    "\n",
    "1. Call `roc_curve()` with arguments `y_test` and `proba_predictions_default`. The `roc_curve` function produces three outputs. Save the three items to the following variables, respectively: `fpr_default` (standing for 'false positive rate'),  `tpr_default` (standing for 'true positive rate'), and `thresholds_default`.\n",
    "\n",
    "2. Call `roc_curve()` with arguments `y_test` and `proba_predictions_best`. The `roc_curve` function produces three outputs. Save the three items to the following variables, respectively: `fpr_best` (standing for 'false positive rate'),  `tpr_best` (standing for 'true positive rate'), and `thresholds_best`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "fpr_default, tpr_default, thresholds_default = roc_curve(y_test, proba_predictions_default)\n",
    "fpr_best, tpr_best, thresholds_best = roc_curve(y_test, proba_predictions_best)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: Create <b>two</b> `seaborn` lineplots to visualize the ROC curve for both models. \n",
    "\n",
    "The plot for the default hyperparameter should be green. The plot for the best hyperparameter should be red.\n",
    "\n",
    "* In each plot, the `fpr` values should be on the $x$-axis.\n",
    "* In each plot, the`tpr` values should be on the $y$-axis. \n",
    "* In each plot, label the $x$-axis \"False positive rate\".\n",
    "* In each plot, label the $y$-axis \"True positive rate\".\n",
    "* Give each plot the title \"Receiver operating characteristic (ROC) curve\".\n",
    "* Create a legend on each plot indicating that the plot represents either the default hyperparameter value or the best hyperparameter value.\n",
    "\n",
    "<b>Note:</b> It may take a few minutes to produce each plot."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Plot ROC Curve for Default Hyperparameter:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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VWJiYrHiXOjfP4uNGjXSmjVr9Pfff5/x98zJDh8+XOIfPk5W1tfuqYpYQUFBictP9X0eOnSo3n77bf30008aPHiwZs2apUaNGqlFixaudc71dy6Af1CcALjYbDZNmjRJPXv21NSpUzVu3DjXX6Td3d2LvKEpSd26dbVx48YzrrNu3Tr17t27VIcu/duQIUP0xBNP6KuvvlJ4eLgyMjJcJ0FLUmhoqHx9fVVQUHDGvGU1YMAATZo0STNmzCixOBUUFOjTTz9VYGCgOnfuXOS2HTt2FFt/+/btrpGYsjzPp/Pll19q+PDheumll1zLsrOzdeTIkSLrnc1zfyaFFzPduXOnevbs6Vqen5+vhISEYoX13y655BLZbDZ9/PHH5XqS/XfffaecnBzNmTOnyJvsshyiVNp91K1bV5K0cePG0/5B4VTP/7n+fJxOZGSk7rrrLt11111KSUlR69at9cwzz7iKU2nvr/C1eqaf9ZIUFow9e/aUav3mzZvrhhtu0Ntvv62xY8e6nvsBAwbos88+04wZM/Too48W2y4jI0PffvutGjVq5Po+DBw4UJ999pk+/vhjjR8/vlT3n5+fr7///luXXXbZadcr62s3MDCw2M+k9M+obWl169ZNkZGRmjlzprp06aJff/1VjzzySJF1KvI1BdQ0nOMEoIgePXqoffv2euWVV5Sdna2wsDD16NFDb7/9tpKTk4utf/DgQdfnV155pdatW6evv/662HqFf/2/5pprlJSUpGnTphVb5/jx467Z4U6lcePGatasmWbOnKmZM2cqMjKySImx2Wy68sor9dVXX5X4xu7kvGXVqVMn9enTRx988IG+//77Yrc/8sgj2r59ux588MFifyH+5ptvipyjtHz5ci1btsz1prUsz/Pp2Gy2YiNA//3vf4v9Jdvb21uSSnzzdrbatm2r4OBgTZs2Tfn5+a7ln3zyySlHGE4WExOj2267Tb/88ov++9//Frvd6XTqpZde0t69e8uUq3BE6t+HLX7wwQflvo+LLrpIvr6+mjRpkrKzs4vcdvK23t7eJR46ea4/HyUpKCgodl9hYWGKiopSTk7OGTP9W2hoqLp166b3339fiYmJRW470+hjdHS0YmJitHLlylLnf/DBB5WXl1dkxOSqq65SkyZN9NxzzxXbl9Pp1J133qnDhw9r4sSJRbZp1qyZnnnmGS1durTY/Rw9erRY6di8ebOys7PVqVOn02Ys62u3bt26Sk9Pd42KSVJycnKJvztPx2q16qqrrtJ3332njz76SPn5+UUO05Mq5jUF1FSMOAEo5oEHHtDVV1+t6dOn64477tDrr7+uLl26qFmzZrrtttsUHx+vAwcOaOnSpdq7d6/WrVvn2u7LL7/U1VdfrZtvvllt2rRRWlqa5syZo7feekstWrTQjTfeqFmzZumOO+7QwoUL1blzZxUUFGjr1q2aNWuWfv75Z9ehg6cyZMgQTZgwQR4eHrrllltktRb9G9Bzzz2nhQsXqkOHDrrtttvUpEkTpaWlafXq1Zo/f77S0tLO+rmZMWOGevfurUGDBum6665T165dlZOTo9mzZ2vRokUaMmSIHnjggWLb1atXT126dNGdd96pnJwcvfLKKwoODtaDDz7oWqe0z/PpDBgwQB999JH8/f3VpEkTLV26VPPnz3cdIlWoZcuWstlsev7555Weni6Hw6FevXopLCzsrJ8bu92uxx9/XKNHj1avXr10zTXXKCEhQdOnT1fdunVL9dful156Sbt27dLdd9+t2bNna8CAAQoMDFRiYqK++OILbd26tcgIY2lcdNFFstvtGjhwoP7zn/8oMzNT06ZNU1hYWIkl9Vz24efnpylTpujWW29Vu3btdN111ykwMFDr1q1TVlaWPvzwQ0lSmzZtNHPmTI0ZM0bt2rWTj4+PBg4cWC4/H/929OhR1apVS1dddZVatGghHx8fzZ8/XytWrCgyMnmqTCV57bXX1KVLF7Vu3Vq333674uLilJCQoB9++EFr1649bZ5Bgwbp66+/LtW5Q9KJQ+369++vd999V4899piCg4Nlt9v15Zdfqnfv3urSpYtGjBihtm3b6siRI/r000+1evVq3X///UVeK+7u7po9e7b69Omjbt266ZprrlHnzp3l7u6uTZs2uUaLT55Ofd68efLy8lLfvn3PmLMsr92hQ4fqoYce0uWXX667775bWVlZevPNN9WgQYMyT+IyZMgQ/fe//9XEiRPVrFmzYpcVqIjXFFBjnf+J/ABUBqe6AK5hnLgyfd26dY26deu6prvetWuXMWzYMCMiIsJwd3c3oqOjjQEDBhhffvllkW0PHTpkjBo1yoiOjnZdaHH48OFFpgbPzc01nn/+eeOCCy4wHA6HERgYaLRp08Z44oknjPT0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vCz5TPIx5b4zx2fK2+59Tvvo1auXWrRoodmzZ+vWW2+VJF199dXy9PTUTz/9JH9/f7399tvq3bu3tm/friFDhigmJkZ9+vTR8uXLFRMTo9DQUG3cuFH9+/fXM888I4fDoRkzZmjgwIHatm2bateuXeZcMTEx+uqrr3TllVdq27Zt8vPzk6en5zk91ltvvVXdunVTcnKyIiMjJUnff/+9srKyNGTIENd6H374oW655RYtX75cK1eu1O23367atWvrtttukySNGjVKmzdv1ueff66oqCh9/fXXuvjii7VhwwbVr19fkpSVlaXnn39e7777roKDgxUWFiZJWrBggTw8PLRo0SIlJCRoxIgRCg4O1jPPPCNJOnbsmMaMGaPmzZsrMzNTEyZM0OWXX661a9cWmX573Lhxeumll9SqVSt5eHgoOztbbdq00UMPPSQ/Pz/98MMPuvHGG1W3bl21b9++yGN78MEHtXz5cs2cOVN33nmnvv76a11++eV6+OGHNWXKFN14441KTEyUl5eXjhw5ol69eunWW2/VlClTdPz4cT300EO65ppr9Ouvv+rVV1/V9u3b1bRpUz355JOSpNDQ0DNud3KeO++8U0uWLJEkffXVV5oyZYo+//xzXXDBBdq/f7+rtAMAAFRmFKcaoFGjRlq/fr0kafHixVq+fLlSUlLkcDgkSS+++KK++eYbffnll7r99tsVHBws6cQb5IiICElSixYt1KJFC9c+n3rqKX399deaM2eORo0aVeZMNptNQUFBkqSwsDAFBASccZtatWoVW5aVleX6vFOnTmrYsKE++ugjPfjgg5KkDz74QFdffbV8fP4pvzExMZoyZYosFosaNmyoDRs2aMqUKbrtttuUmJioDz74QImJiYqKipIkjR07VnPnztUHH3ygZ599VtKJkaw33nijyHMiSXa7Xe+//768vLx0wQUX6Mknn9QDDzygp556SlarVVdeeWWR9d9//32FhoZq8+bNatq0qWv5vffeqyuuuKLIumPHjnV9Pnr0aP3888+aNWtWkeLUokULPfroo5Kk8ePH67nnnlNISIirFE6YMEFvvvmm1q9frwsvvFBTp05Vq1atXI+rMFNMTIy2b9+uBg0ayG63y8vLy/VakFSq7SSpfv36euGFF1zr/PDDD4qIiFCfPn3k7u6u2rVrF8kPAABQWVGczsDL3UuZ4zNNu+/yYBiGLBaLJGndunXKzMx0laNCx48fL3LI2L9lZmbq8ccf1w8//KDk5GTl5+fr+PHjSkxMLJeMpfHHH3/I19e3yLIePXoU+frWW2/VO++8owcffFAHDhzQTz/9VGQERJIuvPBC1/MhSR07dtRLL72kgoICbdiwQQUFBa43/oVycnKKPGd2u13NmzcvlrFFixZFzgnr2LGjMjMz9ffff6tOnTrasWOHJkyYoGXLlik1NdV1blRiYmKR4tS2bdsi+y0oKNCzzz6rWbNmKSkpSbm5ucrJySl2/tnJmWw2m4KDg12HXkpSeHi4JCklJUXSidfDwoULixTLQrt27Sr2PBQq7XZt2rQpctvVV1+tV155RfHx8br44ovVv39/DRw4UG5u/CoCAACVG+9WzsBisZzz4XJm27Jli+Li4iSdKECRkZFFztkpdLpRn7Fjx2revHl68cUXVa9ePXl6euqqq64qcv6OxWIpdl5WeU5sEBcXVyzjv99wDxs2TOPGjdPSpUv1v//9T3FxceratWup7yMzM1M2m02rVq2SzWYrctvJJcHT07NI+SqtgQMHqk6dOpo2bZqioqLkdDrVtGnTYhMneHsXfc1NnjxZr776ql555RU1a9ZM3t7euvfee4tt5+7uXuTrwlkWT/5akquwZWZmauDAgXr++eeLZS083LEkpd3u348jJiZG27Zt0/z58zVv3jzdddddmjx5sn777bdi2QEAACoTilM19+uvv2rDhg267777JEmtW7fW/v375ebmptjY2FLvZ8mSJbrpppt0+eWXSzrxxrlwkoBCoaGhSk5Odn29Y8eOIofS/Zvdbpd0YjSlvAQHB2vw4MH64IMPtHTpUo0YMaLYOsuWLSvy9Z9//qn69evLZrOpVatWKigoUEpKSpkKV6F169bp+PHjrvO1/vzzT/n4+CgmJkaHDh3Stm3bNG3aNNe+Fy9eXKr9LlmyRIMGDdINN9wg6UTx2b59u5o0aVLmjCdr3bq1vvrqK8XGxp5y1Mdutxf7HpVmu1Px9PTUwIEDNXDgQI0cOVKNGjXShg0b1Lp167N+HAAAABWNWfWqkZycHO3fv19JSUlavXq1nn32WQ0aNEgDBgzQsGHDJEl9+vRRx44dNXjwYP3yyy9KSEjQ//73Pz3yyCNauXLlKfddv359zZ49W2vXrtW6det03XXXFZuCu1evXpo6darWrFmjlStX6o477jjtKEKdOnVksVj0/fff6+DBg8rMLJ9DIm+99VZ9+OGH2rJli4YPH17s9sTERI0ZM0bbtm3TZ599pv/+97+65557JEkNGjTQ9ddfr2HDhmn27Nnas2ePli9frkmTJumHH344433n5ubqlltu0ebNm/Xjjz9q4sSJGjVqlKxWqwIDAxUcHKx33nlHO3fu1K+//qoxY8aU6jHVr19f8+bN0//+9z9t2bJF//nPf3TgwIGyPTElGDlypNLS0nTttddqxYoV2rVrl37++WeNGDHCVZZiY2O1bNkyJSQkuA4vLM12JZk+fbree+89bdy4Ubt379bHH38sT09P1alT55wfCwAAQEWiOFUjc+fOVWRkpGJjY3XxxRdr4cKFeu211/Ttt9+6DjuzWCz68ccf1a1bN40YMUINGjTQ0KFD9ddff7nOfynJyy+/rMDAQHXq1EkDBw5Uv379io0QvPTSS4qJiVHXrl113XXXaezYsae9BlR0dLSeeOIJjRs3TuHh4Wc1yURJ+vTpo8jISPXr1881wcPJhg0bpuPHj6t9+/YaOXKk7rnnniIX+f3ggw80bNgw3X///WrYsKEGDx6sFStWlGr2wN69e6t+/frq1q2bhgwZossuu0yPP/64JMlqterzzz/XqlWr1LRpU913332aPHlyqR7To48+qtatW6tfv37q0aOHIiIiNHjw4FJtezpRUVFasmSJCgoKdNFFF6lZs2a69957FRAQ4Jrlb+zYsbLZbGrSpIlCQ0NdE2ecabuSBAQEaNq0aercubOaN2+u+fPn67vvvit2zh0AAEBlYzHKerGgcvT7779r8uTJWrVqlZKTk/X111+f8c3gokWLNGbMGG3atEkxMTF69NFHy3T9n4yMDPn7+ys9PV1+fn5FbsvOztaePXsUFxcnDw+Ps3hEqAwyMzMVHR2tDz74oNjMdD169FDLli31yiuvlPv93nTTTTpy5Eixa1nh7PDzCAAAKtrpusG/mTridOzYMbVo0UKvv/56qdbfs2ePLr30UvXs2VNr167Vvffeq1tvvVU///xzBSdFVeB0OpWSkqKnnnpKAQEBuuyyy8yOBAAAgGrC1MkhLrnkEl1yySWlXv+tt95SXFycXnrpJUlS48aNtXjxYk2ZMkX9+vWrqJioIhITExUXF6datWpp+vTpTHEN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",
      "text/plain": [
       "<Figure size 1000x800 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Create a new figure\n",
    "plt.figure(figsize=(10, 8))\n",
    "\n",
    "# Plot the ROC curve for the default model\n",
    "sns.lineplot(x=fpr_default, y=tpr_default, label='Default Hyperparameters', color='green')\n",
    "\n",
    "# Label the axes and title\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('Receiver Operating Characteristic (ROC) Curve')\n",
    "\n",
    "# Show the legend\n",
    "plt.legend()\n",
    "\n",
    "# Display the plot\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Plot ROC Curve for Best Hyperparameter:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA04AAAK9CAYAAAAT0TyCAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/P9b71AAAACXBIWXMAAA9hAAAPYQGoP6dpAACLSUlEQVR4nOzdd3gUVd/G8Tu9kQKk0AKh995EOgRQBMEG2EDEhmBDVECqiqgoVmxYEB4RENEHFVFEeRRERYoCUqREEGmhJLQkJJn3j/NmISQhWUgyu5vv57r2mtnZmd3fbiawd86Zc7wsy7IEAAAAAMiTt90FAAAAAICrIzgBAAAAQD4ITgAAAACQD4ITAAAAAOSD4AQAAAAA+SA4AQAAAEA+CE4AAAAAkA+CEwAAAADkg+AEAAAAAPkgOAFwWXFxcbrtttvsLqPE6dSpkzp16mR3GfmaOHGivLy8lJiYaHcpLsfLy0sTJ04slOdKSEiQl5eXZs6cWSjPJ0m//vqr/P399ffffxfacxa2AQMGqF+/fnaXAcCFEJyAEmrmzJny8vJy3Hx9fVWxYkXddttt2rt3r93lubSTJ0/qySefVKNGjRQcHKzw8HC1b99es2bNkmVZdpdXIH/++acmTpyohIQEu0vJISMjQ++//746deqkMmXKKCAgQHFxcRo8eLB+++03u8srFHPmzNFLL71kdxnZFGdNjz/+uG688UZVqVLFsa1Tp07Z/k0KCgpSo0aN9NJLLykzMzPX5zl8+LAeeeQR1a5dW4GBgSpTpox69OihL774Is/XTk5O1qRJk9S4cWOVKlVKQUFBatCggR577DH9+++/jv0ee+wxffLJJ/r9998L/L5KwrkLlGRelrv8Lw+gUM2cOVODBw/WE088oapVqyolJUU///yzZs6cqbi4OG3cuFGBgYG21piamipvb2/5+fnZWse5Dhw4oK5du2rz5s0aMGCAOnbsqJSUFH3yySf64Ycf1L9/f3344Yfy8fGxu9QLWrBggW644QZ9//33OVqX0tLSJEn+/v7FXtfp06d17bXXasmSJerQoYN69+6tMmXKKCEhQfPnz9e2bdu0e/duVapUSRMnTtSkSZN06NAhRUZGFnutl6JXr17auHFjkQXXlJQU+fr6ytfX95JrsixLqamp8vPzK5Tzev369WratKl++ukntWnTxrG9U6dO2rFjh6ZMmSJJSkxM1Jw5c7R69WqNGTNGkydPzvY8W7duVdeuXXXo0CENHjxYLVq00LFjx/Thhx9q/fr1GjlypKZOnZrtmJ07dyo+Pl67d+/WDTfcoHbt2snf319//PGHPvroI5UpU0bbtm1z7N+6dWvVrl1bs2bNyvd9OXPuAnBTFoAS6f3337ckWatXr862/bHHHrMkWfPmzbOpMnudPn3aysjIyPPxHj16WN7e3tZ///vfHI+NHDnSkmQ988wzRVlirk6cOOHU/h9//LElyfr++++LpqCLNGzYMEuS9eKLL+Z4LD093Zo6daq1Z88ey7Isa8KECZYk69ChQ0VWT2ZmpnXq1KlCf96rrrrKqlKlSqE+Z0ZGhnX69OmLPr4oasrN/fffb1WuXNnKzMzMtr1jx45W/fr1s207ffq0VaVKFSs0NNRKT093bE9LS7MaNGhgBQcHWz///HO2Y9LT063+/ftbkqy5c+c6tp85c8Zq3LixFRwcbP3444856kpKSrLGjBmTbdvzzz9vhYSEWMePH8/3fTlz7l6KS/05A7h4BCeghMorOH3xxReWJOvpp5/Otn3z5s3WddddZ5UuXdoKCAiwmjdvnmt4OHr0qPXggw9aVapUsfz9/a2KFStat956a7YvtykpKdb48eOt6tWrW/7+/lalSpWsRx55xEpJScn2XFWqVLEGDRpkWZZlrV692pJkzZw5M8drLlmyxJJkff75545t//zzjzV48GArOjra8vf3t+rVq2e9++672Y77/vvvLUnWRx99ZD3++ONWhQoVLC8vL+vo0aO5fmarVq2yJFm33357ro+fOXPGqlmzplW6dGnHl+1du3ZZkqypU6da06ZNsypXrmwFBgZaHTp0sDZs2JDjOQryOWf97JYvX24NHTrUioqKsiIiIizLsqyEhARr6NChVq1atazAwECrTJky1vXXX2/t2rUrx/Hn37JCVMeOHa2OHTvm+JzmzZtnPfXUU1bFihWtgIAAq0uXLtZff/2V4z289tprVtWqVa3AwECrZcuW1g8//JDjOXOzZ88ey9fX1+rWrdsF98uSFZz++usva9CgQVZ4eLgVFhZm3XbbbdbJkyez7fvee+9ZnTt3tqKioix/f3+rbt261uuvv57jOatUqWJdddVV1pIlS6zmzZtbAQEBji/CBX0Oy7KsxYsXWx06dLBKlSplhYaGWi1atLA+/PBDy7LM53v+Z39uYCno74cka9iwYdZ//vMfq169epavr6/16aefOh6bMGGCY9/k5GTrgQcecPxeRkVFWfHx8daaNWvyrSnrHH7//fezvf7mzZutG264wYqMjLQCAwOtWrVq5QgeualcubJ122235dieW3CyLMu6/vrrLUnWv//+69j20UcfWZKsJ554ItfXOHbsmBUREWHVqVPHsW3u3LmWJGvy5Mn51pjl999/tyRZCxcuvOB+zp67gwYNyjWkZp3T58rt5zx//nyrdOnSuX6OSUlJVkBAgPXwww87thX0nAJwYQVvwwdQImR10yldurRj26ZNm9S2bVtVrFhRo0aNUkhIiObPn6++ffvqk08+0TXXXCNJOnHihNq3b6/Nmzfr9ttvV7NmzZSYmKhFixbpn3/+UWRkpDIzM3X11VdrxYoVuuuuu1S3bl1t2LBBL774orZt26bPPvss17patGihatWqaf78+Ro0aFC2x+bNm6fSpUurR48ekkx3ussuu0xeXl4aPny4oqKi9NVXX2nIkCFKTk7Wgw8+mO34J598Uv7+/ho5cqRSU1Pz7KL2+eefS5IGDhyY6+O+vr666aabNGnSJK1cuVLx8fGOx2bNmqXjx49r2LBhSklJ0csvv6wuXbpow4YNiomJcepzznLvvfcqKipK48eP18mTJyVJq1ev1k8//aQBAwaoUqVKSkhI0BtvvKFOnTrpzz//VHBwsDp06KD7779fr7zyisaMGaO6detKkmOZl2eeeUbe3t4aOXKkkpKS9Nxzz+nmm2/WL7/84tjnjTfe0PDhw9W+fXs99NBDSkhIUN++fVW6dOl8uyh99dVXSk9P16233nrB/c7Xr18/Va1aVVOmTNHatWv1zjvvKDo6Ws8++2y2uurXr6+rr75avr6++vzzz3XvvfcqMzNTw4YNy/Z8W7du1Y033qi7775bd955p2rXru3Uc8ycOVO333676tevr9GjRysiIkLr1q3TkiVLdNNNN+nxxx9XUlKS/vnnH7344ouSpFKlSkmS078f3333nebPn6/hw4crMjJScXFxuX5G99xzjxYsWKDhw4erXr16Onz4sFasWKHNmzerWbNmF6wpN3/88Yfat28vPz8/3XXXXYqLi9OOHTv0+eef5+hSd669e/dq9+7datasWZ77nC9rcIqIiAjHtvx+F8PDw9WnTx998MEH2r59u2rUqKFFixZJklPnV7169RQUFKSVK1fm+P0718WeuwV1/s+5Zs2auuaaa7Rw4UK99dZb2f7N+uyzz5SamqoBAwZIcv6cAnABdic3APbIanX49ttvrUOHDll79uyxFixYYEVFRVkBAQHZupR07drVatiwYba/TmZmZlqXX365VbNmTce28ePH5/nX2axuObNnz7a8vb1zdJV58803LUnWypUrHdvObXGyLMsaPXq05efnZx05csSxLTU11YqIiMjWCjRkyBCrfPnyVmJiYrbXGDBggBUeHu5oDcpqSalWrVqBumP17dvXkpRni5RlWdbChQstSdYrr7xiWdbZv9YHBQVZ//zzj2O/X375xZJkPfTQQ45tBf2cs3527dq1y9Z9ybKsXN9HVkvZrFmzHNsu1FUvrxanunXrWqmpqY7tL7/8siXJ0XKWmppqlS1b1mrZsqV15swZx34zZ860JOXb4vTQQw9Zkqx169ZdcL8sWX+dP78F8JprrrHKli2bbVtun0uPHj2satWqZdtWpUoVS5K1ZMmSHPsX5DmOHTtmhYaGWq1bt87Rnercrml5dYtz5vdDkuXt7W1t2rQpx/PovBan8PBwa9iwYTn2O1deNeXW4tShQwcrNDTU+vvvv/N8j7n59ttvc7QOZ+nYsaNVp04d69ChQ9ahQ4esLVu2WI888oglybrqqquy7dukSRMrPDz8gq81bdo0S5K1aNEiy7Isq2nTpvkek5tatWpZV1555QX3cfbcdbbFKbef89dff53rZ9mzZ89s56Qz5xSAC2NUPaCEi4+PV1RUlGJjY3X99dcrJCREixYtcrQOHDlyRN9995369eun48ePKzExUYmJiTp8+LB69Oihv/76yzEK3yeffKLGjRvn+pdZLy8vSdLHH3+sunXrqk6dOo7nSkxMVJcuXSRJ33//fZ619u/fX2fOnNHChQsd27755hsdO3ZM/fv3l2QuZP/kk0/Uu3dvWZaV7TV69OihpKQkrV27NtvzDho0SEFBQfl+VsePH5ckhYaG5rlP1mPJycnZtvft21cVK1Z03G/VqpVat26txYsXS3Luc85y55135rhY/9z3cebMGR0+fFg1atRQREREjvftrMGDB2f7y3b79u0lmQvuJem3337T4cOHdeedd2YblODmm2/O1oKZl6zP7EKfb27uueeebPfbt2+vw4cPZ/sZnPu5JCUlKTExUR07dtTOnTuVlJSU7fiqVas6Wi/PVZDnWLp0qY4fP65Ro0blGFwl63fgQpz9/ejYsaPq1auX7/NGRETol19+yTZq3MU6dOiQfvjhB91+++2qXLlytsfye4+HDx+WpDzPhy1btigqKkpRUVGqU6eOpk6dqquvvjrHUOjHjx/P9zw5/3cxOTnZ6XMrq9b8hry/2HO3oHL7OXfp0kWRkZGaN2+eY9vRo0e1dOlSx7+H0qX9mwsgO7rqASXc9OnTVatWLSUlJem9997TDz/8oICAAMfj27dvl2VZGjdunMaNG5frcxw8eFAVK1bUjh07dN11113w9f766y9t3rxZUVFReT5XXho3bqw6depo3rx5GjJkiCTTTS8yMtLxJeDQoUM6duyY3n77bb399tsFeo2qVatesOYsWV+Kjh8/nq3b0LnyClc1a9bMsW+tWrU0f/58Sc59zheq+/Tp05oyZYref/997d27N9vw6OcHBGed/yU568vv0aNHJckxJ0+NGjWy7efr65tnF7JzhYWFSTr7GRZGXVnPuXLlSk2YMEGrVq3SqVOnsu2flJSk8PBwx/28zoeCPMeOHTskSQ0aNHDqPWRx9vejoOfuc889p0GDBik2NlbNmzdXz549NXDgQFWrVs3pGrOC8sW+R0l5DtsfFxenGTNmKDMzUzt27NDkyZN16NChHCE0NDQ03zBz/u9iWFiYo3Zna80vEF7suVtQuf2cfX19dd1112nOnDlKTU1VQECAFi5cqDNnzmQLTpfyby6A7AhOQAnXqlUrtWjRQpJpFWnXrp1uuukmbd26VaVKlXLMnzJy5Mhc/wov5fyifCGZmZlq2LChpk2bluvjsbGxFzy+f//+mjx5shITExUaGqpFixbpxhtvdLRwZNV7yy235LgWKkujRo2y3S9Ia5NkrgH67LPP9Mcff6hDhw657vPHH39IUoFaAc51MZ9zbnXfd999ev/99/Xggw+qTZs2Cg8Pl5eXlwYMGJDnXDgFlddQ1Hl9CXZWnTp1JEkbNmxQkyZNCnxcfnXt2LFDXbt2VZ06dTRt2jTFxsbK399fixcv1osvvpjjc8ntc3X2OS6Ws78fBT13+/Xrp/bt2+vTTz/VN998o6lTp+rZZ5/VwoULdeWVV15y3QVVtmxZSWfD9vlCQkKyXRvYtm1bNWvWTGPGjNErr7zi2F63bl2tX79eu3fvzhGcs5z/u1inTh2tW7dOe/bsyfffmXMdPXo01z98nMvZczevIJaRkZHr9rx+zgMGDNBbb72lr776Sn379tX8+fNVp04dNW7c2LHPpf6bC+AsghMABx8fH02ZMkWdO3fWa6+9plGjRjn+Iu3n55ftC01uqlevro0bN+a7z++//66uXbsWqOvS+fr3769Jkybpk08+UUxMjJKTkx0XQUtSVFSUQkNDlZGRkW+9zurVq5emTJmiWbNm5RqcMjIyNGfOHJUuXVpt27bN9thff/2VY/9t27Y5WmKc+ZwvZMGCBRo0aJBeeOEFx7aUlBQdO3Ys234X89nnJ2sy0+3bt6tz586O7enp6UpISMgRWM935ZVXysfHR//5z38K9SL7zz//XKmpqVq0aFG2L9nOdFEq6HNUr15dkrRx48YL/kEhr8//Un8/LqR8+fK69957de+99+rgwYNq1qyZJk+e7AhOBX29rHM1v9/13GQFjF27dhVo/0aNGumWW27RW2+9pZEjRzo++169eumjjz7SrFmzNHbs2BzHJScn67///a/q1Knj+Dn07t1bH330kf7zn/9o9OjRBXr99PR07dmzR1dfffUF93P23C1dunSO30npbKttQXXo0EHly5fXvHnz1K5dO3333Xd6/PHHs+1TlOcUUNJwjROAbDp16qRWrVrppZdeUkpKiqKjo9WpUye99dZb2rdvX479Dx065Fi/7rrr9Pvvv+vTTz/NsV/WX//79eunvXv3asaMGTn2OX36tGN0uLzUrVtXDRs21Lx58zRv3jyVL18+W4jx8fHRddddp08++STXL3bn1uusyy+/XPHx8Xr//ff1xRdf5Hj88ccf17Zt2/Too4/m+AvxZ599lu0apV9//VW//PKL40urM5/zhfj4+ORoAXr11Vdz/CU7JCREknL98naxWrRoobJly2rGjBlKT093bP/www/zbGE4V2xsrO6880598803evXVV3M8npmZqRdeeEH//POPU3VltUid323x/fffL/Tn6N69u0JDQzVlyhSlpKRke+zcY0NCQnLtOnmpvx+5ycjIyPFa0dHRqlChglJTU/Ot6XxRUVHq0KGD3nvvPe3evTvbY/m1PlasWFGxsbH67bffClz/o48+qjNnzmRrMbn++utVr149PfPMMzmeKzMzU0OHDtXRo0c1YcKEbMc0bNhQkydP1qpVq3K8zvHjx3OEjj///FMpKSm6/PLLL1ijs+du9erVlZSU5GgVk6R9+/bl+m/nhXh7e+v666/X559/rtmzZys9PT1bNz2paM4poKSixQlADo888ohuuOEGzZw5U/fcc4+mT5+udu3aqWHDhrrzzjtVrVo1HThwQKtWrdI///yj33//3XHcggULdMMNN+j2229X8+bNdeTIES1atEhvvvmmGjdurFtvvVXz58/XPffco++//15t27ZVRkaGtmzZovnz5+vrr792dB3MS//+/TV+/HgFBgZqyJAh8vbO/jegZ555Rt9//71at26tO++8U/Xq1dORI0e0du1affvttzpy5MhFfzazZs1S165d1adPH910001q3769UlNTtXDhQi1fvlz9+/fXI488kuO4GjVqqF27dho6dKhSU1P10ksvqWzZsnr00Ucd+xT0c76QXr16afbs2QoPD1e9evW0atUqffvtt44uUlmaNGkiHx8fPfvss0pKSlJAQIC6dOmi6Ojoi/5s/P39NXHiRN13333q0qWL+vXrp4SEBM2cOVPVq1cv0F+7X3jhBe3YsUP333+/Fi5cqF69eql06dLavXu3Pv74Y23ZsiVbC2NBdO/eXf7+/urdu7fuvvtunThxQjNmzFB0dHSuIfVSniMsLEwvvvii7rjjDrVs2VI33XSTSpcurd9//12nTp3SBx98IElq3ry55s2bpxEjRqhly5YqVaqUevfuXSi/H+c7fvy4KlWqpOuvv16NGzdWqVKl9O2332r16tXZWibzqik3r7zyitq1a6dmzZrprrvuUtWqVZWQkKAvv/xS69evv2A9ffr00aefflqga4ck09WuZ8+eeueddzRu3DiVLVtW/v7+WrBggbp27ap27dpp8ODBatGihY4dO6Y5c+Zo7dq1evjhh7OdK35+flq4cKHi4+PVoUMH9evXT23btpWfn582bdrkaC0+dzj1pUuXKjg4WN26dcu3TmfO3QEDBuixxx7TNddco/vvv1+nTp3SG2+8oVq1ajk9iEv//v316quvasKECWrYsGGOaQWK4pwCSqziH8gPgCvIawJcyzIz01evXt2qXr26Y7jrHTt2WAMHDrTKlStn+fn5WRUrVrR69eplLViwINuxhw8ftoYPH25VrFjRMdHioEGDsg0NnpaW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//13rVy5MltLUUZGRo6azlfQ41q0aJHtuKFDh+q6667T2rVr1b17d/Xt21eXX355rq8BAADgyQhO+fHykkJC7K6iQEJCQrJ1cXvnnXcUHh6uGTNm6KmnntKJEyfUvHlzffjhhzmOjYqKkmS6eN1///1asmSJ5s2bp7Fjx2rp0qW67LLLnKrF398/Wy2S5Oub83S74447dOutt+rFF1/U+++/r/79++f55T83BXlPkvlsnJWQkKBevXpp6NChmjx5ssqUKaMVK1ZoyJAhSktLc9QZFBQkLy+vbMcOGjRIhw8f1ssvv6wqVaooICBAbdq0yTEwhZ+fn2M96zly25Y14MSJEyc0adIkXXvttTnqDQwMzPO9FPS48z+nK6+8Un///bcWL16spUuXqmvXrho2bFiOa9MAAAA8HcHJg3l5ecnb21unT5+WJDVr1kzz5s1TdHS0wsLC8jyuadOmatq0qUaPHq02bdpozpw5uuyyy+Tv76+MjIxCrbFnz54KCQnRG2+8oSVLluiHH37Isc/PP/+sgQMHZrvftGlTp95TXn7++ecc9+vWrStJWrNmjTIzM/XCCy/I29tcDjh//vwCPe/KlSv1+uuvq2fPnpLMABaJiYlO13e+Zs2aaevWrTlC6bn8/Pxy/JwKclxeoqKiNGjQIA0aNEjt27fXI488QnACAAAlDoNDeJDU1FTt379f+/fv1+bNm3XffffpxIkT6t27tyTp5ptvVmRkpPr06aMff/xRu3bt0vLly3X//ffrn3/+0a5duzR69GitWrVKf//9t7755hv99ddfjiARFxenXbt2af369UpMTFRqauol1+zj46PbbrtNo0ePVs2aNXN0nZOkjz/+WO+99562bdumCRMm6Ndff3UMmJDfe8rPypUr9dxzz2nbtm2aPn26Pv74Yz3wwAOSpBo1aujMmTN69dVXtXPnTs2ePVtvvvlmgd5XzZo1NXv2bG3evFm//PKLbr75ZgUFBTnxyeRu/PjxmjVrliZNmqRNmzZp8+bNmjt3rsaOHevYJy4uTsuWLdP+/ft19OjRAh+X1+v997//1fbt27Vp0yZ98cUXjvMBAACgJCE4eZAlS5aofPnyKl++vFq3bq3Vq1fr448/dlxXFBwcrB9++EGVK1fWtddeq7p162rIkCFKSUlRWFiYgoODtWXLFl133XWqVauW7rrrLg0b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Xvrss8/yPWb58uVq1qyZAgICVKNGDc2cObPI63QWwakIzZs3TyNGjNCECRO0du1aNW7cWD169NDBgwdz3f+nn37SjTfeqCFDhmjdunXq27ev+vbtq40bNxZz5bCLs+fM8uXLdeONN+r777/XqlWrFBsbq+7du2vv3r3FXDns4uw5kyUhIUEjR45U+/bti6lSuApnz5m0tDR169ZNCQkJWrBggbZu3aoZM2aoYsWKxVw57OLsOTNnzhyNGjVKEyZM0ObNm/Xuu+9q3rx5GjNmTDFXDrucPHlSjRs31vTp0wu0/65du3TVVVepc+fOWr9+vR588EHdcccd+vrrr4u4UidZKDKtWrWyhg0b5rifkZFhVahQwZoyZUqu+/fr18+66qqrsm1r3bq1dffddxdpnXAdzp4z50tPT7dCQ0OtDz74oKhKhIu5mHMmPT3duvzyy6133nnHGjRokNWnT59iqBSuwtlz5o033rCqVatmpaWlFVeJcDHOnjPDhg2zunTpkm3biBEjrLZt2xZpnXBNkqxPP/30gvs8+uijVv369bNt69+/v9WjR48irMx5tDgVkbS0NK1Zs0bx8fGObd7e3oqPj9eqVatyPWbVqlXZ9pekHj165Lk/PMvFnDPnO3XqlM6cOaMyZcoUVZlwIRd7zjzxxBOKjo7WkCFDiqNMuJCLOWcWLVqkNm3aaNiwYYqJiVGDBg309NNPKyMjo7jKho0u5py5/PLLtWbNGkd3vp07d2rx4sXq2bNnsdQM9+Mu34F97S7AUyUmJiojI0MxMTHZtsfExGjLli25HrN///5c99+/f3+R1QnXcTHnzPkee+wxVahQIcc/PvBMF3POrFixQu+++67Wr19fDBXC1VzMObNz50599913uvnmm7V48WJt375d9957r86cOaMJEyYUR9mw0cWcMzfddJMSExPVrl07WZal9PR03XPPPXTVQ57y+g6cnJys06dPKygoyKbKsqPFCfAQzzzzjObOnatPP/1UgYGBdpcDF3T8+HHdeuutmjFjhiIjI+0uB24iMzNT0dHRevvtt9W8eXP1799fjz/+uN588027S4OLWr58uZ5++mm9/vrrWrt2rRYuXKgvv/xSTz75pN2lAZeEFqciEhkZKR8fHx04cCDb9gMHDqhcuXK5HlOuXDmn9odnuZhzJsvzzz+vZ555Rt9++60aNWpU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",
      "text/plain": [
       "<Figure size 1000x800 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Create a new figure\n",
    "plt.figure(figsize=(10, 8))\n",
    "\n",
    "\n",
    "# Plot the ROC curve for the best model\n",
    "sns.lineplot(x=fpr_best, y=tpr_best, label='Best Hyperparameters', color='red')\n",
    "\n",
    "# Label the axes and title\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('Receiver Operating Characteristic (ROC) Curve')\n",
    "\n",
    "# Show the legend\n",
    "plt.legend()\n",
    "\n",
    "# Display the plot\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: Use the `auc()` function to compute the area under the receiver operating characteristic (ROC) curve for both models.\n",
    "\n",
    "For each model, call the function with the `fpr` argument first and the `tpr` argument second. \n",
    "\n",
    "Save the result of the `auc()` function for `model_default` to the variable `auc_default`.\n",
    "Save the result of the `auc()` function for `model_best` to the variable `auc_best`. \n",
    "Compare the results."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8228632478632479\n",
      "0.8235464726844037\n"
     ]
    }
   ],
   "source": [
    "auc_default = auc(fpr_default, tpr_default)\n",
    "auc_best = auc(fpr_best, tpr_best)\n",
    "\n",
    "print(auc_default)\n",
    "print(auc_best)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Deep Dive: Feature Selection Using SelectKBest"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the code cell below, you will see how to use scikit-learn's `SelectKBest` class to obtain the best features in a given data set using a specified scoring function. For more information on how to use `SelectKBest`, consult the online [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html).\n",
    "\n",
    "We will extract the best 5 features from the Airbnb \"listings\" data set to create new training data, then fit our model with the optimal hyperparameter $C$ to the data and compute the AUC. Walk through the code to see how it works and complete the steps where prompted. Analyze the results."
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
      "Best 5 features:\n",
      "Index(['host_response_rate', 'number_of_reviews', 'number_of_reviews_ltm',\n",
      "       'number_of_reviews_l30d', 'review_scores_cleanliness'],\n",
      "      dtype='object')\n",
      "0.7971924148648286\n"
     ]
    }
   ],
   "source": [
    "from sklearn.feature_selection import SelectKBest\n",
    "from sklearn.feature_selection import f_classif\n",
    "\n",
    "# Note that k=5 is specifying that we want the top 5 features\n",
    "selector = SelectKBest(f_classif, k=5)\n",
    "selector.fit(X, y)\n",
    "filter = selector.get_support()\n",
    "top_5_features = X.columns[filter]\n",
    "\n",
    "print(\"Best 5 features:\")\n",
    "print(top_5_features)\n",
    "\n",
    "# Create new training and test data for features\n",
    "new_X_train = X_train[top_5_features]\n",
    "new_X_test = X_test[top_5_features]\n",
    "\n",
    "\n",
    "# Initialize a LogisticRegression model object with the best value of hyperparameter C \n",
    "# The model object should be named 'model'\n",
    "# Note: Supply max_iter=1000 as an argument when creating the model object\n",
    "model = LogisticRegression(C=1.0, max_iter=1000)\n",
    "\n",
    "# Fit the model to the new training data\n",
    "model.fit(new_X_train, y_train)\n",
    "\n",
    "\n",
    "# Use the predict_proba() method to use your model to make predictions on the new test data \n",
    "# Save the values of the second column to a list called 'proba_predictions'\n",
    "proba_predictions = model.predict_proba(new_X_test)[:, 1]\n",
    "\n",
    "\n",
    "# Compute the auc-roc\n",
    "fpr, tpr, thresholds = roc_curve(y_test, proba_predictions)\n",
    "auc_result = auc(fpr, tpr)\n",
    "print(auc_result)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: Consider the results. Change the specified number of features and re-run your code. Does this change the AUC value? What number of features results in the best AUC value? Record your findings in the cell below."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "By changing the number of specified features and reruning the code, I noticed that the AUC value increase with more features and decreases with less features showing us that the more amount of best performing features, the better the model is a showing us the threshold for values."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 9. Make Your Model Persistent"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You will next practice what you learned in the \"Making Your Model Persistent\" activity, and use the `pickle` module to save `model_best`.\n",
    "\n",
    "First we will import the pickle module."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> Use `pickle` to save your model to a `pkl` file in the current working directory. Choose the name of the file."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "pkl_model_filename = 'finalized_model.pkl'\n",
    "\n",
    "pickle.dump(model_best, open(pkl_model_filename, 'wb'))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> Test that your model is packaged and ready for future use by:\n",
    "\n",
    "1. Loading your model back from the file \n",
    "2. Using your model to make predictions on `X_test`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[False False False ... False  True False]\n"
     ]
    }
   ],
   "source": [
    "# Loads the model from the file\n",
    "loaded_model = pickle.load(open(pkl_model_filename, 'rb'))\n",
    "loaded_model\n",
    "# Uses the model to predict \n",
    "predictions = loaded_model.predict(X_test)\n",
    "print(predictions)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> Download your `pkl` file and your `airbnbData_train` data set, and push these files to your GitHub repository. You can download these files by going to `File -> Open`. A new tab will open in your browser that will allow you to select your files and download them."
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
