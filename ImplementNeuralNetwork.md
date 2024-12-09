{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Assignment 7: Implement a Neural Network Using Keras"
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
    "import os\n",
    "os.environ[\"TF_CPP_MIN_LOG_LEVEL\"] = \"2\" # suppress info and warning messages\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import confusion_matrix\n",
    "import tensorflow.keras as keras\n",
    "import time"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this assignment, you will implement a feedforward neural network using Keras for a binary classification problem. You will complete the following tasks:\n",
    "    \n",
    "1. Build your DataFrame and define your ML problem:\n",
    "    * Load the Airbnb \"listings\" data set\n",
    "    * Define the label - what are you predicting?\n",
    "    * Identify the features\n",
    "2. Prepare your data so that it is ready for modeling.\n",
    "3. Create labeled examples from the data set.\n",
    "4. Split the data into training and test data sets.\n",
    "5. Construct a neural network.\n",
    "6. Train the neural network.\n",
    "7. Evaluate the neural network model's performance on the training, validation and test data.\n",
    "8. Experiment with ways to improve the model's performance.\n",
    "\n",
    "For this assignment, use the demo <i>Implementing a Neural Network in Keras</i> that is contained in this unit as a reference.\n",
    "\n",
    "**<font color='red'>Note: some of the code cells in this notebook may take a while to run</font>**"
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
    "We will work with the data set ``airbnbData_train``. \n",
    "\n",
    "<b>Task</b>: In the code cell below, use the same method you have been using to load the data using `pd.read_csv()` and save it to DataFrame `df`.\n",
    "\n",
    "You will be working with the file named \"airbnbData_train.csv\" that is located in a folder named \"data_NN\"."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "filename = os.path.join(os.getcwd(), \"data_NN\",\"airbnbData_train.csv\")\n",
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
    "## Part 2. Prepare Your Data\n",
    "\n",
    "Many data preparation techniques have already been performed and the data is almost ready for modeling; the data set has one-hot encoded categorical variables, scaled numerical values, and imputed missing values. However, the data set has a few features that have boolean values. When working with Keras, features should have floating point values.\n",
    "\n",
    "Let's convert these features from booleans to floats.\n",
    "\n",
    "<b>Task:</b> Using the Pandas `astype()` method, convert any boolean columns in DataFrame `df` to floating point columns. Use the online [documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html) as a reference.  \n",
    "\n",
    "Note that there are a few different ways that you can accomplish this task. You can convert one boolean column at a time, or you can use the Pandas `select_dtypes()` method to find and return all boolean columns in DataFrame `df` and then convert the columns as a group. Use the online [documentation]( https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html) as a reference. \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Use select_dtypes to select boolean columns\n",
    "bool_cols = df.select_dtypes(include=['bool']).columns\n",
    "\n",
    "# Convert boolean columns to float\n",
    "df[bool_cols] = df[bool_cols].astype(float)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's inspect the columns after the conversion. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
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
       "      <th>host_is_superhost</th>\n",
       "      <th>host_has_profile_pic</th>\n",
       "      <th>host_identity_verified</th>\n",
       "      <th>has_availability</th>\n",
       "      <th>instant_bookable</th>\n",
       "      <th>host_response_rate</th>\n",
       "      <th>host_acceptance_rate</th>\n",
       "      <th>host_listings_count</th>\n",
       "      <th>host_total_listings_count</th>\n",
       "      <th>accommodates</th>\n",
       "      <th>...</th>\n",
       "      <th>n_host_verifications</th>\n",
       "      <th>neighbourhood_group_cleansed_Bronx</th>\n",
       "      <th>neighbourhood_group_cleansed_Brooklyn</th>\n",
       "      <th>neighbourhood_group_cleansed_Manhattan</th>\n",
       "      <th>neighbourhood_group_cleansed_Queens</th>\n",
       "      <th>neighbourhood_group_cleansed_Staten Island</th>\n",
       "      <th>room_type_Entire home/apt</th>\n",
       "      <th>room_type_Hotel room</th>\n",
       "      <th>room_type_Private room</th>\n",
       "      <th>room_type_Shared room</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.578829</td>\n",
       "      <td>-2.845589</td>\n",
       "      <td>-0.054298</td>\n",
       "      <td>-0.054298</td>\n",
       "      <td>-1.007673</td>\n",
       "      <td>...</td>\n",
       "      <td>1.888373</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-4.685756</td>\n",
       "      <td>-0.430024</td>\n",
       "      <td>-0.112284</td>\n",
       "      <td>-0.112284</td>\n",
       "      <td>0.067470</td>\n",
       "      <td>...</td>\n",
       "      <td>0.409419</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.578052</td>\n",
       "      <td>-2.473964</td>\n",
       "      <td>-0.112284</td>\n",
       "      <td>-0.112284</td>\n",
       "      <td>0.605041</td>\n",
       "      <td>...</td>\n",
       "      <td>-1.069535</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.578052</td>\n",
       "      <td>1.010024</td>\n",
       "      <td>-0.112284</td>\n",
       "      <td>-0.112284</td>\n",
       "      <td>-0.470102</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.576550</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.054002</td>\n",
       "      <td>-0.066308</td>\n",
       "      <td>-0.112284</td>\n",
       "      <td>-0.112284</td>\n",
       "      <td>-1.007673</td>\n",
       "      <td>...</td>\n",
       "      <td>0.902404</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 50 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   host_is_superhost  host_has_profile_pic  host_identity_verified  \\\n",
       "0                0.0                   1.0                     1.0   \n",
       "1                0.0                   1.0                     1.0   \n",
       "2                0.0                   1.0                     1.0   \n",
       "3                0.0                   1.0                     0.0   \n",
       "4                0.0                   1.0                     1.0   \n",
       "\n",
       "   has_availability  instant_bookable  host_response_rate  \\\n",
       "0               1.0               0.0           -0.578829   \n",
       "1               1.0               0.0           -4.685756   \n",
       "2               1.0               0.0            0.578052   \n",
       "3               1.0               0.0            0.578052   \n",
       "4               1.0               0.0           -0.054002   \n",
       "\n",
       "   host_acceptance_rate  host_listings_count  host_total_listings_count  \\\n",
       "0             -2.845589            -0.054298                  -0.054298   \n",
       "1             -0.430024            -0.112284                  -0.112284   \n",
       "2             -2.473964            -0.112284                  -0.112284   \n",
       "3              1.010024            -0.112284                  -0.112284   \n",
       "4             -0.066308            -0.112284                  -0.112284   \n",
       "\n",
       "   accommodates  ...  n_host_verifications  \\\n",
       "0     -1.007673  ...              1.888373   \n",
       "1      0.067470  ...              0.409419   \n",
       "2      0.605041  ...             -1.069535   \n",
       "3     -0.470102  ...             -0.576550   \n",
       "4     -1.007673  ...              0.902404   \n",
       "\n",
       "   neighbourhood_group_cleansed_Bronx  neighbourhood_group_cleansed_Brooklyn  \\\n",
       "0                                 0.0                                    0.0   \n",
       "1                                 0.0                                    1.0   \n",
       "2                                 0.0                                    1.0   \n",
       "3                                 0.0                                    0.0   \n",
       "4                                 0.0                                    0.0   \n",
       "\n",
       "   neighbourhood_group_cleansed_Manhattan  \\\n",
       "0                                     1.0   \n",
       "1                                     0.0   \n",
       "2                                     0.0   \n",
       "3                                     1.0   \n",
       "4                                     1.0   \n",
       "\n",
       "   neighbourhood_group_cleansed_Queens  \\\n",
       "0                                  0.0   \n",
       "1                                  0.0   \n",
       "2                                  0.0   \n",
       "3                                  0.0   \n",
       "4                                  0.0   \n",
       "\n",
       "   neighbourhood_group_cleansed_Staten Island  room_type_Entire home/apt  \\\n",
       "0                                         0.0                        1.0   \n",
       "1                                         0.0                        1.0   \n",
       "2                                         0.0                        1.0   \n",
       "3                                         0.0                        0.0   \n",
       "4                                         0.0                        0.0   \n",
       "\n",
       "   room_type_Hotel room  room_type_Private room  room_type_Shared room  \n",
       "0                   0.0                     0.0                    0.0  \n",
       "1                   0.0                     0.0                    0.0  \n",
       "2                   0.0                     0.0                    0.0  \n",
       "3                   0.0                     1.0                    0.0  \n",
       "4                   0.0                     1.0                    0.0  \n",
       "\n",
       "[5 rows x 50 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 3. Create Labeled Examples from the Data Set \n",
    "\n",
    "<b>Task</b>: In the code cell below, create labeled examples from DataFrame `df`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = df['host_is_superhost'] \n",
    "X = df.drop(columns = 'host_is_superhost', axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 4. Create Training and Test Data Sets\n",
    "\n",
    "<b>Task</b>: In the code cell below, create training and test sets out of the labeled examples. Create a test set that is 25 percent of the size of the data set. Save the results to variables `X_train, X_test, y_train, y_test`.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=1234)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(21016, 49)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 5. Construct the Neural Network\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 1.  Define Model Structure\n",
    "\n",
    "Next we will create our neural network structure. We will create an input layer, three hidden layers and an output layer:\n",
    "\n",
    "* <b>Input layer</b>: The input layer will have the input shape corresponding to the number of features. \n",
    "* <b>Hidden layers</b>: We will create three hidden layers of widths (number of nodes) 64, 32, and 16. They will utilize the ReLU activation function. \n",
    "* <b>Output layer</b>: The output layer will have a width of 1. The output layer will utilize the sigmoid activation function. Since we are working with binary classification, we will be using the sigmoid activation function to map the output to a probability between 0.0 and 1.0. We can later set a threshold and assume that the prediction is class 1 if the probability is larger than or equal to our threshold, or class 0 if it is lower than our threshold.\n",
    "\n",
    "To construct the neural network model using Keras, we will do the following:\n",
    "* We will use the Keras `Sequential` class to group a stack of layers. This will be our neural network model object. For more information, consult the Keras online [documentation](https://keras.io/api/models/sequential/#sequential-class).\n",
    "* We will use the `InputLayer` class to create the input layer. For more information, consult  the Keras online [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/InputLayer).\n",
    "* We will use the `Dense` class to create each hidden layer and the output layer. For more information, consult the Keras online [documentation](https://keras.io/api/layers/core_layers/dense/).\n",
    "* We will add each layer to the neural network model object.\n",
    "\n",
    "\n",
    "<b>Task:</b> Follow these steps to complete the code in the cell below:\n",
    "\n",
    "1. Create the neural network model object. \n",
    "    * Use ``keras.Sequential() `` to create a model object, and assign the result to the variable ```nn_model```.\n",
    "    \n",
    "    \n",
    "2. Create the input layer: \n",
    "    * Call `keras.layers.InputLayer()` with the argument `input_shape` to specify the dimensions of the input. In this case, the dimensions will be the number of features (coumns) in `X_train`. Assign the number of features to the argument `input_shape`.\n",
    "    * Assign the results to the variable `input_layer`.\n",
    "    * Use `nn_model.add(input_layer)` to add the layer `input_layer` to the neural network model object.\n",
    "\n",
    "\n",
    "3. Create the first hidden layer:\n",
    "    * Call `keras.layers.Dense()` with the arguments `units=64` and `activation='relu'`. \n",
    "    * Assign the results to the variable `hidden_layer_1`.\n",
    "    * Use `nn_model.add(hidden_layer_1)` to add the layer `hidden_layer_1` to the neural network model object.\n",
    "\n",
    "\n",
    "4. Create the second hidden layer using the same approach that you used to create the first hidden layer, specifying 32 units and the `relu` activation function. \n",
    "    * Assign the results to the variable `hidden_layer_2`.\n",
    "    * Add the layer to the neural network model object.\n",
    "    \n",
    "    \n",
    "5. Create the third hidden layer using the same approach that you used to create the first two hidden layers, specifying 16 units and the `relu` activation function. \n",
    "    * Assign the results to the variable `hidden_layer_3`.\n",
    "    * Add the layer to the neural network model object.\n",
    "\n",
    "\n",
    "6. Create the output layer using the same approach that you used to create the hidden layers, specifying 1 unit and the `sigmoid` activation function. \n",
    "   * Assign the results to the variable `output_layer`.\n",
    "   * Add the layer to the neural network model object.\n",
    "   "
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
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense (Dense)                (None, 64)                3200      \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 32)                2080      \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 16)                528       \n",
      "_________________________________________________________________\n",
      "dense_3 (Dense)              (None, 1)                 17        \n",
      "=================================================================\n",
      "Total params: 5,825\n",
      "Trainable params: 5,825\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "# 1. Create model object:\n",
    "nn_model = keras.Sequential()\n",
    "\n",
    "\n",
    "# 2. Create the input layer and add it to the model object: \n",
    "# Create input layer:\n",
    "input_layer = keras.layers.InputLayer(input_shape=(49,))\n",
    "# Add input_layer to the model object:\n",
    "nn_model.add(input_layer)\n",
    "\n",
    "\n",
    "# 3. Create the first hidden layer and add it to the model object:\n",
    "# Create hidden layer:\n",
    "hidden_layer_1 = keras.layers.Dense(units=64, activation='relu')\n",
    "# Add hidden_layer_1 to the model object:\n",
    "nn_model.add(hidden_layer_1)\n",
    "\n",
    "\n",
    "# 4. Create the second hidden layer and add it to the model object:\n",
    "# Create hidden layer:\n",
    "hidden_layer_2 = keras.layers.Dense(units=32, activation='relu')\n",
    "# Add hidden_layer_2 to the model object:\n",
    "nn_model.add(hidden_layer_2)\n",
    "\n",
    "\n",
    "# 5. Create the third hidden layer and add it to the model object:\n",
    "# Create hidden layer:\n",
    "hidden_layer_3 = keras.layers.Dense(units=16, activation='relu')\n",
    "# Add hidden_layer_3 to the model object:\n",
    "nn_model.add(hidden_layer_3)\n",
    "\n",
    "\n",
    "# 6. Create the output layer and add it to the model object:\n",
    "# Create output layer:\n",
    "output_layer = keras.layers.Dense(units=1, activation='sigmoid')\n",
    "# Add output_layer to the model object:\n",
    "nn_model.add(output_layer)\n",
    "\n",
    "\n",
    "# Print summary of neural network model structure\n",
    "nn_model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2. Define the Optimization Function\n",
    "\n",
    "<b>Task:</b> In the code cell below, create a stochastic gradient descent optimizer using  `keras.optimizers.SGD()`. Specify a learning rate of 0.1 using the `learning_rate` parameter. Assign the result to the variable`sgd_optimizer`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "sgd_optimizer = keras.optimizers.SGD(learning_rate=0.1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 3. Define the Loss Function\n",
    "\n",
    "<b>Task:</b> In the code cell below, create a binary cross entropy loss function using `keras.losses.BinaryCrossentropy()`. Use  the parameter `from_logits=False`. Assign the result to the variable  `loss_fn`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 4. Compile the Model\n",
    "\n",
    "<b>Task:</b> In the code cell below, package the network architecture with the optimizer and the loss function using the `compile()` method. \n",
    "\n",
    "\n",
    "You will specify the optimizer, loss function and accuracy evaluation metric. Call the `nn_model.compile()` method with the following arguments:\n",
    "* Use the `optimizer` parameter and assign it your optimizer variable:`optimizer=sgd_optimizer`\n",
    "* Use the `loss` parameter and assign it your loss function variable: `loss=loss_fn`\n",
    "* Use the `metrics` parameter and assign it the `accuracy` evaluation metric: `metrics=['accuracy']`\n",
    "   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "nn_model.compile(optimizer=sgd_optimizer,loss=loss_fn, metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 6. Fit the Model to the Training Data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We will define our own callback class to output information from our model while it is training. Make sure you execute the code cell below so that it can be used in subsequent cells."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "class ProgBarLoggerNEpochs(keras.callbacks.Callback):\n",
    "    \n",
    "    def __init__(self, num_epochs: int, every_n: int = 50):\n",
    "        self.num_epochs = num_epochs\n",
    "        self.every_n = every_n\n",
    "    \n",
    "    def on_epoch_end(self, epoch, logs=None):\n",
    "        if (epoch + 1) % self.every_n == 0:\n",
    "            s = 'Epoch [{}/ {}]'.format(epoch + 1, self.num_epochs)\n",
    "            logs_s = ['{}: {:.4f}'.format(k.capitalize(), v)\n",
    "                      for k, v in logs.items()]\n",
    "            s_list = [s] + logs_s\n",
    "            print(', '.join(s_list))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> In the code cell below, fit the neural network model to the training data.\n",
    "\n",
    "1. Call `nn_model.fit()` with the training data `X_train` and `y_train` as arguments. \n",
    "\n",
    "2. In addition, specify the following parameters:\n",
    "\n",
    "    * Use the `epochs` parameter and assign it the variable to `epochs`: `epochs=num_epochs`\n",
    "    * Use the `verbose` parameter and assign it the value of  0: `verbose=0`\n",
    "    * Use the `callbacks` parameter and assign it a list containing our logger function: \n",
    "    `callbacks=[ProgBarLoggerNEpochs(num_epochs_M, every_n=5)]`  \n",
    "    * We will use a portion of our training data to serve as validation data. Use the  `validation_split` parameter and assign it the value `0.2`\n",
    "    \n",
    "3. Save the results to the variable `history`. \n",
    "\n",
    "<b>Note</b>: This may take a while to run."
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
      "Epoch [5/ 100], Loss: 0.3565, Accuracy: 0.8384, Val_loss: 0.3718, Val_accuracy: 0.8259\n",
      "Epoch [10/ 100], Loss: 0.3339, Accuracy: 0.8492, Val_loss: 0.3687, Val_accuracy: 0.8399\n",
      "Epoch [15/ 100], Loss: 0.3159, Accuracy: 0.8574, Val_loss: 0.3766, Val_accuracy: 0.8349\n",
      "Epoch [20/ 100], Loss: 0.3039, Accuracy: 0.8625, Val_loss: 0.3615, Val_accuracy: 0.8366\n",
      "Epoch [25/ 100], Loss: 0.2901, Accuracy: 0.8690, Val_loss: 0.3912, Val_accuracy: 0.8311\n",
      "Epoch [30/ 100], Loss: 0.2852, Accuracy: 0.8721, Val_loss: 0.3775, Val_accuracy: 0.8268\n",
      "Epoch [35/ 100], Loss: 0.2743, Accuracy: 0.8778, Val_loss: 0.3847, Val_accuracy: 0.8328\n",
      "Epoch [40/ 100], Loss: 0.2642, Accuracy: 0.8828, Val_loss: 0.4232, Val_accuracy: 0.8363\n",
      "Epoch [45/ 100], Loss: 0.2453, Accuracy: 0.8901, Val_loss: 0.4254, Val_accuracy: 0.8325\n",
      "Epoch [50/ 100], Loss: 0.2421, Accuracy: 0.8928, Val_loss: 0.4410, Val_accuracy: 0.8168\n",
      "Epoch [55/ 100], Loss: 0.2308, Accuracy: 0.8986, Val_loss: 0.4424, Val_accuracy: 0.8076\n",
      "Epoch [60/ 100], Loss: 0.2200, Accuracy: 0.9029, Val_loss: 0.4578, Val_accuracy: 0.8147\n",
      "Epoch [65/ 100], Loss: 0.2111, Accuracy: 0.9079, Val_loss: 0.4753, Val_accuracy: 0.8140\n",
      "Epoch [70/ 100], Loss: 0.2053, Accuracy: 0.9117, Val_loss: 0.5007, Val_accuracy: 0.8133\n",
      "Epoch [75/ 100], Loss: 0.1947, Accuracy: 0.9156, Val_loss: 0.5550, Val_accuracy: 0.8109\n",
      "Epoch [80/ 100], Loss: 0.1882, Accuracy: 0.9177, Val_loss: 0.5701, Val_accuracy: 0.8176\n",
      "Epoch [85/ 100], Loss: 0.1850, Accuracy: 0.9215, Val_loss: 0.5582, Val_accuracy: 0.8183\n",
      "Epoch [90/ 100], Loss: 0.1777, Accuracy: 0.9237, Val_loss: 0.6046, Val_accuracy: 0.8069\n",
      "Epoch [95/ 100], Loss: 0.1901, Accuracy: 0.9217, Val_loss: 0.5568, Val_accuracy: 0.8157\n",
      "Epoch [100/ 100], Loss: 0.1876, Accuracy: 0.9217, Val_loss: 0.5991, Val_accuracy: 0.8149\n",
      "Elapsed time: 31.57s\n"
     ]
    }
   ],
   "source": [
    "t0 = time.time() # start time\n",
    "\n",
    "num_epochs = 100 # epochs\n",
    "\n",
    "history = nn_model.fit(X_train,y_train, epochs=num_epochs,verbose=0,callbacks=[ProgBarLoggerNEpochs(num_epochs, every_n=5)],validation_split=0.2)\n",
    "\n",
    "\n",
    "t1 = time.time() # stop time\n",
    "\n",
    "print('Elapsed time: %.2fs' % (t1-t0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "history.history.keys()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Visualize the Model's Performance Over Time\n",
    "\n",
    "The code below outputs both the training loss and accuracy and the validation loss and accuracy. Let us visualize the model's performance over time:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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n3/i5+fHli1bnNmtp556ii1btvDLL78QGhrKrl27OHmy9qVcRURqhfxs87mWdU0puDlHPPfcc1x++eXO1w0aNKBbt27O188//zzfffcdP/zwA2PHji31OnfccQe33HILAC+88AJvvvkmq1atYtCgQSUen5eXx7Rp02jVqhUAY8eO5bnnnnO+/9ZbbzFhwgSuu+46AKZOnerMolSGI6hZunQp/fr1A+Dzzz8nKiqKOXPmcOONNxIfH8/QoUPp0qULQJFJIOPj4+nRowe9evUCzOyViIiUIu9UcOMIcmoJBTdn4OPhxpbnBrrs3lXF8WPtkJGRwTPPPMPcuXM5cuQI+fn5nDx5kvj4+DKv07VrV+e2n58fgYGBzrWSSuLr6+sMbMBcT8lxfGpqKomJifTp08f5vpubGz179sRut1fo8zls3boVd3d3YmJinPsaNmxIu3bt2Lp1KwAPPvgg9913H7/++iuxsbEMHTrU+bnuu+8+hg4dyrp16/jHP/7BkCFDnEGSiIicppZmblRzcwYWiwVfT3eXPKpyJls/P78irx955BG+++47XnjhBRYvXsyGDRvo0qULubm5ZV7Hw8Oj2PdTViBS0vGunn3g7rvvZs+ePdx+++1s2rSJXr168dZbbwFwxRVXsH//fh5++GEOHz7MZZddxiOPPOLS9oqI1Eq2fDBOLVydn+PatpxGwc05aunSpdxxxx1cd911dOnShYiICPbt21ejbQgKCiI8PJzVq1c799lsNtatW1fpa3bo0IH8/HxWrlzp3Hfs2DG2b99Ox44dnfuioqK49957mT17Nv/617/44IMPnO81atSIkSNH8tlnnzFlyhTef//9SrdHRKTeyj9Z8nYtoG6pc1SbNm2YPXs2gwcPxmKx8NRTT1W6K+hsPPDAA0yePJnWrVvTvn173nrrLU6cOFGurNWmTZsICAhwvrZYLHTr1o1rr72W0aNH89577xEQEMDjjz9OZGSkcw2ycePGccUVV9C2bVtOnDjBggUL6NChAwATJ06kZ8+edOrUiZycHH766SfneyIiUkjhbE2eam6kFnjttde488476devH6GhoTz22GOkpaXVeDsee+wxEhISGDFiBG5ubtxzzz0MHDgQN7cz1xtddNFFRV67ubmRn5/PjBkzeOihh7j66qvJzc3loosu4ueff3Z2kdlsNsaMGcPBgwcJDAxk0KBBvP7664A5V8+ECRPYt28fPj4+XHjhhcycObPqP7iISF1XuIi4lmVutPxCIZoW3/XsdjsdOnTgpptu4vnnn3d1c6qF/p6JSL2QvAum9jS3L/o3XPpktd6uIssvKHMjLrV//35+/fVXLr74YnJycpg6dSp79+7l1ltvdXXTRESkLIUzNxotJVLAarXy8ccf07t3b/r378+mTZv4/fffVeciIlLbFa650Tw3IgWioqJYunSpq5shIiIVVbjOppYVFCtzIyIiIhVXiwuKFdyIiIhIxRXpltIkfiIiIlLXFS4iVkGxiIiI1Hm1uKBYwY2IiIhUXL4yN1IHDBgwgHHjxjlfR0dHM2XKlDLPsVgszJkz56zvXVXXERGRGqKaG6lOgwcPZtCgQSW+t3jxYiwWC3/99VeFr7t69Wruueees21eEc888wzdu3cvtv/IkSNcccUVVXqv03388ccEBwdX6z1ERM4ZGi0l1emuu+7it99+4+DBg8XemzFjBr169aJr164Vvm6jRo3w9fWtiiaeUUREBF5eXjVyLxERqQKF57bRPDdS1a6++moaNWrExx9/XGR/RkYGX3/9NXfddRfHjh3jlltuITIyEl9fX7p06cKXX35Z5nVP75bauXMnF110Ed7e3nTs2JHffvut2DmPPfYYbdu2xdfXl5YtW/LUU0+Rl5cHmJmTZ599lo0bN2KxWLBYLM42n94ttWnTJi699FJ8fHxo2LAh99xzDxkZGc7377jjDoYMGcIrr7xC48aNadiwIWPGjHHeqzLi4+O59tpr8ff3JzAwkJtuuonExETn+xs3buSSSy4hICCAwMBAevbsyZo1awBzGYnBgwcTEhKCn58fnTp14ueff650W0REar1anLnRDMVnYhiQl+Wae3v4gsVyxsPc3d0ZMWIEH3/8MU888QSWU+d8/fXX2Gw2brnlFjIyMujZsyePPfYYgYGBzJ07l9tvv51WrVrRp0+fM97Dbrdz/fXXEx4ezsqVK0lNTS1Sn+MQEBDAxx9/TJMmTdi0aROjR48mICCARx99lGHDhrF582bmzZvH77//DkBQUFCxa2RmZjJw4ED69u3L6tWrSUpK4u6772bs2LFFArgFCxbQuHFjFixYwK5duxg2bBjdu3dn9OjRZ/w8JX0+R2CzaNEi8vPzGTNmDMOGDWPhwoUADB8+nB49evDuu+/i5ubGhg0bnCuNjxkzhtzcXP7880/8/PzYsmUL/v7+FW6HiEidUbjOppZlbhTcnEleFrzQxDX3/s9h8PQr16F33nknL7/8MosWLWLAgAGA2SU1dOhQgoKCCAoK4pFHHnEe/8ADDzB//ny++uqrcgU3v//+O9u2bWP+/Pk0aWJ+Hy+88EKxOpknnyxYFTY6OppHHnmEmTNn8uijj+Lj44O/vz/u7u5ERESUeq8vvviC7OxsPv30U/z8zM8/depUBg8ezEsvvUR4eDgAISEhTJ06FTc3N9q3b89VV11FXFxcpYKbuLg4Nm3axN69e4mKigLg008/pVOnTqxevZrevXsTHx/Pv//9b9q3bw9AmzZtnOfHx8czdOhQunTpAkDLli0r3AYRkTqlcLYmP9tMBpTjH+Q1Qd1S9UT79u3p168f06dPB2DXrl0sXryYu+66CwCbzcbzzz9Ply5daNCgAf7+/syfP5/4+PhyXX/r1q1ERUU5AxuAvn37Fjtu1qxZ9O/fn4iICPz9/XnyySfLfY/C9+rWrZszsAHo378/drud7du3O/d16tQJNzc35+vGjRuTlJRUoXsVvmdUVJQzsAHo2LEjwcHBbN26FYDx48dz9913Exsby4svvsju3budxz744IP897//pX///jz99NOVKuAWEalTioyQMsCW67KmnE6ZmzPx8DUzKK66dwXcddddPPDAA7z99tvMmDGDVq1acfHFFwPw8ssv88YbbzBlyhS6dOmCn58f48aNIze36v4yLl++nOHDh/Pss88ycOBAgoKCmDlzJq+++mqV3aMwR5eQg8ViwW63V8u9wBzpdeuttzJ37lx++eUXnn76aWbOnMl1113H3XffzcCBA5k7dy6//vorkydP5tVXX+WBBx6otvaIiLjU6RP35Z0E99oxMESZmzOxWMyuIVc8Kpjeu+mmm7BarXzxxRd8+umn3Hnnnc76m6VLl3Lttddy22230a1bN1q2bMmOHTvKfe0OHTpw4MABjhw54ty3YsWKIscsW7aM5s2b88QTT9CrVy/atGnD/v37ixzj6emJzWY74702btxIZmamc9/SpUuxWq20a9eu3G2uCMfnO3DggHPfli1bSElJoWPHjs59bdu25eGHH+bXX3/l+uuvZ8aMGc73oqKiuPfee5k9ezb/+te/+OCDD6qlrSIitcLpdTa1aJZi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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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x0xJScGxY8eKPSYRERGZkcEssMo1BGZl6ROYNWsWxo0bhw4dOqBTp05YtmwZ0tPTMWHCBADA2LFj4e/vjyVLlhg8b/Xq1Rg8eDA8PDwMtisUCsycORPvvfcegoKCUL9+fcyfPx9+fn4YPHhwRb0tIiIiqsSzwCweAA0fPhwJCQlYsGABYmNj0aZNG+zcuVNbxBwREQGl0jBRde3aNRw8eBB///230WO+/vrrSE9Px5QpU5CUlIRu3bph586dsLOzK/f3Q0RERHkq8RCYQpIkydInUdmkpKTA1dUVycnJrAciIiIqrQ3Dges7xf2ec4Bec8v15Ury+V2lZ4ERERFRJVaJM0AMgIiIiKh8VOIiaAZAREREVD64GCoRERHVOFYcAiMiIqKaxiADxCEwIiIiqgmsKm8fIAZAREREVD5UXA2eiIiIahr9DBBngREREVGNwAwQERER1ThWnAZPRERENY1BJ2gOgREREVFNYMUhMCIiIqppVCyCJiIioppGxT5AREREVNNwCIyIiIhqHBZBExERUY3DafBERERU47ARIhEREdU4XAqDiIiIahz9DJCSGSAiIiKqCeQMkNIKUFaukMPK0idARERE1ZRrAFCvG+BW19JnUgADICIiIiofShUwYbulz8KoypWPIiIiIqoADICIiIioxmEARERERDUOAyAiIiKqcRgAERERUY3DAIiIiIhqHAZAREREVOMwACIiIqIahwEQERER1TgMgIiIiKjGYQBERERENQ4DICIiIqpxGAARERFRjcMAiIiIiGocK0ufQGUkSRIAICUlxcJnQkRERKaSP7flz/GiMAAyIjU1FQAQEBBg4TMhIiKikkpNTYWrq2uR+ygkU8KkGkaj0SA6OhrOzs5QKBSlPk5KSgoCAgIQGRkJFxcXM54h5cdrXXF4rSsOr3XF4bWuOOV5rSVJQmpqKvz8/KBUFl3lwwyQEUqlEnXq1DHb8VxcXPgfqoLwWlccXuuKw2tdcXitK055XeviMj8yFkETERFRjcMAiIiIiGocBkDlyNbWFgsXLoStra2lT6Xa47WuOLzWFYfXuuLwWlecynKtWQRNRERENQ4zQERERFTjMAAiIiKiGocBEBEREdU4DICIiIioxmEAVI5WrFiBwMBA2NnZISQkBMePH7f0KVVpS5YsQceOHeHs7AwvLy8MHjwY165dM9gnMzMT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kky7romJidLIkSMlJycnycXFRZowYYKUmppaLuerkCS99pdERERENQBrgIiIiKjGYQBERERENQ4DICIiIqpxGAARERFRjcMAiIiIiGocBkBERERU4zAAIiIiohqHARARERHVOAyAiIhMoFAosHXrVkufBhGZCQMgIqr0xo8fD4VCUeCrX79+lj41IqqirCx9AkREpujXrx/Wrl1rsM3W1tZCZ0NEVR0zQERUJdja2sLHx8fgq1atWgDE8NTKlSvRv39/2Nvbo0GDBvjll18Mnn/hwgX07t0b9vb28PDwwJQpU5CWlmawz5o1a9C8eXPY2trC19cX06ZNM3j8/v37GDJkCBwcHBAUFIRt27aV75smonLDAIiIqoX58+dj6NChOHfuHEaNGoURI0bgypUrAID09HSEhoaiVq1aOHHiBDZv3ow9e/YYBDgrV67E1KlTMWXKFFy4cAHbtm1Do0aNDF7j7bffxvPPP4/z589jwIABGDVqFB48eFCh75OIzKRc1pgnIjKjcePGSSqVSnJ0dDT4Wrx4sSRJkgRAeumllwyeExISIr388suSJEnS//73P6lWrVpSWlqa9vHt27dLSqVSio2NlSRJkvz8/KS33nqr0HMAIM2bN0/7fVpamgRA2rFjh9neJxFVHNYAEVGV0KtXL6xcudJgm7u7u/Z+586dDR7r3Lkzzp49CwC4cuUKWrduDUdHR+3jXbt2hUajwbVr16BQKBAdHY0+ffoUeQ6tWrXS3nd0dISLiwvi4+NL+5aIyIIYABFRleDo6FhgSMpc7O3tTdrP2tra4HuFQgGNRlMep0RE5Yw1QERULRw9erTA902bNgUANG3aFOfOnUN6err28UOHDkGpVKJJkyZwdnZGYGAgwsLCKvScichymAEioiohKysLsbGxBtusrKzg6ekJANi8eTM6dOiAbt264ccff8Tx48exevVqAMCoUaOwcOFCjBs3DosWLUJCQgKmT5+OMWPGwNvbGwCwaNEivPTSS/Dy8kL//v2RmpqKQ4cOYfr06RX7RomoQjAAIqIqYefOnfD19TXY1qRJE1y9ehWAmKG1ceNG/N///R98fX3x008/oVmzZgAABwcH7Nq1CzNmzEDHjh3h4OCAoUOHYunSpdpjjRs3DpmZmfjss88we/ZseHp6YtiwYRX3BomoQikkSZIsfRJERGWhUCiwZcsWDB482NKnQkRVBGuAiIiIqMZhAEREREQ1DmuAiKjK40g+EZUUM0BERERU4zAAIiIiohqHARARERHVOAyAiIiIqMZhAEREREQ1DgMgIiIiqnEYABEREVGNwwCIiIiIapz/ByVj/CRDPCddAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot training and validation loss\n",
    "plt.plot(range(1, num_epochs + 1), history.history['loss'], label='Training Loss')\n",
    "plt.plot(range(1, num_epochs + 1), history.history['val_loss'], label='Validation Loss')\n",
    "\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Loss')\n",
    "plt.legend()\n",
    "plt.show()\n",
    "\n",
    "\n",
    "# Plot training and validation accuracy\n",
    "plt.plot(range(1, num_epochs + 1), history.history['accuracy'], label='Training Accuracy')\n",
    "plt.plot(range(1, num_epochs + 1), history.history['val_accuracy'], label='Validation Accuracy')\n",
    "\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.legend()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 7. Evaluate the Model's Performance"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We just evaluated our model's performance on the training and validation data. Let's now evaluate its performance on our test data and compare the results.\n",
    "\n",
    "Keras makes the process of evaluating our model very easy. Recall that when we compiled the model, we specified the metric that we wanted to use to evaluate the model: accuracy. The Keras method `evaluate()` will return the loss and accuracy score of our model on our test data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task:</b> In the code cell below, call `nn_model.evaluate()` with `X_test` and `y_test` as arguments. \n",
    "\n",
    "Note: The `evaluate()` method returns a list containing two values. The first value is the loss and the second value is the accuracy score.\n"
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
      "219/219 [==============================] - 0s 459us/step - loss: 0.6042 - accuracy: 0.8177\n",
      "Loss: 0.6042405366897583 Accuracy: 0.8177276849746704\n"
     ]
    }
   ],
   "source": [
    "loss, accuracy = nn_model.evaluate(X_test,y_test)\n",
    "\n",
    "print('Loss: {0} Accuracy: {1}'.format(loss, accuracy))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, for every example in the test set, we will make a prediction using the `predict()` method, receive a probability between 0.0 and 1.0, and then apply a threshold (we will use a threshold of 0.6) to obtain the predicted class. We will save the class label predictions to list `class_label_predictions`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Make predictions on the test set\n",
    "probability_predictions = nn_model.predict(X_test)\n",
    "class_label_predictions=[]\n",
    "\n",
    "for i in range(0,len(y_test)):\n",
    "    if probability_predictions[i] >= 0.6:\n",
    "        class_label_predictions.append(1)\n",
    "    else:\n",
    "        class_label_predictions.append(0)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Task</b>: In the code cell below, create a confusion matrix out of `y_test` and the list `class_label_predictions`."
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
      "[[4872  399]\n",
      " [ 870  865]]\n"
     ]
    }
   ],
   "source": [
    "conf_matrix = confusion_matrix(y_test, class_label_predictions)\n",
    "print(conf_matrix)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 8. Analysis\n",
    "\n",
    "Experiment with the neural network implementation above and compare your results every time you train the network. Pay attention to the time it takes to train the network, and the resulting loss and accuracy on both the training and test data. \n",
    "\n",
    "Below are some ideas for things you can try:\n",
    "\n",
    "* Adjust the learning rate.\n",
    "* Change the number of epochs by experimenting with different values for the variable `num_epochs`.\n",
    "* Add more hidden layers and/or experiment with different values for the `unit` parameter in the hidden layers to change the number of nodes in the hidden layers.\n",
    "\n",
    "\n",
    "Record your findings in the cell below."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "By adjusting the learning rate, it changes the amount of time it takes after the weight of the model is updated, the smaller it is, the more epochs are needed, but the larger it is, it has a tendency to skip the best model. By adjusting the number of epochs, it can lead to better performance of the model but increases the risk of overfitting. By adding more hidden layers, it can help with learning more complex patterns but increases the risk of overfitting and making the model difficult to train."
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
