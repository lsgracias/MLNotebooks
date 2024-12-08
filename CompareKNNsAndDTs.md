
# Lab 3: ML Life Cycle: Modeling


```python
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```

Decision Trees (DTs) and KNNs have many similarities. They are models that are fairly simple and intuitive to understand, can be used to solve both classification and regression problems, and are non-parametric models, meaning that they don't assume a particular relationship between the features and the label prior to training. However, KNNs and DTs each have their own advantages and disadvantages. In addition, one model may be better suited than the other for a particular machine learning problem based on multiple factors, such as the size and quality of the data, the problem-type and the hyperparameter configuration. For example, KNNs require feature values to be scaled, whereas DTs do not. DTs are also able to handle noisy data better than KNNs. 

Often times, it is beneficial to train multiple models on your training data to find the one that performs the best on the test data. 

In this lab, you will continue practicing the modeling phase of the machine learning life cycle. You will train Decision Trees and KNN models to solve a classification problem. You will experiment training multiple variations of the models with different hyperparameter values to find the best performing model for your predictive problem. You will complete the following tasks:
    
    
1. Build your DataFrame and define your ML problem:
    * Load the Airbnb "listings" data set
    * Define the label - what are you predicting?
    * Identify the features
2. Prepare your data:
    * Perform feature engineering by converting categorical features to one-hot encoded values
3. Create labeled examples from the data set
4. Split the data into training and test data sets
5. Train multiple decision trees and evaluate their performances:
    * Fit Decision Tree classifiers to the training data using different hyperparameter values per classifier
    * Evaluate the accuracy of the models' predictions
    * Plot the accuracy of each DT model as a function of hyperparameter max depth
6. Train multiple KNN classifiers and evaluate their performances:
    * Fit KNN classifiers to the training data using different hyperparameter values per classifier
    * Evaluate the accuracy of the models' predictions
    * Plot the accuracy of each KNN model as a function of hyperparameter $k$
7. Analysis:
   * Determine which is the best performing model 
   * Experiment with other factors that can help determine the best performing model

## Part 1. Build Your DataFrame and Define Your ML Problem

#### Load a Data Set and Save it as a Pandas DataFrame


We will work with a new preprocessed, slimmed down version of the Airbnb NYC "listings" data set. This version is almost ready for modeling, with missing values and outliers taken care of. Also note that unstructured fields have been removed.


```python
# Do not remove or edit the line below:
filename = os.path.join(os.getcwd(), "data", "airbnbData_Prepared.csv")
```

<b>Task</b>: Load the data set into a Pandas DataFrame variable named `df`.


```python
df = pd.read_csv(filename)
```

####  Inspect the Data

<b>Task</b>: In the code cell below, inspect the data in DataFrame `df` by printing the number of rows and columns, the column names, and the first ten rows. You may perform any other techniques you'd like to inspect the data.


```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>host_response_rate</th>
      <th>host_acceptance_rate</th>
      <th>host_is_superhost</th>
      <th>host_listings_count</th>
      <th>host_total_listings_count</th>
      <th>host_has_profile_pic</th>
      <th>host_identity_verified</th>
      <th>neighbourhood_group_cleansed</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>...</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>instant_bookable</th>
      <th>calculated_host_listings_count</th>
      <th>calculated_host_listings_count_entire_homes</th>
      <th>calculated_host_listings_count_private_rooms</th>
      <th>calculated_host_listings_count_shared_rooms</th>
      <th>reviews_per_month</th>
      <th>n_host_verifications</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.800000</td>
      <td>0.170000</td>
      <td>False</td>
      <td>8</td>
      <td>8</td>
      <td>True</td>
      <td>True</td>
      <td>Manhattan</td>
      <td>Entire home/apt</td>
      <td>1</td>
      <td>...</td>
      <td>4.79</td>
      <td>4.86</td>
      <td>4.41</td>
      <td>False</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.33</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.090000</td>
      <td>0.690000</td>
      <td>False</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
      <td>True</td>
      <td>Brooklyn</td>
      <td>Entire home/apt</td>
      <td>3</td>
      <td>...</td>
      <td>4.80</td>
      <td>4.71</td>
      <td>4.64</td>
      <td>False</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4.86</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.000000</td>
      <td>0.250000</td>
      <td>False</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
      <td>True</td>
      <td>Brooklyn</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>...</td>
      <td>5.00</td>
      <td>4.50</td>
      <td>5.00</td>
      <td>False</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.02</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>False</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>Manhattan</td>
      <td>Private room</td>
      <td>2</td>
      <td>...</td>
      <td>4.42</td>
      <td>4.87</td>
      <td>4.36</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3.68</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.890731</td>
      <td>0.768297</td>
      <td>False</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
      <td>True</td>
      <td>Manhattan</td>
      <td>Private room</td>
      <td>1</td>
      <td>...</td>
      <td>4.95</td>
      <td>4.94</td>
      <td>4.92</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.87</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>True</td>
      <td>3</td>
      <td>3</td>
      <td>True</td>
      <td>True</td>
      <td>Brooklyn</td>
      <td>Private room</td>
      <td>2</td>
      <td>...</td>
      <td>4.82</td>
      <td>4.87</td>
      <td>4.73</td>
      <td>False</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1.48</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>False</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
      <td>True</td>
      <td>Brooklyn</td>
      <td>Entire home/apt</td>
      <td>3</td>
      <td>...</td>
      <td>4.80</td>
      <td>4.67</td>
      <td>4.57</td>
      <td>True</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.24</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>False</td>
      <td>3</td>
      <td>3</td>
      <td>True</td>
      <td>True</td>
      <td>Manhattan</td>
      <td>Private room</td>
      <td>1</td>
      <td>...</td>
      <td>4.95</td>
      <td>4.84</td>
      <td>4.84</td>
      <td>True</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.82</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>False</td>
      <td>2</td>
      <td>2</td>
      <td>True</td>
      <td>True</td>
      <td>Brooklyn</td>
      <td>Private room</td>
      <td>1</td>
      <td>...</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>False</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.07</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.000000</td>
      <td>0.990000</td>
      <td>True</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
      <td>True</td>
      <td>Brooklyn</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>...</td>
      <td>4.91</td>
      <td>4.93</td>
      <td>4.78</td>
      <td>True</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3.05</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 43 columns</p>
</div>




```python
df.shape
```




    (28022, 43)



#### Define the Label

Assume that your goal is to train a machine learning model that predicts whether an Airbnb host is a 'super host'. This is an example of supervised learning and is a binary classification problem. In our dataset, our label will be the `host_is_superhost` column and the label will either contain the value `True` or `False`. Let's inspect the values in the `host_is_superhost` column.


```python
df['host_is_superhost']
```




    0        False
    1        False
    2        False
    3        False
    4        False
             ...  
    28017    False
    28018    False
    28019     True
    28020     True
    28021    False
    Name: host_is_superhost, Length: 28022, dtype: bool



#### Identify Features

Our features will be all of the remaining columns in the dataset. 

<b>Task:</b> Create a list of the feature names.


```python
columns = [col for col in df.columns if col != 'host_is_superhost']
columns
```




    ['host_response_rate',
     'host_acceptance_rate',
     'host_listings_count',
     'host_total_listings_count',
     'host_has_profile_pic',
     'host_identity_verified',
     'neighbourhood_group_cleansed',
     'room_type',
     'accommodates',
     'bathrooms',
     'bedrooms',
     'beds',
     'price',
     'minimum_nights',
     'maximum_nights',
     'minimum_minimum_nights',
     'maximum_minimum_nights',
     'minimum_maximum_nights',
     'maximum_maximum_nights',
     'minimum_nights_avg_ntm',
     'maximum_nights_avg_ntm',
     'has_availability',
     'availability_30',
     'availability_60',
     'availability_90',
     'availability_365',
     'number_of_reviews',
     'number_of_reviews_ltm',
     'number_of_reviews_l30d',
     'review_scores_rating',
     'review_scores_cleanliness',
     'review_scores_checkin',
     'review_scores_communication',
     'review_scores_location',
     'review_scores_value',
     'instant_bookable',
     'calculated_host_listings_count',
     'calculated_host_listings_count_entire_homes',
     'calculated_host_listings_count_private_rooms',
     'calculated_host_listings_count_shared_rooms',
     'reviews_per_month',
     'n_host_verifications']



## Part 2. Prepare Your Data

Many of the data preparation techniques that you practiced in Unit two have already been performed and the data is almost ready for modeling. The one exception is that a few string-valued categorical features remain. Let's perform one-hot encoding to transform these features into numerical boolean values. This will result in a data set that we can use for modeling.

#### Identify the Features that Should be One-Hot Encoded

**Task**: Find all of the columns whose values are of type 'object' and add the column names to a list named `to_encode`.


```python
to_encode = df.select_dtypes(include='object').columns.tolist()
to_encode
```




    ['neighbourhood_group_cleansed', 'room_type']



**Task**: Find the number of unique values each column in `to_encode` has:


```python
df[to_encode].nunique()
```




    neighbourhood_group_cleansed    5
    room_type                       4
    dtype: int64



#### One-Hot Encode the Features

Instead of one-hot encoding each column using the NumPy `np.where()` or Pandas `pd.get_dummies()` functions, we can use the more robust `OneHotEncoder` transformation class from `sklearn`. For more information, consult the online [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html). 


<b><i>Note:</i></b> We are working with `sklearn` version 0.22.2. You can find documentation for the `OneHotEncoder` class that that corresponds to our version of `sklearn` [here](https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.OneHotEncoder.html). When choosing which features of the  `OneHotEncoder` class to use, do not use features that have been introduced in newer versions of `sklearn`. For example, you should specify the parameter `sparse=False` when calling `OneHotEncoder()` to create an encoder object. The documentation notes that the latest version of `sklearn` uses the `sparse_ouput` parameter instead of `sparse`, but you should stick with `sparse`.

<b>Task</b>: Refer to the documenation and follow the instructions in the code cell below to create one-hot encoded features.


```python
from sklearn.preprocessing import OneHotEncoder  # Import OneHotEncoder

# Create the encoder:
# Create the  Scikit-learn OneHotEncoder object below and assign to variable 'enc'.
# When calling OneHotEncoder(), specify that the 'sparse' parameter is False
enc = OneHotEncoder(sparse=False)

# Apply the encoder:
# Use the method 'enc.fit_transform() to fit the encoder to the data (the two columns) and transform the data into 
# one-hot encoded values
# Convert the results to a DataFrame and save it to variable 'df_enc'
df_enc = pd.DataFrame(enc.fit_transform(df[to_encode]))

```

Let's inspect our new DataFrame `df_enc` that contains the one-hot encoded columns.


```python
df_enc.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Notice that the column names are numerical. 

<b>Task:</b> Complete the code below to reinstate the original column names.



```python
# Use the method enc.get_feature_names() to resintate the original column names. 
# Call the function with the original two column names as arguments.
# Save the results to 'df_enc.columns'

df_enc.columns = enc.get_feature_names(['neighbourhood_group_cleansed', 'room_type'])
```

Let's inspect our new DataFrame `df_enc` once again.


```python
df_enc.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>neighbourhood_group_cleansed_Bronx</th>
      <th>neighbourhood_group_cleansed_Brooklyn</th>
      <th>neighbourhood_group_cleansed_Manhattan</th>
      <th>neighbourhood_group_cleansed_Queens</th>
      <th>neighbourhood_group_cleansed_Staten Island</th>
      <th>room_type_Entire home/apt</th>
      <th>room_type_Hotel room</th>
      <th>room_type_Private room</th>
      <th>room_type_Shared room</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



<b>Task</b>: You can now remove the original columns that we have just transformed from DataFrame `df`.



```python
df.drop(columns = to_encode ,axis=1, inplace=True)
```

<b>Task</b>: You can now join the transformed features contained in `df_enc` with DataFrame `df`


```python
df = df.join(df_enc)
```

Glance at the resulting column names:


```python
df.columns
```




    Index(['host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
           'host_listings_count', 'host_total_listings_count',
           'host_has_profile_pic', 'host_identity_verified', 'accommodates',
           'bathrooms', 'bedrooms', 'beds', 'price', 'minimum_nights',
           'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights',
           'minimum_maximum_nights', 'maximum_maximum_nights',
           'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'has_availability',
           'availability_30', 'availability_60', 'availability_90',
           'availability_365', 'number_of_reviews', 'number_of_reviews_ltm',
           'number_of_reviews_l30d', 'review_scores_rating',
           'review_scores_cleanliness', 'review_scores_checkin',
           'review_scores_communication', 'review_scores_location',
           'review_scores_value', 'instant_bookable',
           'calculated_host_listings_count',
           'calculated_host_listings_count_entire_homes',
           'calculated_host_listings_count_private_rooms',
           'calculated_host_listings_count_shared_rooms', 'reviews_per_month',
           'n_host_verifications', 'neighbourhood_group_cleansed_Bronx',
           'neighbourhood_group_cleansed_Brooklyn',
           'neighbourhood_group_cleansed_Manhattan',
           'neighbourhood_group_cleansed_Queens',
           'neighbourhood_group_cleansed_Staten Island',
           'room_type_Entire home/apt', 'room_type_Hotel room',
           'room_type_Private room', 'room_type_Shared room'],
          dtype='object')



## Part 3. Create Labeled Examples from the Data Set 

<b>Task</b>: Obtain the feature columns from DataFrame `df` and assign to `X`. Obtain the label column from DataFrame `df` and assign to `y`.


```python
feature_list = df.select_dtypes(include=['float64']).columns.tolist()

y = df['host_is_superhost']
X = df[feature_list]
```


```python
print("Number of examples: " + str(X.shape[0]))
print("\nNumber of Features:" + str(X.shape[1]))
print(str(list(X.columns)))
```

    Number of examples: 28022
    
    Number of Features:23
    ['host_response_rate', 'host_acceptance_rate', 'bathrooms', 'bedrooms', 'beds', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'review_scores_rating', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'reviews_per_month', 'neighbourhood_group_cleansed_Bronx', 'neighbourhood_group_cleansed_Brooklyn', 'neighbourhood_group_cleansed_Manhattan', 'neighbourhood_group_cleansed_Queens', 'neighbourhood_group_cleansed_Staten Island', 'room_type_Entire home/apt', 'room_type_Hotel room', 'room_type_Private room', 'room_type_Shared room']


## Part 4. Create Training and Test Data Sets

<b>Task</b>: In the code cell below create training and test sets out of the labeled examples using Scikit-learn's `train_test_split()` function. Save the results to variables `X_train, X_test, y_train, y_test`.

Specify:
1. A test set that is one third (.33) of the size of the data set.
2. A seed value of '123'. 


```python
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=123)
```

<b>Task</b>: Check the dimensions of the training and test datasets.


```python
X_train.shape
```




    (18774, 23)




```python
X_test.shape
```




    (9248, 23)




```python
y_train.shape
```




    (18774,)




```python
y_test.shape
```




    (9248,)



## Part 5. Train Decision Tree Classifers and Evaluate their Performances

The code cell below contains a function definition named `train_test_DT()`. This function should:
1. train a Decision Tree classifier on the training data (Remember to use ```DecisionTreeClassifier()``` to create a model object.)
2. test the resulting model on the test data
3. compute and return the accuracy score of the resulting predicted class labels on the test data. 

<b>Task:</b> Complete the function to make it work.


```python
def train_test_DT(X_train, X_test, y_train, y_test, depth, leaf=1, crit='entropy'):
    
    #Creates DecisionTreeClassifier object
    dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, criterion=crit)

    # Train the classifier using the training data
    dt.fit(X_train, y_train)

    # Test the model on the test data
    y_pred = dt.predict(X_test)
    
    # Compute and return the accuracy score
    result = accuracy_score(y_test, y_pred)
    
    return result
```

#### Train Two Decision Trees and Evaluate Their Performances

<b>Task:</b> Use your function to train two different decision trees, one with a max depth of $8$ and one with a max depth of $32$. Print the max depth and corresponding accuracy score.


```python
depth1 = 8
depth2 = 32

max_depth_range = [depth1,depth2]
acc_scores = []

for md in max_depth_range:
    score = train_test_DT(X_train,X_test,y_train,y_test,md)
    print(f"Max Depth: {md}, accuracy score = {score}")
    acc_scores.append(float(score))
```

    Max Depth: 8, accuracy score = 0.8240700692041523
    Max Depth: 32, accuracy score = 0.7803849480968859


#### Visualize Accuracy

We will be creating multiple visualizations that plot a specific model's hyperparameter value (such as max depth) and the resulting accuracy score of the model.

To create more clean and maintainable code, we will create one visualization function that can be called every time a plot is needed. 

<b>Task:</b> In the code cell below, create a function called `visualize_accuracy()` that accepts two arguments:

1. a list of hyperparamter values
2. a list of accuracy scores

Both lists must be of the same size.

Inside the function, implement a `seaborn` lineplot in which hyperparameter values will be on the x-axis and accuracy scores will be on the y-axis. <i>Hint</i>: You implemented a lineplot in this week's assignment.


```python
def visualize_accuracy(hype_par, acc_scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    sns.lineplot(x=hype_par, y = acc_scores, marker = 'o', label = "Full Training Set")
    
    plt.title("Test set accuracy of the model's prediction, for $max\_depth$")
    ax.set_xlabel("hyperparameters")
    ax.set_ylabel("accuracy")
    plt.show()
```

<b>Task</b>: Test your visualization function below by calling the function to plot the max depth values and accuracy scores of the two decision trees that you just trained.


```python
visualize_accuracy(max_depth_range, acc_scores)
```


![png](output_63_0.png)


<b>Analysis</b>: Does this graph provide a sufficient visualization for determining a value of max depth that produces a high performing model?

Yes it does because it cover the accuracy scores from having a depth of 8 to a depth of 32. It performs well when the depth is lower compared to when the depth is higher.

#### Train Multiple Decision Trees Using Different Hyperparameter Values and Evaluate Their Performances

<b>Task:</b> Let's train on more values for max depth.

1. Train six different decision trees, using the following values for max depth: $1, 2, 4, 8, 16, 32$
2. Use your visualization function to plot the values of max depth and each model's resulting accuracy score.


```python
max_depth_range = [depth1,depth2]
acc_scores = []

for md in max_depth_range:
    score = train_test_DT(X_train,X_test,y_train,y_test,md)
    print(f"Max Depth: {md}, accuracy score = {score}")
    acc_scores.append(float(score))
```

    Max Depth: 8, accuracy score = 0.8246107266435986
    Max Depth: 32, accuracy score = 0.7806012110726643


<b>Analysis</b>: Analyze this graph. Pay attention to the accuracy scores. Answer the following questions in the cell below.<br>

How would you go about choosing the best model configuration based on this plot? <br>
What other hyperparameters of interest would you want to tune to make sure you are finding the best performing model?

To choose the best model configuration based on the plot, you need to look at the accuracy scores. It's highest point is when the max depth is 8 therefore the best model occurs then. To find the best performing model it is important to fine tune the hyperparameters such as the min_samples_leaf/split as it can help with issues such as overfitting.

## Part 6. Train KNN Classifiers and Evaluate their Performances


The code cell below contains function definition named `train_test_knn()`. This function should:
1. train a KNN classifier on the training data (Remember to use ```KNeighborsClassifier()``` to create a model object).
2. test the resulting model on the test data
3. compute and return the accuracy score of the resulting predicted class labels on the test data. 

<i>Note</i>: You will train KNN classifiers using the same training and test data that you used to train decision trees.

<b>Task:</b> Complete the function to make it work.


```python
def train_test_knn(X_train, X_test, y_train, y_test, k):
    
    knn = KNeighborsClassifier(n_neighbors=k)
    
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test,y_pred)
    
    return accuracy
```

#### Train Three KNN Classifiers and Evaluate Their Performances

<b>Task:</b> Use your function to train three different KNN classifiers, each with a different value for hyperparameter $k$: $3, 30$, and $300$. <i>Note</i>: This make take a second.



```python
k_values = [3, 30, 300]
acc_val = []
for k in k_values:
    accuracy = train_test_knn(X_train, X_test, y_train, y_test, k)
    print(f"Accuracy for k={k}: {accuracy}")
    acc_val.append(float(accuracy))
```

    Accuracy for k=3: 0.7621107266435986
    Accuracy for k=30: 0.7744377162629758
    Accuracy for k=300: 0.7683823529411765


<b>Task:</b> Now call the function `visualize_accuracy()` with the appropriate arguments to plot the results.


```python
visualize_accuracy(k_values,acc_val)
```


![png](output_77_0.png)


#### Train Multiple KNN Classifiers Using Different Hyperparameter Values and Evaluate Their Performances

<b>Task:</b> Let's train on more values for $k$.

1. Array `k_range` contains multiple values for hyperparameter $k$. Train one KNN model per value of $k$
2. Use your visualization function to plot the values of $k$ and each model's resulting accuracy score.

<i>Note</i>: This make take a second.


```python
k_range = np.arange(1, 40, step = 3) 
k_range
```




    array([ 1,  4,  7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37])




```python
acc_val = []
for k in k_range:
    accuracy = train_test_knn(X_train, X_test, y_train, y_test, k)
    print(f"Accuracy for k={k}: {accuracy}")
    acc_val.append(float(accuracy))

visualize_accuracy(k_range,acc_val)
```

    Accuracy for k=1: 0.738538062283737
    Accuracy for k=4: 0.7728157439446367
    Accuracy for k=7: 0.764273356401384
    Accuracy for k=10: 0.77530276816609
    Accuracy for k=13: 0.7689230103806228
    Accuracy for k=16: 0.7732482698961938
    Accuracy for k=19: 0.7725994809688581
    Accuracy for k=22: 0.773356401384083
    Accuracy for k=25: 0.7715181660899654
    Accuracy for k=28: 0.7720588235294118
    Accuracy for k=31: 0.7736807958477508
    Accuracy for k=34: 0.7748702422145328
    Accuracy for k=37: 0.770544982698962



![png](output_81_1.png)


## Part 7. Analysis

1. Compare the performance of the KNN model relative to the Decision Tree model, with various hyperparameter values. Which model performed the best (yielded the highest accuracy score)? Record your findings in the cell below.

2. We tuned hyperparameter $k$ for KNNs and hyperparamter max depth for DTs. Consider other hyperparameters that can be tuned in an attempt to find the best performing model. Try a different combination of hyperparamters for both KNNs and DTs, retrain the models, obtain the accuracy scores and record your findings below. 

    <i>Note:</i> You can consult Scikit-learn documentation for both the [`KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) class and the [`DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) class to see how specific hyperparameters are passed as parameters to the model object.

The KNN model's highest accuracy score was 77.5% while the Decision Tree model's highest accuracy score was 82.46% which means that the Decision Tree model performed the best.

Some other hyperparameters that we could use to determine the best performing model within the KNN model is weights and algorithm which can help with classification and also the way to finding the nearest neighbors also within Decision Tree model is min_samples_leaf/split which can help with preventing overfitting.


```python

```
