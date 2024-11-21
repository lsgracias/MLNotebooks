
# Lab 2: ML Life Cycle: Data Understanding and Data Preparation


```python
import os
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt 
import seaborn as sns
```

In this lab, you will practice the second and third steps of the machine learning life cycle: data understanding and data preparation. You will beging preparing your data so that it can be used to train a machine learning model that solves a regression problem. Note that by the end of the lab, your data set won't be completely ready for the modeling phase, but you will gain experience using some common data preparation techniques. 

You will complete the following tasks to transform your data:

1. Build your data matrix and define your ML problem:
    * Load the Airbnb "listings" data set into a DataFrame and inspect the data
    * Define the label and convert the label's data type to one that is more suitable for modeling
    * Identify features
2. Clean your data:
    * Handle outliers by building a new regression label column by winsorizing outliers
    * Handle missing data by replacing all missing values in the dataset with means
3. Perform feature transformation using one-hot encoding
4. Explore your data:
    * Identify two features with the highest correlation with label
    * Build appropriate bivariate plots to visualize the correlations between features and the label
5. Analysis:
    * Analyze the relationship between the features and the label
    * Brainstorm what else needs to be done to fully prepare the data for modeling

## Part 1. Build Your Data Matrix (DataFrame) and Define Your ML Problem

#### Load a Data Set and Save it as a Pandas DataFrame

We will be working with the Airbnb NYC "listings" data set. Use the specified path and name of the file to load the data. Save it as a Pandas DataFrame called `df`.


```python
# Do not remove or edit the line below:
filename = os.path.join(os.getcwd(), "data", "airbnbData.csv")
```

**Task**: Load the data and save it to DataFrame `df`.

<i>Note:</i> You may receive a warning message. Ignore this warning.


```python
df = pd.read_csv(filename, header=0)
```

    /usr/local/lib/python3.6/dist-packages/IPython/core/interactiveshell.py:2728: DtypeWarning: Columns (67) have mixed types.Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)


####  Inspect the Data


<b>Task</b>: Display the shape of `df` -- that is, the number of rows and columns.


```python
df.shape
```




    (38277, 74)



<b>Task</b>: Display the column names.


```python
df.columns
```




    Index(['id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'description',
           'neighborhood_overview', 'picture_url', 'host_id', 'host_url',
           'host_name', 'host_since', 'host_location', 'host_about',
           'host_response_time', 'host_response_rate', 'host_acceptance_rate',
           'host_is_superhost', 'host_thumbnail_url', 'host_picture_url',
           'host_neighbourhood', 'host_listings_count',
           'host_total_listings_count', 'host_verifications',
           'host_has_profile_pic', 'host_identity_verified', 'neighbourhood',
           'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'latitude',
           'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',
           'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price',
           'minimum_nights', 'maximum_nights', 'minimum_minimum_nights',
           'maximum_minimum_nights', 'minimum_maximum_nights',
           'maximum_maximum_nights', 'minimum_nights_avg_ntm',
           'maximum_nights_avg_ntm', 'calendar_updated', 'has_availability',
           'availability_30', 'availability_60', 'availability_90',
           'availability_365', 'calendar_last_scraped', 'number_of_reviews',
           'number_of_reviews_ltm', 'number_of_reviews_l30d', 'first_review',
           'last_review', 'review_scores_rating', 'review_scores_accuracy',
           'review_scores_cleanliness', 'review_scores_checkin',
           'review_scores_communication', 'review_scores_location',
           'review_scores_value', 'license', 'instant_bookable',
           'calculated_host_listings_count',
           'calculated_host_listings_count_entire_homes',
           'calculated_host_listings_count_private_rooms',
           'calculated_host_listings_count_shared_rooms', 'reviews_per_month'],
          dtype='object')



**Task**: Get a peek at the data by displaying the first few rows, as you usually do.


```python
df.head()
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
      <th>id</th>
      <th>listing_url</th>
      <th>scrape_id</th>
      <th>last_scraped</th>
      <th>name</th>
      <th>description</th>
      <th>neighborhood_overview</th>
      <th>picture_url</th>
      <th>host_id</th>
      <th>host_url</th>
      <th>...</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>license</th>
      <th>instant_bookable</th>
      <th>calculated_host_listings_count</th>
      <th>calculated_host_listings_count_entire_homes</th>
      <th>calculated_host_listings_count_private_rooms</th>
      <th>calculated_host_listings_count_shared_rooms</th>
      <th>reviews_per_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2595</td>
      <td>https://www.airbnb.com/rooms/2595</td>
      <td>20211204143024</td>
      <td>2021-12-05</td>
      <td>Skylit Midtown Castle</td>
      <td>Beautiful, spacious skylit studio in the heart...</td>
      <td>Centrally located in the heart of Manhattan ju...</td>
      <td>https://a0.muscache.com/pictures/f0813a11-40b2...</td>
      <td>2845</td>
      <td>https://www.airbnb.com/users/show/2845</td>
      <td>...</td>
      <td>4.79</td>
      <td>4.86</td>
      <td>4.41</td>
      <td>NaN</td>
      <td>f</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3831</td>
      <td>https://www.airbnb.com/rooms/3831</td>
      <td>20211204143024</td>
      <td>2021-12-05</td>
      <td>Whole flr w/private bdrm, bath &amp; kitchen(pls r...</td>
      <td>Enjoy 500 s.f. top floor in 1899 brownstone, w...</td>
      <td>Just the right mix of urban center and local n...</td>
      <td>https://a0.muscache.com/pictures/e49999c2-9fd5...</td>
      <td>4869</td>
      <td>https://www.airbnb.com/users/show/4869</td>
      <td>...</td>
      <td>4.80</td>
      <td>4.71</td>
      <td>4.64</td>
      <td>NaN</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4.86</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5121</td>
      <td>https://www.airbnb.com/rooms/5121</td>
      <td>20211204143024</td>
      <td>2021-12-05</td>
      <td>BlissArtsSpace!</td>
      <td>&lt;b&gt;The space&lt;/b&gt;&lt;br /&gt;HELLO EVERYONE AND THANK...</td>
      <td>NaN</td>
      <td>https://a0.muscache.com/pictures/2090980c-b68e...</td>
      <td>7356</td>
      <td>https://www.airbnb.com/users/show/7356</td>
      <td>...</td>
      <td>4.91</td>
      <td>4.47</td>
      <td>4.52</td>
      <td>NaN</td>
      <td>f</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5136</td>
      <td>https://www.airbnb.com/rooms/5136</td>
      <td>20211204143024</td>
      <td>2021-12-05</td>
      <td>Spacious Brooklyn Duplex, Patio + Garden</td>
      <td>We welcome you to stay in our lovely 2 br dupl...</td>
      <td>NaN</td>
      <td>https://a0.muscache.com/pictures/miso/Hosting-...</td>
      <td>7378</td>
      <td>https://www.airbnb.com/users/show/7378</td>
      <td>...</td>
      <td>5.00</td>
      <td>4.50</td>
      <td>5.00</td>
      <td>NaN</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5178</td>
      <td>https://www.airbnb.com/rooms/5178</td>
      <td>20211204143024</td>
      <td>2021-12-05</td>
      <td>Large Furnished Room Near B'way</td>
      <td>Please don’t expect the luxury here just a bas...</td>
      <td>Theater district, many restaurants around here.</td>
      <td>https://a0.muscache.com/pictures/12065/f070997...</td>
      <td>8967</td>
      <td>https://www.airbnb.com/users/show/8967</td>
      <td>...</td>
      <td>4.42</td>
      <td>4.87</td>
      <td>4.36</td>
      <td>NaN</td>
      <td>f</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3.68</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 74 columns</p>
</div>



#### Define the Label

Assume that your goal is to train a machine learning model that predicts the price of an Airbnb. This is an example of supervised learning and is a regression problem. In our dataset, our label will be the `price` column. Let's inspect the values in the `price` column.


```python
df['price']
```




    0        $150.00
    1         $75.00
    2         $60.00
    3        $275.00
    4         $68.00
              ...   
    38272     $79.00
    38273     $76.00
    38274    $116.00
    38275    $106.00
    38276    $689.00
    Name: price, Length: 38277, dtype: object



Notice the `price` column contains values that are listed as $<$currency_name$>$$<$numeric_value$>$. 
<br>For example, it contains values that look like this: `$120`. <br>

**Task**:  Obtain the data type of the values in this column:


```python
df['price'].dtype
```




    dtype('O')



Notice that the data type is "object," which in Pandas translates to the String data type.

**Task**:  Display the first 15 unique values of  the `price` column:


```python
df['price'].head(15)

```




    0     $150.00
    1      $75.00
    2      $60.00
    3     $275.00
    4      $68.00
    5      $75.00
    6      $98.00
    7      $89.00
    8      $65.00
    9      $62.00
    10     $90.00
    11    $199.00
    12     $96.00
    13    $299.00
    14    $140.00
    Name: price, dtype: object



In order for us to use the prices for modeling, we will have to transform the values in the `price` column from strings to floats. We will:

* remove the dollar signs (in this case, the platform forces the currency to be the USD, so we do not need to worry about targeting, say, the Japanese Yen sign, nor about converting the values into USD). 
* remove the commas from all values that are in the thousands or above: for example, `$2,500`. 

The code cell below accomplishes this.


```python
df['price'] = df['price'].str.replace(',', '')
df['price'] = df['price'].str.replace('$', '')
df['price'] = df['price'].astype(float)
```

**Task**:  Display the first 15 unique values of  the `price` column again to make sure they have been transformed.


```python
df['price'].head(15)
```




    0     150.0
    1      75.0
    2      60.0
    3     275.0
    4      68.0
    5      75.0
    6      98.0
    7      89.0
    8      65.0
    9      62.0
    10     90.0
    11    199.0
    12     96.0
    13    299.0
    14    140.0
    Name: price, dtype: float64



#### Identify Features

Simply by inspecting the data, let's identify some columns that should not serve as features - those that will not help us solve our predictive ML problem. 

Some that stand out are columns that contain website addresses (URLs).

**Task**: Create a list which contains the names of columns that contain URLs. Save the resulting list to variable `url_colnames`.

*Tip*: There are different ways to accomplish this, including using Python list comprehensions.


```python
url_colnames = [column for column in df if 'url' in column]
url_colnames
```




    ['listing_url',
     'picture_url',
     'host_url',
     'host_thumbnail_url',
     'host_picture_url']



**Task**: Drop the columns with the specified names contained in list `url_colnames` in place (that is, make sure this change applies to the original DataFrame `df`, instead of creating a temporary new DataFrame object with fewer columns).


```python
df = df.drop(columns = url_colnames)
```

**Task**: Display the shape of the data to verify that the new number of columns is what you expected.


```python
df.columns
```




    Index(['id', 'scrape_id', 'last_scraped', 'name', 'description',
           'neighborhood_overview', 'host_id', 'host_name', 'host_since',
           'host_location', 'host_about', 'host_response_time',
           'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
           'host_neighbourhood', 'host_listings_count',
           'host_total_listings_count', 'host_verifications',
           'host_has_profile_pic', 'host_identity_verified', 'neighbourhood',
           'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'latitude',
           'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',
           'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price',
           'minimum_nights', 'maximum_nights', 'minimum_minimum_nights',
           'maximum_minimum_nights', 'minimum_maximum_nights',
           'maximum_maximum_nights', 'minimum_nights_avg_ntm',
           'maximum_nights_avg_ntm', 'calendar_updated', 'has_availability',
           'availability_30', 'availability_60', 'availability_90',
           'availability_365', 'calendar_last_scraped', 'number_of_reviews',
           'number_of_reviews_ltm', 'number_of_reviews_l30d', 'first_review',
           'last_review', 'review_scores_rating', 'review_scores_accuracy',
           'review_scores_cleanliness', 'review_scores_checkin',
           'review_scores_communication', 'review_scores_location',
           'review_scores_value', 'license', 'instant_bookable',
           'calculated_host_listings_count',
           'calculated_host_listings_count_entire_homes',
           'calculated_host_listings_count_private_rooms',
           'calculated_host_listings_count_shared_rooms', 'reviews_per_month'],
          dtype='object')



**Task**: In the code cell below, display the features that we will use to solve our ML problem.


```python
column_features = ['id', 'host_id', 'host_since', 'host_location', 
           'bedrooms', 'beds', 'amenities', 'price', 'minimum_nights',
           'maximum_nights', 'number_of_reviews', 
           'review_scores_rating', 'review_scores_accuracy', 
           'review_scores_location']
df_features = df[column_features]

```

**Task**: Are there any other features that you think may not be well suited for our machine learning problem? Note your findings in the markdown cell below.

I don't think there are several features that are not well suited because it doesn't have an immediate impact to help solve the problem, it is useful data to collect but doesn't change the outcome like the availability.

## Part 2. Clean Your Data

Let's now handle outliers and missing data.

### a. Handle Outliers

Let us prepare the data in our label column. Namely, we will detect and replace outliers in the data using winsorization.

**Task**: Create a new version of the `price` column, named `label_price`, in which you will replace the top and bottom 1% outlier values with the corresponding percentile value. Add this new column to the DataFrame `df`.

Remember, you will first need to load the `stats` module from the `scipy` package:


```python
from scipy import stats

lower = df['price'].quantile(0.01)
upper = df['price'].quantile(0.99)

df['label_price'] = df['price']

df.loc[df['label_price'] < lower, 'label_price'] = lower
df.loc[df['label_price'] > upper, 'label_price'] = upper

```

Let's verify that the new column `label_price` was added to DataFrame `df`:


```python
df.head()
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
      <th>id</th>
      <th>scrape_id</th>
      <th>last_scraped</th>
      <th>name</th>
      <th>description</th>
      <th>neighborhood_overview</th>
      <th>host_id</th>
      <th>host_name</th>
      <th>host_since</th>
      <th>host_location</th>
      <th>...</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>license</th>
      <th>instant_bookable</th>
      <th>calculated_host_listings_count</th>
      <th>calculated_host_listings_count_entire_homes</th>
      <th>calculated_host_listings_count_private_rooms</th>
      <th>calculated_host_listings_count_shared_rooms</th>
      <th>reviews_per_month</th>
      <th>label_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2595</td>
      <td>20211204143024</td>
      <td>2021-12-05</td>
      <td>Skylit Midtown Castle</td>
      <td>Beautiful, spacious skylit studio in the heart...</td>
      <td>Centrally located in the heart of Manhattan ju...</td>
      <td>2845</td>
      <td>Jennifer</td>
      <td>2008-09-09</td>
      <td>New York, New York, United States</td>
      <td>...</td>
      <td>4.86</td>
      <td>4.41</td>
      <td>NaN</td>
      <td>f</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.33</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3831</td>
      <td>20211204143024</td>
      <td>2021-12-05</td>
      <td>Whole flr w/private bdrm, bath &amp; kitchen(pls r...</td>
      <td>Enjoy 500 s.f. top floor in 1899 brownstone, w...</td>
      <td>Just the right mix of urban center and local n...</td>
      <td>4869</td>
      <td>LisaRoxanne</td>
      <td>2008-12-07</td>
      <td>New York, New York, United States</td>
      <td>...</td>
      <td>4.71</td>
      <td>4.64</td>
      <td>NaN</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4.86</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5121</td>
      <td>20211204143024</td>
      <td>2021-12-05</td>
      <td>BlissArtsSpace!</td>
      <td>&lt;b&gt;The space&lt;/b&gt;&lt;br /&gt;HELLO EVERYONE AND THANK...</td>
      <td>NaN</td>
      <td>7356</td>
      <td>Garon</td>
      <td>2009-02-03</td>
      <td>New York, New York, United States</td>
      <td>...</td>
      <td>4.47</td>
      <td>4.52</td>
      <td>NaN</td>
      <td>f</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.52</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5136</td>
      <td>20211204143024</td>
      <td>2021-12-05</td>
      <td>Spacious Brooklyn Duplex, Patio + Garden</td>
      <td>We welcome you to stay in our lovely 2 br dupl...</td>
      <td>NaN</td>
      <td>7378</td>
      <td>Rebecca</td>
      <td>2009-02-03</td>
      <td>Brooklyn, New York, United States</td>
      <td>...</td>
      <td>4.50</td>
      <td>5.00</td>
      <td>NaN</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.02</td>
      <td>275.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5178</td>
      <td>20211204143024</td>
      <td>2021-12-05</td>
      <td>Large Furnished Room Near B'way</td>
      <td>Please don’t expect the luxury here just a bas...</td>
      <td>Theater district, many restaurants around here.</td>
      <td>8967</td>
      <td>Shunichi</td>
      <td>2009-03-03</td>
      <td>New York, New York, United States</td>
      <td>...</td>
      <td>4.87</td>
      <td>4.36</td>
      <td>NaN</td>
      <td>f</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3.68</td>
      <td>68.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 70 columns</p>
</div>



**Task**: Check that the values of `price` and `label_price` are *not* identical. 

You will do this by subtracting the two columns and finding the resulting *unique values*  of the resulting difference. <br>Note: If all values are identical, the difference would not contain unique values. If this is the case, outlier removal did not work.


```python
difference = df['label_price'] - df['price']
unique_values = difference.unique
unique_values
```




    <bound method Series.unique of 0        0.0
    1        0.0
    2        0.0
    3        0.0
    4        0.0
            ... 
    38272    0.0
    38273    0.0
    38274    0.0
    38275    0.0
    38276    0.0
    Length: 38277, dtype: float64>



### b. Handle Missing Data

Next we are going to find missing values in our entire dataset and impute the missing values by
replace them with means.

#### Identifying missingness

**Task**: Check if a given value in the data is missing, and sum up the resulting values by columns. Save this sum to variable `nan_count`. Print the results.


```python
nan_count = df.isnull().sum()
nan_count
```




    id                                                 0
    scrape_id                                          0
    last_scraped                                       0
    name                                              13
    description                                     1192
                                                    ... 
    calculated_host_listings_count_entire_homes        0
    calculated_host_listings_count_private_rooms       0
    calculated_host_listings_count_shared_rooms        0
    reviews_per_month                               9504
    label_price                                        0
    Length: 70, dtype: int64



Those are more columns than we can eyeball! For this exercise, we don't care about the number of missing values -- we just want to get a list of columns that have *any* missing values.

<b>Task</b>: From the variable `nan_count`, create a new series called `nan_detected` that contains `True` or `False` values that indicate whether the number of missing values is *not zero*:


```python
nan_detected = nan_count != 0
nan_detected
```




    id                                              False
    scrape_id                                       False
    last_scraped                                    False
    name                                             True
    description                                      True
                                                    ...  
    calculated_host_listings_count_entire_homes     False
    calculated_host_listings_count_private_rooms    False
    calculated_host_listings_count_shared_rooms     False
    reviews_per_month                                True
    label_price                                     False
    Length: 70, dtype: bool



Since replacing the missing values with the mean only makes sense for the columns that contain numerical values (and not for strings), let us create another condition: the *type* of the column must be `int` or `float`.

**Task**: Create a series that contains `True` if the type of the column is either `int64` or `float64`. Save the results to the variable `is_int_or_float`.


```python
is_int_or_float = df.dtypes.isin([np.dtype('int64'), np.dtype('float64')])
is_int_or_float
```




    id                                               True
    scrape_id                                        True
    last_scraped                                    False
    name                                            False
    description                                     False
                                                    ...  
    calculated_host_listings_count_entire_homes      True
    calculated_host_listings_count_private_rooms     True
    calculated_host_listings_count_shared_rooms      True
    reviews_per_month                                True
    label_price                                      True
    Length: 70, dtype: bool



<b>Task</b>: Combine the two binary series (`nan_detected` and `is_int_or_float`) into a new series named `to_impute`. It will contain the value `True` if a column contains missing values *and* is of type 'int' or 'float'


```python
to_impute = nan_detected & is_int_or_float
to_impute
```




    id                                              False
    scrape_id                                       False
    last_scraped                                    False
    name                                            False
    description                                     False
                                                    ...  
    calculated_host_listings_count_entire_homes     False
    calculated_host_listings_count_private_rooms    False
    calculated_host_listings_count_shared_rooms     False
    reviews_per_month                                True
    label_price                                     False
    Length: 70, dtype: bool



Finally, let's display a list that contains just the selected column names contained in `to_impute`:


```python
df.columns[to_impute]
```




    Index(['host_listings_count', 'host_total_listings_count', 'bathrooms',
           'bedrooms', 'beds', 'minimum_minimum_nights', 'maximum_minimum_nights',
           'minimum_maximum_nights', 'maximum_maximum_nights',
           'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calendar_updated',
           'review_scores_rating', 'review_scores_accuracy',
           'review_scores_cleanliness', 'review_scores_checkin',
           'review_scores_communication', 'review_scores_location',
           'review_scores_value', 'reviews_per_month'],
          dtype='object')



We just identified and displayed the list of candidate columns for potentially replacing missing values with the column mean.

Assume that you have decided that you should impute the values for these specific columns: `host_listings_count`, `host_total_listings_count`, `bathrooms`, `bedrooms`, and `beds`:


```python
to_impute_selected = ['host_listings_count', 'host_total_listings_count', 'bathrooms',
       'bedrooms', 'beds']
```

#### Keeping record of the missingness: creating dummy variables 

As a first step, you will now create dummy variables indicating the missingness of the values.

**Task**: For every column listed in `to_impute_selected`, create a new corresponding column called `<original-column-name>_na`. These columns should contain the a `True`or `False` value in place of `NaN`.


```python
for column in to_impute_selected:
    df[column + '_na'] = df[column].isnull()
```

Check that the DataFrame contains the new variables:


```python
df.head()
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
      <th>id</th>
      <th>scrape_id</th>
      <th>last_scraped</th>
      <th>name</th>
      <th>description</th>
      <th>neighborhood_overview</th>
      <th>host_id</th>
      <th>host_name</th>
      <th>host_since</th>
      <th>host_location</th>
      <th>...</th>
      <th>calculated_host_listings_count_entire_homes</th>
      <th>calculated_host_listings_count_private_rooms</th>
      <th>calculated_host_listings_count_shared_rooms</th>
      <th>reviews_per_month</th>
      <th>label_price</th>
      <th>host_listings_count_na</th>
      <th>host_total_listings_count_na</th>
      <th>bathrooms_na</th>
      <th>bedrooms_na</th>
      <th>beds_na</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2595</td>
      <td>20211204143024</td>
      <td>2021-12-05</td>
      <td>Skylit Midtown Castle</td>
      <td>Beautiful, spacious skylit studio in the heart...</td>
      <td>Centrally located in the heart of Manhattan ju...</td>
      <td>2845</td>
      <td>Jennifer</td>
      <td>2008-09-09</td>
      <td>New York, New York, United States</td>
      <td>...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.33</td>
      <td>150.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3831</td>
      <td>20211204143024</td>
      <td>2021-12-05</td>
      <td>Whole flr w/private bdrm, bath &amp; kitchen(pls r...</td>
      <td>Enjoy 500 s.f. top floor in 1899 brownstone, w...</td>
      <td>Just the right mix of urban center and local n...</td>
      <td>4869</td>
      <td>LisaRoxanne</td>
      <td>2008-12-07</td>
      <td>New York, New York, United States</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4.86</td>
      <td>75.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5121</td>
      <td>20211204143024</td>
      <td>2021-12-05</td>
      <td>BlissArtsSpace!</td>
      <td>&lt;b&gt;The space&lt;/b&gt;&lt;br /&gt;HELLO EVERYONE AND THANK...</td>
      <td>NaN</td>
      <td>7356</td>
      <td>Garon</td>
      <td>2009-02-03</td>
      <td>New York, New York, United States</td>
      <td>...</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.52</td>
      <td>60.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5136</td>
      <td>20211204143024</td>
      <td>2021-12-05</td>
      <td>Spacious Brooklyn Duplex, Patio + Garden</td>
      <td>We welcome you to stay in our lovely 2 br dupl...</td>
      <td>NaN</td>
      <td>7378</td>
      <td>Rebecca</td>
      <td>2009-02-03</td>
      <td>Brooklyn, New York, United States</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.02</td>
      <td>275.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5178</td>
      <td>20211204143024</td>
      <td>2021-12-05</td>
      <td>Large Furnished Room Near B'way</td>
      <td>Please don’t expect the luxury here just a bas...</td>
      <td>Theater district, many restaurants around here.</td>
      <td>8967</td>
      <td>Shunichi</td>
      <td>2009-03-03</td>
      <td>New York, New York, United States</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3.68</td>
      <td>68.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 75 columns</p>
</div>



#### Replacing the missing values with mean values of the column

**Task**: For every column listed in `to_impute_selected`, fill the missing values with the corresponding mean of all values in the column (do not create new columns).


```python
for column in to_impute_selected:
    df[column].fillna(df[column].mean(), inplace=True)
```

Check your results below. The code displays the count of missing values for each of the selected columns. 


```python
for colname in to_impute_selected:
    print("{} missing values count :{}".format(colname, np.sum(df[colname].isnull(), axis = 0)))

```

    host_listings_count missing values count :0
    host_total_listings_count missing values count :0
    bathrooms missing values count :38277
    bedrooms missing values count :0
    beds missing values count :0


Why did the `bathrooms` column retain missing values after our imputation?

**Task**: List the unique values of the `bathrooms` column.


```python
df['bathrooms'].unique
```




    <bound method Series.unique of 0       NaN
    1       NaN
    2       NaN
    3       NaN
    4       NaN
             ..
    38272   NaN
    38273   NaN
    38274   NaN
    38275   NaN
    38276   NaN
    Name: bathrooms, Length: 38277, dtype: float64>



The column did not contain a single value (except the `NaN` indicator) to begin with.

## Part 3. Perform One-Hot Encoding

Machine learning algorithms operate on numerical inputs. Therefore, we have to transform text data into some form of numerical representation to prepare our data for the model training phase. Some features that contain text data are categorical. Others are not. For example, we removed all of the features that contained URLs. These features were not categorical, but rather contained what is called unstructured text. However, not all features that contain unstructured text should be removed, as they can contain useful information for our machine learning problem. Unstructured text data is usually handled by Natural Language Processing (NLP) techniques. You will learn more about NLP later in this course. 

However, for features that contain categorical values, one-hot encoding is a common feature engineering technique that transforms them into binary representations. 

We will first choose one feature column to one-hot encode: `host_response_time`. Let's inspect the unique values this feature can have. 


```python
df['host_response_time'].unique()
```




    array(['within a day', 'a few days or more', 'within an hour', nan,
           'within a few hours'], dtype=object)



Note that each entry can contain one of five possible values. 

**Task**: Since one of these values is `NaN`, replace every entry in the column `host_response_time` that contains a `NaN` value with the string 'unavailable'.


```python
df['host_response_time'].fillna('unavailable', inplace=True)
```

Let's inspect the `host_response_time` column to see the new values.


```python
df['host_response_time'].unique()
```




    array(['within a day', 'a few days or more', 'within an hour',
           'unavailable', 'within a few hours'], dtype=object)



**Task**: Use `pd.get_dummies()` to one-hot encode the `host_response_time` column. Save the result to DataFrame `df_host_response_time`. 


```python
df_host_response_time = pd.get_dummies(df['host_response_time'])
df_host_response_time
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
      <th>a few days or more</th>
      <th>unavailable</th>
      <th>within a day</th>
      <th>within a few hours</th>
      <th>within an hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>38272</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38273</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38274</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>38275</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>38276</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>38277 rows × 5 columns</p>
</div>



**Task**: Since the `pd.get_dummies()` function returned a new DataFrame rather than making the changes to the original DataFrame `df`, add the new DataFrame `df_host_response_time` to DataFrame `df`, and delete the original `host_response_time` column from DataFrame `df`.



```python
df = pd.concat([df, df_host_response_time], axis=1)
df = df.drop('host_response_time', axis=1)
```

Let's inspect DataFrame `df` to see the changes that have been made.


```python
df.columns
```




    Index(['id', 'scrape_id', 'last_scraped', 'name', 'description',
           'neighborhood_overview', 'host_id', 'host_name', 'host_since',
           'host_location', 'host_about', 'host_response_rate',
           'host_acceptance_rate', 'host_is_superhost', 'host_neighbourhood',
           'host_listings_count', 'host_total_listings_count',
           'host_verifications', 'host_has_profile_pic', 'host_identity_verified',
           'neighbourhood', 'neighbourhood_cleansed',
           'neighbourhood_group_cleansed', 'latitude', 'longitude',
           'property_type', 'room_type', 'accommodates', 'bathrooms',
           'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price',
           'minimum_nights', 'maximum_nights', 'minimum_minimum_nights',
           'maximum_minimum_nights', 'minimum_maximum_nights',
           'maximum_maximum_nights', 'minimum_nights_avg_ntm',
           'maximum_nights_avg_ntm', 'calendar_updated', 'has_availability',
           'availability_30', 'availability_60', 'availability_90',
           'availability_365', 'calendar_last_scraped', 'number_of_reviews',
           'number_of_reviews_ltm', 'number_of_reviews_l30d', 'first_review',
           'last_review', 'review_scores_rating', 'review_scores_accuracy',
           'review_scores_cleanliness', 'review_scores_checkin',
           'review_scores_communication', 'review_scores_location',
           'review_scores_value', 'license', 'instant_bookable',
           'calculated_host_listings_count',
           'calculated_host_listings_count_entire_homes',
           'calculated_host_listings_count_private_rooms',
           'calculated_host_listings_count_shared_rooms', 'reviews_per_month',
           'label_price', 'host_listings_count_na', 'host_total_listings_count_na',
           'bathrooms_na', 'bedrooms_na', 'beds_na', 'a few days or more',
           'unavailable', 'within a day', 'within a few hours', 'within an hour'],
          dtype='object')



#### One-hot encode additional features

**Task**: Use the code cell below to find columns that contain string values  (the 'object' data type) and inspect the *number* of unique values each column has.


```python
object_columns = df.select_dtypes(include=['object'])
for column in object_columns.columns:
    print(f"{column}: {df[column].nunique()} unique values")
```

    last_scraped: 2 unique values
    name: 36870 unique values
    description: 34133 unique values
    neighborhood_overview: 18616 unique values
    host_name: 9123 unique values
    host_since: 4289 unique values
    host_location: 1747 unique values
    host_about: 14424 unique values
    host_response_rate: 88 unique values
    host_acceptance_rate: 101 unique values
    host_is_superhost: 2 unique values
    host_neighbourhood: 484 unique values
    host_verifications: 526 unique values
    host_has_profile_pic: 2 unique values
    host_identity_verified: 2 unique values
    neighbourhood: 207 unique values
    neighbourhood_cleansed: 222 unique values
    neighbourhood_group_cleansed: 5 unique values
    property_type: 78 unique values
    room_type: 4 unique values
    bathrooms_text: 30 unique values
    amenities: 31740 unique values
    has_availability: 2 unique values
    calendar_last_scraped: 2 unique values
    first_review: 3171 unique values
    last_review: 2560 unique values
    license: 1 unique values
    instant_bookable: 2 unique values


**Task**: Based on your findings, identify features that you think should be transformed using one-hot encoding.

1. Use the code cell below to inspect the unique *values* that each of these features have.


```python
for column in object_columns.columns:
    print(f"{column}: {df[column].unique()}")
```

    last_scraped: ['2021-12-05' '2021-12-04']
    name: ['Skylit Midtown Castle'
     'Whole flr w/private bdrm, bath & kitchen(pls read)' 'BlissArtsSpace!'
     ... 'King Room - Midtown Manhattan' 'King Room - Bryant Park.'
     '★Luxury in the ❤of Bklyn | Fast Wi-Fi | Sleeps 14★']
    description: ['Beautiful, spacious skylit studio in the heart of Midtown, Manhattan. <br /><br />STUNNING SKYLIT STUDIO / 1 BED + SINGLE / FULL BATH / FULL KITCHEN / FIREPLACE / CENTRALLY LOCATED / WiFi + APPLE TV / SHEETS + TOWELS<br /><br /><b>The space</b><br />- Spacious (500+ft²), immaculate and nicely furnished & designed studio.<br />- Tuck yourself into the ultra comfortable bed under the skylight. Fall in love with a myriad of bright lights in the city night sky. <br />- Single-sized bed/convertible floor mattress with luxury bedding (available upon request).<br />- Gorgeous pyramid skylight with amazing diffused natural light, stunning architectural details, soaring high vaulted ceilings, exposed brick, wood burning fireplace, floor seating area with natural zafu cushions, modern style mixed with eclectic art & antique treasures, large full bath, newly renovated kitchen, air conditioning/heat, high speed WiFi Internet, and Apple TV.<br />- Centrally located in the heart of Midtown Manhattan'
     'Enjoy 500 s.f. top floor in 1899 brownstone, w/ wood & ceramic flooring throughout, roomy bdrm, & upgraded kitchen & bathroom.\xa0 This space is unique but one of the few legal AirBnbs with a totally private bedroom, private full bathroom and private eat-in kitchen, SO PLEASE READ "THE SPACE" CAREFULLY.\xa0 It\'s sunny & loaded with everything you need! Your floor, and the common staircase/hallway/entryway are cleaned/sanitized per Airbnb\'s Enhanced Cleaning Protocol.<br /><br /><b>The space</b><br />We host on the entire top floor of our double-duplex brownstone in Clinton Hill on Gates near Classon Avenue - (7 blocks to C train, 5 blocks to G train, minutes to downtown Brooklyn & lower Manhattan).\xa0 It is not an apartment in the traditional sense, it is more of an efficiency set-up and is TOTALLY LEGAL with all short-term rental laws. The top floor for our guests consists of a sizable bedroom, full bath and eat-in kitchen for your exclusive use - you get the amenities of a private apartment '
     "<b>The space</b><br />HELLO EVERYONE AND THANKS FOR VISITING BLISS ART SPACE! <br /><br />Thank you all for your support. I've traveled a lot in the last year few years, to the  U.K. Germany, Italy and France! Loved Paris, Berlin and Calabria! Highly recommend all these places. <br /><br /><br />One room available for rent in a 2 bedroom apt in Bklyn. We share a common space with kitchen. I am an artist(painter, filmmaker) and curator who is working in the film industry while I'm building my art event production businesses.<br /><br />Price above is nightly for one person. Monthly rates available.  Price is $900 per month for one person. Utilities not included, they are about 50 bucks, payable when the bill arrives mid month.<br /> <br />Couples rates are slightly more for monthly and 90$ per night short term. If you are a couple please Iet me know and I’ll give you the monthly rate for that. Room rental is on a temporary basis, perfect from 2- 6 months - no long term requests please! "
     ...
     'Hi,<br /><br />Welcome to our house!!! The room is very large and spacious, can accommodate up to 4 people, fits 3 beds. Large walk-in closet and a private shower room. Private refrigerator also in your room!!<br /><br />Short walk (3-4minutes) to trains and buses. Easy to find free street parking spot. Laundromat is right across the street. <br /><br />Many local groceries stores near by. Many schools are located around the neighborhood, very safe and family friendly.'
     'You can pack a lot into a minute in this central location. The room feels more like a studio, with a private entrance from the hallway.  It’s spacious, giving the residents the most luxurious living experience without sharing any common spaces. It features a king size bed, ensuite bathroom, study desk, 60" smart TV with a streaming device, mini-fridge, microwave, coffee maker, and kettle. This room is a perfect fit for those looking for ultimate privacy, spacious bed, and comfort.'
     "Stunning newly remodeled apartment in Bushwick, steps from the J train (3 min walk) and a 20-minute commute to Manhattan. The apartment sleeps 12-14 and features 6 cozy bedrooms tastefully designed, luxury bathrooms with glass-enclosed walk-in showers, 75 inches 4K smart TV, blazing-fast wi-fi, and two full kitchens with all major appliances. Walking distance to highland park, walk, run, bike! No parties, gatherings, or get-togethers. No visitors beyond the confirmed guests.<br /><br />Stay here!<br /><br /><b>The space</b><br />Amenities:<br />•\tSmart lock access<br />•\tSecure high-speed wireless internet (400 Mbps) <br />•\t75” 4K smart TV with premium cable channels<br />•\tKeurig coffee maker<br />•\tMicrowave<br />•\tIron and ironing board <br />•\tHairdryer<br />•\tFresh towels, sheets, pillows<br />•\tOutdoor seating<br />•\tStay Here!<br /><br /><b>Guest access</b><br />Street parking. It's NYC so expect it to be busy!<br /><br /><b>Other things to note</b><br />Have fun!"]
    neighborhood_overview: ['Centrally located in the heart of Manhattan just a few blocks from all subway connections in the very desirable Midtown location a few minutes walk to Times Square, the Theater District, Bryant Park and Herald Square.'
     'Just the right mix of urban center and local neighborhood; close to all but enough quiet for a calming walk. 15 to 45 minutes to most parts of Manhattan; 10 to 30 minutes to most Brooklyn points of interest; 45 minutes to 60 minutes to historic Coney Island.'
     nan ...
     'Groceries, delis, bakeries, cafes, restaurants, gourmet/vegan/vegetarian shops, souvenir, dollar stores, bars galore! Bushwick IS the new East Village! This is the place to be with everything you need within a 5 minute walk. CitiBike rentals 1 blk away, in 2 different directions. Main and express (work commute hours) trains 3 min walk from apt door... Myrtle/Bway JMZ trains.'
     'Soho , downtown, close to time square, ltaly town  Chinatown, Koreatown'
     'Spanish Harlem is one of the upcoming trendy neighbourhoods in manhattan. New bars and restaurants are opening all the time! With quick access to a downtown train and only minutes from central park its no wonder people from all over the city are relocating here!']
    host_name: ['Jennifer' 'LisaRoxanne' 'Garon' ... 'Avigail' 'Bridgett' 'Maxinne']
    host_since: ['2008-09-09' '2008-12-07' '2009-02-03' ... '2021-11-30' '2012-12-31'
     '2021-11-06']
    host_location: ['New York, New York, United States' 'Brooklyn, New York, United States'
     'Berkeley, California, United States' ... 'Padua, Veneto, Italy'
     'Neffsville, Pennsylvania, United States'
     'Hartsdale, New York, United States']
    host_about: ["A New Yorker since 2000! My passion is creating beautiful, unique spaces where unforgettable memories are made. It's my pleasure to host people from around the world and meet new faces. Welcome travelers! \r\n\r\nI am a Sound Therapy Practitioner and Kundalini Yoga & Meditation teacher. I work with energy and sound for relaxation and healing, using Symphonic gong, singing bowls, tuning forks, drums, voice and other instruments."
     'Laid-back Native New Yorker (formerly bi-coastal) and AirBnb host of over 6 years and over 400 stays!  Besides being a long-time and attentive AirBnb host, I am an actor, attorney, professor and group fitness instructor.'
     " I am an artist(painter, filmmaker) and curator who is working in the film industry while I'm building my business.\r\n\r\nI am extremely easy going and would like that you are the laid back\r\nand enjoy life kind of person. I also ask that you are open, honest\r\nand easy to communicate with as this is how I like to live my life.And of course creative people are very welcome!\r\n"
     ...
     "I am a responsible teacher living in the great city of New York! I love to travel and see the world and have traveled to Europe 5 times now and miss it when I'm not there. I love food, culture, great conversation, great wine, and great stories to bring home. I can't wait to get to South America someday soon, Greece and return to my favorite place... Italy. I look forward to staying at a great place such as yours and creating new and amazing memories!!!\r\n\r\nCheers! \r\n\r\nChris\r\n\r\nEat, laugh, live, play, breath, dream... love.\r\n\r\n"
     'Hello! We are Yan and Co! My wife Co is a wine expert from China and I am a musician originally from Puerto Rico. We have been married for about a year and have a spare room at a beautiful apartment with views of the Hudson River.  '
     'Y|S- We provide travelers with an awesome experience by providing top-notch accommodations in modern upscale homes across the US.\n']
    host_response_rate: ['80%' '9%' '100%' nan '75%' '0%' '90%' '77%' '98%' '50%' '93%' '67%'
     '38%' '92%' '95%' '89%' '97%' '10%' '60%' '20%' '88%' '86%' '57%' '70%'
     '17%' '87%' '33%' '83%' '96%' '91%' '23%' '40%' '30%' '99%' '73%' '94%'
     '82%' '22%' '11%' '29%' '85%' '71%' '79%' '61%' '25%' '68%' '78%' '76%'
     '81%' '65%' '56%' '43%' '74%' '13%' '35%' '63%' '14%' '5%' '55%' '26%'
     '32%' '72%' '64%' '53%' '31%' '27%' '6%' '46%' '52%' '44%' '84%' '36%'
     '18%' '62%' '59%' '58%' '47%' '69%' '41%' '42%' '7%' '24%' '8%' '39%'
     '4%' '21%' '37%' '45%' '54%']
    host_acceptance_rate: ['17%' '69%' '100%' '25%' nan '0%' '99%' '29%' '61%' '98%' '54%' '84%'
     '75%' '94%' '82%' '83%' '92%' '80%' '9%' '43%' '71%' '95%' '33%' '60%'
     '14%' '50%' '90%' '81%' '77%' '59%' '15%' '56%' '97%' '58%' '3%' '93%'
     '91%' '36%' '62%' '41%' '20%' '19%' '68%' '88%' '78%' '11%' '85%' '63%'
     '44%' '42%' '38%' '40%' '48%' '64%' '87%' '32%' '70%' '89%' '76%' '79%'
     '86%' '35%' '55%' '57%' '74%' '67%' '96%' '30%' '22%' '72%' '65%' '8%'
     '5%' '53%' '73%' '46%' '47%' '52%' '21%' '26%' '18%' '28%' '24%' '45%'
     '66%' '6%' '51%' '27%' '39%' '13%' '23%' '34%' '10%' '37%' '49%' '31%'
     '7%' '2%' '4%' '12%' '1%' '16%']
    host_is_superhost: ['f' 't' nan]
    host_neighbourhood: ['Midtown' 'Clinton Hill' 'Bedford-Stuyvesant' 'Greenwood Heights'
     "Hell's Kitchen" 'Upper West Side' 'Park Slope' 'Williamsburg'
     'East Harlem' 'Fort Greene' 'East Village' 'Harlem' 'Flatbush'
     'Alphabet City' 'Long Island City' 'Jamaica' 'Stuyvesant Heights'
     'East Williamsburg' 'Greenpoint' 'Soho' 'Chelsea' 'Upper East Side'
     'Prospect Heights' 'Washington Heights' 'Meatpacking District' 'Kips Bay'
     'Hamilton Heights' 'Bushwick' 'Carroll Gardens' 'West Village'
     'Lefferts Garden' 'Flatlands' 'Boerum Hill' 'Sunnyside' 'Lower East Side'
     'St. George' 'Tribeca' 'Highbridge' 'Ridgewood' nan 'Morningside Heights'
     'Ditmars Steinway' 'Ditmars / Steinway' 'Cobble Hill' 'Flatiron District'
     'Windsor Terrace' 'Chinatown' 'Greenwich Village' 'Midtown East'
     'Soundview' 'Crown Heights' 'Gowanus' 'Astoria' 'Kingsbridge Heights'
     'South Williamsburg' 'Brooklyn Heights' 'Downtown Brooklyn'
     'Forest Hills' 'Murray Hill' 'North Williamsburg' 'University Heights'
     'Gravesend' 'Baychester' 'East New York' 'Theatre District' 'Yorkville'
     'Sheepshead Bay' 'Bensonhurst' 'Rosebank' 'Richmond Hill' 'Gramercy Park'
     'Fordham' 'South Beach' 'Financial District' 'Brooklyn Navy Yard'
     'Times Square/Theatre District' 'Rego Park' 'Kensington' 'Little Italy'
     'Elmhurst' 'Stapleton' 'Flushing' 'Bay Ridge' 'Sunset Park' 'Maspeth'
     'Brighton Beach' 'Jackson Heights' 'Longwood' 'Inwood' 'Nolita'
     'Battery Park City' 'Bayside' 'Columbia Street Waterfront' 'Times Square'
     'New Springville' 'Red Hook' 'Cambridge' 'Civic Center' 'Grymes Hill'
     'Tottenville' 'Tompkinsville' 'Noho' 'DUMBO' 'Mariners Harbor' 'Woodside'
     'Concord' 'Melrose' 'College Point' 'Mount Eden' 'Union Square'
     'City Island' 'Pacific Beach' 'Canarsie' 'Port Morris' 'East Flatbush'
     'The Rockaways' 'Mott Haven' 'Ipanema' 'Midwood' 'Brownsville'
     'Southside' 'Williamsbridge' 'Woodhaven' 'Parkchester' 'Mid-Wilshire'
     'Bronxdale' 'South Street Seaport' 'Riverdale' 'Hudson Square'
     'Whitestone' 'Ozone Park' 'Corona' 'Manhattan' 'Bedford Park' 'Norwood'
     'Yehuda Hamaccabi' 'Concourse' 'Shoreditch' 'Zona Romántica'
     'Middle Village' 'Tremont' 'Topanga' 'Concourse Village' 'Wakefield'
     'Prospect Lefferts Gardens' 'Lighthouse HIll' 'East Bronx' 'Laguna Woods'
     'West Brighton' 'Utopia' 'South Ozone Park' 'Córcega' 'Kew Garden Hills'
     'Hillcrest' 'Santa Monica' 'The Bronx' 'Hollywood' 'Fresh Meadows'
     'Borough Park' 'Eastchester' 'Morris Park' 'Crotona' 'Woodstock'
     'Far Rockaway' 'East Elmhurst' 'Brooklyn' 'Manhattan Beach' 'Morrisania'
     'Marble Hill' 'Kingsbridge' 'Dongan Hills' 'Dumbo' 'Belmont'
     'Roosevelt Island' 'Castleton Corners' 'Hunts Point' 'Prenzlauer Berg'
     'South Side' 'Miami Beach' 'Howard Beach' 'Great Kills' 'Port Richmond'
     'Grasmere' 'Puerto Madero' 'Central Ward' 'Paddington' 'Belltown'
     'Morris Heights' 'Wicker Park' 'Shaw' 'New Brighton' 'Kailua/Kona'
     'Kauaʻi' 'Dalston' 'Spuyten Duyvil' 'Vinegar Hill'
     'Ludwigsvorstadt - Isarvorstadt' 'Bergen Beach' 'New Dorp' 'Downtown'
     'Coney Island' 'Clifton' 'Country Club' 'Castle Hill ' 'Zona 8'
     'Barrio Santa Lucía' 'Stapleton Heights' 'Sanlitun'
     'Southeast Washington' 'Bristol/Warner' 'Capitol View/Capitol View Manor'
     'Arlington Park' 'Bath Beach' 'San Giovanni' 'Central City' 'Glendale'
     'Midland Beach' 'Oakwood' 'Little Caribbean' 'Kilmainham' 'Pelham Bay'
     'Waikiki' 'Rockaway Beach' 'Annadale' 'Russian Hill'
     'South Hill/Rathnelly' 'Edenwald' 'Barrio Norte' 'Park Versailles'
     'Richmond' 'Josefov' 'Allerton' 'Todt Hill' 'Throgs Neck'
     'Meiers Corners' 'Marine Parade' 'South Lake Tahoe' 'II Arrondissement'
     'St Kilda East' 'Rittenhouse Square' 'Westchester Village' 'Claremont'
     'Westlake Hills' 'Merkaz HaIr' 'Dyker Heights' 'East Lake Terrace'
     'Upper Boggy Creek' 'University City' 'Mid-City' 'Stoke Newington'
     'North Bondi' 'College Park, MD' 'Condesa' 'East Downtown' 'Van Nest'
     'Friedrichshain' 'Greenridge' "Bull's Head" 'Collingwood/Fitzroy'
     'West Town/Noble Square' 'Châtelet - Les Halles - Beaubourg' 'Queens'
     'Hollywood South Central Beach' 'Randall Manor' 'Cypress Hills'
     'Entertainment District' 'Lev HaIr' 'Grant City' 'Dupont Circle'
     'Woollahra' 'Central Area' 'Dapuqiao' 'Peter Howell' 'Elm Park'
     'Rockaway Park' 'Arverne' 'Pelham Gardens' 'Pacific Heights' 'Montmartre'
     'Fort Wadsworth' 'New Dorp Beach' 'Le Port' 'Abbotsford' 'Park Cities'
     'Covent Garden' 'Retiro' 'Sunny Isles Beach' 'Bugis/Kampong Glam'
     'Silver Lake' 'Central LA' 'Ocean Hill' 'Beverly Hills' 'Palermo'
     'Venice' 'City of London' 'Garden Hills/Buckhead Village/Peachtree Park'
     'Browncroft' 'Arden Heights' 'South Richmond Hill' 'Homecrest'
     'North Corona' 'Camden Town' "Prince's Bay" 'South Lake Union'
     'Eltingville' 'Mid Island' 'Pyrmont' 'Columbia Heights' 'Bedford Pine'
     'Loop' 'Chicago Loop' 'Hackney' 'Setagaya District' 'Cerro Gordo'
     'Washington Square West' 'Westerleigh' 'Sunnyvale' 'Westwood' 'Arrochar'
     'Back Bay' 'Northstar Resort' 'West Bronx' 'Springfield Gardens'
     'Clason Point' 'Rossville' 'Graniteville' 'Copacabana' 'Bang Na'
     'Central West End' 'Notting Hill' 'Carlton' 'South First' 'Capitol Hill'
     'Lower Haight' 'West Farms' 'Irvine' 'Frunzensky' 'North Berkeley'
     'Northern Liberties' 'Lindenwood' 'Old Seminole Heights' 'Manchester'
     'Edgewood' 'Del Mar Heights' 'The Heights'
     'Lefferts Manor Historic District' 'Queens Village' 'Napili/Honokowai'
     'Recoleta' 'Laconia' 'São Paulo' 'Fordham Heights' 'Foxhurst' 'Ingleside'
     'Broadway Triangle' 'La Jolla' 'Dutch Kills' 'National Harbor'
     'Tzafon Yashan' 'Oakdale South' 'Marine Park' 'Fairview' 'Hollis'
     'Downtown Las Vegas' 'Shore Acres' 'Tulum Centro' 'Cambria Heights'
     'Cascade Green/Heritage Valley/Old Fairburn Village' 'Fieldston' 'Mantua'
     'West Side' 'Mill Basin' 'Smith Bay' 'Westside' 'Northwest Yonkers'
     'Brickell' 'Little Haiti' 'Mapleton' 'Spring Creek' 'South Slope'
     'Downtown Los Angeles' 'Getty Square' 'North Fork' 'Cherry Creek'
     'Bromley-by-Bow' 'North Philadelphia' 'Sugar House' 'The Bluffs'
     'Sandy Hill' 'Fordham Manor' 'Cannonborough Elliotborough'
     'Hamilton District' 'Fishtown' 'North Riverdale' 'Dietz'
     'Woodland Hills/Warner Center' 'Briarwood' 'French Quarter' 'Watertown'
     'Long Beach' 'Batignolles' 'Madison' 'Homestead' 'St. Albans'
     'Chapmantown' 'Downtown Jersey City' 'Central Business District'
     'Libertad' 'Springdale' 'Normandy Isles' 'Ormond Shores'
     'Pennington Bend' 'Hunter Hills' 'Pennsport' 'Central District'
     'Columbia Street Waterfront District' 'Lake Mohawk' 'Riverside'
     'Travis - Chelsea' 'Bridge Plaza' 'Coral Ridge' 'Shapira'
     'Jeffersonville' 'Laurel Canyon' 'Bel-Air' 'Downtown Austin' 'Table Rock'
     'Glen Park' 'Journal Square' 'Orange' 'Springfield/Belmont' 'West Ashley'
     'Marina Del Rey' 'La Concordia' 'Gerritsen Beach' 'Milwood' 'Park Hill'
     'Carroll Park' 'City of Marco' 'Central Oklahoma City' 'Canadensis'
     'Carnelian Bay' 'Townsite' 'South Scottsdale' 'New Cross' 'SoNo'
     'Laguna Beach On The Gulf Of Mexico' 'Rosedale' 'Ville-Marie' 'Woodrow'
     'Ormewood Park' '3G' 'Palms' 'Queen Village' 'Mid-Beach'
     'Westchester Square' 'Constable Hook' 'Seagate' 'Manchester Center'
     'Santa Cruz' 'Angel Park' 'Douglaston' 'Lakeville' 'Southeast Torrance'
     'Padre Island' 'Lenox Park' 'Fairmount' 'Fonatur' 'Claverack'
     'North Side' 'Old East Dallas' 'Buckhead' 'República' 'Kew Gardens'
     'Perimeter Center' 'Northridge' 'Southeast Yonkers' 'Westshore'
     'Palermo Hollywood' 'Newport' 'Queens Park' 'Kendall-Whittier'
     'Oliver Atlantic City' 'South Montebello' 'NOMO']
    host_verifications: ["['email', 'phone', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'reviews', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews']"
     "['email', 'phone', 'facebook', 'reviews']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'reviews', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id', 'work_email']"
     "['email', 'phone', 'reviews', 'jumio', 'government_id']"
     "['email', 'phone', 'reviews', 'kba', 'work_email']"
     "['email', 'phone', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'reviews', 'kba']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'offline_government_id', 'selfie', 'government_id']"
     "['phone']" "['email', 'phone', 'facebook', 'reviews', 'kba']"
     "['email', 'phone', 'reviews', 'offline_government_id', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id']"
     "['email', 'phone', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['email', 'phone', 'reviews', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'reviews', 'jumio', 'government_id', 'work_email']"
     "['phone', 'reviews']"
     "['email', 'phone', 'reviews', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'facebook', 'reviews', 'kba', 'work_email']"
     "['email', 'phone', 'facebook', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'work_email']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone']" "['phone', 'reviews', 'kba']"
     "['email', 'phone', 'facebook', 'reviews', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'offline_government_id', 'government_id']"
     "['phone', 'facebook', 'reviews', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'reviews', 'jumio', 'kba', 'government_id']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'kba', 'government_id']"
     "['email', 'phone', 'google', 'reviews', 'offline_government_id', 'kba', 'government_id']"
     "['phone', 'facebook', 'reviews', 'jumio', 'government_id']"
     "['email', 'phone', 'reviews', 'work_email']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'kba']"
     "['email', 'phone', 'facebook', 'reviews', 'sent_id']"
     "['email', 'phone', 'facebook', 'reviews', 'offline_government_id', 'selfie', 'government_id']"
     "['email', 'phone', 'reviews', 'jumio', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'reviews', 'manual_offline', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['email', 'phone', 'google', 'reviews', 'kba']"
     "['phone', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id']"
     "['phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id', 'work_email']"
     "['email', 'phone', 'offline_government_id', 'selfie', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'manual_offline', 'jumio', 'government_id']"
     "['email', 'phone', 'reviews', 'jumio', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['phone', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id']"
     "['email', 'phone', 'google', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id']"
     "['email', 'phone', 'kba']" "['email', 'phone', 'facebook']"
     "['phone', 'facebook', 'reviews', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'work_email']"
     "['email', 'phone', 'google', 'reviews', 'kba', 'work_email']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'jumio', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'offline_government_id', 'selfie', 'government_id', 'work_email']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id']"
     'None'
     "['email', 'phone', 'facebook', 'manual_offline', 'jumio', 'government_id']"
     "['email', 'phone', 'reviews', 'sent_id']"
     "['phone', 'facebook', 'reviews']"
     "['email', 'phone', 'jumio', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['phone', 'facebook', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio']" "['reviews', 'kba']"
     "['reviews', 'jumio', 'offline_government_id', 'kba', 'government_id']"
     "['phone', 'reviews', 'offline_government_id', 'kba', 'government_id']"
     "['reviews', 'jumio', 'government_id']"
     "['email', 'phone', 'manual_online', 'reviews', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'sent_id', 'government_id']"
     "['email', 'phone', 'reviews', 'jumio', 'sent_id', 'government_id']"
     "['email', 'phone', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'jumio', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'offline_government_id', 'government_id', 'work_email']"
     "['email', 'phone', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline']"
     "['email', 'phone', 'reviews', 'manual_offline', 'jumio', 'sent_id']"
     "['email', 'phone', 'manual_online', 'facebook', 'reviews', 'manual_offline']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'offline_government_id', 'government_id']"
     "['phone', 'reviews', 'jumio', 'offline_government_id', 'government_id']"
     "['phone', 'facebook', 'reviews', 'kba']"
     "['email', 'phone', 'work_email']"
     "['phone', 'reviews', 'offline_government_id', 'selfie', 'government_id']"
     "['email', 'phone', 'reviews', 'manual_offline', 'jumio', 'government_id']"
     "['email', 'phone', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'jumio']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'kba', 'work_email']"
     "['email', 'phone', 'reviews', 'manual_offline']"
     "['email', 'phone', 'reviews', 'manual_offline', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'manual_online', 'facebook', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'selfie', 'government_id']"
     "['email', 'phone', 'manual_online', 'reviews', 'jumio', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'reviews', 'jumio', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'reviews', 'offline_government_id', 'government_id', 'work_email']"
     "['email', 'phone', 'manual_online', 'facebook', 'reviews', 'manual_offline', 'sent_id', 'kba']"
     "['email', 'phone', 'facebook', 'reviews', 'manual_offline', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['phone', 'kba']"
     "['phone', 'reviews', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'reviews', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'reviews', 'kba', 'selfie', 'work_email']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'government_id', 'work_email']"
     "['phone', 'reviews', 'manual_offline', 'jumio', 'government_id']"
     "['phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'kba']"
     "['email', 'phone', 'facebook', 'kba']"
     "['email', 'phone', 'facebook', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'reviews', 'offline_government_id']"
     "['phone', 'facebook']"
     "['email', 'phone', 'google', 'reviews', 'manual_offline', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'google', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'work_email']"
     "['phone', 'facebook', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'government_id']"
     "['phone', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['phone', 'reviews', 'jumio', 'government_id']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'google', 'jumio', 'government_id']"
     "['email', 'phone', 'jumio', 'kba', 'government_id']"
     "['email', 'phone', 'manual_online', 'facebook', 'reviews', 'manual_offline', 'work_email']"
     "['email', 'phone', 'facebook', 'reviews', 'offline_government_id', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['phone', 'facebook', 'reviews', 'jumio', 'kba', 'government_id']"
     "['email', 'phone', 'jumio', 'offline_government_id', 'selfie', 'government_id']"
     "['phone', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['phone', 'jumio', 'offline_government_id', 'selfie', 'government_id']"
     "['phone', 'reviews', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id']"
     "['email', 'phone', 'facebook', 'offline_government_id', 'selfie', 'government_id']"
     "['email', 'phone', 'reviews', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'facebook', 'reviews', 'manual_offline']"
     "['email', 'phone', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'jumio', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'manual_online', 'reviews', 'kba']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'work_email']"
     "['email', 'phone', 'manual_online', 'reviews', 'jumio', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['phone', 'reviews', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'jumio', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'manual_offline', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'kba', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'jumio', 'selfie', 'government_id']"
     "['email', 'phone', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'google', 'reviews', 'offline_government_id', 'selfie', 'government_id']"
     "['phone', 'reviews', 'kba', 'selfie']"
     "['email', 'phone', 'reviews', 'weibo', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id', 'work_email']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'manual_offline', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'google', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id']"
     "['email', 'phone', 'kba', 'work_email']"
     "['email', 'phone', 'manual_online', 'facebook', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'kba', 'work_email']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'work_email']"
     "['email', 'phone', 'manual_online', 'manual_offline', 'jumio', 'government_id']"
     "['email', 'phone', 'manual_online', 'facebook', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'google', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['google', 'reviews', 'jumio', 'government_id']"
     "['email', 'phone', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['phone', 'jumio', 'government_id']"
     "['email', 'offline_government_id', 'selfie', 'government_id']"
     "['email', 'phone', 'facebook', 'offline_government_id', 'kba', 'selfie', 'government_id', 'work_email']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'offline_government_id', 'kba', 'government_id']"
     "['phone', 'offline_government_id', 'selfie', 'government_id']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['phone', 'reviews', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'manual_offline', 'jumio', 'government_id']"
     "['email', 'phone', 'reviews', 'kba', 'selfie', 'work_email']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'kba', 'government_id']"
     "['phone', 'reviews', 'kba', 'work_email']"
     "['phone', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'reviews', 'kba', 'identity_manual']"
     "['email', 'phone', 'google', 'reviews', 'offline_government_id', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'offline_government_id', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['phone', 'reviews', 'jumio', 'kba', 'government_id']"
     "['reviews', 'kba', 'work_email']" "['email', 'reviews', 'kba']"
     "['email', 'phone', 'facebook', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'phone', 'reviews', 'offline_government_id', 'selfie', 'government_id', 'work_email']"
     "['email', 'phone', 'reviews', 'manual_offline', 'jumio']" "['email']"
     "['email', 'phone', 'facebook', 'google', 'jumio', 'offline_government_id', 'government_id']"
     "['email', 'kba']"
     "['email', 'phone', 'offline_government_id', 'kba', 'selfie', 'government_id', 'work_email']"
     "['email', 'reviews', 'jumio', 'government_id']"
     "['email', 'phone', 'facebook', 'google', 'jumio', 'government_id']"
     "['email', 'phone', 'manual_offline', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'jumio', 'work_email']"
     "['email', 'phone', 'google', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'jumio', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'sent_id', 'work_email']"
     "['email', 'phone', 'offline_government_id', 'selfie', 'government_id', 'work_email']"
     "['phone', 'jumio', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'work_email']"
     "['email', 'reviews', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'kba', 'work_email']"
     "['email', 'phone', 'manual_online', 'facebook', 'reviews', 'manual_offline', 'kba', 'work_email']"
     "['phone', 'reviews', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['phone', 'facebook', 'google', 'jumio', 'government_id']"
     "['email', 'phone', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'reviews', 'jumio', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'facebook', 'work_email']"
     "['phone', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'manual_online', 'facebook', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['phone', 'reviews', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'reviews']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'work_email']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'government_id', 'identity_manual']"
     "['email', 'facebook', 'reviews', 'jumio', 'government_id']"
     "['email', 'phone', 'google', 'kba']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'reviews', 'weibo', 'jumio', 'government_id']"
     "['email', 'phone', 'jumio', 'offline_government_id', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'offline_government_id', 'government_id', 'work_email']"
     "['phone', 'facebook', 'reviews', 'kba', 'work_email']"
     "['email', 'phone', 'reviews', 'manual_offline', 'work_email']"
     "['phone', 'facebook', 'reviews', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'manual_offline', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['phone', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id', 'work_email']"
     "['phone', 'reviews', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['phone', 'reviews', 'weibo', 'jumio', 'government_id']"
     "['email', 'phone', 'jumio', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'phone', 'facebook', 'google', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id']"
     "['reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'offline_government_id', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'google', 'reviews', 'manual_offline', 'jumio', 'government_id']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'identity_manual']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'work_email']"
     "['email', 'phone', 'google', 'jumio', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'jumio']"
     "['email', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'manual_online', 'manual_offline', 'kba']"
     "['email', 'phone', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['reviews', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['email', 'phone', 'offline_government_id', 'kba', 'selfie', 'government_id']"
     "['phone', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'reviews', 'kba', 'work_email']"
     "['email', 'phone', 'facebook', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'manual_online', 'jumio', 'government_id']"
     "['phone', 'facebook', 'reviews', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'google', 'reviews']"
     "['email', 'phone', 'manual_online', 'manual_offline', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'offline_government_id', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'sent_id']"
     "['email', 'phone', 'manual_online', 'facebook', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'facebook', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['jumio', 'offline_government_id', 'selfie', 'government_id']"
     "['phone', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'google', 'reviews', 'offline_government_id', 'government_id']"
     "['email', 'google', 'reviews', 'kba']"
     "['email', 'phone', 'facebook', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['phone', 'google', 'reviews', 'kba']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['phone', 'reviews', 'manual_offline']"
     "['email', 'phone', 'reviews', 'weibo', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'google', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id']"
     "['email', 'phone', 'facebook', 'offline_government_id', 'government_id', 'work_email']"
     "['email', 'google', 'kba']" "['email', 'phone', 'reviews', 'selfie']"
     "['email', 'phone', 'facebook', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'google', 'work_email']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'zhima_selfie']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'work_email']"
     "['email', 'phone', 'reviews', 'identity_manual']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['google', 'reviews', 'kba']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'offline_government_id', 'kba', 'government_id']"
     "['phone', 'google', 'reviews', 'jumio', 'government_id']"
     "['email', 'phone', 'google', 'reviews', 'work_email']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'google', 'reviews', 'jumio', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'sesame', 'sesame_offline']"
     '[]'
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'offline_government_id', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'google', 'reviews']"
     "['email', 'phone', 'sesame', 'sesame_offline']"
     "['phone', 'reviews', 'jumio']"
     "['email', 'phone', 'reviews', 'kba', 'selfie']"
     "['phone', 'facebook', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id']"
     "['google', 'kba']"
     "['email', 'phone', 'facebook', 'jumio', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'google', 'kba']"
     "['email', 'phone', 'manual_online', 'reviews', 'jumio', 'kba', 'government_id', 'identity_manual', 'work_email']"
     "['phone', 'reviews', 'manual_offline', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['reviews']"
     "['email', 'phone', 'facebook', 'offline_government_id', 'selfie', 'government_id', 'work_email']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'sent_id', 'selfie', 'government_id']"
     "['phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['phone', 'facebook', 'reviews', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'manual_offline', 'kba']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id']"
     "['email', 'phone', 'manual_online', 'reviews', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'reviews', 'selfie', 'identity_manual']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'offline_government_id', 'selfie', 'government_id']"
     "['phone', 'facebook', 'reviews', 'offline_government_id', 'government_id', 'work_email']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'google', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'google', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['email', 'phone', 'google']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'jumio', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'manual_online', 'reviews', 'manual_offline']"
     "['email', 'phone', 'facebook', 'reviews', 'weibo', 'jumio', 'government_id', 'work_email']"
     "['jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'manual_online', 'facebook', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['phone', 'facebook', 'google', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'work_email']" "['phone', 'work_email']"
     "['email', 'phone', 'reviews', 'manual_offline', 'kba']"
     "['email', 'phone', 'facebook', 'reviews', 'weibo', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['phone', 'google']"
     "['email', 'phone', 'reviews', 'jumio', 'selfie', 'identity_manual', 'work_email']"
     "['email', 'phone', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'reviews', 'jumio', 'offline_government_id', 'government_id']"
     "['phone', 'facebook', 'jumio', 'offline_government_id', 'government_id']"
     "['phone', 'facebook', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'offline_government_id', 'government_id', 'identity_manual']"
     "['email', 'phone', 'google', 'reviews', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'jumio', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'offline_government_id', 'government_id', 'identity_manual']"
     "['phone', 'facebook', 'google', 'reviews', 'work_email']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'selfie']"
     "['email', 'phone', 'google', 'reviews', 'identity_manual']"
     "['email', 'phone', 'facebook', 'jumio', 'offline_government_id', 'selfie', 'government_id']"
     "['phone', 'google', 'reviews', 'jumio', 'offline_government_id', 'government_id']"
     "['phone', 'facebook', 'reviews', 'jumio', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'manual_online', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'google', 'offline_government_id', 'selfie', 'government_id']"
     "['email', 'kba', 'work_email']"
     "['email', 'phone', 'jumio', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'google', 'reviews', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'government_id', 'identity_manual']"
     "['email', 'phone', 'google', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'facebook', 'jumio', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['phone', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'government_id', 'sesame', 'sesame_offline']"
     "['email', 'facebook', 'reviews', 'kba']"
     "['email', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'zhima_selfie']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'identity_manual']"
     "['email', 'phone', 'google', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['phone', 'facebook', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'reviews', 'sesame', 'sesame_offline']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'work_email']"
     "['phone', 'facebook', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'google', 'jumio', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'facebook', 'google', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'reviews', 'weibo']"
     "['offline_government_id', 'selfie', 'government_id']"
     "['email', 'phone', 'google', 'jumio', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'government_id', 'identity_manual']"
     "['email', 'phone', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'weibo', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'google', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['phone', 'google', 'offline_government_id', 'government_id']"
     "['kba', 'work_email']"
     "['email', 'phone', 'reviews', 'weibo', 'sesame', 'sesame_offline']"
     "['email', 'phone', 'facebook', 'reviews', 'identity_manual']"
     "['email', 'phone', 'manual_online', 'facebook', 'reviews', 'manual_offline', 'offline_government_id', 'sent_id', 'selfie', 'government_id', 'identity_manual']"
     "['phone', 'manual_online', 'facebook', 'reviews', 'manual_offline', 'jumio', 'government_id']"
     "['email', 'reviews', 'kba', 'selfie', 'work_email']"
     "['phone', 'facebook', 'reviews', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['email', 'phone', 'reviews', 'offline_government_id', 'sent_id', 'selfie', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'kba']"
     "['offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id']"
     "['email', 'phone', 'google', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'facebook', 'reviews', 'weibo', 'jumio', 'selfie', 'government_id', 'identity_manual']"
     "['phone', 'zhima_selfie']"
     "['email', 'phone', 'reviews', 'offline_government_id']"
     "['email', 'phone', 'google', 'offline_government_id', 'government_id', 'work_email']"
     "['email', 'phone', 'reviews', 'manual_offline', 'jumio', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'reviews', 'zhima_selfie']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'jumio', 'selfie', 'government_id']"
     "['email', 'phone', 'google', 'reviews', 'offline_government_id', 'selfie', 'government_id', 'work_email']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'government_id', 'work_email']"
     "['phone', 'reviews', 'jumio', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['phone', 'identity_manual']"
     "['phone', 'google', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'reviews', 'jumio', 'sent_id', 'kba', 'government_id']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'sent_id', 'kba']"
     "['email', 'phone', 'offline_government_id']"
     "['email', 'phone', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['phone', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['phone', 'reviews', 'jumio', 'offline_government_id', 'kba', 'government_id', 'work_email']"
     "['phone', 'facebook', 'google', 'jumio', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'reviews', 'jumio', 'government_id', 'sesame', 'sesame_offline', 'work_email']"
     "['email', 'phone', 'facebook', 'google', 'offline_government_id', 'government_id']"
     "['phone', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['reviews', 'jumio', 'offline_government_id', 'government_id']"
     "['phone', 'weibo', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'google']"
     "['email', 'phone', 'google', 'offline_government_id', 'kba', 'selfie', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'manual_offline', 'jumio', 'kba', 'government_id']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'selfie', 'government_id']"
     "['email', 'phone', 'google', 'offline_government_id', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'offline_government_id', 'kba', 'selfie', 'government_id']"
     "['email', 'phone', 'google', 'reviews', 'manual_offline', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['phone', 'offline_government_id', 'government_id', 'work_email']"
     "['phone', 'offline_government_id', 'selfie', 'government_id', 'work_email']"
     "['phone', 'selfie']"
     "['email', 'phone', 'facebook', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['phone', 'facebook', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'weibo', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'offline_government_id', 'government_id']"
     "['email', 'phone', 'manual_online', 'facebook', 'reviews', 'manual_offline', 'jumio', 'government_id']"
     "['reviews', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['jumio', 'government_id']"
     "['email', 'phone', 'facebook', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'google', 'kba', 'work_email']"
     "['email', 'phone', 'jumio', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'offline_government_id', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'facebook', 'google', 'offline_government_id', 'government_id', 'work_email']"
     "['phone', 'google', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'reviews', 'work_email']"
     "['phone', 'facebook', 'reviews', 'manual_offline', 'jumio', 'government_id']"
     "['email', 'phone', 'google', 'reviews', 'offline_government_id']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'sesame', 'sesame_offline', 'work_email']"
     "['email', 'phone', 'facebook', 'reviews', 'jumio', 'kba', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['phone', 'reviews', 'jumio', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'phone', 'google', 'reviews', 'weibo', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'jumio', 'offline_government_id', 'government_id', 'identity_manual']"
     "['phone', 'facebook', 'work_email']"
     "['phone', 'facebook', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['phone', 'reviews', 'offline_government_id', 'kba', 'government_id', 'work_email']"
     "['email', 'phone', 'selfie', 'identity_manual']"
     "['phone', 'manual_online', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'sesame', 'sesame_offline', 'work_email']"
     "['email', 'phone', 'reviews', 'manual_offline', 'jumio', 'work_email']"
     "['email', 'google', 'jumio', 'offline_government_id', 'government_id', 'identity_manual']"
     "['email', 'phone', 'offline_government_id', 'government_id', 'identity_manual']"
     "['email', 'phone', 'google', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'facebook', 'reviews', 'offline_government_id', 'sent_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'google', 'jumio', 'government_id', 'work_email']"
     "['email', 'phone', 'manual_online', 'reviews', 'manual_offline', 'jumio', 'kba', 'government_id']"
     "['email', 'phone', 'manual_online', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'government_id', 'identity_manual', 'work_email']"
     "['email', 'phone', 'identity_manual', 'work_email']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'offline_government_id', 'kba', 'selfie', 'government_id', 'identity_manual']"
     "['phone', 'facebook', 'reviews', 'manual_offline', 'offline_government_id', 'selfie', 'government_id', 'identity_manual']"
     "['email', 'identity_manual']"
     "['email', 'phone', 'manual_online', 'reviews', 'jumio', 'kba', 'government_id']"
     "['email', 'phone', 'google', 'jumio', 'offline_government_id']"
     "['email', 'phone', 'facebook', 'google', 'offline_government_id', 'kba', 'government_id', 'work_email']"
     "['phone', 'facebook', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'phone', 'facebook', 'identity_manual']"
     "['email', 'phone', 'facebook', 'weibo']"
     "['email', 'phone', 'reviews', 'jumio', 'identity_manual']"
     "['phone', 'google', 'reviews', 'jumio', 'government_id', 'work_email']"
     "['phone', 'jumio', 'offline_government_id', 'government_id', 'identity_manual']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'kba', 'government_id', 'identity_manual']"
     "['email', 'phone', 'google', 'jumio', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'google', 'reviews', 'jumio', 'kba', 'government_id']"
     "['phone', 'offline_government_id', 'government_id', 'identity_manual']"
     "['email', 'phone', 'facebook', 'reviews', 'kba', 'photographer']"
     "['email', 'phone', 'reviews', 'manual_offline', 'jumio', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'phone', 'google', 'offline_government_id', 'selfie', 'government_id', 'work_email']"
     "['email', 'phone', 'jumio', 'offline_government_id']"
     "['phone', 'google', 'reviews', 'offline_government_id', 'kba', 'government_id']"
     "['email', 'phone', 'reviews', 'jumio', 'offline_government_id', 'selfie', 'government_id', 'identity_manual', 'sesame', 'sesame_offline']"]
    host_has_profile_pic: ['t' 'f' nan]
    host_identity_verified: ['t' 'f' nan]
    neighbourhood: ['New York, United States' 'Brooklyn, New York, United States' nan
     'Queens, New York, United States'
     'Long Island City, New York, United States'
     'Astoria, New York, United States' 'Bronx, New York, United States'
     'Staten Island, New York, United States'
     'Elmhurst, New York, United States' 'Riverdale , New York, United States'
     'Briarwood, New York, United States' 'Kips Bay, New York, United States'
     'Jackson Heights, New York, United States'
     'New York, Manhattan, United States'
     'Park Slope, Brooklyn, New York, United States'
     'Kew Gardens, New York, United States'
     'Flushing, New York, United States' 'Astoria , New York, United States'
     'Sunnyside, New York, United States' 'Woodside, New York, United States'
     'NY , New York, United States'
     'Bushwick, Brooklyn, New York, United States'
     'Brooklyn , New York, United States' 'United States'
     'Sunnyside , New York, United States'
     'LONG ISLAND CITY, New York, United States'
     'Astoria, Queens, New York, United States'
     'Woodhaven, New York, United States' 'bronx, New York, United States'
     'Harlem, New York, United States' 'brooklyn, New York, United States'
     'Middle Village, New York, United States'
     'BROOKLYN, New York, United States'
     'Brooklyn,  Ny 11221, New York, United States'
     'Staten Island , New York, United States'
     'Greenpoint, Brooklyn, New York, United States'
     'Long Island city, New York, United States'
     'astoria, New York, United States' 'The Bronx, New York, United States'
     'ASTORIA, New York, United States' 'Ridgewood , New York, United States'
     'Ridgewood, New York, United States' 'Jamaica, New York, United States'
     'Bayside, New York, United States'
     'Jackson heights , New York, United States'
     'East Elmhurst, New York, United States'
     'Williamsburg, Brooklyn, New York, United States'
     'Williamsburg, New York, United States' 'LIC, New York, United States'
     'Brooklyn. , New York, United States'
     'Manhattan, New York, United States' 'New-York, New York, United States'
     'Far Rockaway, New York, United States'
     'Richmond Hill, New York, United States'
     'forest hills/corona, New York, United States'
     'Jackson  hights , New York, United States'
     'Clinton Hill Brooklyn, New York, United States'
     'Flushing , New York, United States' 'Elmhurst , New York, United States'
     'Brooklyn, Northern Mariana Islands, United States'
     'queens, New York, United States'
     'Flushing /Kew Gardens Hills, New York, United States'
     'RIVERDALE, New York, United States'
     'East elmhurst, New York, United States'
     'Forest Hills, New York, United States'
     'SUNNYSIDE, New York, United States' 'Maspeth, New York, United States'
     'Fresh Meadows , New York, United States' 'NY, New York, United States'
     'Floral Park, New York, United States'
     'new york, New York, United States'
     'Richmond hill, New York, United States'
     'Jackson heights, New York, United States'
     'Astoria Queens, New York, United States'
     'New York city, New York, United States'
     'Queens Village, New York, United States'
     'New York , New York, United States' 'Corona, New York, United States'
     'Gravesend Brooklyn , New York, United States'
     'MIDDLE VILLAGE, New York, United States'
     'Bronx , New York, United States' 'Bushwick, New York, United States'
     'Queens , New York, United States'
     'Rockaway beach , New York, United States'
     'Arverne, New York, United States' 'flushing , New York, United States'
     'Parkchester , New York, United States'
     'Fresh meadows, New York, United States'
     'flushing, New York, United States' 'Manhattan , New York, United States'
     'Kew Gardens , New York, United States'
     'Rockaway Beach , New York, United States'
     'Rockaway Beach, New York, United States'
     'Manhattan, New York, New York, United States'
     'Jackson Heights , New York, United States'
     'Flush, New York, United States' 'Jamaica , New York, United States'
     'Corona , New York, United States'
     ' Crown Heights,NY, New York, United States'
     'Jamaica , ny, United States'
     'ozone park queens , New York, United States'
     'Bushwick , New York, United States'
     'New York, US, New York, United States'
     'Forest hills, New York, United States'
     'Woodside , New York, United States'
     'Cambria heights , New York, United States'
     '8425 Elmhurst avenue , New York, United States'
     '纽约市, New York, United States' 'Rego Park, New York, United States'
     'Bronx, NY, New York, United States'
     'Springfield Gardens , New York, United States'
     '纽约, New York, United States' 'Hollis, New York, United States'
     'Springfield Gardens, New York, United States'
     'FOREST HILLS, New York, United States'
     'Brookly , New York, United States'
     'elmhurst Queens, New York, United States'
     'Ozone Park, New York, United States'
     'East elmhurst , New York, United States'
     'South Richmond Hill, New York, United States'
     'Staten island , New York, United States'
     'Glendale , New York, United States'
     'Woodhaven , New York, United States'
     'New York City , New York, United States'
     'Pomona, California, United States'
     'Williamsburg, Brooklyn , New York, United States'
     'Bronx New York, New York, United States'
     'Astoria Queens , New York, United States'
     'Fresh Meadows, New York, United States'
     'St. Albans , New York, United States'
     'New York City, New York, United States'
     'Springfield gardens, New York, United States'
     'Richmond Hill, Jamaica, Queens, New York, United States'
     'west new york , New Jersey, United States'
     'East Elmhurst , New York, United States'
     'East Elmhurst or Flushing , New York, United States'
     'Oakland Gardens , New York, United States'
     'Newyork, New York, United States'
     'Long island city , New York, United States'
     'New york, New York, United States' 'bronx , New York, United States'
     'Flushing or east Elmhurst , New York, United States'
     'Laurelton , New York, United States'
     'Brooklyn, New York, New York, United States'
     'Lawrence, New York, United States'
     'Bushwick Brooklyn , New York, United States'
     'Richmond Hill , New York, United States'
     'Brooklyn Heights , New York, United States'
     'Rosedale , New York, United States'
     'Sunnyside, Queens, New York, United States'
     'Middle village, New York, United States'
     'BROOKLYN , New York, United States'
     'Arverne, Queens, New York, United States'
     'Saint Albans , New York, United States'
     'Fort Greene, New York, United States'
     'Saint Albans, New York, United States'
     ' Astoria, New York, United States' 'Maspeth , New York, United States'
     'New York,Manhattan , New York, United States'
     'Williamsburg , New York, United States'
     'Long Island, New York, United States'
     'Howard Beach, New York, United States'
     'Little neck, New York, United States' 'New York , Ny, United States'
     'New York - Sunnyside , New York, United States'
     'Glendale, New York, United States'
     'Queens Village , New York, United States'
     'forest hills, New York, United States' 'NYC , New York, United States'
     'Rosedale, New York, United States'
     'Queens, Flushing , New York, United States'
     'Jamaica queens, New York, United States'
     'NEW YORK, New York, United States'
     'Laurelton , Queens , New York, United States'
     ' Springfield Gardens, New York, United States'
     'Queens, Astoria , New York, United States'
     'The Bronx (Riverdale), New York, United States'
     'Bushwick Brooklyn, New York, United States'
     'Laurelton, New York, United States'
     'Forest Hill, New York, United States'
     ' Forest Hills, New York, United States'
     'Long Island City, Queens, New York, United States'
     'Brooklyn , Ny, United States' 'Queens village, New York, United States'
     'Greenpoint Brooklyn , New York, United States'
     'Elmont, New York, United States' 'WOODSIDE , New York, United States'
     'Queens, New York , United States' 'Broklyn , New York, United States'
     'Queens-Rego Park, New York, United States'
     'Rego Park , New York, United States'
     'North Bronx (Wakefield), New York, United States'
     'Woodside, Queens, New York, United States'
     'South Ozone Park, New York, United States'
     'woodside, New York, United States'
     'Corona queens , New York, United States'
     'Nueva York, New York, United States'
     'Forest hills , New York, United States' 'New york, Ny, United States'
     ' East Elmhurst, New York, United States'
     'South ozone park , New York, United States'
     'Long Island city , New York, United States'
     'New York, Ny, United States' 'Mount Vernon, New York, United States'
     'New York, NY, Argentina' 'Montbel, Lozère, France'
     'Scottsdale, Arizona, United States' 'Yonkers, New York, United States']
    neighbourhood_cleansed: ['Midtown' 'Bedford-Stuyvesant' 'Sunset Park' 'Upper West Side'
     'South Slope' 'Williamsburg' 'East Harlem' 'Fort Greene' "Hell's Kitchen"
     'East Village' 'Harlem' 'Flatbush' 'Long Island City' 'Jamaica'
     'Greenpoint' 'Nolita' 'Chelsea' 'Upper East Side' 'Prospect Heights'
     'Clinton Hill' 'Washington Heights' 'Kips Bay' 'Bushwick'
     'Carroll Gardens' 'West Village' 'Park Slope' 'Prospect-Lefferts Gardens'
     'Lower East Side' 'East Flatbush' 'Boerum Hill' 'Sunnyside' 'St. George'
     'Tribeca' 'Highbridge' 'Ridgewood' 'Mott Haven' 'Morningside Heights'
     'Gowanus' 'Ditmars Steinway' 'Middle Village' 'Brooklyn Heights'
     'Flatiron District' 'Windsor Terrace' 'Chinatown' 'Greenwich Village'
     'Clason Point' 'Crown Heights' 'Astoria' 'Kingsbridge' 'Forest Hills'
     'Murray Hill' 'University Heights' 'Gravesend' 'Allerton' 'East New York'
     'Stuyvesant Town' 'Sheepshead Bay' 'Emerson Hill' 'Bensonhurst'
     'Shore Acres' 'Richmond Hill' 'Gramercy' 'Arrochar' 'Financial District'
     'Theater District' 'Rego Park' 'Kensington' 'Woodside' 'Cypress Hills'
     'SoHo' 'Little Italy' 'Elmhurst' 'Clifton' 'Bayside' 'Bay Ridge'
     'Maspeth' 'Spuyten Duyvil' 'Stapleton' 'Briarwood' 'Battery Park City'
     'Brighton Beach' 'Jackson Heights' 'Longwood' 'Inwood' 'Two Bridges'
     'Fort Hamilton' 'Cobble Hill' 'New Springville' 'Flushing' 'Red Hook'
     'Civic Center' 'Tompkinsville' 'Tottenville' 'NoHo' 'DUMBO' 'Columbia St'
     'Glendale' 'Mariners Harbor' 'East Elmhurst' 'Concord'
     'Downtown Brooklyn' 'Melrose' 'Kew Gardens' 'College Point' 'Mount Eden'
     'Vinegar Hill' 'City Island' 'Canarsie' 'Port Morris' 'Flatlands'
     'Arverne' 'Queens Village' 'Midwood' 'Brownsville' 'Williamsbridge'
     'Soundview' 'Woodhaven' 'Parkchester' 'Bronxdale' 'Bay Terrace'
     'Ozone Park' 'Norwood' 'Rockaway Beach' 'Hollis' 'Claremont Village'
     'Fordham' 'Concourse Village' 'Borough Park' 'Fieldston'
     'Springfield Gardens' 'Huguenot' 'Mount Hope' 'Wakefield' 'Navy Yard'
     'Roosevelt Island' 'Lighthouse Hill' 'Unionport' 'Randall Manor'
     'South Ozone Park' 'Kew Gardens Hills' 'Jamaica Estates' 'Concourse'
     'Bellerose' 'Fresh Meadows' 'Eastchester' 'Morris Park' 'Far Rockaway'
     'East Morrisania' 'Corona' 'Tremont' 'St. Albans' 'West Brighton'
     'Manhattan Beach' 'Marble Hill' 'Dongan Hills' 'Morris Heights' 'Belmont'
     'Castleton Corners' 'Laurelton' 'Hunts Point' 'Howard Beach'
     'Great Kills' 'Pelham Bay' 'Silver Lake' 'Riverdale' 'Morrisania'
     'Grymes Hill' 'Holliswood' 'Edgemere' 'New Brighton' 'Pelham Gardens'
     'Baychester' 'Sea Gate' 'Belle Harbor' 'Bergen Beach' 'Cambria Heights'
     'Richmondtown' 'Olinville' 'Dyker Heights' 'Throgs Neck' 'Coney Island'
     'Rosedale' 'Howland Hook' "Prince's Bay" 'South Beach' 'Bath Beach'
     'Midland Beach' 'Eltingville' 'Oakwood' 'Schuylerville' 'Edenwald'
     'North Riverdale' 'Port Richmond' 'Fort Wadsworth' 'Westchester Square'
     'Van Nest' 'Arden Heights' "Bull's Head" 'Woodlawn' 'New Dorp' 'Neponsit'
     'Grant City' 'Bayswater' 'Douglaston' 'New Dorp Beach' 'Todt Hill'
     'Mill Basin' 'West Farms' 'Little Neck' 'Whitestone' 'Rosebank'
     'Co-op City' 'Jamaica Hills' 'Rossville' 'Castle Hill' 'Westerleigh'
     'Country Club' 'Chelsea, Staten Island' 'Gerritsen Beach' 'Breezy Point'
     'Woodrow' 'Graniteville']
    neighbourhood_group_cleansed: ['Manhattan' 'Brooklyn' 'Queens' 'Staten Island' 'Bronx']
    property_type: ['Entire rental unit' 'Entire guest suite' 'Private room in rental unit'
     'Private room in townhouse' 'Private room in condominium (condo)'
     'Private room in loft' 'Entire loft' 'Private room in residential home'
     'Entire condominium (condo)' 'Entire residential home' 'Entire townhouse'
     'Private room in bed and breakfast' 'Entire guesthouse'
     'Private room in guest suite' 'Room in boutique hotel'
     'Shared room in loft' 'Shared room in rental unit'
     'Shared room in residential home' 'Private room' 'Private room in hostel'
     'Entire place' 'Private room in guesthouse' 'Boat'
     'Entire serviced apartment' 'Room in aparthotel' 'Floor'
     'Private room in vacation home' 'Room in serviced apartment'
     'Entire cottage' 'Private room in serviced apartment' 'Room in hotel'
     'Cave' 'Tiny house' 'Private room in floor'
     'Shared room in condominium (condo)' 'Entire bungalow'
     'Private room in casa particular' 'Shared room in townhouse' 'Houseboat'
     'Private room in bungalow' 'Entire villa' 'Private room in resort'
     'Shared room in guest suite' 'Private room in castle'
     'Private room in villa' 'Shared room in floor' 'Entire bed and breakfast'
     'Entire home/apt' 'Private room in tiny house' 'Private room in tent'
     'Private room in in-law' 'Private room in barn' 'Shared room in hostel'
     'Camper/RV' 'Room in resort' 'Shared room in guesthouse' 'Bus'
     'Shared room in bed and breakfast' 'Private room in farm stay'
     'Private room in dorm' 'Room in bed and breakfast'
     'Shared room in island' 'Shared room in bungalow'
     'Shared room in serviced apartment' 'Private room in earth house'
     'Lighthouse' 'Private room in train' 'Barn' 'Private room in lighthouse'
     'Entire cabin' 'Private room in camper/rv' 'Castle' 'Tent' 'Tower'
     'Casa particular' 'Shared room in casa particular'
     'Private room in cycladic house' 'Entire vacation home']
    room_type: ['Entire home/apt' 'Private room' 'Hotel room' 'Shared room']
    bathrooms_text: ['1 bath' nan '1.5 baths' '1 shared bath' '1 private bath'
     'Shared half-bath' '2 baths' '1.5 shared baths' '3 baths' 'Half-bath'
     '2.5 baths' '2 shared baths' '0 baths' '4 baths' '0 shared baths'
     'Private half-bath' '5 baths' '4.5 baths' '5.5 baths' '2.5 shared baths'
     '3.5 baths' '3 shared baths' '4 shared baths' '6 baths'
     '3.5 shared baths' '4.5 shared baths' '7.5 baths' '6.5 baths' '8 baths'
     '7 baths' '6 shared baths']
    amenities: ['["Extra pillows and blankets", "Baking sheet", "Luggage dropoff allowed", "TV", "Hangers", "Ethernet connection", "Long term stays allowed", "Carbon monoxide alarm", "Wifi", "Heating", "Dishes and silverware", "Air conditioning", "Free street parking", "Essentials", "Hot water", "Bathtub", "Kitchen", "Fire extinguisher", "Cooking basics", "Dedicated workspace", "Hair dryer", "Stove", "Smoke alarm", "Keypad", "Iron", "Oven", "Paid parking off premises", "Refrigerator", "Bed linens", "Cleaning before checkout", "Coffee maker"]'
     '["Extra pillows and blankets", "Luggage dropoff allowed", "Free parking on premises", "Pack \\u2019n play/Travel crib", "Microwave", "Hangers", "Lockbox", "Long term stays allowed", "Carbon monoxide alarm", "High chair", "Wifi", "Heating", "Shampoo", "Dishes and silverware", "Air conditioning", "Free street parking", "Essentials", "Hot water", "Bathtub", "Kitchen", "Cable TV", "Fire extinguisher", "Cooking basics", "Dedicated workspace", "Hair dryer", "Stove", "Children\\u2019s books and toys", "TV with standard cable", "Smoke alarm", "Iron", "Oven", "Refrigerator", "Bed linens", "Baby safety gates", "Coffee maker"]'
     '["Kitchen", "Long term stays allowed", "Wifi", "Heating", "Air conditioning"]'
     ...
     '["Air conditioning", "Fire extinguisher", "Dedicated workspace", "Carbon monoxide alarm", "Lock on bedroom door", "Wifi", "Shampoo", "Hot water", "Hangers", "Iron", "Heating", "First aid kit", "Kitchen", "Hair dryer", "Smoke alarm", "Essentials", "Long term stays allowed"]'
     '["Fire extinguisher", "Cooking basics", "Carbon monoxide alarm", "Lock on bedroom door", "Private entrance", "Shampoo", "Hangers", "Heating", "First aid kit", "Kitchen", "TV", "Wifi", "Smoke alarm", "Essentials", "Long term stays allowed"]'
     '["Stainless steel electric stove", "Security cameras on property", "Private patio or balcony", "Pack \\u2019n play/Travel crib", "Microwave", "First aid kit", "Hangers", "Breakfast", "Long term stays allowed", "Carbon monoxide alarm", "High chair", "Washer", "Wifi", "Heating", "Shampoo", "Dryer", "Dishes and silverware", "Private entrance", "Air conditioning", "Stainless steel oven", "Essentials", "Hot water", "Free street parking", "Kitchen", "Cable TV", "Fire extinguisher", "Cooking basics", "Dedicated workspace", "Hair dryer", "Room-darkening shades", "TV with standard cable", "Smoke alarm", "Dishwasher", "Iron", "Refrigerator", "Bed linens", "Smart lock", "Coffee maker"]']
    has_availability: ['t' 'f']
    calendar_last_scraped: ['2021-12-05' '2021-12-04']
    first_review: ['2009-11-21' '2015-01-05' '2014-01-22' ... '2021-01-13' '2021-03-17'
     '2021-04-21']
    last_review: ['2019-11-04' '2021-10-22' '2016-06-05' ... '2021-01-22' '2021-04-22'
     '2021-12-05']
    license: [nan '41662/AL']
    instant_bookable: ['f' 't']


2.  List these features and explain why they would be suitable for one-hot encoding. Note your findings in the markdown cell below.

The features are name, description, neighborhood overview, host name, host since, host location, host about, host response rate, host acceptance rate, host is superhost, host neighbour, host verifications, host has profile pic, host idenity verified, neighbourhood, neighbourhood group cleansed, property type, room type, bathrooms text, amenities, has availability, calendar last scraped, first review, last review, license, and instant bookable are all categorical variable and could be condensed into binary data.

**Task**: In the code cell below, one-hot encode one of the features you have identified and replace the original column in DataFrame `df` with the new one-hot encoded columns. 


```python
one_hot = pd.get_dummies(df['room_type'])

df = df.drop('room_type', axis=1)

df = df.join(one_hot)
```

## Part 4. Explore Your Data

You will now perform exploratory data analysis in preparation for selecting your features as part of feature engineering. 

#### Identify Correlations

We will focus on identifying which features in the data have the highest correlation with the label.

Let's first run the `corr()` method on DataFrame `df` and save the result to the variable `corr_matrix`. Let's round the resulting correlations to five decimal places:


```python
corr_matrix = round(df.corr(),5)
corr_matrix
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
      <th>id</th>
      <th>scrape_id</th>
      <th>host_id</th>
      <th>host_listings_count</th>
      <th>host_total_listings_count</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>...</th>
      <th>beds_na</th>
      <th>a few days or more</th>
      <th>unavailable</th>
      <th>within a day</th>
      <th>within a few hours</th>
      <th>within an hour</th>
      <th>Entire home/apt</th>
      <th>Hotel room</th>
      <th>Private room</th>
      <th>Shared room</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>1.00000</td>
      <td>-0.0</td>
      <td>0.58617</td>
      <td>0.12986</td>
      <td>0.12986</td>
      <td>0.01000</td>
      <td>0.08708</td>
      <td>0.03540</td>
      <td>NaN</td>
      <td>0.04503</td>
      <td>...</td>
      <td>0.13640</td>
      <td>0.01215</td>
      <td>-0.35410</td>
      <td>-0.01164</td>
      <td>0.12780</td>
      <td>0.29187</td>
      <td>-0.04284</td>
      <td>0.01698</td>
      <td>0.03813</td>
      <td>0.00958</td>
    </tr>
    <tr>
      <th>scrape_id</th>
      <td>-0.00000</td>
      <td>1.0</td>
      <td>0.00000</td>
      <td>-0.00000</td>
      <td>-0.00000</td>
      <td>0.00000</td>
      <td>-0.00000</td>
      <td>0.00000</td>
      <td>NaN</td>
      <td>0.00000</td>
      <td>...</td>
      <td>-0.00000</td>
      <td>0.00000</td>
      <td>-0.00000</td>
      <td>-0.00000</td>
      <td>-0.00000</td>
      <td>-0.00000</td>
      <td>-0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>host_id</th>
      <td>0.58617</td>
      <td>0.0</td>
      <td>1.00000</td>
      <td>0.03189</td>
      <td>0.03189</td>
      <td>0.04148</td>
      <td>0.11620</td>
      <td>0.02723</td>
      <td>NaN</td>
      <td>0.02202</td>
      <td>...</td>
      <td>0.09218</td>
      <td>0.04055</td>
      <td>-0.24094</td>
      <td>-0.05562</td>
      <td>0.01844</td>
      <td>0.26491</td>
      <td>-0.12862</td>
      <td>0.07086</td>
      <td>0.10957</td>
      <td>0.03676</td>
    </tr>
    <tr>
      <th>host_listings_count</th>
      <td>0.12986</td>
      <td>-0.0</td>
      <td>0.03189</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.03475</td>
      <td>-0.08843</td>
      <td>-0.02621</td>
      <td>NaN</td>
      <td>-0.01710</td>
      <td>...</td>
      <td>-0.01032</td>
      <td>-0.03124</td>
      <td>-0.11686</td>
      <td>-0.03119</td>
      <td>-0.01468</td>
      <td>0.17132</td>
      <td>0.01040</td>
      <td>-0.00877</td>
      <td>-0.00468</td>
      <td>-0.01825</td>
    </tr>
    <tr>
      <th>host_total_listings_count</th>
      <td>0.12986</td>
      <td>-0.0</td>
      <td>0.03189</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.03475</td>
      <td>-0.08843</td>
      <td>-0.02621</td>
      <td>NaN</td>
      <td>-0.01710</td>
      <td>...</td>
      <td>-0.01032</td>
      <td>-0.03124</td>
      <td>-0.11686</td>
      <td>-0.03119</td>
      <td>-0.01468</td>
      <td>0.17132</td>
      <td>0.01040</td>
      <td>-0.00877</td>
      <td>-0.00468</td>
      <td>-0.01825</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>0.01000</td>
      <td>0.0</td>
      <td>0.04148</td>
      <td>0.03475</td>
      <td>0.03475</td>
      <td>1.00000</td>
      <td>0.05718</td>
      <td>-0.04745</td>
      <td>NaN</td>
      <td>-0.07150</td>
      <td>...</td>
      <td>0.02258</td>
      <td>0.02052</td>
      <td>0.01134</td>
      <td>0.01410</td>
      <td>-0.00499</td>
      <td>-0.02598</td>
      <td>-0.02656</td>
      <td>0.02825</td>
      <td>0.01830</td>
      <td>0.01707</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>0.08708</td>
      <td>-0.0</td>
      <td>0.11620</td>
      <td>-0.08843</td>
      <td>-0.08843</td>
      <td>0.05718</td>
      <td>1.00000</td>
      <td>0.00374</td>
      <td>NaN</td>
      <td>0.00752</td>
      <td>...</td>
      <td>0.00221</td>
      <td>-0.01400</td>
      <td>-0.07471</td>
      <td>-0.03805</td>
      <td>0.03534</td>
      <td>0.08358</td>
      <td>-0.14909</td>
      <td>-0.04860</td>
      <td>0.15128</td>
      <td>0.02280</td>
    </tr>
    <tr>
      <th>accommodates</th>
      <td>0.03540</td>
      <td>0.0</td>
      <td>0.02723</td>
      <td>-0.02621</td>
      <td>-0.02621</td>
      <td>-0.04745</td>
      <td>0.00374</td>
      <td>1.00000</td>
      <td>NaN</td>
      <td>0.70586</td>
      <td>...</td>
      <td>-0.06916</td>
      <td>0.01101</td>
      <td>-0.11168</td>
      <td>0.01642</td>
      <td>-0.00382</td>
      <td>0.11060</td>
      <td>0.45742</td>
      <td>-0.01671</td>
      <td>-0.44105</td>
      <td>-0.06358</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>0.04503</td>
      <td>0.0</td>
      <td>0.02202</td>
      <td>-0.01710</td>
      <td>-0.01710</td>
      <td>-0.07150</td>
      <td>0.00752</td>
      <td>0.70586</td>
      <td>NaN</td>
      <td>1.00000</td>
      <td>...</td>
      <td>-0.04571</td>
      <td>0.01969</td>
      <td>-0.09343</td>
      <td>0.03512</td>
      <td>0.01114</td>
      <td>0.06432</td>
      <td>0.35604</td>
      <td>-0.02448</td>
      <td>-0.33917</td>
      <td>-0.05944</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>0.03289</td>
      <td>0.0</td>
      <td>0.03689</td>
      <td>-0.03151</td>
      <td>-0.03151</td>
      <td>-0.05388</td>
      <td>0.03136</td>
      <td>0.73665</td>
      <td>NaN</td>
      <td>0.72914</td>
      <td>...</td>
      <td>0.00000</td>
      <td>0.02056</td>
      <td>-0.10810</td>
      <td>0.01886</td>
      <td>0.00242</td>
      <td>0.09628</td>
      <td>0.32487</td>
      <td>-0.01256</td>
      <td>-0.32660</td>
      <td>0.01000</td>
    </tr>
    <tr>
      <th>price</th>
      <td>0.04256</td>
      <td>-0.0</td>
      <td>0.02907</td>
      <td>0.07492</td>
      <td>0.07492</td>
      <td>0.02734</td>
      <td>-0.11484</td>
      <td>0.30803</td>
      <td>NaN</td>
      <td>0.25383</td>
      <td>...</td>
      <td>-0.01596</td>
      <td>0.02432</td>
      <td>-0.05266</td>
      <td>-0.00026</td>
      <td>-0.01433</td>
      <td>0.05805</td>
      <td>0.17365</td>
      <td>0.05119</td>
      <td>-0.18024</td>
      <td>-0.00669</td>
    </tr>
    <tr>
      <th>minimum_nights</th>
      <td>-0.12067</td>
      <td>0.0</td>
      <td>-0.10640</td>
      <td>0.19739</td>
      <td>0.19739</td>
      <td>0.03422</td>
      <td>-0.08550</td>
      <td>-0.08474</td>
      <td>NaN</td>
      <td>-0.02749</td>
      <td>...</td>
      <td>-0.01830</td>
      <td>0.03087</td>
      <td>0.18254</td>
      <td>-0.00695</td>
      <td>0.00592</td>
      <td>-0.21377</td>
      <td>0.00925</td>
      <td>-0.03447</td>
      <td>-0.00313</td>
      <td>-0.00423</td>
    </tr>
    <tr>
      <th>maximum_nights</th>
      <td>-0.00696</td>
      <td>0.0</td>
      <td>-0.00385</td>
      <td>-0.00080</td>
      <td>-0.00080</td>
      <td>0.00561</td>
      <td>-0.00296</td>
      <td>-0.00494</td>
      <td>NaN</td>
      <td>0.00002</td>
      <td>...</td>
      <td>-0.00135</td>
      <td>-0.00108</td>
      <td>0.00577</td>
      <td>-0.00153</td>
      <td>-0.00210</td>
      <td>-0.00334</td>
      <td>0.00478</td>
      <td>-0.00039</td>
      <td>-0.00458</td>
      <td>-0.00064</td>
    </tr>
    <tr>
      <th>minimum_minimum_nights</th>
      <td>-0.10234</td>
      <td>0.0</td>
      <td>-0.09188</td>
      <td>0.26125</td>
      <td>0.26125</td>
      <td>0.03317</td>
      <td>-0.08397</td>
      <td>-0.07485</td>
      <td>NaN</td>
      <td>-0.02546</td>
      <td>...</td>
      <td>-0.01823</td>
      <td>0.02434</td>
      <td>0.15024</td>
      <td>-0.01002</td>
      <td>-0.00678</td>
      <td>-0.16408</td>
      <td>0.02079</td>
      <td>-0.02844</td>
      <td>-0.01574</td>
      <td>-0.00443</td>
    </tr>
    <tr>
      <th>maximum_minimum_nights</th>
      <td>-0.00041</td>
      <td>-0.0</td>
      <td>-0.04521</td>
      <td>0.65300</td>
      <td>0.65300</td>
      <td>0.04352</td>
      <td>-0.09520</td>
      <td>-0.05134</td>
      <td>NaN</td>
      <td>-0.01708</td>
      <td>...</td>
      <td>-0.02851</td>
      <td>-0.00457</td>
      <td>0.00076</td>
      <td>-0.02714</td>
      <td>-0.02551</td>
      <td>0.03672</td>
      <td>0.07891</td>
      <td>-0.01886</td>
      <td>-0.07349</td>
      <td>-0.01233</td>
    </tr>
    <tr>
      <th>minimum_maximum_nights</th>
      <td>0.00747</td>
      <td>-0.0</td>
      <td>0.02572</td>
      <td>-0.00349</td>
      <td>-0.00349</td>
      <td>0.01735</td>
      <td>-0.00780</td>
      <td>-0.00249</td>
      <td>NaN</td>
      <td>-0.01161</td>
      <td>...</td>
      <td>-0.00673</td>
      <td>-0.00543</td>
      <td>-0.00942</td>
      <td>0.02956</td>
      <td>-0.01049</td>
      <td>0.00315</td>
      <td>-0.02184</td>
      <td>0.14009</td>
      <td>0.00279</td>
      <td>-0.00322</td>
    </tr>
    <tr>
      <th>maximum_maximum_nights</th>
      <td>0.01461</td>
      <td>0.0</td>
      <td>0.04267</td>
      <td>-0.00529</td>
      <td>-0.00529</td>
      <td>0.01598</td>
      <td>-0.01993</td>
      <td>-0.00931</td>
      <td>NaN</td>
      <td>-0.01705</td>
      <td>...</td>
      <td>-0.00781</td>
      <td>-0.00845</td>
      <td>-0.02371</td>
      <td>0.02398</td>
      <td>-0.01634</td>
      <td>0.02789</td>
      <td>-0.03952</td>
      <td>0.11571</td>
      <td>0.02444</td>
      <td>-0.00500</td>
    </tr>
    <tr>
      <th>minimum_nights_avg_ntm</th>
      <td>-0.00338</td>
      <td>-0.0</td>
      <td>-0.04707</td>
      <td>0.65239</td>
      <td>0.65239</td>
      <td>0.04379</td>
      <td>-0.09507</td>
      <td>-0.05266</td>
      <td>NaN</td>
      <td>-0.01782</td>
      <td>...</td>
      <td>-0.02848</td>
      <td>-0.00364</td>
      <td>0.00547</td>
      <td>-0.02691</td>
      <td>-0.02692</td>
      <td>0.03209</td>
      <td>0.07834</td>
      <td>-0.01984</td>
      <td>-0.07289</td>
      <td>-0.01192</td>
    </tr>
    <tr>
      <th>maximum_nights_avg_ntm</th>
      <td>0.01149</td>
      <td>0.0</td>
      <td>0.03438</td>
      <td>-0.00451</td>
      <td>-0.00451</td>
      <td>0.01828</td>
      <td>-0.01401</td>
      <td>-0.00558</td>
      <td>NaN</td>
      <td>-0.01465</td>
      <td>...</td>
      <td>-0.00831</td>
      <td>-0.00717</td>
      <td>-0.01380</td>
      <td>0.02215</td>
      <td>-0.01386</td>
      <td>0.01568</td>
      <td>-0.03164</td>
      <td>0.15595</td>
      <td>0.01063</td>
      <td>-0.00425</td>
    </tr>
    <tr>
      <th>calendar_updated</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>availability_30</th>
      <td>0.25190</td>
      <td>-0.0</td>
      <td>0.26850</td>
      <td>0.07148</td>
      <td>0.07148</td>
      <td>0.00261</td>
      <td>0.13025</td>
      <td>0.04429</td>
      <td>NaN</td>
      <td>0.01816</td>
      <td>...</td>
      <td>0.09611</td>
      <td>0.20254</td>
      <td>-0.29428</td>
      <td>0.04232</td>
      <td>0.10312</td>
      <td>0.12962</td>
      <td>-0.10800</td>
      <td>0.04272</td>
      <td>0.08909</td>
      <td>0.05305</td>
    </tr>
    <tr>
      <th>availability_60</th>
      <td>0.32793</td>
      <td>-0.0</td>
      <td>0.32728</td>
      <td>0.06218</td>
      <td>0.06218</td>
      <td>0.00026</td>
      <td>0.15062</td>
      <td>0.07983</td>
      <td>NaN</td>
      <td>0.04432</td>
      <td>...</td>
      <td>0.09098</td>
      <td>0.18352</td>
      <td>-0.43295</td>
      <td>0.05946</td>
      <td>0.13745</td>
      <td>0.25344</td>
      <td>-0.08439</td>
      <td>0.03851</td>
      <td>0.06938</td>
      <td>0.03931</td>
    </tr>
    <tr>
      <th>availability_90</th>
      <td>0.34401</td>
      <td>-0.0</td>
      <td>0.33395</td>
      <td>0.06279</td>
      <td>0.06279</td>
      <td>-0.00157</td>
      <td>0.14953</td>
      <td>0.09096</td>
      <td>NaN</td>
      <td>0.05567</td>
      <td>...</td>
      <td>0.09143</td>
      <td>0.17710</td>
      <td>-0.47929</td>
      <td>0.08130</td>
      <td>0.14265</td>
      <td>0.29008</td>
      <td>-0.06442</td>
      <td>0.03578</td>
      <td>0.05103</td>
      <td>0.03405</td>
    </tr>
    <tr>
      <th>availability_365</th>
      <td>0.28722</td>
      <td>0.0</td>
      <td>0.27332</td>
      <td>0.14287</td>
      <td>0.14287</td>
      <td>0.01383</td>
      <td>0.09596</td>
      <td>0.10293</td>
      <td>NaN</td>
      <td>0.08280</td>
      <td>...</td>
      <td>0.08961</td>
      <td>0.12545</td>
      <td>-0.47520</td>
      <td>0.10797</td>
      <td>0.17218</td>
      <td>0.26996</td>
      <td>-0.00816</td>
      <td>0.05067</td>
      <td>-0.00435</td>
      <td>0.02056</td>
    </tr>
    <tr>
      <th>number_of_reviews</th>
      <td>-0.29164</td>
      <td>0.0</td>
      <td>-0.12215</td>
      <td>-0.06617</td>
      <td>-0.06617</td>
      <td>-0.04801</td>
      <td>0.06759</td>
      <td>0.07255</td>
      <td>NaN</td>
      <td>0.00408</td>
      <td>...</td>
      <td>-0.05311</td>
      <td>-0.03115</td>
      <td>-0.16121</td>
      <td>0.00818</td>
      <td>-0.00846</td>
      <td>0.19174</td>
      <td>0.02319</td>
      <td>0.03582</td>
      <td>-0.02639</td>
      <td>-0.00903</td>
    </tr>
    <tr>
      <th>number_of_reviews_ltm</th>
      <td>0.07737</td>
      <td>0.0</td>
      <td>0.11469</td>
      <td>-0.04448</td>
      <td>-0.04448</td>
      <td>-0.04884</td>
      <td>0.06458</td>
      <td>0.08118</td>
      <td>NaN</td>
      <td>0.02836</td>
      <td>...</td>
      <td>-0.03291</td>
      <td>-0.06060</td>
      <td>-0.22794</td>
      <td>-0.03950</td>
      <td>-0.02346</td>
      <td>0.31743</td>
      <td>0.02510</td>
      <td>0.08765</td>
      <td>-0.03482</td>
      <td>-0.01391</td>
    </tr>
    <tr>
      <th>number_of_reviews_l30d</th>
      <td>0.15257</td>
      <td>-0.0</td>
      <td>0.15333</td>
      <td>-0.04962</td>
      <td>-0.04962</td>
      <td>-0.04339</td>
      <td>0.07309</td>
      <td>0.08552</td>
      <td>NaN</td>
      <td>0.03271</td>
      <td>...</td>
      <td>-0.01860</td>
      <td>-0.07216</td>
      <td>-0.24822</td>
      <td>-0.05445</td>
      <td>-0.02925</td>
      <td>0.35797</td>
      <td>0.03656</td>
      <td>0.00086</td>
      <td>-0.03389</td>
      <td>-0.01197</td>
    </tr>
    <tr>
      <th>review_scores_rating</th>
      <td>0.01187</td>
      <td>0.0</td>
      <td>-0.04397</td>
      <td>-0.00742</td>
      <td>-0.00742</td>
      <td>-0.03767</td>
      <td>0.00523</td>
      <td>0.03097</td>
      <td>NaN</td>
      <td>0.01686</td>
      <td>...</td>
      <td>-0.01925</td>
      <td>-0.06101</td>
      <td>-0.09901</td>
      <td>0.02862</td>
      <td>0.02229</td>
      <td>0.09629</td>
      <td>0.08109</td>
      <td>-0.01071</td>
      <td>-0.07572</td>
      <td>-0.01767</td>
    </tr>
    <tr>
      <th>review_scores_accuracy</th>
      <td>-0.08867</td>
      <td>0.0</td>
      <td>-0.15428</td>
      <td>-0.02365</td>
      <td>-0.02365</td>
      <td>-0.04076</td>
      <td>-0.01136</td>
      <td>-0.00422</td>
      <td>NaN</td>
      <td>-0.00323</td>
      <td>...</td>
      <td>-0.04077</td>
      <td>-0.07606</td>
      <td>0.04080</td>
      <td>0.00761</td>
      <td>-0.04651</td>
      <td>0.01555</td>
      <td>0.09148</td>
      <td>-0.03556</td>
      <td>-0.08241</td>
      <td>-0.01757</td>
    </tr>
    <tr>
      <th>review_scores_cleanliness</th>
      <td>0.00424</td>
      <td>0.0</td>
      <td>-0.05183</td>
      <td>-0.00694</td>
      <td>-0.00694</td>
      <td>-0.03469</td>
      <td>0.00772</td>
      <td>0.03702</td>
      <td>NaN</td>
      <td>0.03206</td>
      <td>...</td>
      <td>-0.03027</td>
      <td>-0.06482</td>
      <td>-0.06196</td>
      <td>0.01355</td>
      <td>-0.01300</td>
      <td>0.09180</td>
      <td>0.10695</td>
      <td>0.00819</td>
      <td>-0.10530</td>
      <td>-0.01476</td>
    </tr>
    <tr>
      <th>review_scores_checkin</th>
      <td>-0.09156</td>
      <td>0.0</td>
      <td>-0.14890</td>
      <td>-0.01701</td>
      <td>-0.01701</td>
      <td>-0.04612</td>
      <td>-0.00525</td>
      <td>-0.00125</td>
      <td>NaN</td>
      <td>0.00638</td>
      <td>...</td>
      <td>-0.04050</td>
      <td>-0.08196</td>
      <td>0.02230</td>
      <td>0.01290</td>
      <td>-0.01974</td>
      <td>0.01498</td>
      <td>0.07370</td>
      <td>-0.02068</td>
      <td>-0.06553</td>
      <td>-0.02305</td>
    </tr>
    <tr>
      <th>review_scores_communication</th>
      <td>-0.11950</td>
      <td>0.0</td>
      <td>-0.17420</td>
      <td>-0.05032</td>
      <td>-0.05032</td>
      <td>-0.04250</td>
      <td>-0.01358</td>
      <td>-0.00067</td>
      <td>NaN</td>
      <td>-0.00019</td>
      <td>...</td>
      <td>-0.03904</td>
      <td>-0.08031</td>
      <td>0.05199</td>
      <td>-0.00556</td>
      <td>-0.04243</td>
      <td>0.01028</td>
      <td>0.08425</td>
      <td>-0.02970</td>
      <td>-0.07540</td>
      <td>-0.02031</td>
    </tr>
    <tr>
      <th>review_scores_location</th>
      <td>0.00322</td>
      <td>0.0</td>
      <td>-0.07864</td>
      <td>0.00638</td>
      <td>0.00638</td>
      <td>0.01355</td>
      <td>-0.13822</td>
      <td>-0.01220</td>
      <td>NaN</td>
      <td>-0.01053</td>
      <td>...</td>
      <td>-0.02043</td>
      <td>-0.04102</td>
      <td>0.01118</td>
      <td>0.00999</td>
      <td>-0.01360</td>
      <td>0.00806</td>
      <td>0.09444</td>
      <td>0.01197</td>
      <td>-0.09296</td>
      <td>-0.01618</td>
    </tr>
    <tr>
      <th>review_scores_value</th>
      <td>-0.07080</td>
      <td>0.0</td>
      <td>-0.13340</td>
      <td>-0.07391</td>
      <td>-0.07391</td>
      <td>-0.04887</td>
      <td>0.00052</td>
      <td>-0.00778</td>
      <td>NaN</td>
      <td>0.00074</td>
      <td>...</td>
      <td>-0.03429</td>
      <td>-0.06118</td>
      <td>0.04111</td>
      <td>-0.00564</td>
      <td>-0.05498</td>
      <td>0.02334</td>
      <td>0.04539</td>
      <td>-0.03393</td>
      <td>-0.03770</td>
      <td>-0.01173</td>
    </tr>
    <tr>
      <th>calculated_host_listings_count</th>
      <td>0.23667</td>
      <td>-0.0</td>
      <td>0.15754</td>
      <td>0.42944</td>
      <td>0.42944</td>
      <td>0.07954</td>
      <td>-0.06543</td>
      <td>-0.11818</td>
      <td>NaN</td>
      <td>-0.05754</td>
      <td>...</td>
      <td>0.12938</td>
      <td>-0.05406</td>
      <td>-0.08352</td>
      <td>-0.01243</td>
      <td>0.09949</td>
      <td>0.04675</td>
      <td>-0.04794</td>
      <td>-0.00784</td>
      <td>0.05666</td>
      <td>-0.03027</td>
    </tr>
    <tr>
      <th>calculated_host_listings_count_entire_homes</th>
      <td>0.13713</td>
      <td>0.0</td>
      <td>0.02524</td>
      <td>0.54188</td>
      <td>0.54188</td>
      <td>0.07065</td>
      <td>-0.12713</td>
      <td>-0.01929</td>
      <td>NaN</td>
      <td>-0.00212</td>
      <td>...</td>
      <td>0.01163</td>
      <td>-0.04190</td>
      <td>-0.14256</td>
      <td>0.04999</td>
      <td>0.01930</td>
      <td>0.13010</td>
      <td>0.16276</td>
      <td>-0.00853</td>
      <td>-0.15528</td>
      <td>-0.02785</td>
    </tr>
    <tr>
      <th>calculated_host_listings_count_private_rooms</th>
      <td>0.21188</td>
      <td>-0.0</td>
      <td>0.19320</td>
      <td>0.14915</td>
      <td>0.14915</td>
      <td>0.05096</td>
      <td>0.01401</td>
      <td>-0.14499</td>
      <td>NaN</td>
      <td>-0.07591</td>
      <td>...</td>
      <td>0.16732</td>
      <td>-0.03991</td>
      <td>0.00213</td>
      <td>-0.05728</td>
      <td>0.11927</td>
      <td>-0.04169</td>
      <td>-0.19529</td>
      <td>-0.01535</td>
      <td>0.20438</td>
      <td>-0.02503</td>
    </tr>
    <tr>
      <th>calculated_host_listings_count_shared_rooms</th>
      <td>0.04671</td>
      <td>-0.0</td>
      <td>0.07831</td>
      <td>-0.01595</td>
      <td>-0.01595</td>
      <td>0.00762</td>
      <td>0.02066</td>
      <td>-0.05161</td>
      <td>NaN</td>
      <td>-0.04902</td>
      <td>...</td>
      <td>0.01101</td>
      <td>0.02082</td>
      <td>-0.01928</td>
      <td>-0.01131</td>
      <td>0.01389</td>
      <td>0.00810</td>
      <td>-0.11059</td>
      <td>-0.00835</td>
      <td>-0.04520</td>
      <td>0.64509</td>
    </tr>
    <tr>
      <th>reviews_per_month</th>
      <td>0.23169</td>
      <td>0.0</td>
      <td>0.20844</td>
      <td>-0.02096</td>
      <td>-0.02096</td>
      <td>-0.03667</td>
      <td>0.07121</td>
      <td>0.06850</td>
      <td>NaN</td>
      <td>0.03030</td>
      <td>...</td>
      <td>-0.00329</td>
      <td>-0.04892</td>
      <td>-0.20592</td>
      <td>-0.04801</td>
      <td>-0.02663</td>
      <td>0.28524</td>
      <td>-0.00268</td>
      <td>0.03322</td>
      <td>-0.00053</td>
      <td>-0.00766</td>
    </tr>
    <tr>
      <th>label_price</th>
      <td>0.07907</td>
      <td>-0.0</td>
      <td>0.04053</td>
      <td>0.13104</td>
      <td>0.13104</td>
      <td>0.04330</td>
      <td>-0.20695</td>
      <td>0.50062</td>
      <td>NaN</td>
      <td>0.41996</td>
      <td>...</td>
      <td>-0.03461</td>
      <td>0.00792</td>
      <td>-0.10279</td>
      <td>0.01335</td>
      <td>-0.02111</td>
      <td>0.11721</td>
      <td>0.33529</td>
      <td>0.10587</td>
      <td>-0.34108</td>
      <td>-0.04563</td>
    </tr>
    <tr>
      <th>host_listings_count_na</th>
      <td>-0.00830</td>
      <td>-0.0</td>
      <td>-0.00371</td>
      <td>-0.00000</td>
      <td>-0.00000</td>
      <td>0.00199</td>
      <td>-0.01261</td>
      <td>0.00519</td>
      <td>NaN</td>
      <td>-0.00089</td>
      <td>...</td>
      <td>-0.00772</td>
      <td>-0.00620</td>
      <td>0.03302</td>
      <td>-0.00873</td>
      <td>-0.01199</td>
      <td>-0.01912</td>
      <td>0.01561</td>
      <td>-0.00221</td>
      <td>-0.01444</td>
      <td>-0.00367</td>
    </tr>
    <tr>
      <th>host_total_listings_count_na</th>
      <td>-0.00830</td>
      <td>-0.0</td>
      <td>-0.00371</td>
      <td>-0.00000</td>
      <td>-0.00000</td>
      <td>0.00199</td>
      <td>-0.01261</td>
      <td>0.00519</td>
      <td>NaN</td>
      <td>-0.00089</td>
      <td>...</td>
      <td>-0.00772</td>
      <td>-0.00620</td>
      <td>0.03302</td>
      <td>-0.00873</td>
      <td>-0.01199</td>
      <td>-0.01912</td>
      <td>0.01561</td>
      <td>-0.00221</td>
      <td>-0.01444</td>
      <td>-0.00367</td>
    </tr>
    <tr>
      <th>bathrooms_na</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>bedrooms_na</th>
      <td>0.03343</td>
      <td>0.0</td>
      <td>0.03354</td>
      <td>0.01297</td>
      <td>0.01297</td>
      <td>0.05533</td>
      <td>-0.10992</td>
      <td>-0.05957</td>
      <td>NaN</td>
      <td>-0.00000</td>
      <td>...</td>
      <td>0.04985</td>
      <td>-0.00898</td>
      <td>-0.02418</td>
      <td>0.02494</td>
      <td>-0.00819</td>
      <td>0.02186</td>
      <td>0.20509</td>
      <td>0.03037</td>
      <td>-0.20010</td>
      <td>-0.04193</td>
    </tr>
    <tr>
      <th>beds_na</th>
      <td>0.13640</td>
      <td>-0.0</td>
      <td>0.09218</td>
      <td>-0.01032</td>
      <td>-0.01032</td>
      <td>0.02258</td>
      <td>0.00221</td>
      <td>-0.06916</td>
      <td>NaN</td>
      <td>-0.04571</td>
      <td>...</td>
      <td>1.00000</td>
      <td>0.04385</td>
      <td>-0.02452</td>
      <td>0.02050</td>
      <td>0.00105</td>
      <td>-0.00536</td>
      <td>-0.06572</td>
      <td>0.03615</td>
      <td>0.05451</td>
      <td>0.02490</td>
    </tr>
    <tr>
      <th>a few days or more</th>
      <td>0.01215</td>
      <td>0.0</td>
      <td>0.04055</td>
      <td>-0.03124</td>
      <td>-0.03124</td>
      <td>0.02052</td>
      <td>-0.01400</td>
      <td>0.01101</td>
      <td>NaN</td>
      <td>0.01969</td>
      <td>...</td>
      <td>0.04385</td>
      <td>1.00000</td>
      <td>-0.18787</td>
      <td>-0.06088</td>
      <td>-0.08364</td>
      <td>-0.13339</td>
      <td>-0.00872</td>
      <td>-0.01191</td>
      <td>0.00571</td>
      <td>0.01973</td>
    </tr>
    <tr>
      <th>unavailable</th>
      <td>-0.35410</td>
      <td>-0.0</td>
      <td>-0.24094</td>
      <td>-0.11686</td>
      <td>-0.11686</td>
      <td>0.01134</td>
      <td>-0.07471</td>
      <td>-0.11168</td>
      <td>NaN</td>
      <td>-0.09343</td>
      <td>...</td>
      <td>-0.02452</td>
      <td>-0.18787</td>
      <td>1.00000</td>
      <td>-0.26424</td>
      <td>-0.36305</td>
      <td>-0.57898</td>
      <td>-0.04946</td>
      <td>-0.03010</td>
      <td>0.05008</td>
      <td>0.01648</td>
    </tr>
    <tr>
      <th>within a day</th>
      <td>-0.01164</td>
      <td>-0.0</td>
      <td>-0.05562</td>
      <td>-0.03119</td>
      <td>-0.03119</td>
      <td>0.01410</td>
      <td>-0.03805</td>
      <td>0.01642</td>
      <td>NaN</td>
      <td>0.03512</td>
      <td>...</td>
      <td>0.02050</td>
      <td>-0.06088</td>
      <td>-0.26424</td>
      <td>1.00000</td>
      <td>-0.11764</td>
      <td>-0.18761</td>
      <td>0.06668</td>
      <td>0.00451</td>
      <td>-0.06601</td>
      <td>-0.00648</td>
    </tr>
    <tr>
      <th>within a few hours</th>
      <td>0.12780</td>
      <td>-0.0</td>
      <td>0.01844</td>
      <td>-0.01468</td>
      <td>-0.01468</td>
      <td>-0.00499</td>
      <td>0.03534</td>
      <td>-0.00382</td>
      <td>NaN</td>
      <td>0.01114</td>
      <td>...</td>
      <td>0.00105</td>
      <td>-0.08364</td>
      <td>-0.36305</td>
      <td>-0.11764</td>
      <td>1.00000</td>
      <td>-0.25777</td>
      <td>0.00195</td>
      <td>-0.01658</td>
      <td>0.00196</td>
      <td>-0.00597</td>
    </tr>
    <tr>
      <th>within an hour</th>
      <td>0.29187</td>
      <td>-0.0</td>
      <td>0.26491</td>
      <td>0.17132</td>
      <td>0.17132</td>
      <td>-0.02598</td>
      <td>0.08358</td>
      <td>0.11060</td>
      <td>NaN</td>
      <td>0.06432</td>
      <td>...</td>
      <td>-0.00536</td>
      <td>-0.13339</td>
      <td>-0.57898</td>
      <td>-0.18761</td>
      <td>-0.25777</td>
      <td>1.00000</td>
      <td>0.01693</td>
      <td>0.04812</td>
      <td>-0.01967</td>
      <td>-0.01831</td>
    </tr>
    <tr>
      <th>Entire home/apt</th>
      <td>-0.04284</td>
      <td>-0.0</td>
      <td>-0.12862</td>
      <td>0.01040</td>
      <td>0.01040</td>
      <td>-0.02656</td>
      <td>-0.14909</td>
      <td>0.45742</td>
      <td>NaN</td>
      <td>0.35604</td>
      <td>...</td>
      <td>-0.06572</td>
      <td>-0.00872</td>
      <td>-0.04946</td>
      <td>0.06668</td>
      <td>0.00195</td>
      <td>0.01693</td>
      <td>1.00000</td>
      <td>-0.07933</td>
      <td>-0.95966</td>
      <td>-0.13155</td>
    </tr>
    <tr>
      <th>Hotel room</th>
      <td>0.01698</td>
      <td>0.0</td>
      <td>0.07086</td>
      <td>-0.00877</td>
      <td>-0.00877</td>
      <td>0.02825</td>
      <td>-0.04860</td>
      <td>-0.01671</td>
      <td>NaN</td>
      <td>-0.02448</td>
      <td>...</td>
      <td>0.03615</td>
      <td>-0.01191</td>
      <td>-0.03010</td>
      <td>0.00451</td>
      <td>-0.01658</td>
      <td>0.04812</td>
      <td>-0.07933</td>
      <td>1.00000</td>
      <td>-0.06674</td>
      <td>-0.00915</td>
    </tr>
    <tr>
      <th>Private room</th>
      <td>0.03813</td>
      <td>0.0</td>
      <td>0.10957</td>
      <td>-0.00468</td>
      <td>-0.00468</td>
      <td>0.01830</td>
      <td>0.15128</td>
      <td>-0.44105</td>
      <td>NaN</td>
      <td>-0.33917</td>
      <td>...</td>
      <td>0.05451</td>
      <td>0.00571</td>
      <td>0.05008</td>
      <td>-0.06601</td>
      <td>0.00196</td>
      <td>-0.01967</td>
      <td>-0.95966</td>
      <td>-0.06674</td>
      <td>1.00000</td>
      <td>-0.11067</td>
    </tr>
    <tr>
      <th>Shared room</th>
      <td>0.00958</td>
      <td>0.0</td>
      <td>0.03676</td>
      <td>-0.01825</td>
      <td>-0.01825</td>
      <td>0.01707</td>
      <td>0.02280</td>
      <td>-0.06358</td>
      <td>NaN</td>
      <td>-0.05944</td>
      <td>...</td>
      <td>0.02490</td>
      <td>0.01973</td>
      <td>0.01648</td>
      <td>-0.00648</td>
      <td>-0.00597</td>
      <td>-0.01831</td>
      <td>-0.13155</td>
      <td>-0.00915</td>
      <td>-0.11067</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
<p>55 rows × 55 columns</p>
</div>



The result is a computed *correlation matrix*. The values on the diagonal are all equal to 1 because they represent the correlations between each column with itself. The matrix is symmetrical with respect to the diagonal.<br>

We only need to observe correlations of all features with the column `label_price` (as opposed to every possible pairwise correlation). Se let's query the `label_price` column of this matrix:

**Task**: Extract the `label_price` column of the correlation matrix and save the results to the variable `corrs`.


```python
corrs = corr_matrix['label_price']
corrs
```




    id                                              0.07907
    scrape_id                                      -0.00000
    host_id                                         0.04053
    host_listings_count                             0.13104
    host_total_listings_count                       0.13104
    latitude                                        0.04330
    longitude                                      -0.20695
    accommodates                                    0.50062
    bathrooms                                           NaN
    bedrooms                                        0.41996
    beds                                            0.37370
    price                                           0.71112
    minimum_nights                                 -0.07589
    maximum_nights                                 -0.00097
    minimum_minimum_nights                         -0.03804
    maximum_minimum_nights                          0.06554
    minimum_maximum_nights                          0.06582
    maximum_maximum_nights                          0.11169
    minimum_nights_avg_ntm                          0.06388
    maximum_nights_avg_ntm                          0.08210
    calendar_updated                                    NaN
    availability_30                                 0.14569
    availability_60                                 0.14701
    availability_90                                 0.14391
    availability_365                                0.12356
    number_of_reviews                              -0.04197
    number_of_reviews_ltm                           0.02757
    number_of_reviews_l30d                          0.02159
    review_scores_rating                            0.04320
    review_scores_accuracy                          0.00536
    review_scores_cleanliness                       0.08254
    review_scores_checkin                          -0.00367
    review_scores_communication                     0.00012
    review_scores_location                          0.09724
    review_scores_value                            -0.00482
    calculated_host_listings_count                 -0.01582
    calculated_host_listings_count_entire_homes     0.09509
    calculated_host_listings_count_private_rooms   -0.09978
    calculated_host_listings_count_shared_rooms    -0.04334
    reviews_per_month                               0.03114
    label_price                                     1.00000
    host_listings_count_na                          0.04450
    host_total_listings_count_na                    0.04450
    bathrooms_na                                        NaN
    bedrooms_na                                     0.02381
    beds_na                                        -0.03461
    a few days or more                              0.00792
    unavailable                                    -0.10279
    within a day                                    0.01335
    within a few hours                             -0.02111
    within an hour                                  0.11721
    Entire home/apt                                 0.33529
    Hotel room                                      0.10587
    Private room                                   -0.34108
    Shared room                                    -0.04563
    Name: label_price, dtype: float64



**Task**: Sort the values of the series we just obtained in the descending order and save the results to the variable `corrs_sorted`.


```python
corrs_sorted = corrs.sort_values(ascending=False)
corrs_sorted
```




    label_price                                     1.00000
    price                                           0.71112
    accommodates                                    0.50062
    bedrooms                                        0.41996
    beds                                            0.37370
    Entire home/apt                                 0.33529
    availability_60                                 0.14701
    availability_30                                 0.14569
    availability_90                                 0.14391
    host_total_listings_count                       0.13104
    host_listings_count                             0.13104
    availability_365                                0.12356
    within an hour                                  0.11721
    maximum_maximum_nights                          0.11169
    Hotel room                                      0.10587
    review_scores_location                          0.09724
    calculated_host_listings_count_entire_homes     0.09509
    review_scores_cleanliness                       0.08254
    maximum_nights_avg_ntm                          0.08210
    id                                              0.07907
    minimum_maximum_nights                          0.06582
    maximum_minimum_nights                          0.06554
    minimum_nights_avg_ntm                          0.06388
    host_listings_count_na                          0.04450
    host_total_listings_count_na                    0.04450
    latitude                                        0.04330
    review_scores_rating                            0.04320
    host_id                                         0.04053
    reviews_per_month                               0.03114
    number_of_reviews_ltm                           0.02757
    bedrooms_na                                     0.02381
    number_of_reviews_l30d                          0.02159
    within a day                                    0.01335
    a few days or more                              0.00792
    review_scores_accuracy                          0.00536
    review_scores_communication                     0.00012
    scrape_id                                      -0.00000
    maximum_nights                                 -0.00097
    review_scores_checkin                          -0.00367
    review_scores_value                            -0.00482
    calculated_host_listings_count                 -0.01582
    within a few hours                             -0.02111
    beds_na                                        -0.03461
    minimum_minimum_nights                         -0.03804
    number_of_reviews                              -0.04197
    calculated_host_listings_count_shared_rooms    -0.04334
    Shared room                                    -0.04563
    minimum_nights                                 -0.07589
    calculated_host_listings_count_private_rooms   -0.09978
    unavailable                                    -0.10279
    longitude                                      -0.20695
    Private room                                   -0.34108
    bathrooms                                           NaN
    calendar_updated                                    NaN
    bathrooms_na                                        NaN
    Name: label_price, dtype: float64



**Task**: Use Pandas indexing to extract the column names for the top two correlation values and save the results to the Python list `top_two_corr`. Add the feature names to the list in the order in which they appear in the output above. <br> 

<b>Note</b>: Do not count the correlation of `label` column with itself, nor the `price` column -- which is the `label` column prior to outlier removal.


```python
corrs_sorted = corrs_sorted.drop(['label', 'price'], errors='ignore')
top_two_corr = corrs_sorted.index[:2].tolist()
top_two_corr
```




    ['label_price', 'accommodates']



#### Bivariate Plotting: Produce Plots for the Label and Its Top Correlates

Let us visualize our data.

We will use the `pairplot()` function in `seaborn` to plot the relationships between the two features and the label.

**Task**: Create a DataFrame `df_corrs` that contains only three columns from DataFrame `df`: the label, and the two columns which correlate with it the most.


```python
df_corrs = df[['label_price'] + top_two_corr]
df_corrs
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
      <th>label_price</th>
      <th>label_price</th>
      <th>accommodates</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>150.0</td>
      <td>150.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>75.0</td>
      <td>75.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60.0</td>
      <td>60.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>275.0</td>
      <td>275.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>68.0</td>
      <td>68.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>38272</th>
      <td>79.0</td>
      <td>79.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>38273</th>
      <td>76.0</td>
      <td>76.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>38274</th>
      <td>116.0</td>
      <td>116.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>38275</th>
      <td>106.0</td>
      <td>106.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>38276</th>
      <td>689.0</td>
      <td>689.0</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
<p>38277 rows × 3 columns</p>
</div>



**Task**: Create a `seaborn` pairplot of the data subset you just created. Specify the *kernel density estimator* as the kind of the plot, and make sure that you don't plot redundant plots.

<i>Note</i>: It will take a few minutes to run and produce a plot.


```python
sns.pairplot(df_corrs, kind="kde", diag_kind="kde", corner=True)
sns.plt.show()
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-45-0f561a7cee96> in <module>()
    ----> 1 sns.pairplot(df_corrs, kind="kde", diag_kind="kde", corner=True)
          2 sns.plt.show()


    ~/.local/lib/python3.6/site-packages/seaborn/_decorators.py in inner_f(*args, **kwargs)
         44             )
         45         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 46         return f(**kwargs)
         47     return inner_f
         48 


    ~/.local/lib/python3.6/site-packages/seaborn/axisgrid.py in pairplot(data, hue, hue_order, palette, vars, x_vars, y_vars, kind, diag_kind, markers, height, aspect, corner, dropna, plot_kws, diag_kws, grid_kws, size)
       2096     grid = PairGrid(data, vars=vars, x_vars=x_vars, y_vars=y_vars, hue=hue,
       2097                     hue_order=hue_order, palette=palette, corner=corner,
    -> 2098                     height=height, aspect=aspect, dropna=dropna, **grid_kws)
       2099 
       2100     # Add the markers here as PairGrid has figured out how many levels of the


    ~/.local/lib/python3.6/site-packages/seaborn/_decorators.py in inner_f(*args, **kwargs)
         44             )
         45         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 46         return f(**kwargs)
         47     return inner_f
         48 


    ~/.local/lib/python3.6/site-packages/seaborn/axisgrid.py in __init__(self, data, hue, hue_order, palette, hue_kws, vars, x_vars, y_vars, corner, diag_sharey, height, aspect, layout_pad, despine, dropna, size)
       1210 
       1211         # Sort out the variables that define the grid
    -> 1212         numeric_cols = self._find_numeric_cols(data)
       1213         if hue in numeric_cols:
       1214             numeric_cols.remove(hue)


    ~/.local/lib/python3.6/site-packages/seaborn/axisgrid.py in _find_numeric_cols(self, data)
       1638         numeric_cols = []
       1639         for col in data:
    -> 1640             if variable_type(data[col]) == "numeric":
       1641                 numeric_cols.append(col)
       1642         return numeric_cols


    ~/.local/lib/python3.6/site-packages/seaborn/_core.py in variable_type(vector, boolean_type)
       1227 
       1228     # Special-case all-na data, which is always "numeric"
    -> 1229     if pd.isna(vector).all():
       1230         return "numeric"
       1231 


    ~/.local/lib/python3.6/site-packages/pandas/core/generic.py in __nonzero__(self)
       1328     def __nonzero__(self):
       1329         raise ValueError(
    -> 1330             f"The truth value of a {type(self).__name__} is ambiguous. "
       1331             "Use a.empty, a.bool(), a.item(), a.any() or a.all()."
       1332         )


    ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().


## Part 5: Analysis

1. Think about the possible interpretation of the plot. Recall that the label is the listing price. <br> How would you explain the relationship between the label and the two features? Is there a slight tilt to the points cluster, as the price goes up?<br>
2. Are the top two correlated features strongly or weakly correlated with the label? Are they features that should be used for our predictive machine learning problem?
3. Inspect your data matrix. It has a few features that contain unstructured text, meaning text data that is neither numerical nor categorical. List some features that contain unstructured text that you think are valuable for our predictive machine learning problem. Are there other remaining features that you think need to be prepared for the modeling phase? Do you have any suggestions on how to prepare these features?

Record your findings in the cell below.

<Double click this Markdown cell to make it editable, and record your findings here.>
