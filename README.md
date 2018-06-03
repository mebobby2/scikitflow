# Machine Learning with Scikit-Learn and Tensorflow

# Requirements
## python3
Comes installed on OSX. To run a .py file, use *python3 file.py*.

To run a function inside the REPL:
1. Start REPL ```python3```
2. ```from housing import *```
3. ```fetch_housing_data()```

4. If the file you're loading is just a python script without any functions, ```from housing import *``` will run it, and you can access variables set up by the script in side the REPL

To run a script from CLI
1. cd into script folder (so all the relative paths, if any, the script uses will work)
2. ```python3 run.py```

## pip3
Comes installed on OSX. To install a library, use *pip3 install library-name*.

# Machine Learning Concepts
## Feature scaling
Machine learning algorithms do not perform well when the attributes have different scales. Hence, we need to get all the attibutes to have the same scale.

There are two common ways to do this: *min-max* scaling and *standardization*.

Min-max bounds values to the range of 0-1 whereas Standardization does not bound values to a specific range. Standardization is much less affected by outliers as compared to Min-max.

# Statistic Concepts
## Stratified Sampling
Random sampling methods to choose your training and test set is fine for large datasets, especially relative to the number of attributes, but it can run the risk of introducing a significant sampling bias. For example, if the population is composed of 51.3% males and 48.7% females, when selecting the test set of 1000 people, you want to make sure your sample maintains this ratio as well. This is called Stratified Sampling.

## Root Mean Squared error
Is a common metric used to measure accuracy for continuous variables. That is, it can be used to measure the accuracy of a linear regression model. RMSE is a quadratic scoring rule that measures the average magnitude of the error. It’s the square root of the average of squared differences between prediction and actual observation.

# Basic Maths Concepts

## Standard Deviation
The Standard Deviation is a measure of how spread out numbers are

1. Work out the Mean (the simple average of the numbers)
2. Then for each number: subtract the Mean and square the result
3. Then work out the mean of those squared differences.
4. Take the square root of that and we are done!

## Percentiles
A percentile indicates the value below which a given percentage of observations in a group of observations falls. These are often called the 25th percentile (or 1st quartile), the median, and the 75th percentile (or 3rd qaurtile). E.g. If an attribute's 25th percentile is 18, this means 25% of the attribute's values are lower than 18.

## Nominal
Nominal numbers or categorical numbers are numeric codes, meaning numerals used for labelling or identification only. The values of the numerals are irrelevant, and they do not indicate quantity, rank, or any other measurement.

This is opposite of an ordinal number: a number denoting relative position in a sequence, such as first, second, third.

## Upto

Page 120

Performance Measures
