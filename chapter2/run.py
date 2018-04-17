from housing import *
from combined_attributes_adder import CombinedAttributesAdder
from data_frame_selector import DataFrameSelector
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

print("### 1. Load the data")
housing = load_housing_data()

print("### 2. Take quick look at the data to understand the nature of it")
# print(housing.info())
print(housing.describe())
print(housing.head())  # look at the first 5 rows

# histograms offers a quick way to see how your data is distributed
# so you can use stratified sampling to obtain your training and
# test sets.
housing.hist(bins=50, figsize=(20, 15))
save_fig("1-attribute_histogram_plots")
# plt.show()

print("### 3. Split data into training and test set using Stratified Sampling")
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(housing["income_cat"].value_counts() / len(housing))

# Cleanup: remove the income_cat attribute
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

print("### 4. Visual data to discover and gain insights")

housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude")
save_fig("2-bad_visualization_plot")

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("3-better_visualization_plot")

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
              s=housing["population"]/100, label="population",
              c="median_house_value", cmap=plt.get_cmap("jet"),
              colorbar=True)
plt.legend()
save_fig("4-housing_prices_scatterplot")

print("Fig 4 tells us that housing prices are related to location")
print(" (e.g., close to the ocean), and to the population density.")
print("We can also look for Correlations, using standard correlation coefficiient")

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

print("Correlations tells is that house value goes up when median income goes up")
print("Now lets plot the numerical attributes that seem correlated against one another")

attributes = ["median_house_value", "median_income",
              "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("5-scatter_matrix_plot")

print("Zoom into fig5 for median-house-value and median-income scatter plot")
print("There is indeed a strong correlation as there is an upwards trend")
print("and the data points are not too dispersed.")
print("We have also identified some data quirks ie straight horizontal lines")
print("at 450k and 350k")

print("### 5. Experimenting with attribute combinations")

housing["rooms_per_household"] = housing["total_rooms"] / \
    housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / \
    housing["total_rooms"]
housing["population_per_household"] = housing["population"] / \
    housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

print("### 6. Preparig data for ML algorithms")
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

print("6a. Handling missing features")
# housing.dropna(subset=["total_bedrooms"]) # get rid of rows with missing total_bedrooms
# housing.drop("total_bedrooms", axis=1) # Get rid of the whole attribute
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median) # Set missing values to median

# Instead of above - use the Imputer class
housing_num = housing.drop("ocean_proximity", axis=1)
# imputer = Imputer(strategy="median")
# imputer.fit(housing_num)
# X = imputer.transform(housing_num)
# housing_tr = pd.DataFrame(X, columns=housing_num.columns)

print("6a. Converting Text and Categorical Attributes to numbers")
# housing_cat = housing["ocean_proximity"]
# encoder = LabelEncoder()
# housing_cat_encoded = encoder.fit_transform(housing_cat)
# encoder = OneHotEncoder()
# housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))

# Instead of above, the LabelBinarizer combines the LabelEncoder and the OneHotEncoder
# encoder = LabelBinarizer(sparse_output=True)
# housing_cat_1hot = encoder.fit_transform(housing_cat)
# print(housing_cat_1hot)

print("6a. Add custom transformers to get combined features")
# add_bedrooms_per_room is a hyperparameter
# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
# housing_extra_attribs = attr_adder.transform(housing.values)
# print(housing_extra_attribs)

print("6a. Feature scaling")
# Machine learning algos do not perform well when the attributes
# have different scales. E.g. total rooms range from 6 to 39320
# where as median incomr is 0 to 15.

# Instead of all the manual steps to Prepare the data, we will
# use a data Pipeline
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaller', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', LabelBinarizer()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing_prepared.shape)

# return housing
