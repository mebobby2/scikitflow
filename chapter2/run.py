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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

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


print("### 7. Select a model")
print("Linear Regression")

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse) # rmse = root mean square error

# 68628.19819848922. This means a typical prediction error of $68,628.
# This is not very good considering the most median_housing_values are
# between $120,000 and $265,000. This is an exampe of underfitting the
# training data.
# This usually means the features do not provide enough info to make
# good predictions, or the model is not powerful enough.
# To fix:
# 1. Select a more powerful model
# 2. Feed algo better features (e.g. the log of the population)
# 3. Reduce the constraints on the model
# Linear regression is unregularized, so option 3 is out.
print("lin_rmse = ", lin_rmse) # 68628.19819848922

print("Pick a more powerful model: Decision Tree Regression")
# Decision tree regression is capable of finding complex nonlinear
# relationships in data
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

# 0. No error rate? Sounds to good to be true. It's much more likely
# that the model has badly overfitted the data.
print("tree_rmse = ", tree_rmse) # 0

print("Lets use better techniques to evaluate the decision tree regression model")
print("Cross-Validation")
print("Decision Tree")

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(rmse_scores)

print("Linear Regression")
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

print("Seems like the Decision Tree is overfitting so badly that it performs worst than the Regression model")

print("Let's try Random Forest Regression")

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

print("Wow - looks like Random Forest Regression yields the best results")

print("### 8. Fine tune the model")

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error")

grid_search.fit(housing_prepared, housing_labels)

print("The best set of hyperparamters for Random Forest Regression is", grid_search.best_params_)
print("And the evaluation scores:")
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

print("### 9. Analyze the Best Models and Their Errors")
feature_importances = grid_search.best_estimator_.feature_importances_

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
label_binarizer = cat_pipeline.named_steps["label_binarizer"]
cat_one_hot_attribs = list(label_binarizer.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))

print("### 10. Evaluate system on the Test Set")

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
print("FINAL RMSE:", np.sqrt(final_mse))
