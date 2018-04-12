import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

IMAGES_PATH = os.path.join(".", "images")
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("..", "datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# train_set, test_set = split_train_test(housing, 0.2)
# print(len(train_set), "train +", len(test_set), "test")

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

# housing_with_id = housing.reset_index() # adds an 'index' column
# housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

# Or use the sklearn provided ones to split data
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

def run():
    print("### 1. Load the data")
    housing = load_housing_data()

    print("### 2. Take quick look at the data to understand the nature of it")
    # print(housing.info())
    print(housing.describe())
    print(housing.head()) # look at the first 5 rows

    # histograms offers a quick way to see how your data is distributed
    # so you can use stratified sampling to obtain your training and
    # test sets.
    housing.hist(bins=50, figsize=(20,15))
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

    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    save_fig("5-scatter_matrix_plot")

    print("Zoom into fig5 for median-house-value and median-income scatter plot")
    print("There is indeed a strong correlation as there is an upwards trend")
    print("and the data points are not too dispersed.")
    print("We have also identified some data quirks ie straight horizontal lines")
    print("at 450k and 350k")

    print("### 5. Experimenting with attribute combinations")

    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"] = housing["population"]/housing["households"]

    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    # return housing
