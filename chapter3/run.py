from sklearn.datasets import fetch_mldata

print("Downloading the MNIST data")
print("")

mnist = fetch_mldata('MNIST original')

print("Each instance of the dataset has 784 features. This is because each image is 28x28 pixels, and each feature represents one pixel's intensity, from 0 (white) to 255 (black)")
print("")

print("Build an image: grab an instance's feature vector, reshape it to 28x28 array and display it using matplotlib")
print("")
import matplotlib
import matplotlib.pyplot as plt

x, y = mnist["data"], mnist["target"]
some_digit = x[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

print("Split data into training set and test set")
print("")

x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]


print("Shuffle the training set; to ensure cross-validation folds will be similar (you don't want one fold to be missing some digits)")
print("Moreover, some algorithsm are sensitive to the order of the training instances, and perform poorly if the get many similar instances in a row")
print("")
import numpy as np
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
