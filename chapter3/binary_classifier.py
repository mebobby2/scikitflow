from run import *

print("Use Binary Classifier to detect the number 5")
print("")

y_train_5 = (y_train == 5) # True for all 5s, False for all other digits
y_test_5 = (y_test == 5)

print("Lets use Stochastic Gradient Descent (SGD) classifier")
print("Efficient in handling large dataset because it deals with training instances independently, one at a time (which makes SGD well suited for online learning")
print("")

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)

# Play:
# test_digit = x_test[4]
# sgd_clf.predict(test_digit)
# Verify using the label: y_test[4]
