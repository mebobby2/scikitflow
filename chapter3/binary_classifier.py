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

print("Use K-fold cross-validation to evaluate model (3 folds)")
print("")

from sklearn.model_selection import cross_val_score
results = cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring="accuracy")
print(results) # [0.9515  0.9646  0.96705]
print("WOW!, above 95% accuracy (ratio of correct predictions) on all 3 folds!")
print("Seems to good to be true. Upon looking at the data, we find that about 10%")
print("of the images are 5s, so if the algorithm always guess that an image is not a 5,")
print("it will be right about 90% of the time. This demostrates why accurracy is generally")
print("not the preferred performance measure for classifiers, esp when you are dealing")
print("with skewed datasets (ie when some classes are much more frequent than others.")


print("A better way: confusion matrix")

print("Generate the predictions first in order to set up the matrix")
print("We use sklearn's cross_val_predict method. Like cross_val_score, it also performs")
print("k-fold cross validation but instead of returning the evaluation scores, it returns")
print("the predictions made on each test fold.")
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix
# Each row represents the actual class, and each column represents the predicted class
results = confusion_matrix(y_train_5, y_train_pred)
print(results)
print("The confusion matrix gives you ALOT of information")

print("A more concise metric is the precision of the classifier, ie accuracy of position predictions")
print("The prediction, and another metric, the recall, can be inferred from the confusion matrix")
from sklearn.metrics import precision_score, recall_score
print("Precision:")
print(precision_score(y_train_5, y_train_pred))

print("Recall:")
print(recall_score(y_train_5, y_train_pred))

print("F1 score: harmonic mean precision and recall")
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

print("Increasing Precision reduces recall, and vice versa. You can't have it both ways.")
print("This is called the precision/recall tradeoff.")
