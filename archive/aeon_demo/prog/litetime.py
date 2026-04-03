from aeon.datasets import load_classification
from aeon.classification.deep_learning import LITETimeClassifier
xtrain, ytrain, _ = load_classification(name="Coffee", split="train", return_metadata=True)
xtest, ytest, _ = load_classification(name="Coffee", split="test", return_metadata=True)
# using LITETime
clf = LITETimeClassifier(n_classifiers=5, n_epochs=10)
clf.fit(xtrain, ytrain)
print(clf.score(xtest, ytest))
# using LITEMVTime for multivariate
clf = LITETimeClassifier(n_classifiers=5, n_epochs=10, use_litemv=True)
clf.fit(xtrain, ytrain)
print(clf.score(xtest, ytest))