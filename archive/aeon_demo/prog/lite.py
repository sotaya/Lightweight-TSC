from aeon.datasets import load_classification
from aeon.classification.deep_learning import LITETimeClassifier
xtrain, ytrain = load_classification(name="Coffee", split="train")
xtest, ytest = load_classification(name="Coffee", split="test")
# using LITE
clf = LITETimeClassifier(n_classifiers=1, n_epochs=10)
clf.fit(xtrain, ytrain)
print(clf.score(xtest, ytest))
# using LITEMV
clf = LITETimeClassifier(n_classifiers=1, n_epochs=10, use_litemv=True)
clf.fit(xtrain, ytrain)
print(clf.score(xtest, ytest))