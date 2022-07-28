import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

data = pd.read_csv("dataset.csv")
a = data.values[:, 0:56]
b = data.values[:, 57]

# Decicion Tree
dec_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
aTrain, aTest, bTrain, bTest = train_test_split(a, b, test_size=0.25)
dec_tree = dec_tree.fit(aTrain, bTrain)
predict = dec_tree.predict(aTest)
print("Decicion_Tree accuracy:", accuracy_score(bTest, predict))

crossVal = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
crossValidationPoints = cross_val_score(dec_tree, a, b, cv=crossVal)
print("Decicion_Tree Accuracy Test Ten Fold:", crossValidationPoints)
print("Decicion_Tree Accuracy Test Ten Fold avg:", crossValidationPoints.mean())

# knn
knn = KNeighborsClassifier(n_neighbors=4)
knn = knn.fit(aTrain, bTrain)

predict = knn.predict(aTest)
print("\nKNeighbours accuracy:", accuracy_score(bTest, predict))

crossVal = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
crossValidationPoints = cross_val_score(knn, a, b, cv=crossVal)
print("KNeighbours Accuracy Test Ten Fold:", crossValidationPoints)
print("KNeighbours Accuracy Test Ten Fold avg:", crossValidationPoints.mean())

# stochastic gradient descent

stochastic = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
stochastic.fit(aTrain, bTrain)

predict = stochastic.predict(aTest)
print("\nStochastic accuracy:", accuracy_score(bTest, predict))

crossVal = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
crossValidationPoints = cross_val_score(stochastic, a, b, cv=crossVal)
print("Stochastic Accuracy Test Ten Fold:", crossValidationPoints)
print("Stochastic Accuracy Test Ten Fold avg:", crossValidationPoints.mean())

# ensemble
warnings.simplefilter('ignore')

lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[dec_tree, knn, stochastic],
                          meta_classifier=lr)

print('10-fold cross validation:\n')

for clf, label in zip([dec_tree, knn, stochastic, sclf],
                      ['KNN',
                       'Decision Tree',
                       'Stochastic',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, a, b,
                                              cv=10, scoring='accuracy')
    print("\nAccuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))

    scores = model_selection.cross_val_score(clf, a, b,
                                              cv=10, scoring='precision')
    print("\nPrecision: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))