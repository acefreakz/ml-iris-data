# 1. Prepare Problem
# a) Load libraries
import numpy
from matplotlib import pyplot
from pandas import read_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from numpy import set_printoptions
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import scipy


def evaluate_models_accuracy_cv(models, X, y=None, n_splits=10, seed=None):
    for name, model in models:
        kfold = KFold(n_splits=n_splits, random_state=seed)
        cv_results = cross_val_score(model, X, y=y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


# b) Load dataset
filename = 'iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names)
# print(dataset.shape)
print(dataset.head(10))

# 2. Summarize Data
# a) Descriptive statistics
# print(dataset.describe())
# print(dataset.groupby('class').size())
# b) Data visualizations
# -- box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# -- line plots
# dataset.plot(kind='line', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# -- density plots
# dataset.plot(kind='density', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()
# -- histogramss
# dataset.hist()
# pyplot.show()
# -- scatter plot matrix
# scatter_matrix(dataset)
# pyplot.show()
# -- plot correlation matrix
# correlations = dataset.corr()
# fig = pyplot.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(correlations, vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = numpy.arange(0,4,1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(names)
# ax.set_yticklabels(names)
# pyplot.show()

# 3. Prepare Data
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
# a) Data Cleaning
# b) Feature Selection
# SelectKBest
# test = SelectKBest(score_func=chi2, k=3)
# fit = test.fit(X, Y)
# set_printoptions(precision=3)
# print(fit.scores_)
# X = fit.transform(X)
# print(X[0:3, :]) # Notice the lowest score's feature is eliminated (sepal-width)
# RFE
# model = LogisticRegression()
# rfe = RFE(model, 3)
# fit = rfe.fit(X, Y)
# print("Num Features: %d") % fit.n_features_
# print("Selected Features: %s") % fit.support_
# print("Feature Ranking: %s") % fit.ranking_
# PCA
# pca = PCA(n_components=3)
# fit = pca.fit(X)
# print("Explained Variance: %s") % fit.explained_variance_ratio_
# print(fit.components_)
# c) Data Transforms

# 4. Evaluate Algorithms
# a) Split-out validation dataset
# Split-out validation dataset
validation_size = 0.33
seed = 7

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
print Y_validation
# b) Test options and evaluation metric
# c) Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
n_splits = 10
scoring='accuracy'
print 'Evaluating algorithms against training data'
evaluate_models_accuracy_cv(models, X_train, y=Y_train, n_splits=n_splits, seed=seed)
print 'Evaluating algorithms against validation data'
evaluate_models_accuracy_cv(models, X_validation, y=Y_validation, n_splits=n_splits, seed=seed)

# d) Compare Algorithms
# fig = pyplot.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# pyplot.boxplot(results)
# ax.set_xticklabels(names)
# pyplot.show()

print 'Evaluating tuned algorithms against training data'
models = []
models.append(('KNN(tuned)', KNeighborsClassifier(n_neighbors=23, weights='uniform')))
models.append(('SVM(tuned)', SVC(kernel='sigmoid', C=1000, gamma=0.001)))
evaluate_models_accuracy_cv(models, X_train, y=Y_train, n_splits=n_splits, seed=seed)

print 'Evaluating tuned algorithms against validation data'
models = []
models.append(('KNN(tuned)', KNeighborsClassifier(n_neighbors=23, weights='uniform')))
models.append(('SVM(tuned)', SVC(kernel='sigmoid', C=1000, gamma=0.001)))
evaluate_models_accuracy_cv(models, X_validation, y=Y_validation, n_splits=n_splits, seed=seed)

# 5. Improve Accuracy
# a) Algorithm Tuning
# -- KNN Algorithm tuning (GRID)
param_grid = [{'n_neighbors': list(range(2, 30)), 'weights': ['uniform', 'distance']}]
model = KNeighborsClassifier()
kfold = KFold(n_splits=n_splits, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train, Y_train)
print("Tuning KNN with GridSearch. Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

# -- KNN Algorithm tuning (RAND)
# param_grid = { 'n_neighbors': list(range(2, 30)), 'weights': ['uniform', 'distance'] }
# model = KNeighborsClassifier()
# kfold = KFold(n_splits=n_splits, random_state=seed)
# rand = RandomizedSearchCV(estimator=model, n_iter=56, param_distributions=param_grid, scoring=scoring, cv=kfold, random_state=seed)
# rand_result = rand.fit(X_train, Y_train)
# print("Best: %f using %s" % (rand_result.best_score_, rand_result.best_params_))
# means = rand_result.cv_results_['mean_test_score']
# stds = rand_result.cv_results_['std_test_score']
# params = rand_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

# -- SVC Algorithm tuning (GRID)
# param_grid = [
#   {'C': [10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['linear', 'sigmoid', 'rbf']},
#  ]
# model = SVC(random_state=seed)
# kfold = KFold(n_splits=n_splits, random_state=seed)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
# grid_result = grid.fit(X_train, Y_train)
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

# -- SVC Algorithm tuning (RAND)
# param_grid = { 'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.001), 'kernel': ['rbf', 'sigmoid'] }
# model = SVC(random_state=seed)
# kfold = KFold(n_splits=n_splits, random_state=seed)
# rand = RandomizedSearchCV(estimator=model, n_iter=30, param_distributions=param_grid, scoring=scoring, cv=kfold, random_state=seed)
# rand_result = rand.fit(X_train, Y_train)
# print("Best: %f using %s" % (rand_result.best_score_, rand_result.best_params_))
# means = rand_result.cv_results_['mean_test_score']
# stds = rand_result.cv_results_['std_test_score']
# params = rand_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

# knn = KNeighborsClassifier(n_neighbors=23, weights='uniform')
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
#
# svc = SVC(kernel='sigmoid', C=1000, gamma=0.001)
# svc.fit(X_train, Y_train)
# predictions = svc.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))

# b) Ensembles

# 6. Finalize Model
# a) Predictions on validation dataset
# b) Create standalone model on entire training dataset
# c) Save model for later use
