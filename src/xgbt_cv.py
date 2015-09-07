from __future__ import division

__author__ = 'Vladimir Iglovikov'

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import xgboost as xgb
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_squared_error
import math
import numpy as np

# load training and test datasets
train = pd.read_csv('../data/train_set.csv', parse_dates=[2, ])
test = pd.read_csv('../data/test_set.csv', parse_dates=[3, ])
tube_data = pd.read_csv('../data/tube.csv')
bill_of_materials_data = pd.read_csv('../data/bill_of_materials.csv')
specs_data = pd.read_csv('../data/specs.csv')

print("train columns")
print(train.columns)
print("test columns")
print(test.columns)
print("tube.csv df columns")
print(tube_data.columns)
print("bill_of_materials.csv df columns")
print(bill_of_materials_data.columns)
print("specs.csv df columns")
print(specs_data.columns)

print(specs_data[2:3])

train = pd.merge(train, tube_data, on='tube_assembly_id')
train = pd.merge(train, bill_of_materials_data, on='tube_assembly_id')
test = pd.merge(test, tube_data, on='tube_assembly_id')
test = pd.merge(test, bill_of_materials_data, on='tube_assembly_id')

print("new train columns")
print(train.columns)
print(train[1:10])
print(train.columns.to_series().groupby(train.dtypes).groups)

# create some new features
train['year'] = train.quote_date.dt.year
train['month'] = train.quote_date.dt.month
#train['dayofyear'] = train.quote_date.dt.dayofyear
#train['dayofweek'] = train.quote_date.dt.dayofweek
#train['day'] = train.quote_date.dt.day

test['year'] = test.quote_date.dt.year
test['month'] = test.quote_date.dt.month
#test['dayofyear'] = test.quote_date.dt.dayofyear
#test['dayofweek'] = test.quote_date.dt.dayofweek
#test['day'] = test.quote_date.dt.day

# drop useless columns and create labels
idx = test.id.values.astype(int)
test = test.drop(['id', 'tube_assembly_id', 'quote_date'], axis=1)
y = np.log1p(train.cost.values)

#'tube_assembly_id', 'supplier', 'bracket_pricing', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x'
#for some reason material_id cannot be converted to categorical variable
train = train.drop(['quote_date', 'cost', 'tube_assembly_id'], axis=1)

train['material_id'].replace(np.nan,' ', regex=True, inplace=True)
test['material_id'].replace(np.nan,' ', regex=True, inplace=True)

for i in range(1, 9):
    column_label = 'component_id_' + str(i)
    print(column_label)
    train[column_label].replace(np.nan, ' ', regex=True, inplace=True)
    test[column_label].replace(np.nan, ' ', regex=True, inplace=True)

train.fillna(0, inplace = True)
test.fillna(0, inplace = True)

print("train columns")
print(train.columns)

# convert data to numpy array
X = np.array(train)
X_test = np.array(test)


# label encode the categorical variables
for i in range(train.shape[1]):
    if i in [0, 3, 5, 11, 12, 13, 14, 15, 16, 20, 22, 24, 26, 28, 30, 32, 34]:
        print(i, list(X[1:5, i]) + list(X_test[1:5, i]))
        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X[:, i]) + list(X_test[:, i]))

        X[:, i] = lbl.transform(X[:, i])
        X_test[:, i] = lbl.transform(X_test[:, i])


# object array to float
X = X.astype(float)
X_test = X_test.astype(float)

params = {
  'objective': 'reg:linear',
  # 'objective': 'count:poisson',
  # 'eta': 0.005,
  # 'min_child_weight': 6,
  # 'subsample': 0.7,
  # 'colsabsample_bytree': 0.7,
  # 'scal_pos_weight': 1,
  'silent': 1,
  # 'max_depth': 9
  # "max_delta_step": 2
}

num_rounds = 10000
random_state = 42
offset = 5000
test_size = 0.2

ind = 1
if ind == 1:
  n_iter = 5
  rs = ShuffleSplit(len(y), n_iter=n_iter, test_size=test_size, random_state=random_state)

  result = []

  for scale_pos_weight in [1]:
    for min_child_weight in [5]:
      for eta in [0.1]:
        for colsample_bytree in [0.6]:
          for max_depth in [7, 8, 9]:
            for subsample in [1]:
              for gamma in [1]:
                params['min_child_weight'] = min_child_weight
                params['eta'] = eta
                params['colsample_bytree'] = colsample_bytree
                params['max_depth'] = max_depth
                params['subsample'] = subsample
                params['gamma'] = gamma
                params['scale_pos_weight'] = scale_pos_weight

                params_new = list(params.items())
                score = []

                for train_index, test_index in rs:

                  X_train = X[train_index]
                  X_test = X[test_index]
                  y_train = y[train_index]
                  y_test = y[test_index]

                  xgtest = xgb.DMatrix(X_test)

                  xgtrain = xgb.DMatrix(X_train[offset:, :], label=y_train[offset:])
                  xgval = xgb.DMatrix(X_train[:offset, :], label=y_train[:offset])

                  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

                  model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

                  preds1 = model.predict(xgtest, ntree_limit=model.best_iteration)

                  # X_train = X_train[::-1, :]
                  # labels = y_train[::-1]
                  #
                  # xgtrain = xgb.DMatrix(X_train[offset:, :], label=labels[offset:])
                  #
                  # xgval = xgb.DMatrix(X_train[:offset, :], label=labels[:offset])
                  #
                  # watchlist = [(xgtrain, 'train'), (xgval, 'val')]
                  #
                  # model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

                  # preds2 = model.predict(xgtest, ntree_limit=model.best_iteration)

                  # preds = 0.5 * preds1 + 0.5 * preds2
                  preds = preds1

                  score += [math.sqrt(mean_squared_error(y_test, preds))]
                result += [(math.ceil(100000 * np.mean(score)) / 100000,
                            math.ceil(100000 * np.std(score)) / 100000,
                            min_child_weight,
                            eta,
                            colsample_bytree,
                            max_depth, subsample, gamma, n_iter, params['objective'],
                            test_size, scale_pos_weight)]


    result.sort()
    print
    print('result')
    print(result)


elif ind == 2:
  # y = np.log1p(y)

  xgtrain = xgb.DMatrix(X[offset:, :], label=y[offset:])
  xgval = xgb.DMatrix(X[:offset, :], label=y[:offset])
  xgtest = xgb.DMatrix(X_test)

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  params = {
  'objective': 'reg:linear',
  #   'objective': 'count:poisson',
  'eta': 0.005,
  'min_child_weight': 6,
  'subsample': 0.6,
  'colsample_bytree': 0.6,
  # 'scal_pos_weight': 1,
  'silent': 1,
  'max_depth': 8,
  'gamma': 1
  }
  params_new = list(params.items())
  model1 = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
  prediction_test_1 = model1.predict(xgtest, ntree_limit=model1.best_iteration)

  X_train = X[::-1, :]
  labels = y[::-1]

  xgtrain = xgb.DMatrix(X_train[offset:, :], label=labels[offset:])
  xgval = xgb.DMatrix(X_train[:offset, :], label=labels[:offset])

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  model2 = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

  prediction_test_2 = model2.predict(xgtest, ntree_limit=model2.best_iteration)

  prediction_test = 0.5 * np.expm1(prediction_test_1) + 0.5 * np.expm1(prediction_test_2)

  submission = pd.DataFrame()
  submission['id'] = idx
  submission['cost'] = prediction_test
  submission.to_csv("predictions/xgbt_script.csv", index=False)

#
# # i like to train on log(1+x) for RMSLE ;)
# # The choice is yours :)
# label_log = np.log1p(labels)
#
# plst = list(params.items())
#
# xgtrain = xgb.DMatrix(train, label=label_log)
# xgtest = xgb.DMatrix(test)
#
# print('1500')
#
#
# num_rounds = 1500
# model = xgb.train(plst, xgtrain, num_rounds)
# preds1 = model.predict(xgtest)
#
# print('3000')
#
# num_rounds = 3000
# model = xgb.train(plst, xgtrain, num_rounds)
# preds2 = model.predict(xgtest)
#
# print('4000')
#
# num_rounds = 4000
# model = xgb.train(plst, xgtrain, num_rounds)
# preds4 = model.predict(xgtest)
#
# label_log = np.power(labels,1.0/16.0)
#
# xgtrain = xgb.DMatrix(train, label=label_log)
# xgtest = xgb.DMatrix(test)
#
# print('power 1/16 4000')
#
# num_rounds = 4000
# model = xgb.train(plst, xgtrain, num_rounds)
# preds3 = model.predict(xgtest)
#
# preds = 0.4*np.expm1(preds4)+.1*np.expm1(preds1)+0.1*np.expm1(preds2)+0.4*np.power(preds3,16)
#
#
# preds = pd.DataFrame({"id": idx, "cost": preds})
# preds.to_csv('benchmark.csv', index=False)