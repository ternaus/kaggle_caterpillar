from __future__ import division
__author__ = 'Vladimir Iglovikov'

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_squared_error
import math

joined = pd.read_csv('data/joined_simple.csv')

#add flag if ends are of the same type

def flag(x):
  if x['end_a'] == x['end_x']:
    return 1
  else:
    return 0

joined['end_flag'] = joined.apply(lambda x: flag(x), 1)

#rename forming a and x to 1 and 0
joined['forming_a'] = joined['forming_a'].map({'Yes': 1, 'No': 0})

#use label encoder on ends

le = LabelEncoder()
le.fit(np.hstack([joined['end_a'].values, joined['end_x'].values]))

joined['end_x'] = le.transform(joined['end_x'].values)
joined['end_a'] = le.transform(joined['end_a'].values)

train = joined[joined['id'] == -1]
test = joined[joined['cost'] == -1]

y = train['cost'].apply(lambda x: math.log(x + 1), 1)

X = train.drop(['id', 'spec', 'tube_assembly_id', 'quote_date'], 1)

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
}

num_rounds = 10000
random_state = 42
offset = 5000

ind = 1
if ind == 1:
  n_iter = 10
  rs = ShuffleSplit(len(y), n_iter=n_iter, test_size=0.1, random_state=random_state)

  result = []
  # result_truncated_up = []
  # result_truncated_down = []
  result_truncated_both = []
  # result_truncated_both_round = []
  # result_truncated_both_int = []


  for min_child_weight in [3]:
    for eta in [0.01]:
      for colsample_bytree in [0.5]:
        for max_depth in [7]:
          for subsample in [0.7]:
            for gamma in [1]:
              params['min_child_weight'] = min_child_weight
              params['eta'] = eta
              params['colsample_bytree'] = colsample_bytree
              params['max_depth'] = max_depth
              params['subsample'] = subsample
              params['gamma'] = gamma

              params_new = list(params.items())
              score = []
              # score_truncated_up = []
              # score_truncated_down = []
              score_truncated_both = []
              # score_truncated_both_round = []
              # score_truncated_both_int = []

              for train_index, test_index in rs:

                X_train = X.values[train_index]
                X_test = X.values[test_index]
                y_train = y.values[train_index]
                y_test = y.values[test_index]

                xgtest = xgb.DMatrix(X_test)

                xgtrain = xgb.DMatrix(X_train[offset:, :], label=y_train[offset:])
                xgval = xgb.DMatrix(X_train[:offset, :], label=y_train[:offset])

                watchlist = [(xgtrain, 'train'), (xgval, 'val')]

                model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

                preds1 = model.predict(xgtest, ntree_limit=model.best_iteration)

                X_train = X_train[::-1, :]
                labels = y_train[::-1]

                xgtrain = xgb.DMatrix(X_train[offset:, :], label=labels[offset:])
                xgval = xgb.DMatrix(X_train[:offset, :], label=labels[:offset])

                watchlist = [(xgtrain, 'train'), (xgval, 'val')]

                model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

                preds2 = model.predict(xgtest, ntree_limit=model.best_iteration)

                # preds = model.predict(xgval, ntree_limit=model.best_iteration)

                preds = 0.5 * preds1 + 0.5 * preds2

                tp = mean_squared_error(y_test, preds)

                score += [tp]

                print tp

              sc = math.ceil(10000 * np.mean(score)) / 10000
              sc_std = math.ceil(10000 * np.std(score)) / 10000
              result += [(sc, sc_std, min_child_weight, eta, colsample_bytree, max_depth, subsample, gamma, n_iter, params['objective'])]

  result.sort()

  print
  print 'result'
  print result

elif ind == 2:
  xgtrain = xgb.DMatrix(X.values[offset:, :], label=y.values[offset:])
  xgval = xgb.DMatrix(X.values[:offset, :], label=y.values[:offset])
  xghold = xgb.DMatrix(X_hold.values)
  xgtest = xgb.DMatrix(X_test.values)

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  params = {
  # 'objective': 'reg:linear',
    'objective': 'count:poisson',
  'eta': 0.005,
  'min_child_weight': 3,
  'subsample': 0.7,
  'colsample_bytree': 0.5,
  # 'scal_pos_weight': 1,
  'silent': 1,
  'max_depth': 7,
  'gamma': 1
  }
  params_new = list(params.items())
  model1 = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
  prediction_hold_1 = model1.predict(xghold, ntree_limit=model1.best_iteration)
  prediction_test_1 = model1.predict(xgtest, ntree_limit=model1.best_iteration)

  X_train = X.values[::-1, :]
  labels = y.values[::-1]

  xgtrain = xgb.DMatrix(X_train[offset:, :], label=labels[offset:])
  xgval = xgb.DMatrix(X_train[:offset, :], label=labels[:offset])

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  model2 = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

  prediction_hold_2 = model2.predict(xghold, ntree_limit=model2.best_iteration)
  prediction_test_2 = model2.predict(xgtest, ntree_limit=model2.best_iteration)

  prediction_hold = 0.5 * prediction_hold_1 + 0.5 * prediction_hold_2

  submission = pd.DataFrame()
  submission['Id'] = hold['Id']
  submission['Hazard'] = prediction_hold
  submission.to_csv("preds_on_hold/xgbt.csv", index=False)

  prediction_test = 0.5 * prediction_test_1 + 0.5 * prediction_test_2
  submission = pd.DataFrame()
  submission['Id'] = test['Id']
  submission['Hazard'] = prediction_test
  submission.to_csv("preds_on_test/xgbt.csv", index=False)
