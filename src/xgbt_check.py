from __future__ import division
__author__ = 'Vladimir Iglovikov'

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_squared_error
import math

joined = pd.read_csv('../data/joined_simple.csv', parse_dates=['quote_date'])

#add flag if ends are of the same type

def flag(x):
  if x['end_a'] == x['end_x']:
    return 1
  else:
    return 0

joined['end_flag'] = joined.apply(lambda x: flag(x), 1)

#rename forming a and x to 1 and 0
joined['forming_a'] = joined['forming_a'].map({'Yes': 1, 'No': 0})
joined['forming_x'] = joined['forming_x'].map({'Yes': 1, 'No': 0})

#use label encoder on ends
le = LabelEncoder()
le.fit(np.hstack([joined['end_a'].values, joined['end_x'].values]))

joined['end_x'] = le.transform(joined['end_x'].values)
joined['end_a'] = le.transform(joined['end_a'].values)

#use label encoder on supplier
le = LabelEncoder()
joined['supplier'] = le.fit_transform(joined['supplier'].values)

#map bracket pricing to Yes=1, No=0
joined['bracket_pricing'] = joined['bracket_pricing'].map({'Yes': 1, 'No': 0})


#add datetime features
joined['year'] = joined['quote_date'].dt.year
joined['month'] = joined['quote_date'].dt.month
joined['dayofyear'] = joined['quote_date'].dt.dayofyear
joined['dayofweek'] = joined['quote_date'].dt.dayofweek
joined['day'] = joined['quote_date'].dt.day


train = joined[joined['id'] == -1]
test = joined[joined['cost'] == -1]

features = [
  'year',
  'month',
  'dayofweek',
  'dayofyear',
  'day',
  'annual_usage',
            'bracket_pricing',
            # 'cost',
            # 'id',
            'min_order_quantity',
            'quantity',
            # 'quote_date',
            'supplier',
            # 'tube_assembly_id',
            # 'component_id_1',
            'quantity_1',
            # 'component_id_2',
            'quantity_2',
            # 'component_id_3',
            'quantity_3',
            # 'component_id_4',
            'quantity_4',
            # 'component_id_5',
            'quantity_5',
            # 'material_id',
            'diameter',
            'wall',
            'length',
            'num_bends',
            'bend_radius',
            # 'end_a_1x',
            # 'end_a_2x',
            # 'end_x_1x',
            # 'end_x_2x',
            'end_a',
            'end_x',
            'num_boss',
            'num_bracket',
            # 'other',
            'forming_a',
            'forming_x',
            # 'spec',
            '0039', '0038', '0035', '0037', '0036', '0030', '0033', '0014', '0066', '0067', '0064', '0065', '0062', '0063', '0060', '0061', '0068', '0069', '0004', '0049', '0006', '0007', '0001', '0002', '0003', '0040', '0042', '0043', '0044', '0009', '0005', '0047', '0019', '0054', '0071', '0070', '0073', '0072', '0075', '0074', '0077', '0076', '0079', '0078', '0017', '0016', '0059', '0058', '0013', '0012', '0011', '0010', '0053', '0052', '0051', '0050', '0057', '0056', '0055', '0018', '0088', '0084', '0085', '0086', '0087', '0080', '0081', '0082', '0083', '0022', '0023', '0020', '0021', '0026', '0027', '0024', '0025', '0028', '0029', '0046', '0091', '0096', '0094', '0092', '0015',
            'end_flag'
            ]

y = train['cost'].apply(lambda x: math.log(x + 1), 1)


X = train[features]
X_test = test[features]
# print X.info()
# print X.head9()

X.fillna(-1, inplace=True)
X_test.fillna(-1, inplace=True)

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
offset = 4000

ind = 1
if ind == 1:
  n_iter = 5
  rs = ShuffleSplit(len(y), n_iter=n_iter, test_size=0.5, random_state=random_state)

  result = []

  for min_child_weight in [5]:
    for eta in [0.1]:
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

                # X_train = X_train[::-1, :]
                # labels = y_train[::-1]
                #
                # xgtrain = xgb.DMatrix(X_train[offset:, :], label=labels[offset:])
                # xgval = xgb.DMatrix(X_train[:offset, :], label=labels[:offset])
                #
                # watchlist = [(xgtrain, 'train'), (xgval, 'val')]
                #
                # model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
                #
                # preds2 = model.predict(xgtest, ntree_limit=model.best_iteration)
                #
                # # preds = model.predict(xgval, ntree_limit=model.best_iteration)
                #
                # preds = 0.5 * preds1 + 0.5 * preds2

                tp = mean_squared_error(y_test, preds1)

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
  xgtest = xgb.DMatrix(X_test.values)

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  params = {
  'objective': 'reg:linear',
  #   'objective': 'count:poisson',
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
  prediction_test_1 = model1.predict(xgtest, ntree_limit=model1.best_iteration)

  X_train = X.values[::-1, :]
  labels = y.values[::-1]

  xgtrain = xgb.DMatrix(X_train[offset:, :], label=labels[offset:])
  xgval = xgb.DMatrix(X_train[:offset, :], label=labels[:offset])

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  model2 = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)

  prediction_test_2 = model2.predict(xgtest, ntree_limit=model2.best_iteration)



  prediction_test = 0.5 * prediction_test_1 + 0.5 * prediction_test_2
  submission = pd.DataFrame()
  submission['id'] = test['id']
  submission['cost'] = prediction_test
  submission['cost'] = submission['cost'].apply(lambda x: math.exp(x) - 1)
  submission.to_csv("predictions/xgbt.csv", index=False)
