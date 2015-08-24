from __future__ import division
__author__ = 'Vladimir Iglovikov'

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_squared_error
import math
from sklearn.ensemble import RandomForestRegressor

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

random_state = 42

X = train[features]
X_test = test[features]
# print X.info()
# print X.head9()

X.fillna(-1, inplace=True)
X_test.fillna(-1, inplace=True)

ind = 1

if ind == 1:
  rs = ShuffleSplit(len(y), n_iter=10, test_size=0.5, random_state=random_state)

  result = []

  for n_estimators in [10]:
    for min_samples_split in [2]:
      for max_features in [0.7]:
        for max_depth in [7]:
          for min_samples_leaf in [1]:
            score = []
            for train_index, test_index in rs:

              a_train = X.values[train_index]
              a_test = X.values[test_index]
              b_train = y.values[train_index]
              b_test = y.values[test_index]

              clf = RandomForestRegressor(n_estimators=n_estimators,
                                          min_samples_split=min_samples_split,
                                          max_features=max_features,
                                          max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf,
                                          n_jobs=-1,
                                          random_state=random_state)

              clf.fit(a_train, b_train)

              preds = clf.predict(a_test)

              score += [math.sqrt(mean_squared_error(b_test, preds))]

            result += [(np.mean(score), np.std(score), n_estimators, min_samples_split, min_samples_leaf, max_depth, max_features)]

  result.sort()
  print result

elif ind == 2:
  clf = RandomForestRegressor(n_estimators=100,
                              min_samples_split=2,
                              max_features=0.4,
                              max_depth=7,
                              min_samples_leaf=1,
                              n_jobs=-1,
                              random_state=random_state)
  clf.fit(X, y)

  prediction_test = clf.predict(X_test)
  submission = pd.DataFrame()
  submission['id'] = hold['id']
  submission['cost'] = prediction_test
  submission.to_csv("preds_on_test/RF.csv", index=False)
