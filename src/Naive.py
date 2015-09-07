from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer

__author__ = 'Vladimir Iglovikov'

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_squared_error
import math
from sklearn.decomposition import TruncatedSVD
import time

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

#label encode 'component_id'
le = LabelEncoder()
le.fit(np.hstack([joined['component_id_1'],
                  joined['component_id_2'],
                  joined['component_id_3'],
                  joined['component_id_4'],
                  joined['component_id_5']]))

joined['component_id_1'] = le.transform(joined['component_id_1'])
joined['component_id_2'] = le.transform(joined['component_id_2'])
joined['component_id_3'] = le.transform(joined['component_id_3'])
joined['component_id_4'] = le.transform(joined['component_id_4'])
joined['component_id_5'] = le.transform(joined['component_id_5'])

#map bracket pricing to Yes=1, No=0
joined['bracket_pricing'] = joined['bracket_pricing'].map({'Yes': 1, 'No': 0})


#add datetime features
joined['year'] = joined['quote_date'].dt.year
joined['month'] = joined['quote_date'].dt.month
joined['dayofyear'] = joined['quote_date'].dt.dayofyear
joined['dayofweek'] = joined['quote_date'].dt.dayofweek
joined['day'] = joined['quote_date'].dt.day
joined['weekday'] = joined['quote_date'].apply(lambda x: x.weekday(), 1)

cv = CountVectorizer()

x = cv.fit_transform(joined['spec'])

svd = TruncatedSVD(n_components=10)
x = svd.fit_transform(x)

x = pd.DataFrame(x)

joined = pd.concat([joined, x], 1)

train = joined[joined['id'] == -1]
test = joined[joined['cost'] == -1]
idx = test.id.values.astype(int)

features = [
  'weekday',
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
            'component_id_1',
            'quantity_1',
            'component_id_2',
            'quantity_2',
            'component_id_3',
            'quantity_3',
            'component_id_4',
            'quantity_4',
            'component_id_5',
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
            # '0039', '0038', '0035', '0037', '0036', '0030', '0033', '0014', '0066', '0067', '0064', '0065', '0062', '0063', '0060', '0061', '0068', '0069', '0004', '0049', '0006', '0007', '0001', '0002', '0003', '0040', '0042', '0043', '0044', '0009', '0005', '0047', '0019', '0054', '0071', '0070', '0073', '0072', '0075', '0074', '0077', '0076', '0079', '0078', '0017', '0016', '0059', '0058', '0013', '0012', '0011', '0010', '0053', '0052', '0051', '0050', '0057', '0056', '0055', '0018', '0088', '0084', '0085', '0086', '0087', '0080', '0081', '0082', '0083', '0022', '0023', '0020', '0021', '0026', '0027', '0024', '0025', '0028', '0029', '0046', '0091', '0096', '0094', '0092', '0015',
            'end_flag',
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
            ]

labels = train['cost'].values

X = train[features]
X_test = test[features]

print X.shape
print X.columns
# print X.info()
# print X.head9()

X.fillna(-1, inplace=True)
X_test.fillna(-1, inplace=True)

train = X.astype(float)
test = X_test.astype(float)
label_log = np.log1p(labels)

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.02
params["min_child_weight"] = 6
params["subsample"] = 0.7
params["colsample_bytree"] = 0.6
params["scale_pos_weight"] = 0.8
params["silent"] = 1
params["max_depth"] = 8
params["max_delta_step"] = 2

plst = list(params.items())

xgtrain = xgb.DMatrix(train, label=label_log)
xgtest = xgb.DMatrix(test)

print('1500')


num_rounds = 1500
model = xgb.train(plst, xgtrain, num_rounds)
preds1 = model.predict(xgtest)

print('3000')

num_rounds = 3000
model = xgb.train(plst, xgtrain, num_rounds)
preds2 = model.predict(xgtest)

print('4000')

num_rounds = 4000
model = xgb.train(plst, xgtrain, num_rounds)
preds4 = model.predict(xgtest)

label_log = np.power(labels, 1.0/16.0)

xgtrain = xgb.DMatrix(train, label=label_log)
xgtest = xgb.DMatrix(test)

print('power 1/16 4000')

num_rounds = 4000
model = xgb.train(plst, xgtrain, num_rounds)
preds3 = model.predict(xgtest)

preds = 0.4*np.expm1(preds4)+.1*np.expm1(preds1)+0.1*np.expm1(preds2)+0.4*np.power(preds3,16)


preds = pd.DataFrame({"id": idx, "cost": preds})
preds.to_csv('predictions/benchmark.csv', index=False)
