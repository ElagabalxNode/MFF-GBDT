import lightgbm as lgb
import pandas as pd
import random
import numpy as np
import pickle

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



# df_test = pd.read_csv('GBDT/csvData/20210206-200-1198-manuals/20210206-1198-test-0.2.csv')# manual features
df_test = pd.read_csv('GBDT/csvData/20210206-200-1198-manuals/20210206-1198-normal-test-0.2.csv') # normalized manual features

# y_train = df_train['weight'] # get y for the training set
# x_train = df_train.drop(['weight','imgName','equi_diameter'],axis=1) # get x for the training set, 1 means drop by column
x_test = df_test.drop(['weight', 'imgName'], axis=1) # get x for the test set
y_test = df_test['weight'] # get y for the test set
# print(x_test['ellipse_long'])

# 2D特征
# x_test = pd.DataFrame(x_test.values[:, 0:15])

#  3D特征
# x_test = pd.DataFrame(x_test.values[:, 15:24])


columns  = df_test.drop(['weight','imgName'],axis=1).columns.values
print(columns,len(columns))

# load model with pickle to predict
# model_weight = 'GBDT/exps/lgbm_data_20210206-1198/2021-12-16_21-11/result.pkl' # no normal
model_weight = 'GBDT/exps/lgbm_data_20210206-1198/2025-11-20_15-22/result.pkl' # with normal

# model_weight = 'GBDT/exps/xgb_data_20210206-1198/2021-12-16_21-18/xgb.pkl' # no normal
# model_weight = 'GBDT/exps/xgb_data_20210206-1198/2022-09-16_15-36/xgb.pkl' # with normal



with open(model_weight, 'rb') as fin:
    pkl_bst = pickle.load(fin)

print(pkl_bst)
features = list(pkl_bst.feature_importances_)
print(len(features))

preds = pkl_bst.predict(x_test)

print('Mean Absolute Error (MAE):', mean_absolute_error(y_test, preds))
print('Mean Squared Error (MSE):', mean_squared_error(y_test, preds))
print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(y_test, preds)))
print('R^2 Score:', r2_score(y_test, preds))


# Save prediction results
# resList = []
# for p in preds:
#     resList.append(str(p)+'\n')
# with open('exps/20210911-90-model-1068-2d-re360_640_lgb200-predict.txt','w') as f:
#     f.writelines(resList)

# lgb.plot_importance(pkl_bst)
# plt.show()


# Calculate the MAE for each of the 5 intervals
y_test=y_test.to_numpy()
print(len(y_test))
print(len(preds))
min_weight = min(y_test)
max_weight = max(y_test)
inteval = (max_weight - min_weight)/5
print(min_weight,max_weight,inteval)
class_results = {
    0:[[],[]],
    1:[[],[]],
    2:[[],[]],
    3:[[],[]],
    4:[[],[]]
}

for i in range(len(y_test)):
    start = min_weight
    # end = min_weight
    end = min_weight + inteval
    for j in range(5):
        # end = end + inteval
        if j == 4 :
            if y_test[i] >= start and y_test[i] <= max_weight:
                class_results[j][0].append(y_test[i])
                class_results[j][1].append(preds[i])
        else:
            if y_test[i] >= start and y_test[i] <= end:
                class_results[j][0].append(y_test[i])
                class_results[j][1].append(preds[i])

        start = end
        end = end + inteval


for j in range(5):
    print('Mean Absolute Error:', j, mean_absolute_error(class_results[j][0], class_results[j][1]), "Count:", len(class_results[j][1]))

print(class_results[4])

