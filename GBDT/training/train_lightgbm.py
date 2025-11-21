import pandas as pd
import random
import numpy as np
import pickle
import joblib
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import os

expName = 'lgbm_data_20251121-1198'
nowTime = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
expPath = os.path.join('data/outputs/exps',expName,nowTime)
if not os.path.exists(expPath):
    os.makedirs(expPath)

def logger(log_str):
    with open(expPath + '/log.txt','a',encoding='utf-8') as file:
        file.write(log_str)



# Normalized manual features + automatic features
train_data_path = 'data/processed/csvData/20251121-withauto/20251121-withauto-withnormal-train.csv'
val_data_path = 'data/processed/csvData/20251121-withauto/20251121-withauto-withnormal-val.csv'
test_data_path = 'data/processed/csvData/20251121-withauto/20251121-withauto-withnormal-val.csv'

df_train = pd.read_csv(train_data_path)
df_val = pd.read_csv(val_data_path)
df_test = pd.read_csv(test_data_path)
logger('train_data_path: '+train_data_path+'\nval_data_path: '+val_data_path+'\ntest_data_path: '+ test_data_path+'\n')
print(df_train.head())


# print(df_train.info())
# print(df_train.head())
print(df_train.columns)

# -----------
# Split dataset
# -----------
# 2D + 3D 特征
# print(df_train.columns)
y_train = df_train['weight']  # Get y for the training set
x_train = df_train.drop(['weight', 'imgName'], axis=1)  # Get x for the training set, 1 means drop by column
# x_train = df_train.drop(['weight', 'imgName', 'equi_diameter'], axis=1)  # Get x for the training set, 1 means drop by column
y_test = df_test['weight']  # Get y for the test set
x_test = df_test.drop(['weight', 'imgName'], axis=1)  # Get x for the test set
# x_test = df_test.drop(['weight', 'imgName', 'equi_diameter'], axis=1)  # Get x for the test set
y_val = df_val['weight']
x_val = df_val.drop(['weight', 'imgName'], axis=1)  # Get x for the validation set, 1 means drop by column
# x_val = df_val.drop(['weight', 'imgName', 'equi_diameter'], axis=1)  # Get x for the validation set, 1 means drop by column
logger("Using all features\n")



# 2D features
# x_train = pd.DataFrame(x_train.values[:, 0:16]) # 舍弃'equi_diameter'的话是 15
# x_test = pd.DataFrame(x_test.values[:, 0:16])
# x_val = pd.DataFrame(x_val.values[:, 0:16])
# x_train = pd.concat([x_train,train_life] ,axis=1)
# x_test = pd.concat([x_test,test_life],axis=1)
# x_val = pd.concat([x_val,val_life],axis=1)
# logger("只使用2D特征\n")

print(x_train.shape,x_test.shape)       # Print the shape of the training and test sets

# 3D features
# x_train = pd.DataFrame(x_train.values[:, 16:24])
# x_test = pd.DataFrame(x_test.values[:, 16:24])
# x_val = pd.DataFrame(x_val.values[:, 16:24])
# print(x_train.shape,x_test.shape)       # Print the shape of the training and test sets
# logger("Using 3D features\n")

all_manuals = ['area', 'perimeter', 'min_rect_width', 'min_rect_high', 'approx_area', 'approx_perimeter', 'extent', 'hull_perimeter', 'hull_area', 'solidity', 'max_defect_dist', 'sum_defect_dist', 'equi_diameter', 'ellipse_long', 'ellipse_short', 'eccentricity', 'volume', 'maxHeight', 'minHeight', 'max2min', 'meanHeight', 'mean2min', 'mean2max', 'stdHeight', 'heightSum']
new_manuals = ['approx_area', 'approx_perimeter', 'extent', 'hull_perimeter', 'hull_area', 'solidity', 'max_defect_dist', 'sum_defect_dist', 'equi_diameter', 'ellipse_long', 'ellipse_short', 'maxHeight', 'minHeight', 'max2min', 'meanHeight', 'mean2min', 'mean2max', 'stdHeight', 'heightSum']
old_manuals = ['area', 'perimeter', 'min_rect_width', 'min_rect_high', 'eccentricity', 'volume']

# # Manual features
# x_train = pd.DataFrame(x_train.values[:, :25],columns=all_manuals)
# x_test = pd.DataFrame(x_test.values[:, :25],columns=all_manuals)
# x_val = pd.DataFrame(x_val.values[:, :25],columns=all_manuals)
# logger("Using manual features\n")

# Automatic features
# x_train = pd.DataFrame(x_train.values[:, 25:2074])
# x_test = pd.DataFrame(x_test.values[:, 25:2074])
# x_val = pd.DataFrame(x_val.values[:, 25:2074])
# logger("Using automatic features\n")


# print(x_train.columns)
# New features 19
# x_train = x_train.drop(old_manuals,axis=1) # Get training set x, 1 means drop by column
# x_test = x_test.drop(old_manuals,axis=1) # Get test set x
# x_val = x_val.drop(old_manuals,axis=1) # Get training set x, 1 means drop by column
# logger("Using new manual features\n")


# Old features 6
# x_train = x_train.drop(new_manuals,axis=1) # Get training set x, 1 means drop by column
# x_test = x_test.drop(new_manuals,axis=1) # Get test set x
# x_val = x_val.drop(new_manuals,axis=1) # Get training set x, 1 means drop by column
# logger("Using old manual features\n")


# ------------------------------
# Construct dataset for lightgbm
# ------------------------------
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

# Define a utility function to analyze data distribution
def show_stats(data):
    """data is the input data, then calculate the following statistics"""
    print("Minimum value: {}".format(np.min(data)))
    print("Maximum value: {}".format(np.max(data)))
    print("Range: {}".format(np.ptp(data)))
    print("Mean: {}".format(np.mean(data)))
    print("Standard deviation: {}".format(np.std(data)))
    print("Variance: {}\n".format(np.var(data)))

# Write a function to input model and output the grid search model and MAE
def model_train(model):
    # Set the range of hyperparameters
    ## param_grid={'learning_rate':[0.01,0.05,0.1,0.2]}
    ##model=GridSearchCV(model,param_grid)
    # Start training the model
    # For LightGBM 4.0+, use callbacks for early stopping
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        eval_metric='l1',
        callbacks=[early_stopping(stopping_rounds=500)]
    )
    # model.fit(x_train,y_train)

    test_predict=model.predict(x_test)# Get the predicted value here, no need to return
    show_stats(y_train)# Show the predicted effect
    #model=model.best_estimator_# Get the best parameters from the grid search
    train_predict=model.predict(x_train)# Get the predicted value here, no need to return
    # On the training set:
    train_log_str = 'Training:\n' \
                    'Mean absolute error (MAE): {:.6f}\n' \
                    'Mean squared error (MSE): {:.6f}\n' \
                    'Root mean squared error (RMSE): {:.6f}\n' \
                    'R2: {:.6f}\n'.format(mean_absolute_error(y_train,train_predict),mean_squared_error(y_train, train_predict),
                                          mean_squared_error(y_train, train_predict) ** 0.5, r2_score(y_train,train_predict))

    print(train_log_str),logger(train_log_str)
    # On the test set:
    val_log_str = 'Testing:\n' \
                    'Mean absolute error (MAE): {:.6f}\n' \
                    'Mean squared error (MSE): {:.6f}\n' \
                    'Root mean squared error (RMSE): {:.6f}\n' \
                    'R^2 Score: {:.6f}\n\n'.format(mean_absolute_error(y_test,test_predict),mean_squared_error(y_test, test_predict),
                                          mean_squared_error(y_test, test_predict) ** 0.5, r2_score(y_test,test_predict))
    print(val_log_str),logger(val_log_str)
    feature_log = 'Feature importances:' + str(list(model.feature_importances_))

    print(feature_log),logger(feature_log)
    pd.DataFrame({'gt':y_test,'pr':test_predict}).to_csv(expPath+'/test_predict.csv',index_label='index')
    pd.DataFrame({'gt':y_train,'pr':train_predict}).to_csv(expPath+'/train_predict.csv',index_label='index')
    return model


model_lgb = lgb.LGBMRegressor(
    n_estimators=4000,
    learning_rate=0.1,
    num_leaves=15,
    max_depth=5,
    min_child_samples=15,
    min_child_weight=0.01,
    subsample=0.8,
    colsample_bytree=1
)
model_lgb=model_train(model_lgb)
print(model_lgb.get_params())


lgb.plot_importance(model_lgb)
plt.savefig(expPath+'/importance.png')
plt.show()



# Save the model
joblib.dump(model_lgb, expPath+'/result.pkl')

# # Load the model
# gbm = joblib.load('model_lgb2.pkl')
# # Predict the model
# y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration_)
# # 模型评估
# print('测试集上的平均绝对误差:',mean_absolute_error(y_test,y_pred))
# print('Root mean squared error (RMSE):', mean_squared_error(y_test, y_pred) ** 0.5)
# print('Mean squared error (MSE):', mean_squared_error(y_test, y_pred))
# # Print the feature importance
# print('Feature importances:', list(gbm.feature_importances_))
