import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time
import os

# Experiment setup
expName = 'lgbm_fixed_pipeline'
nowTime = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
expPath = os.path.join('data/outputs/exps', expName, nowTime)
if not os.path.exists(expPath):
    os.makedirs(expPath)

def logger(log_str):
    with open(os.path.join(expPath, 'log.txt'), 'a', encoding='utf-8') as file:
        file.write(log_str)

# ------------------------------
# Load Fixed Data (Robust Pipeline)
# ------------------------------
# Пути к новым, нормализованным данным
train_data_path = 'data/processed/csvData/processed_fixed/train_fixed.csv'
val_data_path = 'data/processed/csvData/processed_fixed/val_fixed.csv'
test_data_path = 'data/processed/csvData/processed_fixed/test_fixed.csv'

logger(f'train_data_path: {train_data_path}\nval_data_path: {val_data_path}\ntest_data_path: {test_data_path}\n')

# Load datasets
df_train = pd.read_csv(train_data_path)
df_val = pd.read_csv(val_data_path)
df_test = pd.read_csv(test_data_path)

print("Train shape:", df_train.shape)
print("Val shape:", df_val.shape)
print("Test shape:", df_test.shape)

# Prepare X and y
# Данные уже нормализованы, просто разделяем фичи и таргет
def get_xy(df):
    if 'weight' not in df.columns:
        raise ValueError("Column 'weight' not found")
    y = df['weight']
    # Удаляем таргет и мета-информацию (имя файла)
    drop_cols = ['weight']
    if 'imgName' in df.columns:
        drop_cols.append('imgName')
    
    X = df.drop(drop_cols, axis=1)
    return X, y

x_train, y_train = get_xy(df_train)
x_val, y_val = get_xy(df_val)
x_test, y_test = get_xy(df_test)

logger(f"Features used: {list(x_train.columns)}\n")

# ------------------------------
# Train LightGBM
# ------------------------------
# Create LightGBM datasets
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)

def show_stats(data):
    print("Min: {:.4f}, Max: {:.4f}, Mean: {:.4f}, Std: {:.4f}".format(
        np.min(data), np.max(data), np.mean(data), np.std(data)
    ))

def train_and_evaluate(model):
    print("Starting training...")
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        eval_metric='l1', # L1 = MAE
        callbacks=[early_stopping(stopping_rounds=500)]
    )
    
    # Predictions
    test_predict = model.predict(x_test)
    train_predict = model.predict(x_train)
    
    # --- Evaluation ---
    def get_metrics(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_true, y_pred)
        return f'{name}:\nMAE: {mae:.6f}\nMSE: {mse:.6f}\nRMSE: {rmse:.6f}\nR2: {r2:.6f}\n'

    train_log = get_metrics(y_train, train_predict, "Training")
    test_log = get_metrics(y_test, test_predict, "Testing")
    
    print(train_log)
    logger(train_log)
    print(test_log)
    logger(test_log)
    
    # Feature Importance
    feature_imp = list(model.feature_importances_)
    logger(f'Feature importances: {feature_imp}\n')
    
    # Save predictions for analysis
    pd.DataFrame({'gt': y_test, 'pr': test_predict}).to_csv(os.path.join(expPath, 'test_predict.csv'), index_label='index')
    pd.DataFrame({'gt': y_train, 'pr': train_predict}).to_csv(os.path.join(expPath, 'train_predict.csv'), index_label='index')
    
    return model

# Initialize model
model_lgb = lgb.LGBMRegressor(
    n_estimators=4000,
    learning_rate=0.1,
    num_leaves=15,
    max_depth=5,
    min_child_samples=15,
    min_child_weight=0.01,
    subsample=0.8,
    colsample_bytree=1,
    random_state=42 # Fix seed for model as well
)

# Run training
trained_model = train_and_evaluate(model_lgb)

# Plot importance
lgb.plot_importance(trained_model, max_num_features=20)
plt.tight_layout()
plt.savefig(os.path.join(expPath, 'importance.png'))
# plt.show() # Commented out for non-interactive environments

# Save model
model_path = os.path.join(expPath, 'result.pkl')
joblib.dump(trained_model, model_path)
print(f"Model saved to {model_path}")
