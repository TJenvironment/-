# -*- coding: utf-8 -*-
"""
Created on Sun May 25 13:10:16 2025

@author: hj
"""


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray


import matplotlib.pyplot as plt
# Cross-validation and hyperparameter optimization
import optuna 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, learning_curve  # Splitting dataset
from sklearn.model_selection import cross_validate, KFold 
# ==============================================================================
# 1. **Modeling Libraries**: These libraries are used to build and train machine learning models.
# ==============================================================================
# Regression models
from xgboost import XGBRegressor  # XGBoost regressor
from catboost import CatBoostRegressor  # CatBoost regressor
import lightgbm as lgb  # LightGBM package
from lightgbm import LGBMRegressor  # LightGBM regressor
from sklearn.ensemble import RandomForestRegressor  # Random Forest regressor
from sklearn.linear_model import Lasso  # Lasso regression
from sklearn.linear_model import LinearRegression  # Linear regression
import optuna
# ==============================================================================
# 2. **Model Interpretation Libraries**: Used for model explainability and feature importance.
# ==============================================================================
# SHAP values for model interpretability
import shap

# ==============================================================================
# 3. **Data Processing and Preprocessing**: Libraries to handle and manipulate data.
# ==============================================================================
import pandas as pd  # DataFrame handling
import numpy as np  # Array manipulation and mathematical functions
from sklearn.impute import SimpleImputer  # Handling missing data
from sklearn.pipeline import Pipeline  # Creating machine learning pipelines

# ==============================================================================
# 4. **Statistical and Mathematical Functions**: Used for statistical testing and math calculations.
# ==============================================================================
from scipy.stats import spearmanr  # Spearman correlation for non-parametric rank correlation
from scipy.stats import binom  # Binomial distribution for probability modeling
import math  # Basic mathematical functions

# ==============================================================================
# 5. **Model Evaluation and Validation**: Used to evaluate model performance.
# ==============================================================================
# Cross-validation methods and data splitting
from sklearn.model_selection import train_test_split, cross_validate, KFold  # Train-test split, cross-validation, and KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Metrics to evaluate models

# Permutation importance for feature importance calculation
from sklearn.inspection import permutation_importance

# ==============================================================================
# 6. **Visualization Libraries**: Used for creating visualizations.
# ==============================================================================
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Statistical data visualization

def RMSE(y, y_pred):
    RMSE = np.sqrt(mean_squared_error(y, y_pred))
    return RMSE

def MAPE(y, y_pred):
    e = np.abs((y_pred - y) / y)
    MAPE = np.sum(e) / len(e)
    return MAPE


# 加载数据集（假设第一列为SMILES，最后一列为回归目标值）
data=pd.read_excel('D:\data4','LD50_dataset',index_col=None,keep_default_na=True)
smiles_list = data.iloc[:, 2]  # SMILES表达式在第一列
y = data.iloc[:, -1]  # 最后一列为目标值

# 将SMILES转换为摩根指纹（ECFP4, 半径=2, 长度=2048）
def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

# 构建特征矩阵
X = []
valid_indices = []

for i, smi in enumerate(smiles_list):
    fp = smiles_to_morgan(smi)
    if fp is not None:
        X.append(fp)
        valid_indices.append(i)

# 转换为 Numpy 数组
X_array = np.zeros((len(X), 2048))
for i, fp in enumerate(X):
    ConvertToNumpyArray(fp, X_array[i])

# 过滤y以匹配有效SMILES
y_filtered = y.iloc[valid_indices].reset_index(drop=True)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_array, y_filtered, test_size=0.2, random_state=42)



import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # 确保y为列向量
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 1. 定义多层感知机（MLP）模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 定义一个多层感知机
        self.fc1 = nn.Linear(2048, 512)  # 输入2048维，输出512维
        self.fc2 = nn.Linear(512, 128)   # 输入512维，输出128维
        self.fc3 = nn.Linear(128, 1)     # 输出1个数值（回归任务）
        
        # 使用 Xavier 初始化
        init.xavier_uniform_(self.fc1.weight)  # 对fc1层的权重进行 Xavier 均匀初始化
        init.xavier_uniform_(self.fc2.weight)  # 对fc2层的权重进行 Xavier 均匀初始化
        init.xavier_uniform_(self.fc3.weight)  # 对fc3层的权重进行 Xavier 均匀初始化

        # 偏置项通常初始化为零
        if self.fc1.bias is not None:
            init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)
        if self.fc3.bias is not None:
            init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 第一层激活
        x = torch.relu(self.fc2(x))  # 第二层激活
        x = self.fc3(x)  # 输出层
        return x
# 2. 初始化模型，损失函数和优化器
model = MLP()
criterion = nn.MSELoss()  # 使用均方误差作为回归任务的损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 3. 训练模型
num_epochs = 100
batch_size = 64
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)



# 初始化记录列表
train_mse_list = []
test_mse_list = []

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()  # 清除之前的梯度
        output = model(data)   # 模型输出
        loss = criterion(output, target)  # 计算损失
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数
        
        running_loss += loss.item()  # 累计损失

    # 记录每个epoch的平均训练集MSE
    avg_train_mse = running_loss / len(train_loader)

    # 每10个epoch打印一次损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_mse:.4f}')

    # 计算测试集 MSE
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不需要计算梯度
        y_train_pred = model(X_train_tensor)
        y_test_pred = model(X_test_tensor)

        # 计算训练集和测试集的MSE
        train_mse = mean_squared_error(y_train_tensor, y_train_pred)
        test_mse = mean_squared_error(y_test_tensor, y_test_pred)

        train_mse_list.append(train_mse)
        test_mse_list.append(test_mse)

    # 打印训练集和测试集 MSE
    print(f'Training MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')

# 绘制训练集和测试集MSE随epoch变化的图
# 提取最后一个偶数位置的元素


plt.figure(figsize=(10, 6))

# 画训练MSE的线（黑色实线，线宽为2）
plt.plot(range(1, num_epochs + 1), train_mse_list, label='Train MSE', color='black', linestyle='-', linewidth=2)

# 画测试MSE的线（黑色虚线，线宽为2）
plt.plot(range(1, num_epochs + 1), test_mse_list, label='Test MSE', color='black', linestyle='--', linewidth=2)


# 调整横纵坐标轴数字的字体大小
plt.tick_params(axis='both', labelsize=22)
# 显示网格
plt.grid(True)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()



# 超参数网格设置
learning_rates = [0.001, 0.0005, 0.01]
batch_sizes = [32, 64, 128]

# 记录每种超参数组合的MSE和R2
results = []

# 训练过程
for lr in learning_rates:
    for batch_size in batch_sizes:
        print(f"Training with learning rate: {lr}, batch size: {batch_size}")
        
        # 2. 初始化模型，损失函数和优化器
        model = MLP()  # 根据实际模型进行实例化
        criterion = nn.MSELoss()  # 使用均方误差作为回归任务的损失函数
        optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用Adam优化器
        
        # 3. 数据加载器
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        
        # 初始化记录列表
        train_mse_list = []
        test_mse_list = []
        train_r2_list = []
        test_r2_list = []
        
        # 训练模型
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for data, target in train_loader:
                optimizer.zero_grad()  # 清除之前的梯度
                output = model(data)   # 模型输出
                loss = criterion(output, target)  # 计算损失
                loss.backward()        # 反向传播
                optimizer.step()       # 更新参数
                
                running_loss += loss.item()  # 累计损失

            # 记录每个epoch的平均训练集MSE
            avg_train_mse = running_loss / len(train_loader)

            # 每10个epoch打印一次损失
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_mse:.4f}')

            # 计算测试集 MSE 和 R²
            model.eval()  # 设置模型为评估模式
            with torch.no_grad():  # 不需要计算梯度
                y_train_pred = model(X_train_tensor)
                y_test_pred = model(X_test_tensor)

                # 计算训练集和测试集的MSE
                train_mse = mean_squared_error(y_train_tensor, y_train_pred)
                test_mse = mean_squared_error(y_test_tensor, y_test_pred)
                
                # 计算训练集和测试集的R²
                train_r2 = r2_score(y_train_tensor, y_train_pred)
                test_r2 = r2_score(y_test_tensor, y_test_pred)

                # 保存每个epoch的MSE和R²
                train_mse_list.append(train_mse)
                test_mse_list.append(test_mse)
                train_r2_list.append(train_r2)
                test_r2_list.append(test_r2)

        # 保存每种超参数组合下的最终结果
        results.append({
            'learning_rate': lr,
            'batch_size': batch_size,
            'train_mse': train_mse_list[-1],  # 取最后一个epoch的MSE
            'test_mse': test_mse_list[-1],    # 取最后一个epoch的MSE
            'train_r2': train_r2_list[-1],    # 取最后一个epoch的R²
            'test_r2': test_r2_list[-1]       # 取最后一个epoch的R²
        })

# 打印每种超参数组合的结果
for result in results:
    print(f"Learning Rate: {result['learning_rate']}, Batch Size: {result['batch_size']}")
    print(f"Training MSE: {result['train_mse']:.4f}, Test MSE: {result['test_mse']:.4f}")
    print(f"Training R²: {result['train_r2']:.4f}, Test R²: {result['test_r2']:.4f}")


# 初始化并训练XGBoost模型（回归任务）
model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)

# 假设 y_test 和 y_pred 已经定义
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# 计算均方根误差 (RMSE)
rmse = np.sqrt(mse)

# 计算决定系数 (R²)
r2 = r2_score(y_test, y_pred)

# 输出结果
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")


# Hyperparameter Tuning and Model Optimization with Bayesian Optimization
# Step 1: Define the Objective Function and Parameter Space
def optuna_objective(x_train,y_train,trial,model): 
    # Define the parameter space based on the selected model
    if model == 'xgb':
        params = {
            "max_depth": trial.suggest_int("max_depth", 10, 20, step=1),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 800, 1500, step=50),
            "subsample": trial.suggest_float("subsample", 0.5, 1, log=False),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1, log=False),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 4, step=1),
            "reg_lambda": trial.suggest_int("reg_lambda", 6, 12, 1),
            "objective": "reg:squarederror",
            "booster": "gbtree",
            "gamma": 0,
            "reg_alpha": 0
            }
        # Define the estimator
        reg = XGBRegressor(**params)   
        
    elif model == 'lgb':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 600, 1200, step=100),
            'max_depth': trial.suggest_int('max_depth', 8, 15,step=1),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
            'min_child_samples':trial.suggest_int("min_child_samples", 1, 7, step=1),
            "subsample": trial.suggest_float("subsample", 0.5, 1, log=False),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1, log=False),
            'num_leaves': trial.suggest_int('num_leaves', 200, 700,step=50),
            'metric':'rmse','boosting_type':'gbdt','objective':'regression','force_row_wise':'True'
            # Add other LightGBM parameters
        }
        reg = lgb.LGBMRegressor(**params)
        
    elif model == 'catb':
        params = {
            'iterations': trial.suggest_int('iterations', 400, 900,step=50),
            'depth': trial.suggest_int('depth', 6, 12,step=1),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.25, log=True),
            'rsm':trial.suggest_float('rsm', 0.5, 1.0, log=False) ,
            'early_stopping_rounds': 20,
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
            'loss_function':'RMSE'}
        reg = CatBoostRegressor(**params, verbose=False)
        
    elif model == 'RF':
       # Define the parameter space for Random Forest
        # 用SimpleImputer对象拟合数据并转换数据
        imputer = SimpleImputer(strategy='mean')
        x_train = imputer.fit_transform(x_train)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 9, 18,step=1),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 5,step=1),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5,step=1),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1, log=False),
            "criterion": 'squared_error',
            "min_weight_fraction_leaf": 0.0,
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "random_state": None,
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
           
        reg = RandomForestRegressor(**params)
    else:
        raise ValueError("Invalid model type. Supported models are 'xgb', 'lgb', 'catb', and 'RF'.")
    
    # 5-fold cross-validation process, output negative root mean squared error (-RMSE)
    cv = KFold(n_splits=5, shuffle=True)
    validation_loss = cross_validate(reg, x_train, y_train,
                                     scoring="neg_root_mean_squared_error",
                                     cv=cv,  # Cross-validation mode
                                     verbose=False,  # Print process
                                     n_jobs=-1,  # Number of threads
                                     error_score='raise'
                                     )
    # Final output: RMSE
    return np.mean(abs(validation_loss["test_score"]))
# Step 2: Define the Specific Workflow for Optimizing the Objective Function
def optimizer_optuna(x_train,y_train, n_trials, algo, model):
    
    # Define the sampler
    algo = optuna.samplers.TPESampler(n_startup_trials=15, n_ei_candidates=30)
    
    # Create an Optuna study for optimization
    study = optuna.create_study(sampler=algo, direction="minimize")
    
    # Optimize the objective function
    study.optimize(lambda trial: optuna_objective(x_train,y_train,trial, model), n_trials=n_trials,show_progress_bar=True)
    
    # Display the best parameters and score
    print("\n", "\n", "best params: ", study.best_trial.params,
          "\n", "\n", "best score: ", study.best_trial.values,
          "\n")
    
    return study.best_trial.params, study.best_trial.values
def evaluate_model_performance(y_true, y_pred):
    result=[]
    # Reshape y_pred to match the shape of y_true
    y_pred = np.reshape(y_pred, (len(y_pred), 1))   
    # Clip predicted values to be between 0 and 100
    y_pred = np.where(y_pred > 100, 100, y_pred)
    y_pred = np.where(y_pred < 0, 0, y_pred)
    # Calculate MAE
    MAE = mean_absolute_error(y_true, y_pred)
    print("Testing Set MAE: %.2f%%" % MAE) 
    result.append(MAE)
    # Calculate RMSE
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    print("Testing Set RMSE: %.2f%%" % RMSE)
    result.append(RMSE)
    # Calculate R2 score
    R2 = r2_score(y_true, y_pred)
    print("Testing Set R2: %.4f" % R2)
    result.append(R2)
    return result

'''XGBoost model construction and training'''
param = {}
param, best_score = optimizer_optuna(X_train,y_train,40, "TPE", 'xgb')
model_xgb = XGBRegressor(**param)
model_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              eval_metric='rmse', early_stopping_rounds=100,
              verbose=True)
y_pred_train = model_xgb.predict(X_train)
evaluate_model_performance(y_train, y_pred_train)

y_pred_test = model_xgb.predict(X_test)
evaluate_model_performance(y_test, y_pred_test)


# 创建 SHAP 解释器
explainer = shap.Explainer(model_xgb, X_test)

# 计算 SHAP 值
shap_values = explainer(X_train)

# 获取特征重要性（SHAP值的平均绝对值）
feature_importance = shap_values.abs.mean(axis=0).values

# 如果 X_train 是一个稀疏矩阵，需要提取特征名。假设特征名为 'f_0', 'f_1', ... 'f_n'，你可以根据实际数据格式调整。
num_features = X_test.shape[1]  # 特征的数量
feature_names = [f'f_{i}' for i in range(num_features)]  # 假设特征名是 f_0, f_1, f_2, ...

# 获取前十个特征，按 SHAP 值的绝对值排序
top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:10]


# 输出排名前十的特征
print("Top 10 Features based on SHAP importance:")
print(top_features)


