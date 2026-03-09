# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 09:22:17 2025

@author: hj
"""

# ==============================================================================
# 1. **Modeling Libraries**: These libraries are used to build and train machine learning models.
# ==============================================================================
# Regression models
from xgboost import XGBRegressor  # XGBoost regressor
from catboost import CatBoostRegressor  # CatBoost regressor
import lightgbm as lgb  # LightGBM package
from lightgbm import LGBMRegressor  # LightGBM regressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,BaggingRegressor 
from sklearn.tree import DecisionTreeRegressor
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
def optuna_objective(x_train,y_train,trial,model): 
    # Define the parameter space based on the selected model
    if model == 'xgb':
        params = {
            "max_depth": trial.suggest_int("max_depth", 8, 17, step=1),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
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
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 8, 17,step=1),
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

    
def train_and_evaluate_model(x, y, modeltype,splitting='Random splitting',iterations=20):
    
    columns = ['Iteration', 'MAE_train', 'RMSE_train', 'R2_train', 'MAE_test', 'RMSE_test', 'R2_test']
    results_df = pd.DataFrame(columns=columns)
    for iters in range(0, iterations):
        i = iters * 10
        x_train, y_train, x_test, y_test = train_test_split(x, y, random_state=i)     

        # Hyperparameter optimization
        param = {}
        param, best_score = optimizer_optuna(x_train,y_train,40, "TPE",modeltype)  
        if modeltype == 'xgb':
            model = XGBRegressor(**param)
            model.fit(x_train, y_train, eval_set=[(x_test, y_test)],
                      eval_metric='rmse', early_stopping_rounds=20,
                      verbose=True)  
            
        elif modeltype == 'lgb':
            # 创建模型
            model = LGBMRegressor(**param)
            # 训练模型
            model.fit(x_train, y_train)
            
        elif modeltype == 'catb':
            model = CatBoostRegressor(**param)
            y_train = y_train.ravel()
            y_test = y_test.ravel()
            model.fit(x_train, y_train)

        elif modeltype == 'RF':
            imputer = SimpleImputer(strategy='mean')
            model_rf = RandomForestRegressor(**param)
            model = Pipeline([('imputer', imputer), ('model', model_rf)])
            
            # 在管道中进行数据拟合
            model.fit(x_train, y_train)

        else:
            raise ValueError("Invalid model type. Supported models are 'xgb', 'lgb', 'catb', and 'RF'.")
        # Model construction     
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        result_train=evaluate_model_performance(y_train, y_pred_train)
        result_test=evaluate_model_performance(y_test, y_pred_test)
        
        # Save results
        results_df.loc[iters] = [i, result_train[0], result_train[1], result_train[2], result_test[0], result_test[1],result_test[2]]
    
    return results_df

dataset = pd.read_excel('D:/data3.xlsx', 'dataset_all', index_col=None, keep_default_na=True)
x = dataset.iloc[:, np.r_[3:19]]
y = dataset.iloc[:, 22:23].to_numpy(dtype=np.float64).ravel()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=60)



# 创建一个基本的决策树回归器
base_regressor = DecisionTreeRegressor(max_depth=13,random_state=42)
# 使用 Bagging 回归训练模型
bagging_regressor = BaggingRegressor(base_regressor, n_estimators=1000, random_state=42)
bagging_regressor.fit(x_train, y_train)
# 对测试集进行预测
y_pred_train = bagging_regressor.predict(x_train)
evaluate_model_performance(y_train, y_pred_train)
y_pred = bagging_regressor.predict(x_test)
evaluate_model_performance(y_test, y_pred)

from scipy.interpolate import RegularGridInterpolator
from sklearn.model_selection import GridSearchCV
# 定义决策树回归器
dt = DecisionTreeRegressor(random_state=42)

# 定义Bagging回归器，基学习器为决策树
bagging_regressor = BaggingRegressor(estimator=dt, random_state=42)

# 定义超参数的搜索空间
depth_range = range(8, 18, 1)  # 决策树最大深度的范围
estimator_range = range(10,300, 20)  # AdaBoost的基学习器数目的范围

param_grid = {
    'estimator__max_depth': depth_range,
    'n_estimators': estimator_range
    }

# 使用网格搜索进行超参数调优
grid_search = GridSearchCV(bagging_regressor, param_grid, cv=5, scoring='r2')
grid_search.fit(x_train, y_train)

# 获取最佳参数组合和结果
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Cross-validation Score: {best_score}")

# 获取所有结果并整理为矩阵形式
results = grid_search.cv_results_
mean_test_scores = results['mean_test_score']

# 获取所有不同的 max_depth 和 n_estimators 组合的 MSE
score_matrix = np.array(mean_test_scores).reshape(len(depth_range), len(estimator_range))

# 可视化
from scipy import interpolate

# 对 score_matrix 进行插值
interp_func = interpolate.interp2d(estimator_range, depth_range, score_matrix, kind='cubic')

# 生成更高分辨率的网格
estimator_range_fine = np.linspace(min(estimator_range), max(estimator_range), 100)
depth_range_fine = np.linspace(min(depth_range), max(depth_range), 100)

# 插值后的分数矩阵
score_matrix_fine = interp_func(estimator_range_fine, depth_range_fine)

# 绘制插值后的等高线图
X_grid_fine, Y_grid_fine = np.meshgrid(estimator_range_fine, depth_range_fine)
plt.figure(figsize=(7, 6))
contour = plt.contourf(X_grid_fine, Y_grid_fine, score_matrix_fine, cmap='viridis', vmin=0.736, vmax=0.800)
plt.title("Interpolated Mean CV Score (Neg MSE) Contour Plot")
plt.xlabel("Number of Estimators")
plt.ylabel("Max Depth")
plt.colorbar(contour, label="Negative MSE")
plt.show()

'''AdaBoost模型训练与超参数优化'''
# 创建SimpleImputer对象，选择均值填充策略
imputer = SimpleImputer(strategy='mean')

# 使用均值填充训练集和测试集的缺失值
x_train = imputer.fit_transform(x_train)  # 训练集填充
x_test = imputer.transform(x_test)  # 测试集填充
# 定义决策树回归器作为基学习器
dt = DecisionTreeRegressor(max_depth=12,random_state=42)
# 定义AdaBoost回归器，基学习器为决策树
ada_boost_regressor = AdaBoostRegressor(n_estimators=350,learning_rate=0.1,estimator=dt, random_state=42)
ada_boost_regressor.fit(x_train, y_train)
# 对测试集进行预测
y_pred_train = ada_boost_regressor.predict(x_train)
evaluate_model_performance(y_train, y_pred_train)
y_pred = ada_boost_regressor.predict(x_test)
evaluate_model_performance(y_test, y_pred)

'''第一轮网格搜索'''
# 定义决策树回归器作为基学习器
dt = DecisionTreeRegressor(random_state=42)

# 定义AdaBoost回归器，基学习器为决策树
ada_boost_regressor = AdaBoostRegressor(estimator=dt, random_state=42)

# 定义超参数的搜索空间
depth_range = range(8, 18, 1)  # 决策树最大深度的范围
estimator_range = range(100, 1001, 100)  # AdaBoost的基学习器数目的范围

param_grid = {
    'estimator__max_depth': depth_range,  # 决策树最大深度
    'n_estimators': estimator_range  # 基学习器的数量
}

# 使用网格搜索进行超参数调优
grid_search = GridSearchCV(ada_boost_regressor, param_grid, cv=5, scoring='r2')
grid_search.fit(x_train, y_train)

# 获取最佳参数组合和结果
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Cross-validation Score: {best_score}")

# 获取所有结果并整理为矩阵形式
results = grid_search.cv_results_
mean_test_scores = results['mean_test_score']

# 获取所有不同的 max_depth 和 n_estimators 组合的 MSE
score_matrix = np.array(mean_test_scores).reshape(len(depth_range), len(estimator_range))

# 使用 RegularGridInterpolator 进行插值
interp_func = RegularGridInterpolator((estimator_range, depth_range), score_matrix, method='cubic' )

# 生成更高分辨率的网格
estimator_range_fine = np.linspace(min(estimator_range), max(estimator_range), 100)
depth_range_fine = np.linspace(min(depth_range), max(depth_range), 100)

# 生成二维网格并进行插值
X_grid_fine, Y_grid_fine = np.meshgrid( depth_range_fine,estimator_range_fine)

# 使用插值函数对整个网格进行插值，得到一个与新网格大小一致的二维矩阵
points_fine = np.array([Y_grid_fine.ravel(), X_grid_fine.ravel()]).T
score_matrix_fine = interp_func(points_fine)

# 将一维的插值结果重新整理成二维矩阵
score_matrix_fine_1 = score_matrix_fine.reshape(X_grid_fine.shape)


'''第二轮网格搜索'''
# 定义决策树回归器作为基学习器，最大深度固定为12
dt = DecisionTreeRegressor(max_depth=12, random_state=42)

# 定义AdaBoost回归器，基学习器为决策树
ada_boost_regressor = AdaBoostRegressor(estimator=dt, random_state=42)

# 定义超参数的搜索空间
estimator_range = range(100, 1001,100)  # AdaBoost的基学习器数目的范围
learning_rate_range = [0.01,0.05,0.1,0.2,0.3,0.4,0.5]# 学习率的范围，从0.01到0.5，步长不固定

param_grid = {
    'n_estimators': estimator_range,  # 基学习器的数量
    'learning_rate': learning_rate_range  # 学习率
}

# 使用网格搜索进行超参数调优
grid_search = GridSearchCV(ada_boost_regressor, param_grid, cv=5, scoring='r2')
grid_search.fit(x_train, y_train)

# 获取最佳参数组合和结果
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Cross-validation Score: {best_score}")



# 获取所有结果并整理为矩阵形式
results = grid_search.cv_results_
mean_test_scores = results['mean_test_score']

# 获取所有不同的 n_estimators 和 learning_rate 组合的 MSE
score_matrix = np.array(mean_test_scores).reshape(len(estimator_range), len(learning_rate_range))


# 转置 score_matrix，使其符合插值函数的要求
score_matrix = score_matrix.T  # 转置

# 使用 RegularGridInterpolator 进行插值
interp_func = RegularGridInterpolator((learning_rate_range,estimator_range ), score_matrix, method='cubic')

# 生成更高分辨率的网格
estimator_range_fine = np.linspace(min(estimator_range), max(estimator_range), 100)
learning_rate_range_fine = np.linspace(min(learning_rate_range), max(learning_rate_range), 100)

# 生成二维网格并进行插值
X_grid_fine, Y_grid_fine = np.meshgrid( estimator_range_fine,learning_rate_range_fine)

# 使用插值函数对整个网格进行插值，得到一个与新网格大小一致的二维矩阵
points_fine = np.array([Y_grid_fine.ravel(), X_grid_fine.ravel()]).T
score_matrix_fine = interp_func(points_fine)

# 将一维的插值结果重新整理成二维矩阵
score_matrix_fine = score_matrix_fine.reshape(X_grid_fine.shape)


# 获取两个图的全局最大最小值
vmin = min(score_matrix_fine_1.min(), score_matrix_fine.min())
vmax = max(score_matrix_fine_1.max(), score_matrix_fine.max())

# 绘制插值后的等高线图_第一轮优化
X_grid_fine_1, Y_grid_fine_1 = np.meshgrid(estimator_range_fine, depth_range_fine)
plt.figure(figsize=(7, 6))
contour = plt.contourf(X_grid_fine_1, Y_grid_fine_1, score_matrix_fine_1, cmap='viridis', vmin=0.74, vmax=0.81)
plt.title("for AdaBoost")
plt.xlabel("Number of Estimators")
plt.ylabel("Max Depth")
plt.colorbar(contour, label="Negative MSE")
plt.show()

# 绘制插值后的等高线图_第二轮优化
X_grid_fine, Y_grid_fine = np.meshgrid(estimator_range_fine, learning_rate_range_fine)
plt.figure(figsize=(7, 6))
contour = plt.contourf(X_grid_fine, Y_grid_fine, score_matrix_fine, cmap='viridis', vmin=0.74, vmax=0.81)
plt.title("for AdaBoost")
plt.xlabel("Number of Estimators")
plt.ylabel("Learning Rate")
plt.colorbar(contour, label="Negative MSE")
plt.show()

'''随机森林算法超参数优化'''
# Create a simple imputer to fill missing values with the mean
param = {}
param, best_score = optimizer_optuna(x_train,y_train,40, "TPE", 'RF')

# Build a pipeline combining the imputer and model
imputer = SimpleImputer(strategy='mean')
model_rf = RandomForestRegressor(**param)
pipeline = Pipeline([('imputer', imputer), ('model', model_rf)])

# Fit the model using the pipeline
pipeline.fit(x_train, y_train)

# Model prediction and evaluation
y_pred_train_rf = pipeline.predict(x_train)
print("Random Forest - Training Set:")
evaluate_model_performance(y_train, y_pred_train_rf)

# Predict on the test set
y_pred_test_rf = pipeline.predict(x_test)
print("Random Forest - Testing Set:")
evaluate_model_performance(y_test, y_pred_test_rf)

'''XGBoost算法超参数优化'''
param = {}
param, best_score = optimizer_optuna(x_train,y_train,40, "TPE", 'xgb')
model_xgb = XGBRegressor(**param)
model_xgb.fit(x_train, y_train, eval_set=[(x_test, y_test)],
              eval_metric='rmse', early_stopping_rounds=100,
              verbose=True)
y_pred_train = model_xgb.predict(x_train)
evaluate_model_performance(y_train, y_pred_train)

y_pred_test = model_xgb.predict(x_test)
evaluate_model_performance(y_test, y_pred_test)


