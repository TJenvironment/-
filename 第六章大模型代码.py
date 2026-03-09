# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:02:05 2024

@author: hj
"""
# Main imports
# Helper imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product,combinations
from sklearn.linear_model import (Lasso, MultiTaskElasticNetCV)
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, learning_curve,cross_validate, KFold   # Splitting dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # MSE, MAE, R2
from xgboost import XGBRegressor

from openpyxl import load_workbook
# Cross-validation and hyperparameter optimization
import re
import optuna
# SHAP model explanation method
import shap
# Functions for model performance evaluation metrics
import seaborn as sns
import os
import json  # 引入 JSON 库来处理 JSON 数据
def RMSE(y, y_pred):
    RMSE = np.sqrt(mean_squared_error(y, y_pred))
    return RMSE

def MAPE(y, y_pred):
    e = np.abs((y_pred - y) / y)
    MAPE = np.sum(e) / len(e)
    return MAPE

def custom_train_test_split(x, y, splitting='Random splitting', random_state=None,groups=None):
    """
    Custom train-test split function.
    
    Parameters:
    - x: Features
    - y: Labels
    - splitting: Type of splitting ('Random splitting' or 'Grouped random splitting')
    - random_state: Random seed for reproducibility
    
    Returns:
    - x_train, y_train: Features and labels for training set
    - x_test, y_test: Features and labels for testing set
    """
    
    if splitting == 'Random splitting':
        # Random splitting
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    elif splitting == 'Grouped random splitting':
        # Data Leakage Management: Grouped random splitting
        groups = groups
        group_id = np.unique(groups)
        group_trainid, group_testid = train_test_split(group_id, test_size=0.15, random_state=random_state)
        
        # Get training and testing set indices
        group_train = np.where(np.isin(groups, group_trainid))[0]
        group_test = np.where(np.isin(groups, group_testid))[0]
        
        # Get training and testing sets
        x_train, y_train = x[group_train], y[group_train]
        x_test, y_test = x[group_test], y[group_test]
    
    return x_train, y_train, x_test, y_test


# Hyperparameter Tuning and Model Optimization with Bayesian Optimization
# Step 1: Define the Objective Function and Parameter Space
def optuna_objective(trial): 
    # Define the parameter space
    max_depth = trial.suggest_int("max_depth", 8, 12, 1)
    learning_rate = trial.suggest_float("learning_rate", 0.1, 0.3, log=True)
    n_estimators = trial.suggest_int("n_estimators", 400, 700, 50)
    subsample = trial.suggest_float("subsample", 0.5, 1, log=False)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1, log=False)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 4, 1)
    reg_lambda = trial.suggest_int("reg_lambda", 6, 12, 1)
    
    # Define the estimator
    reg = XGBRegressor(max_depth=max_depth, 
                       learning_rate=learning_rate,      
                       n_estimators=n_estimators,
                       objective='reg:squarederror', 
                       booster='gbtree', 
                       gamma=0,
                       min_child_weight=min_child_weight,
                       subsample=subsample, 
                       colsample_bytree=colsample_bytree,       
                       reg_alpha=0,
                       reg_lambda=reg_lambda
                       )   
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
def optimizer_optuna(n_trials, algo):
    # Define the sampler
    algo = optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=24)
    
    # Create an Optuna study for optimization
    study = optuna.create_study(sampler=algo, direction="minimize")
    
    # Optimize the objective function
    study.optimize(optuna_objective,  # Objective function
                   n_trials=n_trials,   # Maximum number of iterations (including initial observations)
                   show_progress_bar=True  # Show progress bar
                  )
    
    # Display the best parameters and score
    print("\n", "\n", "best params: ", study.best_trial.params,
          "\n", "\n", "best score: ", study.best_trial.values,
          "\n")
    
    return study.best_trial.params, study.best_trial.values

def train_and_evaluate_model(x, y, splitting='Random splitting',iterations=10):
    
    columns = ['Iteration', 'MAE_train', 'RMSE_train', 'R2_train', 'MAE_test', 'RMSE_test', 'R2_test']
    results_df = pd.DataFrame(columns=columns)
    for iters in range(0, iterations):
        i = iters * 10
        global x_train, y_train
        x_train, y_train, x_test, y_test = custom_train_test_split(x, y, splitting=splitting, random_state=i)            
        # Hyperparameter optimization
        param = {}
        param, best_score = optimizer_optuna(30, "TPE")
        
        # Model construction
        model = XGBRegressor(
            max_depth=param['max_depth'],
            learning_rate=param['learning_rate'],
            n_estimators=param['n_estimators'],
            objective='reg:squarederror',
            booster='gbtree',
            gamma=0.2,
            min_child_weight=param['min_child_weight'],
            subsample=param['subsample'],
            colsample_bytree=param['colsample_bytree'],
            reg_alpha=0,
            reg_lambda=param['reg_lambda']
        )
        
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)],
                  eval_metric='rmse', early_stopping_rounds=20,
                  verbose=True)
        
        y_pred_train = model.predict(x_train)
        y_pred_train = np.reshape(y_pred_train, (len(y_pred_train), 1))
        y_pred_train = np.where(y_pred_train > 100, 100, y_pred_train)
        y_pred_train = np.where(y_pred_train < 0, 0, y_pred_train)
        MAE_train = mean_absolute_error(y_train, y_pred_train)
        print("Training Set MAE: %.2f%%" % (MAE_train))
        RMSE_train = RMSE(y_train, y_pred_train)
        print("Training Set RMSE: %.2f%%" % (RMSE_train))
        R2_train = r2_score(y_train, y_pred_train)
        print("Training Set R2: %.4f" % (R2_train))
        
        y_pred_test = model.predict(x_test)
        y_pred_test = np.reshape(y_pred_test, (len(y_pred_test), 1))
        y_pred_test = np.where(y_pred_test > 100, 100, y_pred_test)
        y_pred_test = np.where(y_pred_test < 0, 0, y_pred_test)
        MAE_test = mean_absolute_error(y_test, y_pred_test)
        print("Testing Set MAE: %.2f%%" % (MAE_test))
        RMSE_test = RMSE(y_test, y_pred_test)
        print("Testing Set RMSE: %.2f%%" % (RMSE_test))
        R2_test = r2_score(y_test, y_pred_test)
        print("Testing Set R2: %.4f" % (R2_test))
        
        # Save results
        results_df.loc[iters] = [i, MAE_train, RMSE_train, R2_train, MAE_test, RMSE_test, R2_test]
    
    return results_df

from decimal import Decimal
def convert_to_jsonl(x, y):
    """
    Convert x and y DataFrames to a list of dictionaries in JSONL format.
    
    Args:
        x (pd.DataFrame): Input features DataFrame.
        y (pd.DataFrame or np.ndarray): Corresponding rejection values.
        
    Returns:
        List of dictionaries in JSONL format.
    """
    jsonl_data = []

    for i in range(len(y)):
        # Prepare the feature-content string for x
        # 修改列名：按空格分割
        columns = [col.split('(', 1)[0].rstrip() for col in x.columns]
        units = [re.search(r'\((.*?)\)', col)  # 正则提取括号中的内容
         .group(1) if re.search(r'\((.*?)\)', col) else ''  # 如果没有括号则为空字符串
         for col in x.columns]
        x_content = ', '.join([f"{col}={Decimal(value):.{3}g} {unit}" 
                       for col, value, unit in zip(columns, x.iloc[i], units)])
        
        # Prepare the rejection value for y
        y_value = y[i] if isinstance(y, pd.DataFrame) else y[i]

        # Create the dictionary based on the template
        item = [
            {
                "role": "system",
                "content": "You should model the relationship between input features (membrane characteristics, OMP properties, opreation conditions) and the corresponding OMP rejection, and provide precise, data-driven responses based on them."
            },
            {
                "role": "user",
                "content": f"Predict the rejection of this OMP sample under the condition: {x_content}?"
            },
            {
                "role": "assistant",
                "content": f"{y_value[0]:.1f}%"
            }
        ]
        jsonl_data.append({"messages":item})

    return jsonl_data

def convert_to_jsonl_sque(x, y):
    """
    Convert x and y DataFrames to a list of dictionaries in JSONL format.
    
    Args:
        x (pd.DataFrame): Input features DataFrame.
        y (pd.DataFrame or np.ndarray): Corresponding rejection values.
        
    Returns:
        List of dictionaries in JSONL format.
    """
    jsonl_data = []

    for i in range(len(y)):
        # Prepare the feature-content string for x
        # 修改列名：按空格分割
        col_list = [f"{col}" for col in x.columns]
        columns = ', '.join(col_list[:-1]) + ' and ' + col_list[-1]
        x_content = ', '.join([f"{Decimal(value):.{3}g}" 
                       for value in x.iloc[i]])
        
        # Prepare the rejection value for y
        y_value = y[i] if isinstance(y, pd.DataFrame) else y[i]

        # Create the dictionary based on the template
        item = [
            {
                "role": "system",
                "content": f"You should provide precise, data-driven prediction of OMP rejection based on the input features, which are {columns}."
            },
            {
                "role": "user",
                "content": f"{x_content}"
            },
            {
                "role": "assistant",
                "content": f"{y_value[0]:.1f}%"
            }
        ]
        jsonl_data.append({"messages":item})

    return jsonl_data


def convert_to_jsonl2(x, y):
    """
    Convert x and y DataFrames to a list of dictionaries in JSONL format.
    
    Args:
        x (pd.DataFrame): Input features DataFrame.
        y (pd.DataFrame or np.ndarray): Corresponding rejection values.
        
    Returns:
        List of dictionaries in JSONL format.
    """
    jsonl_data = []

    for i in range(len(y)):
        # Prepare the feature-content string for x
        # 修改列名：按空格分割
        columns = [col.split('(', 1)[0].rstrip() for col in x.columns]
        units = [re.search(r'\((.*?)\)', col)  # 正则提取括号中的内容
         .group(1) if re.search(r'\((.*?)\)', col) else ''  # 如果没有括号则为空字符串
         for col in x.columns]
        x_content = ', '.join([f"{col}={Decimal(value):.{3}g} {unit}" 
                       for col, value, unit in zip(columns, x.iloc[i], units)])
        
        # Prepare the rejection value for y
        y_value = y[i] if isinstance(y, pd.DataFrame) else y[i]

        # Create the dictionary based on the template
        
        item = [ {"system": "You are an expert that focus on data-driven modeling of membrane separation. Your task is perform systematic OMP rejection prediction through modeling the relationship between input features (membrane characteristics, OMP properties, opreation conditions) and OMP rejection.",
                "prompt": f"Predict the rejection of this OMP sample under the condition: {x_content}.",
                "response":f"{y_value[0]:.1f}%"}]

        jsonl_data.append(item)
    return jsonl_data
def convert_to_jsonl_R(x, y):
    """
    Convert x and y DataFrames to a list of dictionaries in JSONL format.
    
    Args:
        x (pd.DataFrame): Input features DataFrame.
        y (pd.DataFrame or np.ndarray): Corresponding rejection values.
        
    Returns:
        List of dictionaries in JSONL format.
    """
    jsonl_data = []

    for i in range(len(y)):
        # Prepare the feature-content string for x
        # 修改列名：按空格分割
        columns = [col.split('(', 1)[0].rstrip() for col in x.columns]
        units = [re.search(r'\((.*?)\)', col)  # 正则提取括号中的内容
         .group(1) if re.search(r'\((.*?)\)', col) else ''  # 如果没有括号则为空字符串
         for col in x.columns]
        x_content = ', '.join([f"{col}={Decimal(value):.{3}g} {unit}" 
                       for col, value, unit in zip(columns, x.iloc[i], units)])
        
        # Prepare the rejection value for y
        y_value = y[i] if isinstance(y, pd.DataFrame) else y[i]

        # Create the dictionary based on the template
        
        item = [ {"system": "You are an expert that focus on data-driven modeling of membrane separation. Your task is perform systematic OMP rejection prediction through modeling the relationship between input features (membrane characteristics, OMP properties, opreation conditions) and OMP rejection.",
                "prompt": f"Predict the rejection of this OMP sample under the condition: {x_content}.",
                "reasoning_content":"To predict the OMP rejection, I need to carefully analyze the interactions between the input features and consider how each factor influences the overall rejection process, and apply ny knowledge of membrane filtration principles to make accurate predictions.",
                "response":f"{y_value[0]:.1f}%"}]

        jsonl_data.append(item)
    return jsonl_data
# 定义微调模型性能评价方法
def fine_tuned_model_evaluation(jsonl_train,jsonl_test, ft_model, pattern):
    """
    评估微调模型的性能。

    参数：
    jsonl_train (list): 包含训练数据的列表，每个元素是一个字典，包含“messages”键。
    ft_model: 已微调的模型，用于预测。
    pattern (str): 用于从消息中提取百分比的正则表达式模式。

    返回：
    dict: 包含MAE, RMSE, R2的字典，分别对应训练集的绝对误差、均方根误差和R2得分。
    """
    # 初始化空列表，用于存储提取的结果
    train_x = []
    train_y = []
    train_y_predicted = []

    # 遍历jsonl_train的每个元素
    for item in jsonl_train:
        # 获取"messages"键对应的列表
        message = item["messages"]

        # 提取前两个元素并将其存储为train_x
        train_x.append({"message": message[:2]})

        # 提取第三个元素并将其存储为train_y
        # 使用正则表达式提取百分比值
        match = re.search(pattern, message[2]["content"])
        if match:
            # 将提取出的百分比值转为浮动数，并存储到train_y
            percentage = float(match.group(1))  # group(1)是匹配到的数字部分
            train_y.append(percentage)

        # 使用调优模型预测
        completion = ft_model.chat.completions.create(
            model=ft_model,
            messages=message[:2]
        )

        match = re.search(pattern, completion.choices[0].message.content)
        if match:
            # 将提取出的百分比值转为浮动数，并存储到train_y_predicted
            percentage = float(match.group(1))  # group(1)是匹配到的数字部分
            train_y_predicted.append(percentage)

    # 将train_y和train_y_predicted转换为numpy数组
    train_y = np.array(train_y)
    train_y_predicted = np.array(train_y_predicted)

    # 计算MAE、RMSE和R2
    MAE_train = mean_absolute_error(train_y, train_y_predicted)
    RMSE_train = RMSE(train_y, train_y_predicted)
    R2_train = r2_score(train_y, train_y_predicted)

    # 打印并返回性能评价
    print(f"Training Set MAE: {MAE_train:.2f}%")
    print(f"Training Set RMSE: {RMSE_train:.2f}%")
    print(f"Training Set R2: {R2_train:.4f}")

    return {
        "MAE": MAE_train,
        "RMSE": RMSE_train,
        "R2": R2_train
    }

dataset = pd.read_excel('D:/data6.xlsx', 'dataset_all', index_col=None, keep_default_na=True)
x = dataset.iloc[:, np.r_[3:19]]
y = dataset.iloc[:, 22:23].to_numpy(dtype=np.float64)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=60)



# 检查'Pore radius (nm)'列是否缺失，缺失为1，不缺失为0
missing_values = x_test['Pore radius (nm)'].isna().astype(int)

# 将结果转换为列表
missing_values_list = missing_values.tolist()




# '大模型输入数据预处理'
# # Convert to JSONL format
jsonl_train = convert_to_jsonl(x_train, y_train)
jsonl_test = convert_to_jsonl(x_test, y_test)
jsonl_test_sque = convert_to_jsonl_sque(x_test, y_test)
jsonl_train_sque = convert_to_jsonl_sque(x_train, y_train)
# # Save to a file or print
# with open('GPT_train.jsonl', 'w') as f:
#     for line in jsonl_train:
#         f.write(json.dumps(line) + '\n')
        
# # Save to a file or print
# with open('GPT_test.jsonl', 'w') as f:
#     for line in jsonl_test:
#         f.write(json.dumps(line) + '\n')
# print("Conversion to JSONL complete.")
# # Save to a file or print
# with open('gpt_sq_train.jsonl', 'w') as f:
#     for line in jsonl_train_sque:
#         f.write(json.dumps(line) + '\n')

# # Save to a file or print
# with open('gpt_sq_test.jsonl', 'w') as f:
#     for line in jsonl_test_sque:
#         f.write(json.dumps(line) + '\n')
# print("Conversion to JSONLR complete.")








'大模型API调用'
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
client = OpenAI()





'微调数据上传与训练'
# client.files.create(
#   file=open("GPT_train.jsonl", "rb"),
#   purpose="fine-tune"
# )
# #'file-6etLcwxSZGtuQ8XN8mdtxU'
# client.files.create(
#   file=open("GPT_test.jsonl", "rb"),
#   purpose="fine-tune"
# )
# #'file-1XhCXCaVZmZHfkwedApGR7'

# client.files.create(
#   file=open("gpt_sq_train.jsonl", "rb"),
#   purpose="fine-tune"
# )
# #'file-AUehHXmEeknWZgate2ScKx'
# client.files.create(
#   file=open("gpt_sq_test.jsonl", "rb"),
#   purpose="fine-tune"
# )
# #'file-M62fEhNSYpVPHDhApNKrYK'

# client.files.list()
# client.files.delete('file-V1rsTLK5xiKvWUA2ztB8Re')

'''使用结构化形式的数据微调'''
# # gpt-4o-mini
# # id='ftjob-bNfIs2MSgiaw4K7UTAGCqTbx'
# client.fine_tuning.jobs.create(
#     training_file='file-6etLcwxSZGtuQ8XN8mdtxU',
#     model="gpt-4o-mini-2024-07-18",
#     validation_file= 'file-1XhCXCaVZmZHfkwedApGR7'
# )
# # gpt-4.1-mini
# # id='ftjob-elsU3F6tWjOYAUmAdwusJePq'
# client.fine_tuning.jobs.create(
#     training_file='file-6etLcwxSZGtuQ8XN8mdtxU',
#     model="gpt-4.1-mini-2025-04-14",
#     validation_file= 'file-1XhCXCaVZmZHfkwedApGR7'
# )

'''使用序列形式的数据微调'''
# # id='ftjob-AminzvKWFQliFcj2H04S8a4O'
# client.fine_tuning.jobs.create(
#     training_file='file-AUehHXmEeknWZgate2ScKx',
#     model="gpt-4.1-mini-2025-04-14",
#     validation_file= 'file-M62fEhNSYpVPHDhApNKrYK'
# )

# id='ftjob-UzKycUROESauGo22WEV1RzG1'
# client.fine_tuning.jobs.create(
#     training_file='file-AUehHXmEeknWZgate2ScKx',
#     model="gpt-4o-mini-2024-07-18",
#     validation_file= 'file-M62fEhNSYpVPHDhApNKrYK'
# )
# 

# gpt-4.1
# id=
# client.fine_tuning.jobs.create(
#     training_file='file-6etLcwxSZGtuQ8XN8mdtxU',
#     model="gpt-4.1-2025-04-14",
#     validation_file= 'file-1XhCXCaVZmZHfkwedApGR7'
# )



'微调任务与进程查看'
# client.fine_tuning.jobs.cancel('ftjob-0f6VEdCV927SlqqBlEr9Ln8N')

# # List 10 fine-tuning jobs
client.fine_tuning.jobs.list(limit=10)
# # Retrieve the state of a fine-tune,第一次，n_epochs=3
# client.fine_tuning.jobs.retrieve("ftjob-HhpKdE8kcED9J3VQjfLE9Vce")
# ftmodel_mini='ft:gpt-4o-mini-2024-07-18:tongji-university::AfoVlLAU'

# # Retrieve the state of a fine-tune,第二次，n_epochs+1
# client.fine_tuning.jobs.retrieve('ftjob-KGXEY6wu89NaNouZwrxQQfc7')

# client.fine_tuning.jobs.retrieve('ftjob-7rqvcZYGppEbrlhPPWYeCB8q')
# ftmodel_4o='ft:gpt-4o-2024-08-06:tongji-university::Aj04dgak'


'大模型预测性能测试'


def test_ft_model_performance(jsonl_test, model, client):
    """
    测试Fine-tuned模型的性能
    :param jsonl_test: jsonl 格式的测试数据，包含消息列表
    :param model: 微调后的模型名称或ID
    :param client: 客户端实例，用于调用模型进行预测
    :return: None
    """
    pattern = r"(\d+\.?\d*)%"  # 匹配数字及其后的百分号
    # 初始化空列表，用于存储提取的结果
    test_y = []  # 实际的标签（百分比）
    test_y_predicted = []  # 模型预测的标签（百分比）
    
    # 遍历测试集中的每个元素
    for item in jsonl_test:
        # 获取"messages"键对应的列表
        message = item["messages"]
        
        # 提取真实标签（第三个元素中的百分比）
        match = re.search(pattern, message[2]["content"])
        if match:
            percentage = float(match.group(1))  # 提取并转换为浮动数
            test_y.append(percentage)
        
        # 使用Fine-tuned模型进行预测
        completion = client.chat.completions.create(
            model=model,
            messages=message[:2]
        )
        
        # 提取模型预测的结果中的百分比
        match = re.search(pattern, completion.choices[0].message.content)
        if match:
            percentage = float(match.group(1))  # 提取并转换为浮动数
            test_y_predicted.append(percentage)
    
    # 将test_y和test_y_predicted转换为numpy数组
    test_y = np.array(test_y)
    test_y_predicted = np.array(test_y_predicted)
    
    # 计算MAE（平均绝对误差）
    MAE_test = mean_absolute_error(test_y, test_y_predicted)
    print("Testing Set MAE: %.2f%%" % (MAE_test))
    
    # 计算RMSE（均方根误差）
    RMSE_test = np.sqrt(mean_squared_error(test_y, test_y_predicted))
    print("Testing Set RMSE: %.2f%%" % (RMSE_test))
    
    # 计算R2分数
    R2_test = r2_score(test_y, test_y_predicted)
    print("Testing Set R2: %.4f" % (R2_test))
    
    # 返回评估结果
    return MAE_test, RMSE_test, R2_test, test_y, test_y_predicted 

# 调用方法来评估模型性能
# 假设jsonl_test是测试数据，client是API客户端，ftmodel_241220是微调后的模型

fine_tuned_model_41mini='ft:gpt-4.1-mini-2025-04-14:tongji-university::BU9RYQim'
fine_tuned_model_41mini_sque='ft:gpt-4.1-mini-2025-04-14:tongji-university::BU7PF81s'
fine_tuned_model_4omini='ft:gpt-4o-mini-2024-07-18:tongji-university::BU7IRk0l'
fine_tuned_model_4omini_sque='ft:gpt-4o-mini-2024-07-18:tongji-university::BUXE0mZX'
# MAE_test, RMSE_test, R2_test, test_y, test_y_predicted  = test_ft_model_performance(jsonl_test, fine_tuned_model_41mini, client)
# MAE_test, RMSE_test, R2_test, test_y, test_y_predicted  = test_ft_model_performance(jsonl_test, fine_tuned_model_4omini, client)
# MAE_test, RMSE_test, R2_test, test_y, test_y_predicted  = test_ft_model_performance(jsonl_test_sque, fine_tuned_model_41mini_sque, client)
# MAE_test, RMSE_test, R2_test, test_y, test_y_predicted  = test_ft_model_performance(jsonl_test_sque, fine_tuned_model_4omini_sque, client)

message={"messages": [{"role": "system", "content": "You should model the relationship between input features (membrane characteristics, OMP properties, opreation conditions) and the corresponding OMP rejection, and provide precise, data-driven responses based on them."}, {"role": "user", "content": "Predict the rejection of this OMP sample under the condition: Pressure=5 bar, pH=7 , Temperature=25 oC, Filtration duration=24 h, OMP concentration=0.0200 mg/L, Cross-flow velocity=33.5 cm/s, MWCO=330 Da, Pore radius=0.420 nm, Pure water permeability=15.2 L m-2 h-1 bar-1, Zeta potential=-54.8 mV, Water contact angle=44.8 o, MW=267 Da, Molecular radius=0.406 nm, pKa1=NaN , pKa2=9.56 , log Kow=1.88 ?"}]}
# 使用Fine-tuned模型进行预测
completion =client.chat.completions.create(
        model=fine_tuned_model_4omini,
        messages=message)

# 提取模型预测的结果中的百分比
match = completion.choices[0].message.content
print(match)


'''代理模型评估大模型特征重要性变化'''
# results = pd.read_excel('D:/data.xlsx', 'Results', index_col=None, keep_default_na=True)
# dataset = pd.read_excel('D:/data.xlsx', 'dataset_all', index_col=None, keep_default_na=True)
# for i in range(6):

#     x = dataset.iloc[:, np.r_[3:19]]
#     y_test = results.iloc[:,2:3].to_numpy(dtype=np.float64)
#     x_train, x_test = train_test_split(x, test_size=0.1, random_state=60)
    
#     param = {}
#     param, best_score = optimizer_optuna(x_test, y_test, 40, "TPE", 'xgb')
    
#     # 构建并训练模型
#     model_xgb = XGBRegressor(**param, eval_metric='rmse')
#     model_xgb.fit(x_test,y_test, verbose=False)# 全量预测
    
#     # 评估保真度
#     y_pred_test = model_xgb.predict(x_test) 
#     result_test=evaluate_model_performance(y_test, y_pred_test)
    
#     # 使用plot_importance方法绘制XGBoost特征重要性排序图
#     plt.figure(figsize=(12, 8))
#     plot_importance(model_xgb, max_num_features=16, importance_type='weight', xlabel='F score', title='XGBoost Feature Importance (Weight)')
#     plt.show()
#     '''SHAP interpretability method'''
#     explainer = shap.TreeExplainer(model_xgb) # Initialize the explainer
#     expected_value = explainer.expected_value # Calculate the baseline value for the entire sample
#     shap_values = explainer.shap_values(x_test) # Calculate SHAP values for each feature of each sample
#     shap_explanation = shap.Explanation(shap_values, data=x_test, feature_names=np.array(x.columns))
    
#     '''Global importance ranking plot, indicating the overall importance of each feature (agnostic to positive or negative, averaging all SHAP values)'''
#     shap.summary_plot(shap_values, x_test, plot_type="bar", show=True)


'''数据集格式检查及成本估算'''
# import json
# import tiktoken # for token counting
# import numpy as np
# from collections import defaultdict

# data_path = "GPT_train.jsonl"

# # Load the dataset
# with open(data_path, 'r', encoding='utf-8') as f:
#     dataset = [json.loads(line) for line in f]

# # Initial dataset stats
# print("Num examples:", len(dataset))
# print("First example:")
# for message in dataset[0]["messages"]:
#     print(message)

# # Format error checks
# format_errors = defaultdict(int)

# for ex in dataset:
#     if not isinstance(ex, dict):
#         format_errors["data_type"] += 1
#         continue
        
#     messages = ex.get("messages", None)
#     if not messages:
#         format_errors["missing_messages_list"] += 1
#         continue
        
#     for message in messages:
#         if "role" not in message or "content" not in message:
#             format_errors["message_missing_key"] += 1
        
#         if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
#             format_errors["message_unrecognized_key"] += 1
        
#         if message.get("role", None) not in ("system", "user", "assistant", "function"):
#             format_errors["unrecognized_role"] += 1
            
#         content = message.get("content", None)
#         function_call = message.get("function_call", None)
        
#         if (not content and not function_call) or not isinstance(content, str):
#             format_errors["missing_content"] += 1
    
#     if not any(message.get("role", None) == "assistant" for message in messages):
#         format_errors["example_missing_assistant_message"] += 1

# if format_errors:
#     print("Found errors:")
#     for k, v in format_errors.items():
#         print(f"{k}: {v}")
# else:
#     print("No errors found")

# encoding = tiktoken.get_encoding("cl100k_base")

# # not exact!
# # simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
# def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
#     num_tokens = 0
#     for message in messages:
#         num_tokens += tokens_per_message
#         for key, value in message.items():
#             num_tokens += len(encoding.encode(value))
#             if key == "name":
#                 num_tokens += tokens_per_name
#     num_tokens += 3
#     return num_tokens

# def num_assistant_tokens_from_messages(messages):
#     num_tokens = 0
#     for message in messages:
#         if message["role"] == "assistant":
#             num_tokens += len(encoding.encode(message["content"]))
#     return num_tokens

# def print_distribution(values, name):
#     print(f"\n#### Distribution of {name}:")
#     print(f"min / max: {min(values)}, {max(values)}")
#     print(f"mean / median: {np.mean(values)}, {np.median(values)}")
#     print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")
    
# # Warnings and tokens counts
# n_missing_system = 0
# n_missing_user = 0
# n_messages = []
# convo_lens = []
# assistant_message_lens = []

# for ex in dataset:
#     messages = ex["messages"]
#     if not any(message["role"] == "system" for message in messages):
#         n_missing_system += 1
#     if not any(message["role"] == "user" for message in messages):
#         n_missing_user += 1
#     n_messages.append(len(messages))
#     convo_lens.append(num_tokens_from_messages(messages))
#     assistant_message_lens.append(num_assistant_tokens_from_messages(messages))
    
# print("Num examples missing system message:", n_missing_system)
# print("Num examples missing user message:", n_missing_user)
# print_distribution(n_messages, "num_messages_per_example")
# print_distribution(convo_lens, "num_total_tokens_per_example")
# print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
# n_too_long = sum(l > 16385 for l in convo_lens)
# print(f"\n{n_too_long} examples may be over the 16,385 token limit, they will be truncated during fine-tuning")

# # Pricing and default n_epochs estimate
# MAX_TOKENS_PER_EXAMPLE = 16385

# TARGET_EPOCHS = 3
# MIN_TARGET_EXAMPLES = 100
# MAX_TARGET_EXAMPLES = 25000
# MIN_DEFAULT_EPOCHS = 1
# MAX_DEFAULT_EPOCHS = 25

# n_epochs = TARGET_EPOCHS
# n_train_examples = len(dataset)
# if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
#     n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
# elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
#     n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

# n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
# print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
# print(f"By default, you'll train for {n_epochs} epochs on this dataset")
# print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")