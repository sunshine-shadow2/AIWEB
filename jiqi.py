import pandas as pd
import numpy as np
import os
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

def detect_ddos(input_data):
    """
    DDoS攻击检测函数，当4个以上模型预测为1时判定为攻击
    :param input_data: 无label列的DataFrame对象
    :return: 包含预测结果的DataFrame，新增'ddos_prediction'列(1表示攻击，0表示正常)
    """
    # 加载预处理工具
    imputer = load('model/imputer.joblib')
    scaler = load('model/scaler.joblib')
    
    # 数据预处理
    X_processed = imputer.transform(input_data)
    X_processed = scaler.transform(X_processed)
    
    # 加载所有模型
    model_paths = {
        'LogisticRegression': 'model/LogisticRegression.joblib',
        'SVM': 'model/SVM.joblib',
        'KNN': 'model/KNN.joblib',
        'DecisionTree': 'model/DecisionTree.joblib',
        'RandomForest': 'model/RandomForest.joblib'
    }
    models = {name: load(path) for name, path in model_paths.items()}
    
    # 存储各模型预测结果
    predictions = pd.DataFrame()
    
    # 各模型预测
    for name, model in models.items():
        y_pred = model.predict(X_processed)
        predictions[name] = y_pred
    
    # 计算预测为1的模型数量
    predictions['positive_count'] = predictions.sum(axis=1)
    
    # 当4个以上模型预测为1时判定为DDoS攻击
    input_data_with_pred = input_data.copy()
    input_data_with_pred['ddos_prediction'] = np.where(predictions['positive_count'] >= 4, 1, 0)
    input_data_with_pred['model_positive_count'] = predictions['positive_count']
    
    return input_data_with_pred

def detect_ddos_logistic_regression(input_data):
    """使用逻辑回归模型检测DDoS攻击"""
    # 加载预处理工具
    imputer = load('model/imputer.joblib')
    scaler = load('model/scaler.joblib')
    
    # 数据预处理
    X_processed = imputer.transform(input_data)
    X_processed = scaler.transform(X_processed)
    
    # 加载模型
    model = load('model/LogisticRegression.joblib')
    
    # 预测
    y_pred = model.predict(X_processed)
    
    # 构建结果DataFrame
    result = input_data.copy()
    result['ddos_prediction'] = y_pred
    result['model_positive_count'] = y_pred  # 单个模型的预测结果
    
    return result

def detect_ddos_svm(input_data):
    """使用SVM模型检测DDoS攻击"""
    # 加载预处理工具
    imputer = load('model/imputer.joblib')
    scaler = load('model/scaler.joblib')
    
    # 数据预处理
    X_processed = imputer.transform(input_data)
    X_processed = scaler.transform(X_processed)
    
    # 加载模型
    model = load('model/SVM.joblib')
    
    # 预测
    y_pred = model.predict(X_processed)
    
    # 构建结果DataFrame
    result = input_data.copy()
    result['ddos_prediction'] = y_pred
    result['model_positive_count'] = y_pred  # 单个模型的预测结果
    
    return result

def detect_ddos_knn(input_data):
    """使用KNN模型检测DDoS攻击"""
    # 加载预处理工具
    imputer = load('model/imputer.joblib')
    scaler = load('model/scaler.joblib')
    
    # 数据预处理
    X_processed = imputer.transform(input_data)
    X_processed = scaler.transform(X_processed)
    
    # 加载模型
    model = load('model/KNN.joblib')
    
    # 预测
    y_pred = model.predict(X_processed)
    
    # 构建结果DataFrame
    result = input_data.copy()
    result['ddos_prediction'] = y_pred
    result['model_positive_count'] = y_pred  # 单个模型的预测结果
    
    return result

def detect_ddos_decision_tree(input_data):
    """使用决策树模型检测DDoS攻击"""
    # 加载预处理工具
    imputer = load('model/imputer.joblib')
    scaler = load('model/scaler.joblib')
    
    # 数据预处理
    X_processed = imputer.transform(input_data)
    X_processed = scaler.transform(X_processed)
    
    # 加载模型
    model = load('model/DecisionTree.joblib')
    
    # 预测
    y_pred = model.predict(X_processed)
    
    # 构建结果DataFrame
    result = input_data.copy()
    result['ddos_prediction'] = y_pred
    result['model_positive_count'] = y_pred  # 单个模型的预测结果
    
    return result

def detect_ddos_random_forest(input_data):
    """使用随机森林模型检测DDoS攻击"""
    # 加载预处理工具
    imputer = load('model/imputer.joblib')
    scaler = load('model/scaler.joblib')
    
    # 数据预处理
    X_processed = imputer.transform(input_data)
    X_processed = scaler.transform(X_processed)
    
    # 加载模型
    model = load('model/RandomForest.joblib')
    
    # 预测
    y_pred = model.predict(X_processed)
    
    # 构建结果DataFrame
    result = input_data.copy()
    result['ddos_prediction'] = y_pred
    result['model_positive_count'] = y_pred  # 单个模型的预测结果
    
    return result

# 示例：使用shujuji1.csv数据进行检测
if __name__ == '__main__':
    # 加载测试数据（无label列）
    test_data = pd.read_csv('shujuji1.csv').drop('label', axis=1)
    
    # 调用DDoS检测函数
    result = detect_ddos(test_data)
    
    # 打印检测结果
    print('DDoS攻击检测结果:\n')
    print(result[['ddos_prediction', 'model_positive_count']].to_string(index=False))