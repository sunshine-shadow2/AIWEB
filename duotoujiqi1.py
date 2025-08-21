import pandas as pd
import numpy as np
import torch
import joblib
import os
import time
import psutil
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from duotou import MultiHeadAttentionModel, DDoSDataset, CONFIG

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(use_pruned_model=False):
    # 加载预处理工具和特征选择器
    try:
        preprocessor = joblib.load(CONFIG['preprocessor_save_path'])
    except AttributeError as e:
        if "_RemainderColsList" in str(e):
            print("错误: 检测到scikit-learn版本不兼容。预处理模型是用旧版本保存的。")
            print("解决方案1: 降低scikit-learn版本到1.6.1")
            print("解决方案2: 运行duotou.py重新训练模型并生成新的预处理工具")
            raise RuntimeError("scikit-learn版本不兼容，请尝试上述解决方案") from e
        else:
            raise
    try:
        feature_selector = joblib.load(CONFIG['feature_selector_save_path'])
    except AttributeError as e:
        if "_RemainderColsList" in str(e):
            print("错误: 检测到scikit-learn版本不兼容。特征选择器模型是用旧版本保存的。")
            print("解决方案1: 降低scikit-learn版本到1.6.1")
            print("解决方案2: 运行duotou.py重新训练模型并生成新的特征选择器")
            raise RuntimeError("scikit-learn版本不兼容，请尝试上述解决方案") from e
        else:
            raise
    try:
        svd = joblib.load('Tuo/svd_model.joblib')
    except AttributeError as e:
        if "_RemainderColsList" in str(e):
            print("错误: 检测到scikit-learn版本不兼容。SVD模型是用旧版本保存的。")
            print("解决方案1: 降低scikit-learn版本到1.6.1")
            print("解决方案2: 运行duotou.py重新训练模型并生成新的SVD模型")
            raise RuntimeError("scikit-learn版本不兼容，请尝试上述解决方案") from e
        else:
            raise
    
    # 加载测试数据
    test_df = pd.read_csv('D:\\Project\\PythonProject\\AIwab4(1)\\AIwab4\\http\\duotou.csv')
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']
    
    # 预处理数据
    X_test_processed = preprocessor.transform(X_test).astype('float32')
    X_test_processed = feature_selector.transform(X_test_processed)
    X_test_processed = svd.transform(X_test_processed)
    
    # 获取输入维度
    input_dim = X_test_processed.shape[1]
    
    # 动态调整序列长度以匹配可用数据量
    data_length = len(X_test_processed)
    adjusted_seq_length = min(CONFIG['seq_length'], data_length)
    
    if adjusted_seq_length < 1:
        raise ValueError(f"测试数据样本数({data_length})不足，无法创建有效序列")
    
    # 创建数据集和数据加载器
    test_dataset = DDoSDataset(X_test_processed, y_test.values, seq_length=adjusted_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 根据参数选择加载剪枝或未剪枝模型
    if use_pruned_model:
        model_path = 'Tuo/multi_head_attention_model_pruned_best.pth'
        print("使用剪枝后的模型进行预测")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"剪枝模型文件不存在: {model_path}")
        
        # 加载检查点
        model_path = 'Tuo/multi_head_attention_model_pruned_best.pth'
        print(f"加载剪枝后的最佳模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 从检查点中获取剪枝后的头数量和模型维度
        pruned_num_heads = checkpoint['num_heads']
        pruned_model_dim = checkpoint['model_dim']
        state_dict = checkpoint['state_dict']

        print(f"从检查点中加载剪枝参数: 头数量={pruned_num_heads}, 模型维度={pruned_model_dim}")

        # 使用剪枝后的参数初始化模型
        model = MultiHeadAttentionModel(
            input_dim=input_dim,
            model_dim=pruned_model_dim,
            num_heads=pruned_num_heads,
            hidden_dim=CONFIG['hidden_dim'],
            dropout_rate=CONFIG['dropout_rate']
        ).to(device)

        # 验证模型维度与权重形状是否匹配
        if 'attention.in_proj_weight' in state_dict:
            expected_in_proj_shape = (3 * pruned_model_dim, state_dict['attention.in_proj_weight'].shape[1])
            if state_dict['attention.in_proj_weight'].shape != expected_in_proj_shape:
                print(f"警告: 模型维度({pruned_model_dim})与in_proj_weight形状({state_dict['attention.in_proj_weight'].shape})不匹配")
                print(f"期望形状: {expected_in_proj_shape}")
        
        # 处理权重格式不匹配问题
        # 如果状态字典中包含q_proj_weight等键，而不是in_proj_weight
        if 'attention.q_proj_weight' in state_dict and 'attention.k_proj_weight' in state_dict and 'attention.v_proj_weight' in state_dict:
            print("检测到剪枝模型使用了分离的q/k/v权重，正在合并为in_proj_weight...")
            # 获取q、k、v权重
            q_weight = state_dict.pop('attention.q_proj_weight')
            k_weight = state_dict.pop('attention.k_proj_weight')
            v_weight = state_dict.pop('attention.v_proj_weight')

            # 合并q、k、v权重为in_proj_weight
            in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

            # 检查合并后的形状是否与模型维度匹配
            expected_in_proj_shape = (3 * pruned_model_dim, q_weight.shape[1])
            if in_proj_weight.shape != expected_in_proj_shape:
                print(f"警告: 合并后的in_proj_weight形状({in_proj_weight.shape})与期望形状({expected_in_proj_shape})不匹配")
                # 调整权重形状以匹配模型维度
                if in_proj_weight.shape[0] < expected_in_proj_shape[0]:
                    # 如果权重行数不足，填充零
                    padding = torch.zeros(expected_in_proj_shape[0] - in_proj_weight.shape[0], in_proj_weight.shape[1], device=in_proj_weight.device)
                    in_proj_weight = torch.cat([in_proj_weight, padding], dim=0)
                    print("已填充权重以匹配期望形状")
                else:
                    # 如果权重行数过多，截断
                    in_proj_weight = in_proj_weight[:expected_in_proj_shape[0], :]
                    print("已截断权重以匹配期望形状")

            state_dict['attention.in_proj_weight'] = in_proj_weight

            # 处理偏置参数（如果存在）
            if 'attention.q_proj_bias' in state_dict and 'attention.k_proj_bias' in state_dict and 'attention.v_proj_bias' in state_dict:
                q_bias = state_dict.pop('attention.q_proj_bias')
                k_bias = state_dict.pop('attention.k_proj_bias')
                v_bias = state_dict.pop('attention.v_proj_bias')

                # 合并q、k、v偏置为in_proj_bias
                in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

                # 检查合并后的偏置形状是否与模型维度匹配
                expected_in_proj_bias_shape = (3 * pruned_model_dim,)
                if in_proj_bias.shape != expected_in_proj_bias_shape:
                    print(f"警告: 合并后的in_proj_bias形状({in_proj_bias.shape})与期望形状({expected_in_proj_bias_shape})不匹配")
                    # 调整偏置形状以匹配模型维度
                    if in_proj_bias.shape[0] < expected_in_proj_bias_shape[0]:
                        # 如果偏置长度不足，填充零
                        padding = torch.zeros(expected_in_proj_bias_shape[0] - in_proj_bias.shape[0], device=in_proj_bias.device)
                        in_proj_bias = torch.cat([in_proj_bias, padding], dim=0)
                        print("已填充偏置以匹配期望形状")
                    else:
                        # 如果偏置长度过多，截断
                        in_proj_bias = in_proj_bias[:expected_in_proj_bias_shape[0]]
                        print("已截断偏置以匹配期望形状")

                state_dict['attention.in_proj_bias'] = in_proj_bias

        # 加载权重
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"加载剪枝模型权重时出错: {e}")
            raise
    else:
        model_path = CONFIG['model_save_path']
        print("使用未剪枝的模型进行预测")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未剪枝模型文件不存在: {model_path}")
        
        # 使用原始参数初始化模型
        model = MultiHeadAttentionModel(
            input_dim=input_dim,
            model_dim=CONFIG['model_dim'],
            num_heads=CONFIG['num_heads'],
            hidden_dim=CONFIG['hidden_dim'],
            dropout_rate=CONFIG['dropout_rate']
        ).to(device)
        # 加载权重
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError as e:
            print(f"加载未剪枝模型权重时出错: {e}")
            raise
    model.eval()
    
    # 预测
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 输出评估指标
    print('混淆矩阵:\n', confusion_matrix(all_labels, all_preds))
    print('分类报告:\n', classification_report(all_labels, all_preds))
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'准确率: {accuracy:.4f}')
    return accuracy

def get_memory_usage():
    """获取当前内存使用情况"""
    memory = psutil.virtual_memory()
    return memory.used / (1024 ** 3)  # 转换为GB

def predict_from_input_not_lable(input_df, use_pruned_model=True):
    """从输入数据框进行预测，不使用标签列

    参数:
        input_df: 输入数据框，包含所有特征列
        use_pruned_model: 是否使用剪枝模型，默认为True

    返回:
        预测结果列表
    """
    # 加载预处理工具和特征选择器
    preprocessor = joblib.load(CONFIG['preprocessor_save_path'])
    feature_selector = joblib.load(CONFIG['feature_selector_save_path'])
    svd = joblib.load('Tuo/svd_model.joblib')

    # 预处理数据
    X_test_processed = preprocessor.transform(input_df).astype('float32')
    X_test_processed = feature_selector.transform(X_test_processed)
    X_test_processed = svd.transform(X_test_processed)

    # 获取输入维度
    input_dim = X_test_processed.shape[1]

    # 动态调整序列长度以匹配可用数据量
    data_length = len(X_test_processed)
    adjusted_seq_length = min(CONFIG['seq_length'], data_length)

    if adjusted_seq_length < 1:
        raise ValueError(f"数据样本数({data_length})不足，无法创建有效序列")

    # 创建数据集（使用虚拟标签）
    dummy_labels = np.zeros(data_length)
    test_dataset = DDoSDataset(X_test_processed, dummy_labels, seq_length=adjusted_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # 根据参数选择加载剪枝或未剪枝模型
    if use_pruned_model:
        model_path = 'Tuo/multi_head_attention_model_pruned_best.pth'
        print("使用剪枝后的模型进行预测")

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"剪枝模型文件不存在: {model_path}")

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=device)

        # 从检查点中获取剪枝后的头数量和模型维度
        pruned_num_heads = checkpoint['num_heads']
        pruned_model_dim = checkpoint['model_dim']
        state_dict = checkpoint['state_dict']

        # 使用剪枝后的参数初始化模型
        model = MultiHeadAttentionModel(
            input_dim=input_dim,
            model_dim=pruned_model_dim,
            num_heads=pruned_num_heads,
            hidden_dim=CONFIG['hidden_dim'],
            dropout_rate=CONFIG['dropout_rate']
        ).to(device)

        # 处理权重格式不匹配问题
        # 如果状态字典中包含q_proj_weight等键，而不是in_proj_weight
        if 'attention.q_proj_weight' in state_dict and 'attention.k_proj_weight' in state_dict and 'attention.v_proj_weight' in state_dict:
            # 获取q、k、v权重
            q_weight = state_dict.pop('attention.q_proj_weight')
            k_weight = state_dict.pop('attention.k_proj_weight')
            v_weight = state_dict.pop('attention.v_proj_weight')

            # 合并q、k、v权重为in_proj_weight
            in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

            # 检查合并后的形状是否与模型维度匹配
            expected_in_proj_shape = (3 * pruned_model_dim, q_weight.shape[1])
            if in_proj_weight.shape != expected_in_proj_shape:
                if in_proj_weight.shape[0] < expected_in_proj_shape[0]:
                    # 如果权重行数不足，填充零
                    padding = torch.zeros(expected_in_proj_shape[0] - in_proj_weight.shape[0], in_proj_weight.shape[1], device=in_proj_weight.device)
                    in_proj_weight = torch.cat([in_proj_weight, padding], dim=0)
                else:
                    # 如果权重行数过多，截断
                    in_proj_weight = in_proj_weight[:expected_in_proj_shape[0], :]

            state_dict['attention.in_proj_weight'] = in_proj_weight

            # 处理偏置参数（如果存在）
            if 'attention.q_proj_bias' in state_dict and 'attention.k_proj_bias' in state_dict and 'attention.v_proj_bias' in state_dict:
                q_bias = state_dict.pop('attention.q_proj_bias')
                k_bias = state_dict.pop('attention.k_proj_bias')
                v_bias = state_dict.pop('attention.v_proj_bias')

                # 合并q、k、v偏置为in_proj_bias
                in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

                # 检查合并后的偏置形状是否与模型维度匹配
                expected_in_proj_bias_shape = (3 * pruned_model_dim,)
                if in_proj_bias.shape != expected_in_proj_bias_shape:
                    if in_proj_bias.shape[0] < expected_in_proj_bias_shape[0]:
                        # 如果偏置长度不足，填充零
                        padding = torch.zeros(expected_in_proj_bias_shape[0] - in_proj_bias.shape[0], device=in_proj_bias.device)
                        in_proj_bias = torch.cat([in_proj_bias, padding], dim=0)
                    else:
                        # 如果偏置长度过多，截断
                        in_proj_bias = in_proj_bias[:expected_in_proj_bias_shape[0]]

                state_dict['attention.in_proj_bias'] = in_proj_bias

        # 加载权重
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"加载剪枝模型权重时出错: {e}")
            raise
    else:
        model_path = CONFIG['model_save_path']
        print("使用未剪枝的模型进行预测")

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未剪枝模型文件不存在: {model_path}")

        # 使用原始参数初始化模型
        model = MultiHeadAttentionModel(
            input_dim=input_dim,
            model_dim=CONFIG['model_dim'],
            num_heads=CONFIG['num_heads'],
            hidden_dim=CONFIG['hidden_dim'],
            dropout_rate=CONFIG['dropout_rate']
        ).to(device)
        # 加载权重
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError as e:
            print(f"加载未剪枝模型权重时出错: {e}")
            raise
    model.eval()

    # 预测
    all_preds = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())

    return all_preds

def get_gpu_memory_usage():
    """获取当前GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # 转换为GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # 转换为GB
        return allocated, max_allocated
    return 0, 0

if __name__ == '__main__':
    print("===== 模型性能对比 =====")
    
    # 测试未剪枝模型
    print("\n[未剪枝模型]")
    # 记录初始内存和显存使用
    initial_memory = get_memory_usage()
    initial_gpu_allocated, initial_gpu_max = get_gpu_memory_usage()
    
    start_time = time.time()
    original_accuracy = predict(use_pruned_model=False)
    original_time = time.time() - start_time
    
    # 记录结束内存和显存使用
    final_memory = get_memory_usage()
    final_gpu_allocated, final_gpu_max = get_gpu_memory_usage()
    
    # 计算内存和显存使用差值
    memory_used = final_memory - initial_memory
    gpu_allocated_used = final_gpu_allocated - initial_gpu_allocated
    gpu_max_used = final_gpu_max - initial_gpu_max if initial_gpu_max > 0 else final_gpu_max
    
    print(f"未剪枝模型运算时间: {original_time:.4f} 秒")
    print(f"未剪枝模型内存使用: {memory_used:.4f} GB")
    if torch.cuda.is_available():
        print(f"未剪枝模型GPU内存分配: {gpu_allocated_used:.4f} GB")
        print(f"未剪枝模型GPU最大内存使用: {gpu_max_used:.4f} GB")
    else:
        print("未剪枝模型: 未使用GPU")
    
    # 测试剪枝模型
    print("\n[剪枝模型]")
    # 记录初始内存和显存使用
    initial_memory_pruned = get_memory_usage()
    initial_gpu_allocated_pruned, initial_gpu_max_pruned = get_gpu_memory_usage()
    
    start_time = time.time()
    pruned_accuracy = predict(use_pruned_model=True)
    pruned_time = time.time() - start_time
    
    # 记录结束内存和显存使用
    final_memory_pruned = get_memory_usage()
    final_gpu_allocated_pruned, final_gpu_max_pruned = get_gpu_memory_usage()
    
    # 计算内存和显存使用差值
    memory_used_pruned = final_memory_pruned - initial_memory_pruned
    gpu_allocated_used_pruned = final_gpu_allocated_pruned - initial_gpu_allocated_pruned
    gpu_max_used_pruned = final_gpu_max_pruned - initial_gpu_max_pruned if initial_gpu_max_pruned > 0 else final_gpu_max_pruned
    
    print(f"剪枝模型运算时间: {pruned_time:.4f} 秒")
    print(f"剪枝模型内存使用: {memory_used_pruned:.4f} GB")
    if torch.cuda.is_available():
        print(f"剪枝模型GPU内存分配: {gpu_allocated_used_pruned:.4f} GB")
        print(f"剪枝模型GPU最大内存使用: {gpu_max_used_pruned:.4f} GB")
    else:
        print("剪枝模型: 未使用GPU")
    
    # 输出对比结果
    print("\n===== 对比结果 =====")
    print(f"未剪枝模型准确率: {original_accuracy:.4f}")
    print(f"剪枝模型准确率: {pruned_accuracy:.4f}")
    
    # 计算准确率变化
    accuracy_diff = pruned_accuracy - original_accuracy
    if accuracy_diff > 0:
        print(f"剪枝后准确率提升: {accuracy_diff:.4f}")
    elif accuracy_diff < 0:
        print(f"剪枝后准确率下降: {abs(accuracy_diff):.4f}")
    else:
        print("剪枝前后准确率相同")
    
    # 计算时间变化
    time_diff = pruned_time - original_time
    time_ratio = pruned_time / original_time if original_time > 0 else 0
    if time_diff < 0:
        print(f"剪枝后运算速度提升: {abs(time_diff):.4f} 秒 ({(1-time_ratio)*100:.2f}%)")
    elif time_diff > 0:
        print(f"剪枝后运算速度下降: {time_diff:.4f} 秒 ({(time_ratio-1)*100:.2f}%)")
    else:
        print("剪枝前后运算速度相同")
    
    # 计算内存变化
    memory_diff = memory_used_pruned - memory_used
    memory_ratio = memory_used_pruned / memory_used if memory_used > 0 else 0
    if memory_diff < 0:
        print(f"剪枝后内存使用减少: {abs(memory_diff):.4f} GB ({(1-memory_ratio)*100:.2f}%)")
    elif memory_diff > 0:
        print(f"剪枝后内存使用增加: {memory_diff:.4f} GB ({(memory_ratio-1)*100:.2f}%)")
    else:
        print("剪枝前后内存使用相同")
    
    # 计算GPU内存变化
    if torch.cuda.is_available():
        gpu_diff = gpu_allocated_used_pruned - gpu_allocated_used
        gpu_ratio = gpu_allocated_used_pruned / gpu_allocated_used if gpu_allocated_used > 0 else 0
        if gpu_diff < 0:
            print(f"剪枝后GPU内存分配减少: {abs(gpu_diff):.4f} GB ({(1-gpu_ratio)*100:.2f}%)")
        elif gpu_diff > 0:
            print(f"剪枝后GPU内存分配增加: {gpu_diff:.4f} GB ({(gpu_ratio-1)*100:.2f}%)")
        else:
            print("剪枝前后GPU内存分配相同")
    else:
        print("未使用GPU，无法比较GPU内存使用")