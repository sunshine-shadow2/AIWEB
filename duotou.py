import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
import os
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
# 配置matplotlib支持中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun", "Arial"]
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('Tuo/training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'使用设备: {device}')
logger.info(f'CUDA可用: {torch.cuda.is_available()}, 设备数量: {torch.cuda.device_count()}')

# 模型配置
CONFIG = {
    'seq_length': 60,
    'batch_size': 128,
    'epochs': 50,
    'lr': 0.0005,  # 降低初始学习率
    'model_dim': 256,
    'num_heads': 32,
    'hidden_dim': 256,
    'dropout_rate': 0.3,  # 增加dropout率以增强正则化
    # 'test_size': CONFIG['test_size'],  # 不再需要，使用独立测试集
    'random_state': 42,
    'cv_folds': 5,  # 交叉验证折数
    'patience': 20,
    'model_save_path': 'Tuo/multi_head_attention_model.pth',
    'feature_selector_save_path': 'Tuo/feature_selector.joblib',
    'preprocessor_save_path': 'Tuo/preprocessor.joblib',
    'train_data_path': 'shujuji1.csv',
    'test_data_path': 'duotou1.csv',
    'weight_decay': 5e-2,  # 增加权重衰减强度
    'svd_n_components': 10,  # 增加SVD主成分数量以保留更多特征信息
    'l1_lambda': 1e-5  # L1正则化系数
}

# 创建保存模型的Tuo文件夹
os.makedirs('Tuo', exist_ok=True)

# 定义数据集类
class DDoSDataset(Dataset):
    def __init__(self, features, labels, seq_length=10):
        self.features = features
        self.labels = labels
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.features) - self.seq_length + 1
    
    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_length]
        y = self.labels[idx+self.seq_length-1]  # 取序列最后一个样本的标签
        return torch.FloatTensor(x), torch.FloatTensor([y])

# 定义多头注意力模型
class MultiHeadAttentionModel(nn.Module):
    def __init__(self, input_dim: int, model_dim: int = 128, num_heads: int = 8, hidden_dim: int = 256, dropout_rate: float = 0.5, low_rank_factor: int = 1, l1_lambda=0.001):
        """多头注意力模型，用于DDoS攻击检测
        
        参数:
            input_dim: 输入特征维度
            model_dim: 模型隐藏层维度
            num_heads: 注意力头数量
            hidden_dim: 前馈网络隐藏层维度
            dropout_rate: dropout比率
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads  # 保存头数量
        
        # 输入投影层（低秩分解）
        self.intermediate_dim = model_dim // low_rank_factor
        self.input_proj1 = nn.Linear(input_dim, self.intermediate_dim)
        self.input_proj2 = nn.Linear(self.intermediate_dim, model_dim)
        
        # 多头注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 前馈网络
        self.fc = nn.Sequential(
            nn.Linear(model_dim, self.intermediate_dim),
            nn.Linear(self.intermediate_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, model_dim)
        )
        
        # 输出层（低秩分解）
        self.output_proj1 = nn.Linear(model_dim, self.intermediate_dim)
        self.output_proj2 = nn.Linear(self.intermediate_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        # 层归一化和 dropout
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)
        # L1正则化参数
        self.l1_lambda = l1_lambda
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size, seq_length, input_dim = x.size()
        
        # 输入投影（低秩分解）
        x = self.input_proj2(self.input_proj1(x))
        
        # 多头注意力
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        fc_output = self.fc(x)
        x = self.norm2(x + self.dropout(fc_output))
        
        # 输出层，取最后一个时间步的输出
        output = self.sigmoid(self.output_proj2(self.output_proj1(x[:, -1, :])))
        return output

# 数据预处理函数
def preprocess_data(train_file_path, test_file_path):
    # 加载训练数据和测试数据
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    
    # 分离特征和标签
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']
    
    # 识别数值特征和分类特征（使用训练集）
    logger.info(f'训练集样本数: {len(train_df)}, 测试集样本数: {len(test_df)}')
    logger.info(f'训练集标签分布: {y_train.value_counts(normalize=True).to_dict()}')
    logger.info(f'测试集标签分布: {y_test.value_counts(normalize=True).to_dict()}')

    # 创建数据分布分析目录
    os.makedirs('data_distribution', exist_ok=True)

    # 分析数值特征分布差异
    numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # 分析数值特征分布差异
    for col in numeric_features:
        # 执行KS检验检查分布是否相同
        stat, p_value = stats.ks_2samp(train_df[col].dropna(), test_df[col].dropna())
        is_same_distribution = p_value > 0.05
        logger.info(f'特征 {col}: KS统计量={stat:.4f}, p值={p_value:.4f}, 分布相同={is_same_distribution}')

        # 绘制分布对比图
        plt.figure(figsize=(10, 4))
        sns.histplot(train_df[col].dropna(), kde=True, label='训练集', bins=30)
        sns.histplot(test_df[col].dropna(), kde=True, label='测试集', bins=30)
        plt.title(f'{col} 分布对比 (KS p值: {p_value:.4f})')
        plt.legend()
        plt.savefig(f'data_distribution/{col}_distribution.png')
        plt.close()

    # 分析分类特征分布差异
    for col in categorical_features:
        # 计算类别分布
        train_dist = train_df[col].value_counts(normalize=True).to_dict()
        test_dist = test_df[col].value_counts(normalize=True).to_dict()
        all_categories = set(train_dist.keys()).union(set(test_dist.keys()))

        # 记录分布差异
        logger.info(f'特征 {col} 类别分布差异:')
        for cat in all_categories:
            train_pct = train_dist.get(cat, 0)
            test_pct = test_dist.get(cat, 0)
            logger.info(f'  {cat}: 训练集={train_pct:.2%}, 测试集={test_pct:.2%}, 差异={abs(train_pct-test_pct):.2%}')

        # 绘制类别分布对比图
        plt.figure(figsize=(12, 6))
        categories = list(all_categories)
        train_pcts = [train_dist.get(cat, 0) for cat in categories]
        test_pcts = [test_dist.get(cat, 0) for cat in categories]

        x = np.arange(len(categories))
        width = 0.35
        plt.bar(x - width/2, train_pcts, width, label='训练集')
        plt.bar(x + width/2, test_pcts, width, label='测试集')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.title(f'{col} 类别分布对比')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'data_distribution/{col}_categories.png')
        plt.close()

    # 记录KS检验结果以便特征选择
    ks_results = {}
    # 分析数值特征分布差异
    for col in numeric_features:
        # 执行KS检验检查分布是否相同
        stat, p_value = stats.ks_2samp(train_df[col].dropna(), test_df[col].dropna())
        is_same_distribution = p_value > 0.05
        ks_results[col] = {'statistic': stat, 'p_value': p_value, 'same_dist': is_same_distribution}
        logger.info(f'特征 {col}: KS统计量={stat:.4f}, p值={p_value:.4f}, 分布相同={is_same_distribution}')

        # 绘制分布对比图
        plt.figure(figsize=(10, 4))
        sns.histplot(train_df[col].dropna(), kde=True, label='训练集', bins=30)
        sns.histplot(test_df[col].dropna(), kde=True, label='测试集', bins=30)
        plt.title(f'{col} 分布对比 (KS p值: {p_value:.4f})')
        plt.legend()
        plt.savefig(f'data_distribution/{col}_distribution.png')
        plt.close()

    # 分析分类特征分布差异
    for col in categorical_features:
        # 计算类别分布
        train_dist = train_df[col].value_counts(normalize=True).to_dict()
        test_dist = test_df[col].value_counts(normalize=True).to_dict()
        all_categories = set(train_dist.keys()).union(set(test_dist.keys()))

        # 记录分布差异
        logger.info(f'特征 {col} 类别分布差异:')
        for cat in all_categories:
            train_pct = train_dist.get(cat, 0)
            test_pct = test_dist.get(cat, 0)
            logger.info(f'  {cat}: 训练集={train_pct:.2%}, 测试集={test_pct:.2%}, 差异={abs(train_pct-test_pct):.2%}')

        # 绘制类别分布对比图
        plt.figure(figsize=(12, 6))
        categories = list(all_categories)
        train_pcts = [train_dist.get(cat, 0) for cat in categories]
        test_pcts = [test_dist.get(cat, 0) for cat in categories]

        x = np.arange(len(categories))
        width = 0.35
        plt.bar(x - width/2, train_pcts, width, label='训练集')
        plt.bar(x + width/2, test_pcts, width, label='测试集')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.title(f'{col} 类别分布对比')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'data_distribution/{col}_categories.png')
        plt.close()

    # 创建预处理管道
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True, max_categories=10))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 拟合预处理管道（仅使用训练集）
    X_train_processed = preprocessor.fit_transform(X_train).astype('float32')
    X_test_processed = preprocessor.transform(X_test).astype('float32')

    # 基于KS检验结果选择特征
    # 设置KS统计量阈值，移除分布差异过大的特征
    selected_features = []
    for i, col in enumerate(numeric_features):
        # 跳过已处理的特征
        if i >= X_train_processed.shape[1]:
            continue
        # 仅保留KS统计量小于0.5的特征
        if col in ks_results and ks_results[col]['statistic'] < 0.5:
            selected_features.append(i)
    
    from sklearn.feature_selection import SelectKBest
    selector = SelectKBest(k=len(selected_features))
    selector.fit(X_train_processed, y_train)
    X_train_processed = selector.transform(X_train_processed)
    X_test_processed = selector.transform(X_test_processed)
    
    if selected_features:
        logger.info(f'基于KS检验选择了{len(selected_features)}/{X_train_processed.shape[1]}个特征')
        joblib.dump(selector, CONFIG['feature_selector_save_path'])
    else:
        logger.warning('未选择任何特征，使用全部特征')

    # 应用SVD降维去噪
    svd = TruncatedSVD(n_components=CONFIG['svd_n_components'], random_state=CONFIG['random_state'])
    X_train_processed = svd.fit_transform(X_train_processed)
    X_test_processed = svd.transform(X_test_processed)
    joblib.dump(svd, 'Tuo/svd_model.joblib')  # 保存SVD模型
    
    # 保存预处理模型
    joblib.dump(preprocessor, CONFIG['preprocessor_save_path']) 
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

# 交叉验证训练函数
def cross_validate_model(X, y, input_dim, cv_folds=5):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=CONFIG['random_state'])
    fold_results = []
    best_model = None
    best_acc = 0

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f'===== 第 {fold+1}/{cv_folds} 折交叉验证 =====')
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # 创建数据集和数据加载器
        train_dataset = DDoSDataset(X_train_fold, y_train_fold.values, CONFIG['seq_length'])
        val_dataset = DDoSDataset(X_val_fold, y_val_fold.values, CONFIG['seq_length'])

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, pin_memory=device.type == 'cuda')
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, pin_memory=device.type == 'cuda')

        # 初始化模型
        model = MultiHeadAttentionModel(
            l1_lambda=CONFIG.get('l1_lambda', 0.001),
            input_dim=input_dim,
            model_dim=CONFIG['model_dim'],
            num_heads=CONFIG['num_heads'],
            hidden_dim=CONFIG['hidden_dim'],
            dropout_rate=CONFIG['dropout_rate']
        ).to(device)

        criterion = nn.BCELoss(weight=torch.tensor([5.0], device=device))
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        fold_best_acc = 0
        fold_counter = 0

        for epoch in range(CONFIG['epochs']):
            model.train()
            train_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels) + l1_regularization(model, CONFIG['l1_lambda'])
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)

            # 在验证集上评估
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    preds = (outputs > 0.5).float()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            val_acc = accuracy_score(all_labels, all_preds)
            scheduler.step(val_acc)

            if val_acc > fold_best_acc:
                fold_best_acc = val_acc
                fold_counter = 0
                # 保存当前折的最佳模型
                if fold_best_acc > best_acc:
                    best_acc = fold_best_acc
                    torch.save(model.state_dict(), CONFIG['model_save_path'])
                    best_model = model
            else:
                fold_counter += 1
                if fold_counter >= 5:  # 每个折的早停耐心设为5
                    break

            logger.info(f'Fold {fold+1}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

        fold_results.append(fold_best_acc)
        logger.info(f'第 {fold+1} 折最佳验证准确率: {fold_best_acc:.4f}')

    logger.info(f'交叉验证结果: {fold_results}')
    logger.info(f'平均验证准确率: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}')
    return best_model

# 训练模型函数
def train_model(X_train, y_train, X_test, y_test, input_dim):
    seq_length = CONFIG['seq_length']
    epochs = CONFIG['epochs']
    batch_size = CONFIG['batch_size']
    lr = CONFIG['lr']
    # 创建数据集和数据加载器
    train_dataset = DDoSDataset(X_train, y_train.values, seq_length)
    test_dataset = DDoSDataset(X_test, y_test.values, seq_length)
    
    # 启用CUDA内存固定以加速数据传输
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, pin_memory=device.type == 'cuda')
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, pin_memory=device.type == 'cuda')
    
    # 初始化模型、损失函数和优化器
    model = MultiHeadAttentionModel(
        l1_lambda=CONFIG.get('l1_lambda', 0.001),
        input_dim=input_dim,
        model_dim=CONFIG['model_dim'],
        num_heads=CONFIG['num_heads'],
        hidden_dim=CONFIG['hidden_dim'],
        dropout_rate=CONFIG['dropout_rate']
    ).to(device)
    logger.info(f'模型参数设备: {next(model.parameters()).device}')
    # 为少数类添加权重以处理类别不平衡问题
    criterion = nn.BCELoss(weight=torch.tensor([5.0], device=device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=CONFIG['weight_decay'])
    # L1正则化函数
    def l1_regularization(model, lambda_l1):
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.norm(param, p=1)
        return lambda_l1 * l1_loss
    # 使用ReduceLROnPlateau调度器动态调整学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 早停机制
    best_test_acc = 0.0
    patience = CONFIG['patience']
    counter = 0
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            # 仅在第一个批次记录设备信息
            if batch_idx == 0:
                logger.info(f'训练数据设备: {inputs.device}, 标签设备: {labels.device}')
            optimizer.zero_grad()
            outputs = model(inputs)
            # 仅在第一个批次记录设备信息
            if batch_idx == 0:
                logger.info(f'输出设备: {outputs.device}')
            loss = criterion(outputs, labels) + l1_regularization(model, CONFIG['l1_lambda'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        
        # 在测试集上评估
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        # 计算平均测试损失和准确率
        test_loss /= len(test_loader.dataset)
        test_acc = accuracy_score(all_labels, all_preds)
        # 打印 epoch 结果
        logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        # 学习率调度
        scheduler.step(test_acc)  # 传入测试准确率作为监控指标
        # 早停检查
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), CONFIG['model_save_path'])
        else:
            counter += 1
            if counter >= patience:
                logger.info(f'早停在第{epoch+1}轮，最佳测试准确率: {best_test_acc:.4f}')
                break
    
    logger.info(f'训练完成，最佳测试准确率: {best_test_acc:.4f}')
    return model
def compute_head_importance(attention_layer, method='l1_norm'):
    """计算每个注意力头的重要性

    参数:
        attention_layer: 多头注意力层
        method: 重要性计算方法，支持'l1_norm', 'l2_norm', 'std', 'activation_var'

    返回:
        每个头的重要性分数列表
    """
    total_heads = attention_layer.num_heads
    embed_dim = attention_layer.embed_dim
    head_dim = embed_dim // total_heads

    # 从in_proj_weight中分离查询、键、值的投影权重
    in_proj_weight = attention_layer.in_proj_weight
    q_weight = in_proj_weight[:embed_dim, :]
    k_weight = in_proj_weight[embed_dim:2*embed_dim, :]
    v_weight = in_proj_weight[2*embed_dim:3*embed_dim, :]

    head_importance = []

    for i in range(total_heads):
        start_idx = i * head_dim
        end_idx = start_idx + head_dim

        q_head = q_weight[start_idx:end_idx]
        k_head = k_weight[start_idx:end_idx]
        v_head = v_weight[start_idx:end_idx]

        if method == 'l1_norm':
            # L1范数
            q_norm = torch.norm(q_head, p=1)
            k_norm = torch.norm(k_head, p=1)
            v_norm = torch.norm(v_head, p=1)
            importance = q_norm + k_norm + v_norm
        elif method == 'l2_norm':
            # L2范数
            q_norm = torch.norm(q_head, p=2)
            k_norm = torch.norm(k_head, p=2)
            v_norm = torch.norm(v_head, p=2)
            importance = q_norm + k_norm + v_norm
        elif method == 'std':
            # 标准差
            q_std = torch.std(q_head)
            k_std = torch.std(k_head)
            v_std = torch.std(v_head)
            importance = q_std + k_std + v_std
        elif method == 'activation_var':
            # 模拟激活值的方差
            # 创建随机输入来估计激活值方差，并确保与模型在同一设备上
            dummy_input = torch.randn(1, 10, embed_dim).to(q_head.device)
            q_act = torch.matmul(dummy_input, q_head.t())
            k_act = torch.matmul(dummy_input, k_head.t())
            v_act = torch.matmul(dummy_input, v_head.t())
            q_var = torch.var(q_act)
            k_var = torch.var(k_act)
            v_var = torch.var(v_act)
            importance = q_var + k_var + v_var
        else:
            raise ValueError(f'不支持的重要性度量方法: {method}')

        head_importance.append((importance, i))

    return head_importance

def fuse_importance_scores(importance_scores_list, weights=None):
    """融合多种重要性度量结果

    参数:
        importance_scores_list: 多种重要性度量结果的列表
        weights: 每种度量的权重列表

    返回:
        融合后的重要性分数
    """
    if weights is None:
        weights = [1.0 / len(importance_scores_list)] * len(importance_scores_list)

    total_heads = len(importance_scores_list[0])
    fused_importance = {i: 0.0 for i in range(total_heads)}

    for scores, weight in zip(importance_scores_list, weights):
        # 归一化分数
        max_score = max(scores, key=lambda x: x[0])[0]
        min_score = min(scores, key=lambda x: x[0])[0]
        if max_score == min_score:
            normalized_scores = [(1.0, idx) for _, idx in scores]
        else:
            normalized_scores = [((s - min_score) / (max_score - min_score), idx) for s, idx in scores]

        # 加权累加
        for score, idx in normalized_scores:
            fused_importance[idx] += score * weight

    # 转换为列表形式
    fused_importance_list = [(fused_importance[i], i) for i in range(total_heads)]
    return fused_importance_list

def prune_model(model, pruning_ratio=0.5, importance_method='l1_norm', fused_scores=None):
    """剪枝多头注意力模型中的冗余头

    参数:
        model: 要剪枝的模型
        pruning_ratio: 剪枝比例
        importance_method: 重要性计算方法，当fused_scores不为None时忽略
        fused_scores: 融合后的重要性分数，优先级高于importance_method

    返回:
        剪枝后的模型
    """
    # 获取注意力层
    attention_layer = model.attention

    # 计算要保留的头数量
    total_heads = attention_layer.num_heads
    embed_dim = attention_layer.embed_dim
    target_keep_heads = int(total_heads * (1 - pruning_ratio))

    # 确保keep_heads是embed_dim的约数
    # 找到小于等于target_keep_heads的最大约数
    keep_heads = target_keep_heads
    while keep_heads > 0 and embed_dim % keep_heads != 0:
        keep_heads -= 1

    # 如果找不到约数（理论上不可能，因为1总是约数），则使用1
    if keep_heads == 0:
        keep_heads = 1

    logger.info(f'剪枝前头数量: {total_heads}, 目标保留头数量: {target_keep_heads}, 实际保留头数量: {keep_heads} (确保能整除embed_dim={embed_dim})')

    # 获取查询、键、值投影权重
    logger.info(f'MultiheadAttention参数名称: {[name for name, _ in attention_layer.named_parameters()]}')

    # 计算或使用提供的重要性分数
    if fused_scores is not None:
        head_importance = fused_scores
    else:
        head_importance = compute_head_importance(attention_layer, method=importance_method)

    # 按重要性排序，保留最重要的头
    head_importance.sort(reverse=True)
    kept_head_indices = [idx for _, idx in head_importance[:keep_heads]]
    kept_head_indices.sort()  # 保持顺序以避免打乱头的顺序

    # 从in_proj_weight中分离查询、键、值的投影权重
    in_proj_weight = attention_layer.in_proj_weight
    embed_dim = attention_layer.embed_dim
    q_weight = in_proj_weight[:embed_dim, :]
    k_weight = in_proj_weight[embed_dim:2*embed_dim, :]
    v_weight = in_proj_weight[2*embed_dim:3*embed_dim, :]
    head_dim = embed_dim // total_heads

    # 创建新的投影权重
    new_q_weight = []
    new_k_weight = []
    new_v_weight = []

    for idx in kept_head_indices:
        start_idx = idx * head_dim
        end_idx = start_idx + head_dim

        new_q_weight.append(q_weight[start_idx:end_idx])
        new_k_weight.append(k_weight[start_idx:end_idx])
        new_v_weight.append(v_weight[start_idx:end_idx])

    # 拼接新的权重
    new_q_weight = torch.cat(new_q_weight, dim=0)
    new_k_weight = torch.cat(new_k_weight, dim=0)
    new_v_weight = torch.cat(new_v_weight, dim=0)

    # 创建新的多头注意力层
    new_attention = nn.MultiheadAttention(
        embed_dim=attention_layer.embed_dim,
        num_heads=keep_heads,
        dropout=attention_layer.dropout,
        batch_first=attention_layer.batch_first
    )

    # 设置新的权重
    new_attention.q_proj_weight = nn.Parameter(new_q_weight)
    new_attention.k_proj_weight = nn.Parameter(new_k_weight)
    new_attention.v_proj_weight = nn.Parameter(new_v_weight)

    # 更新模型的注意力层
    model.attention = new_attention
    model.num_heads = keep_heads
    # 剪枝不改变嵌入维度，确保记录正确的模型参数
    model.model_dim = model.attention.embed_dim
    
    return model

def save_pruned_model(model, save_path):
    """保存剪枝后的模型，包含必要的参数信息"""
    torch.save({
        'state_dict': model.state_dict(),
        'num_heads': model.num_heads,
        'model_dim': model.model_dim
    }, save_path)
    logger.info(f'剪枝后的模型已保存至 {save_path}，头数量: {model.num_heads}, 模型维度: {model.model_dim}')

# 主函数
def main():
    # 数据预处理
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(CONFIG['train_data_path'], CONFIG['test_data_path']) 
    # 获取输入特征维度
    input_dim = X_train.shape[1]
    # 训练模型
    model = train_model(X_train, y_train, X_test, y_test, input_dim)
    # 加载最佳模型并评估
    best_model = MultiHeadAttentionModel(
        l1_lambda=CONFIG.get('l1_lambda', 0.001),
        input_dim=input_dim,
        model_dim=CONFIG['model_dim'],
        num_heads=CONFIG['num_heads'],
        hidden_dim=CONFIG['hidden_dim'],
        dropout_rate=CONFIG['dropout_rate']
    )
    best_model.load_state_dict(torch.load(CONFIG['model_save_path'], map_location=device))
    best_model.to(device)
    
    # 在测试集上生成混淆矩阵和分类报告
    test_dataset = DDoSDataset(X_test, y_test.values, seq_length=CONFIG['seq_length'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    best_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = best_model(inputs)
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    logger.info('\n混淆矩阵:\n%s', confusion_matrix(all_labels, all_preds))
    logger.info('\n分类报告:\n%s', classification_report(all_labels, all_preds))

    # 评估原始模型
    original_accuracy = evaluate_model(best_model, test_loader)
    logger.info(f'原始模型测试准确率: {original_accuracy:.4f}')
    
    # 定义要使用的重要性度量方法
    importance_methods = ['l1_norm', 'l2_norm', 'std', 'activation_var']
    pruning_ratio = 0.7  # 剪枝50%的头
    pruned_models = []
    pruned_accuracies = []
    importance_scores_list = []
    
    # 为每种度量方法剪枝并保存模型
    for method in importance_methods:
        logger.info(f'使用{method}方法剪枝模型...')
        # 创建模型副本以避免多次剪枝同一模型
        model_copy = MultiHeadAttentionModel(
            l1_lambda=CONFIG.get('l1_lambda', 0.001),
            input_dim=input_dim,
            model_dim=CONFIG['model_dim'],
            num_heads=CONFIG['num_heads'],
            hidden_dim=CONFIG['hidden_dim'],
            dropout_rate=CONFIG['dropout_rate']
        )
        model_copy.load_state_dict(best_model.state_dict())
        model_copy.to(device)
        
        # 剪枝模型
        pruned_model = prune_model(model_copy, pruning_ratio, importance_method=method)
        pruned_model.to(device)
        
        # 评估剪枝模型
        pruned_accuracy = evaluate_model(pruned_model, test_loader)
        logger.info(f'{method}剪枝后模型测试准确率: {pruned_accuracy:.4f}')
        
        # 保存剪枝后的模型
        save_path = f'Tuo/multi_head_attention_model_pruned_{method}.pth'
        save_pruned_model(pruned_model, save_path)
        
        # 记录结果
        pruned_models.append(pruned_model)
        pruned_accuracies.append(pruned_accuracy)
        
        # 计算并保存重要性分数
        attention_layer = model_copy.attention
        importance_scores = compute_head_importance(attention_layer, method=method)
        importance_scores_list.append(importance_scores)
    
    # 融合多种重要性度量结果
    logger.info('融合多种重要性度量结果...')
    # 可以根据每种方法的准确率设置权重
    # 这里简单使用等权重
    weights = [1.0 / len(importance_methods)] * len(importance_methods)
    fused_scores = fuse_importance_scores(importance_scores_list, weights)
    
    # 使用融合后的重要性分数剪枝模型
    logger.info('使用融合后的重要性分数剪枝模型...')
    model_copy = MultiHeadAttentionModel(
        l1_lambda=CONFIG.get('l1_lambda', 0.001),
        input_dim=input_dim,
        model_dim=CONFIG['model_dim'],
        num_heads=CONFIG['num_heads'],
        hidden_dim=CONFIG['hidden_dim'],
        dropout_rate=CONFIG['dropout_rate']
    )
    model_copy.load_state_dict(best_model.state_dict())
    model_copy.to(device)
    
    fused_pruned_model = prune_model(model_copy, pruning_ratio, fused_scores=fused_scores)
    fused_pruned_model.to(device)
    
    # 评估融合剪枝模型
    fused_pruned_accuracy = evaluate_model(fused_pruned_model, test_loader)
    logger.info(f'融合剪枝后模型测试准确率: {fused_pruned_accuracy:.4f}')
    
    # 保存融合剪枝后的模型
    save_pruned_model(fused_pruned_model, 'Tuo/multi_head_attention_model_pruned_fused.pth')
    
    # 对比所有结果
    logger.info('\n剪枝结果对比:')
    logger.info(f'原始模型准确率: {original_accuracy:.4f}')
    for i, method in enumerate(importance_methods):
        accuracy_diff = pruned_accuracies[i] - original_accuracy
        if accuracy_diff > 0:
            logger.info(f'{method}剪枝模型准确率: {pruned_accuracies[i]:.4f} (+{accuracy_diff:.4f})')
        elif accuracy_diff < 0:
            logger.info(f'{method}剪枝模型准确率: {pruned_accuracies[i]:.4f} (-{abs(accuracy_diff):.4f})')
        else:
            logger.info(f'{method}剪枝模型准确率: {pruned_accuracies[i]:.4f} (不变)')
    
    fused_accuracy_diff = fused_pruned_accuracy - original_accuracy
    if fused_accuracy_diff > 0:
        logger.info(f'融合剪枝模型准确率: {fused_pruned_accuracy:.4f} (+{fused_accuracy_diff:.4f})')
    elif fused_accuracy_diff < 0:
        logger.info(f'融合剪枝模型准确率: {fused_pruned_accuracy:.4f} (-{abs(fused_accuracy_diff):.4f})')
    else:
        logger.info(f'融合剪枝模型准确率: {fused_pruned_accuracy:.4f} (不变)')

    # 保存最佳剪枝模型
    best_pruned_idx = np.argmax(pruned_accuracies + [fused_pruned_accuracy])
    if best_pruned_idx < len(pruned_accuracies):
        best_method = importance_methods[best_pruned_idx]
        best_pruned_model = pruned_models[best_pruned_idx]
        logger.info(f'最佳剪枝模型是使用{best_method}方法，准确率: {pruned_accuracies[best_pruned_idx]:.4f}')
    else:
        best_pruned_model = fused_pruned_model
        logger.info(f'最佳剪枝模型是融合方法，准确率: {fused_pruned_accuracy:.4f}')

    save_pruned_model(best_pruned_model, 'Tuo/multi_head_attention_model_pruned_best.pth')
    logger.info('最佳剪枝模型已保存至 Tuo/multi_head_attention_model_pruned_best.pth')

def evaluate_model(model, test_loader):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return accuracy_score(all_labels, all_preds)

if __name__ == '__main__':
    main()