import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
import optuna
from sklearn.metrics import confusion_matrix, classification_report
import os
import json
import math
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('output.log', mode='w')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)


# 定义日期编码函数
def encode_dates(dates, num_time_features=4):
    if not isinstance(dates, (pd.Series, np.ndarray)):
        raise ValueError("dates should be a pandas Series or numpy array")
    time_features = np.zeros((len(dates), num_time_features * 2))
    for i, date in enumerate(dates):
        timestamp = (date - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s')
        for j in range(num_time_features):
            omega = 1.0 / (10000 ** (j / num_time_features))
            time_features[i, 2 * j] = math.sin(omega * timestamp)
            time_features[i, 2 * j + 1] = math.cos(omega * timestamp)
    return time_features


# 定义数据集类
class TipDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.label_encoder = LabelEncoder()

        if 'tip' not in self.data.columns:
            raise ValueError("CSV file must contain a 'tip' column")
        if 'date' not in self.data.columns:
            raise ValueError("CSV file must contain a 'date' column")

        self.labels = self.label_encoder.fit_transform(self.data['tip'])

        # 将日期信息转换为数值特征
        self.dates = pd.to_datetime(self.data['date'], format='%Y%m%d')
        self.date_features = encode_dates(self.dates)

        # 标准化特征
        numeric_features = self.data.drop(columns=['id', 'tip', 'date'])
        if numeric_features.shape[1] == 0:
            raise ValueError("No numeric features found after dropping 'id', 'tip', and 'date' columns")
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(numeric_features)

        # 计算时间差分特征
        time_diffs = (self.dates - self.dates.min()).dt.total_seconds().values
        self.time_diff_features = (time_diffs - time_diffs.mean()) / time_diffs.std()

        # 绝对时间编码
        absolute_time_features = (self.dates - pd.Timestamp("2010-01-01")).total_seconds().values
        self.absolute_time_features = (
                                                  absolute_time_features - absolute_time_features.mean()) / absolute_time_features.std()

        # 生成多项式特征
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        self.interaction_features = poly.fit_transform(self.features)

        # 使用 SelectKBest 选择前 k 个最有用的特征
        k = 10  # 选择前 10 个最有用的特征
        selector = SelectKBest(f_classif, k=k)
        self.selected_features = selector.fit_transform(self.interaction_features, self.labels)

        # 将所有时间特征与原始特征拼接
        self.combined_features = np.hstack([self.selected_features, self.date_features,
                                            self.time_diff_features.reshape(-1, 1),
                                            self.absolute_time_features.reshape(-1, 1)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.combined_features[idx]
        label = self.labels[idx]

        if self.transform:
            features = self.transform(features)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, dropout=0.1, hidden_dim=512):
        super(TransformerModel, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert input_dim > 0, "input_dim must be greater than 0"
        assert num_heads > 0, "num_heads must be greater than 0"
        assert num_layers > 0, "num_layers must be greater than 0"
        assert num_classes > 0, "num_classes must be greater than 0"

        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # 添加序列维度
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x


# 训练模型
def train(model, device, train_loader, optimizer, criterion, scheduler, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)
    scheduler.step(loss)  # 学习率调度


# 测试模型
def test(model, device, test_loader, criterion, label_encoder, writer, epoch, phase='Test'):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\n{phase} set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n')

    cm = confusion_matrix(all_targets, all_preds)
    class_accuracies = cm.diagonal() / cm.sum(axis=1) * 100
    class_labels = label_encoder.inverse_transform(range(len(class_accuracies)))

    for i, label in enumerate(class_labels):
        logger.info(f'Class: {label}, Accuracy: {class_accuracies[i]:.2f}%')

    logger.info("\nClassification Report:")
    logger.info(classification_report(all_targets, all_preds, target_names=class_labels, zero_division=0))

    writer.add_scalar(f'{phase} Loss', test_loss, epoch)
    writer.add_scalar(f'{phase} Accuracy', accuracy, epoch)

    return accuracy, class_accuracies, class_labels


# 创建数据加载器
def create_data_loaders(dataset, batch_size):
    train_dataset, temp_dataset = train_test_split(dataset, test_size=0.3, random_state=42)
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# Optuna目标函数
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1000])
    num_heads = trial.suggest_int('num_heads', 2, 8)
    num_layers = trial.suggest_int('num_layers', 1, 6)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512, 1024])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    optimizer_type = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])

    # 参数验证
    if hidden_dim % num_heads != 0:
        logger.error(f"hidden_dim ({hidden_dim}) is not divisible by num_heads ({num_heads}).")
        return 0.0

    csv_path = r"E:\postgraduate\小论文\小论文\表格文件\id_tip_date_features.csv"
    dataset = TipDataset(csv_path)
    train_loader, val_loader, test_loader = create_data_loaders(dataset, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = TransformerModel(input_dim=dataset.combined_features.shape[1], num_heads=num_heads,
                                 num_layers=num_layers, num_classes=len(np.unique(dataset.labels)), dropout=dropout,
                                 hidden_dim=hidden_dim).to(device)
    except AssertionError as e:
        logger.error(f"Failed to create model with num_heads={num_heads}, hidden_dim={hidden_dim}: {e}")
        return 0.0

    criterion = nn.CrossEntropyLoss()

    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    num_epochs = 100
    best_accuracy = 0.0
    early_stopping_patience = 10
    no_improvement_count = 0
    writer = SummaryWriter(log_dir=f"runs/trial_{trial.number}")

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, scheduler, epoch, writer)
        val_accuracy, _, _ = test(model, device, val_loader, criterion, dataset.label_encoder, writer, epoch,
                                  phase='Validation')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            no_improvement_count = 0
            save_dir = f"trial_{trial.number}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    trial_params = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'dropout': dropout,
        'hidden_dim': hidden_dim,
        'weight_decay': weight_decay,
        'optimizer': optimizer_type,
        'best_accuracy': best_accuracy
    }
    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(trial_params, f, indent=4)

    return best_accuracy


# 创建研究对象并优化
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")

# 使用最佳超参数训练最终模型
best_params = study.best_params
learning_rate = best_params['learning_rate']
batch_size = best_params['batch_size']
num_heads = best_params['num_heads']
num_layers = best_params['num_layers']
dropout = best_params['dropout']
hidden_dim = best_params['hidden_dim']
weight_decay = best_params['weight_decay']
optimizer_type = best_params['optimizer']

# 参数验证
if hidden_dim % num_heads != 0:
    logger.error(f"hidden_dim ({hidden_dim}) is not divisible by num_heads ({num_heads}).")
    exit(1)

csv_path = r"E:\postgraduate\小论文\小论文\表格文件\id_tip_date_features.csv"
dataset = TipDataset(csv_path)
train_loader, val_loader, test_loader = create_data_loaders(dataset, batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerModel(input_dim=dataset.combined_features.shape[1], num_heads=num_heads, num_layers=num_layers,
                         num_classes=len(np.unique(dataset.labels)), dropout=dropout, hidden_dim=hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()

if optimizer_type == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer_type == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
elif optimizer_type == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

num_epochs = 100
best_accuracy = 0.0
early_stopping_patience = 10
no_improvement_count = 0
os.makedirs("final_model", exist_ok=True)
writer = SummaryWriter(log_dir="runs/final_model")

for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, scheduler, epoch, writer)
    val_accuracy, _, _ = test(model, device, val_loader, criterion, dataset.label_encoder, writer, epoch,
                              phase='Validation')

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        no_improvement_count = 0
        torch.save(model.state_dict(), 'final_model/best_model.pth')
        print(f'Best model saved with accuracy: {best_accuracy:.2f}%')
    else:
        no_improvement_count += 1

    if no_improvement_count >= early_stopping_patience:
        logger.info(f"Early stopping at epoch {epoch}")
        break

print(f'Final Best Accuracy: {best_accuracy:.2f}%')
