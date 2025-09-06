import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class PendantDataset(Dataset):
    """The dataset for pendant authenticity classification."""
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: 数据目录
            split: 'train' 或 'val'
            transform: 数据变换
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 加载元数据
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # 加载正品数据
        authentic_data = [(os.path.join(data_dir, 'authentic', item['filename']), 1) 
                         for item in metadata]
        
        # 加载赝品数据（如果存在）
        fake_data = []
        fake_dir = os.path.join(data_dir, 'fake')
        if os.path.exists(fake_dir):
            for filename in os.listdir(fake_dir):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    fake_data.append((os.path.join(fake_dir, filename), 0))
        
        # 合并数据
        all_data = authentic_data + fake_data
        
        # 划分训练集和验证集
        train_data, val_data = train_test_split(
            all_data, test_size=0.2, random_state=42, stratify=[label for _, label in all_data]
        )
        
        self.data = train_data if split == 'train' else val_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

class GeometricFeatureExtractor(nn.Module):
    """提取几何特征的额外网络分支"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # 边缘检测卷积层
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 形状特征提取
        self.shape_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def forward(self, x):
        edge_features = self.edge_conv(x)
        shape_features = self.shape_conv(edge_features)
        return shape_features.squeeze(-1).squeeze(-1)

class PendantAuthenticator(nn.Module):
    """吊坠真伪鉴定模型"""
    
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        
        # 使用预训练的EfficientNet作为主干网络
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # 获取特征维度
        backbone_features = self.backbone.classifier[1].in_features
        
        # 移除原始分类器
        self.backbone.classifier = nn.Identity()
        
        # 几何特征提取器
        self.geometric_extractor = GeometricFeatureExtractor()
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(backbone_features + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # 提取视觉特征
        visual_features = self.backbone(x)
        
        # 提取几何特征
        geometric_features = self.geometric_extractor(x)
        
        # 特征融合
        combined = torch.cat([visual_features, geometric_features], dim=1)
        fused_features = self.fusion(combined)
        
        # 分类
        output = self.classifier(fused_features)
        
        return output

class Trainer:
    """模型训练器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # 损失函数：使用BCE损失用于二分类
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        # 记录训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predicted = torch.sigmoid(outputs) > 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        return total_loss / len(dataloader), correct / total
    
    def validate(self, dataloader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predicted = torch.sigmoid(outputs) > 0.5
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return total_loss / len(dataloader), correct / total, all_outputs, all_labels
    
    def fit(self, train_loader, val_loader, epochs=50):
        """训练模型"""
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, _, _ = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, 'best_pendant_model.pth')
                print(f"Saved best model with accuracy: {val_acc:.4f}")

class PendantPredictor:
    """用于推理的预测器"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # 加载模型
        self.model = PendantAuthenticator()
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """
        预测单张图片
        Args:
            image_path: 图片路径
        Returns:
            authenticity_score: 0-1之间的真品概率
            is_authentic: 布尔值判断
        """
        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = torch.sigmoid(output).item()
        
        return {
            'authenticity_score': probability,
            'is_authentic': probability > 0.5,
            'confidence': abs(probability - 0.5) * 2  # 0-1的置信度
        }
    
    def batch_predict(self, image_paths):
        """批量预测"""
        results = []
        for path in tqdm(image_paths, desc="Predicting"):
            result = self.predict(path)
            result['image_path'] = path
            results.append(result)
        return results

# 训练脚本
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = PendantDataset('./training_data', split='train', 
                                  transform=train_transform)
    val_dataset = PendantDataset('./training_data', split='val',
                                transform=val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                          num_workers=4, pin_memory=True)
    
    # 创建模型
    model = PendantAuthenticator()
    
    # 创建训练器并训练
    trainer = Trainer(model, device)
    trainer.fit(train_loader, val_loader, epochs=50)
    
    print("Training complete!")

if __name__ == "__main__":
    main()