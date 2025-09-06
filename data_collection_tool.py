import os
import shutil
import requests
from PIL import Image
import hashlib
import json
from typing import List, Dict, Tuple
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class DataCollector:
    """数据收集和整理工具"""
    
    def __init__(self, base_dir: str = "./pendant_data"):
        """
        初始化数据收集器
        Args:
            base_dir: 基础数据目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 创建目录结构
        self.dirs = {
            'raw': self.base_dir / 'raw',
            'processed': self.base_dir / 'processed',
            'authentic': self.base_dir / 'processed' / 'authentic',
            'fake': self.base_dir / 'processed' / 'fake',
            'uncertain': self.base_dir / 'processed' / 'uncertain',
            'metadata': self.base_dir / 'metadata'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # 初始化元数据
        self.metadata_file = self.dirs['metadata'] / 'collection_log.json'
        self.load_metadata()
    
    def load_metadata(self):
        """加载现有元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'authentic_count': 0,
                'fake_count': 0,
                'uncertain_count': 0,
                'processed_hashes': [],
                'sources': []
            }
    
    def save_metadata(self):
        """保存元数据"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_image_hash(self, image_path: str) -> str:
        """计算图像哈希值以避免重复"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def validate_pendant_image(self, image_path: str) -> Tuple[bool, Dict]:
        """
        验证图像是否包含吊坠
        Args:
            image_path: 图像路径
        Returns:
            (is_valid, info_dict)
        """
        try:
            # 加载图像
            img = cv2.imread(image_path)
            if img is None:
                return False, {'reason': 'Cannot load image'}
            
            # 检查图像尺寸
            h, w = img.shape[:2]
            if min(h, w) < 200:
                return False, {'reason': 'Image too small'}
            
            # 检测主要物体（使用简单的轮廓检测）
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return False, {'reason': 'No clear object detected'}
            
            # 找到最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # 检查物体大小是否合理
            image_area = h * w
            object_ratio = area / image_area
            
            if object_ratio < 0.05 or object_ratio > 0.9:
                return False, {'reason': f'Object size ratio: {object_ratio:.2f}'}
            
            # 获取边界框
            x, y, bbox_w, bbox_h = cv2.boundingRect(largest_contour)
            
            info = {
                'image_size': (w, h),
                'object_area_ratio': object_ratio,
                'bbox': (x, y, bbox_w, bbox_h),
                'aspect_ratio': bbox_w / bbox_h if bbox_h > 0 else 0
            }
            
            return True, info
            
        except Exception as e:
            return False, {'reason': str(e)}
    
    def preprocess_image(self, image_path: str, output_path: str, 
                        target_size: Tuple[int, int] = (800, 800)) -> bool:
        """
        预处理图像
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            target_size: 目标尺寸
        Returns:
            是否成功
        """
        try:
            # 打开图像
            img = Image.open(image_path).convert('RGB')
            
            # 获取原始尺寸
            w, h = img.size
            
            # 计算缩放比例，保持长宽比
            scale = min(target_size[0] / w, target_size[1] / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 缩放图像
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # 创建白色背景
            background = Image.new('RGB', target_size, (255, 255, 255))
            
            # 将图像粘贴到中心
            paste_x = (target_size[0] - new_w) // 2
            paste_y = (target_size[1] - new_h) // 2
            background.paste(img, (paste_x, paste_y))
            
            # 保存
            background.save(output_path, quality=95)
            return True
            
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return False
    
    def organize_local_images(self, source_dir: str, category: str = 'uncertain'):
        """
        组织本地图像文件
        Args:
            source_dir: 源目录
            category: 分类 ('authentic', 'fake', 'uncertain')
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"Source directory {source_dir} does not exist")
            return
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        # 获取所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_path.glob(f'*{ext}'))
            image_files.extend(source_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images in {source_dir}")
        
        processed_count = 0
        skipped_count = 0
        
        for img_path in tqdm(image_files, desc=f"Processing {category} images"):
            # 计算哈希值
            img_hash = self.get_image_hash(str(img_path))
            
            # 检查是否已处理
            if img_hash in self.metadata['processed_hashes']:
                skipped_count += 1
                continue
            
            # 验证图像
            is_valid, info = self.validate_pendant_image(str(img_path))
            
            if not is_valid:
                print(f"Skipping {img_path.name}: {info.get('reason', 'Unknown')}")
                skipped_count += 1
                continue
            
            # 生成输出文件名
            output_name = f"{category}_{img_hash[:8]}_{processed_count:05d}.jpg"
            output_path = self.dirs[category] / output_name
            
            # 预处理并保存
            if self.preprocess_image(str(img_path), str(output_path)):
                processed_count += 1
                self.metadata['processed_hashes'].append(img_hash)
                self.metadata[f'{category}_count'] += 1
                
                # 保存图像信息
                image_info = {
                    'filename': output_name,
                    'original_name': img_path.name,
                    'category': category,
                    'hash': img_hash,
                    'validation_info': info
                }
                
                # 保存单个图像的元数据
                info_file = self.dirs['metadata'] / f"{output_name}.json"
                with open(info_file, 'w') as f:
                    json.dump(image_info, f, indent=2)
        
        print(f"Processed: {processed_count}, Skipped: {skipped_count}")
        self.save_metadata()
    
    def create_balanced_dataset(self, max_samples_per_class: int = None):
        """
        创建平衡的数据集
        Args:
            max_samples_per_class: 每个类别的最大样本数
        """
        # 统计各类别数量
        authentic_count = len(list(self.dirs['authentic'].glob('*.jpg')))
        fake_count = len(list(self.dirs['fake'].glob('*.jpg')))
        
        print(f"Current distribution - Authentic: {authentic_count}, Fake: {fake_count}")
        
        # 确定每个类别的样本数
        if max_samples_per_class:
            samples_per_class = min(max_samples_per_class, authentic_count, fake_count)
        else:
            samples_per_class = min(authentic_count, fake_count)
        
        # 创建平衡数据集目录
        balanced_dir = self.base_dir / 'balanced_dataset'
        balanced_dir.mkdir(exist_ok=True)
        
        # 复制平衡的样本
        for category in ['authentic', 'fake']:
            category_dir = balanced_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # 获取所有图像
            images = list(self.dirs[category].glob('*.jpg'))
            
            # 随机选择
            import random
            random.seed(42)
            selected = random.sample(images, min(samples_per_class, len(images)))
            
            # 复制文件
            for img_path in tqdm(selected, desc=f"Copying {category} images"):
                shutil.copy2(img_path, category_dir / img_path.name)
        
        print(f"Balanced dataset created with {samples_per_class} samples per class")
        
        # 创建训练集元数据
        metadata = {
            'samples_per_class': samples_per_class,
            'total_samples': samples_per_class * 2,
            'classes': ['authentic', 'fake'],
            'balanced': True
        }
        
        with open(balanced_dir / 'dataset_info.json', 'w') as f:
            json.dump(metadata, f, indent=2)

class DataAugmentor:
    """数据增强工具"""
    
    @staticmethod
    def augment_image(image: np.ndarray, augmentation_type: str) -> np.ndarray:
        """
        应用特定的数据增强
        Args:
            image: 输入图像
            augmentation_type: 增强类型
        Returns:
            增强后的图像
        """
        if augmentation_type == 'rotate':
            angle = np.random.uniform(-15, 15)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                                 borderValue=(255, 255, 255))
        
        elif augmentation_type == 'perspective':
            h, w = image.shape[:2]
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            
            # 随机透视变换
            offset = 0.1
            pts2 = np.float32([
                [np.random.uniform(0, w*offset), np.random.uniform(0, h*offset)],
                [np.random.uniform(w*(1-offset), w), np.random.uniform(0, h*offset)],
                [np.random.uniform(0, w*offset), np.random.uniform(h*(1-offset), h)],
                [np.random.uniform(w*(1-offset), w), np.random.uniform(h*(1-offset), h)]
            ])
            
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            return cv2.warpPerspective(image, matrix, (w, h),
                                     borderValue=(255, 255, 255))
        
        elif augmentation_type == 'lighting':
            # 随机调整亮度和对比度
            alpha = np.random.uniform(0.7, 1.3)  # 对比度
            beta = np.random.uniform(-30, 30)    # 亮度
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        elif augmentation_type == 'blur':
            kernel_size = np.random.choice([3, 5, 7])
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        elif augmentation_type == 'noise':
            noise = np.random.randn(*image.shape) * 10
            noisy = image + noise
            return np.clip(noisy, 0, 255).astype(np.uint8)
        
        else:
            return image
    
    @staticmethod
    def generate_augmented_dataset(source_dir: str, output_dir: str, 
                                  augmentations_per_image: int = 5):
        """
        生成增强数据集
        Args:
            source_dir: 源目录
            output_dir: 输出目录
            augmentations_per_image: 每张图像的增强次数
        """
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 增强类型
        augmentation_types = ['rotate', 'perspective', 'lighting', 'blur', 'noise']
        
        # 处理每个类别
        for category in ['authentic', 'fake']:
            category_source = source_path / category
            category_output = output_path / category
            category_output.mkdir(exist_ok=True)
            
            if not category_source.exists():
                continue
            
            images = list(category_source.glob('*.jpg'))
            
            for img_path in tqdm(images, desc=f"Augmenting {category} images"):
                # 加载原始图像
                img = cv2.imread(str(img_path))
                
                # 保存原始图像
                base_name = img_path.stem
                cv2.imwrite(str(category_output / f"{base_name}_original.jpg"), img)
                
                # 生成增强版本
                for i in range(augmentations_per_image):
                    # 随机选择增强类型
                    aug_type = np.random.choice(augmentation_types)
                    
                    # 应用增强
                    augmented = DataAugmentor.augment_image(img, aug_type)
                    
                    # 保存
                    output_name = f"{base_name}_aug_{aug_type}_{i}.jpg"
                    cv2.imwrite(str(category_output / output_name), augmented)
        
        print(f"Augmented dataset created in {output_dir}")

# 使用示例
def main():
    # 初始化数据收集器
    collector = DataCollector("./pendant_data")
    
    # 1. 组织本地正品图像（从3D渲染生成的）
    print("Step 1: Organizing authentic images from 3D renders...")
    collector.organize_local_images("./training_data/authentic", category='authentic')
    
    # 2. 组织本地赝品图像（需要您手动收集）
    print("\nStep 2: Organizing fake images...")
    print("Please place fake pendant images in './fake_pendants' directory")
    # collector.organize_local_images("./fake_pendants", category='fake')
    
    # 3. 创建平衡数据集
    print("\nStep 3: Creating balanced dataset...")
    collector.create_balanced_dataset()
    
    # 4. 数据增强
    print("\nStep 4: Augmenting dataset...")
    augmentor = DataAugmentor()
    augmentor.generate_augmented_dataset(
        "./pendant_data/balanced_dataset",
        "./pendant_data/augmented_dataset",
        augmentations_per_image=3
    )
    
    print("\nData preparation complete!")

if __name__ == "__main__":
    main()