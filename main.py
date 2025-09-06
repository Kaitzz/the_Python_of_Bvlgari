"""
宝格丽Divas' Dream吊坠真伪鉴定系统
Main Project File and Complete Pipeline
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import json
from datetime import datetime

# 导入项目模块
from pendant_renderer import PendantRenderer
from pendant_classifier import PendantAuthenticator, Trainer, PendantPredictor
from data_collection_tool import DataCollector, DataAugmentor

class PendantAuthenticationSystem:
    """完整的吊坠鉴定系统"""
    
    def __init__(self, project_dir="."):
        """
        初始化系统
        Args:
            project_dir: 项目根目录
        """
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(exist_ok=True)
        
        # 创建项目结构
        self.dirs = {
            'models': self.project_dir / 'models',
            'data': self.project_dir / 'data',
            'results': self.project_dir / 'results',
            'logs': self.project_dir / 'logs',
            'config': self.project_dir / 'config'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # 初始化配置
        self.config = self.load_config()
        
    def load_config(self):
        """加载或创建配置文件"""
        config_file = self.dirs['config'] / 'config.json'
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # 默认配置
            config = {
                'glb_path': 'bulgari_pendant.glb',
                'render_views': 300,
                'image_size': 800,
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 1e-4,
                'confidence_threshold': 0.7
            }
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            return config
    
    def step1_generate_3d_renders(self, glb_path):
        """
        步骤1: 从3D模型生成渲染图像
        """
        print("\n" + "="*50)
        print("STEP 1: Generating 3D Renders")
        print("="*50)
        
        output_dir = self.dirs['data'] / '3d_renders'
        
        # 初始化渲染器
        renderer = PendantRenderer(glb_path)
        
        # 生成训练数据
        renderer.generate_training_data(
            output_dir=str(output_dir),
            num_views=self.config['render_views']
        )
        
        print(f"✓ Generated {self.config['render_views']} 3D renders")
        print(f"✓ Saved to: {output_dir}")
        
    def step2_collect_fake_samples(self, fake_images_dir):
        """
        步骤2: 收集和组织赝品样本
        """
        print("\n" + "="*50)
        print("STEP 2: Collecting Fake Samples")
        print("="*50)
        
        collector = DataCollector(str(self.dirs['data']))
        
        # 组织正品图像（从3D渲染）
        print("Organizing authentic images...")
        collector.organize_local_images(
            str(self.dirs['data'] / '3d_renders' / 'authentic'),
            category='authentic'
        )
        
        # 组织赝品图像
        if fake_images_dir and Path(fake_images_dir).exists():
            print("Organizing fake images...")
            collector.organize_local_images(fake_images_dir, category='fake')
        else:
            print("⚠ No fake images directory provided")
            print("Please collect fake pendant images and run:")
            print(f"  python main.py collect-fakes --fake-dir YOUR_FAKE_IMAGES_DIR")
        
    def step3_prepare_dataset(self):
        """
        步骤3: 准备平衡的训练数据集
        """
        print("\n" + "="*50)
        print("STEP 3: Preparing Dataset")
        print("="*50)
        
        collector = DataCollector(str(self.dirs['data']))
        
        # 创建平衡数据集
        print("Creating balanced dataset...")
        collector.create_balanced_dataset()
        
        # 数据增强
        print("Applying data augmentation...")
        augmentor = DataAugmentor()
        augmentor.generate_augmented_dataset(
            str(self.dirs['data'] / 'balanced_dataset'),
            str(self.dirs['data'] / 'augmented_dataset'),
            augmentations_per_image=3
        )
        
        print("✓ Dataset prepared and augmented")
        
    def step4_train_model(self):
        """
        步骤4: 训练鉴定模型
        """
        print("\n" + "="*50)
        print("STEP 4: Training Model")
        print("="*50)
        
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from pendant_classifier import PendantDataset
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 数据变换
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
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
        data_dir = self.dirs['data'] / 'augmented_dataset'
        if not data_dir.exists():
            data_dir = self.dirs['data'] / 'processed'
        
        train_dataset = PendantDataset(str(data_dir), split='train', 
                                      transform=train_transform)
        val_dataset = PendantDataset(str(data_dir), split='val',
                                    transform=val_transform)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, 
                                batch_size=self.config['batch_size'],
                                shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset,
                              batch_size=self.config['batch_size'],
                              shuffle=False, num_workers=4)
        
        # 创建模型
        model = PendantAuthenticator()
        
        # 训练
        trainer = Trainer(model, device)
        trainer.fit(train_loader, val_loader, epochs=self.config['epochs'])
        
        # 保存最终模型
        final_model_path = self.dirs['models'] / 'pendant_authenticator.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'training_date': datetime.now().isoformat()
        }, final_model_path)
        
        print(f"✓ Model trained and saved to: {final_model_path}")
        
    def step5_evaluate(self, test_image_path):
        """
        步骤5: 评估单张图像
        """
        print("\n" + "="*50)
        print("STEP 5: Evaluation")
        print("="*50)
        
        model_path = self.dirs['models'] / 'pendant_authenticator.pth'
        if not model_path.exists():
            model_path = 'best_pendant_model.pth'
        
        if not Path(model_path).exists():
            print("⚠ No trained model found. Please train the model first.")
            return
        
        # 初始化预测器
        predictor = PendantPredictor(str(model_path))
        
        # 预测
        result = predictor.predict(test_image_path)
        
        # 显示结果
        print("\n" + "-"*40)
        print("AUTHENTICATION RESULT")
        print("-"*40)
        print(f"Image: {test_image_path}")
        print(f"Authenticity Score: {result['authenticity_score']:.3f}")
        print(f"Verdict: {'AUTHENTIC ✓' if result['is_authentic'] else 'FAKE ✗'}")
        print(f"Confidence: {result['confidence']:.1%}")
        
        # 保存结果
        result_file = self.dirs['results'] / f"result_{datetime.now():%Y%m%d_%H%M%S}.json"
        result['image_path'] = test_image_path
        result['timestamp'] = datetime.now().isoformat()
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✓ Result saved to: {result_file}")

# 命令行接口
def main():
    parser = argparse.ArgumentParser(
        description="Bulgari Divas' Dream Pendant Authentication System"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 完整流程
    parser_full = subparsers.add_parser('full-pipeline', 
                                        help='Run complete pipeline')
    parser_full.add_argument('--glb-path', required=True,
                            help='Path to GLB 3D model file')
    parser_full.add_argument('--fake-dir', 
                            help='Directory containing fake pendant images')
    
    # 单独步骤
    parser_render = subparsers.add_parser('render',
                                          help='Generate 3D renders only')
    parser_render.add_argument('--glb-path', required=True,
                              help='Path to GLB 3D model file')
    
    parser_collect = subparsers.add_parser('collect-fakes',
                                          help='Collect fake samples')
    parser_collect.add_argument('--fake-dir', required=True,
                              help='Directory containing fake images')
    
    parser_train = subparsers.add_parser('train',
                                        help='Train the model')
    
    parser_predict = subparsers.add_parser('predict',
                                          help='Predict single image')
    parser_predict.add_argument('--image', required=True,
                              help='Path to image to authenticate')
    
    # 批量预测
    parser_batch = subparsers.add_parser('batch-predict',
                                        help='Predict multiple images')
    parser_batch.add_argument('--dir', required=True,
                            help='Directory containing images')
    
    args = parser.parse_args()
    
    # 初始化系统
    system = PendantAuthenticationSystem()
    
    if args.command == 'full-pipeline':
        # 运行完整流程
        system.step1_generate_3d_renders(args.glb_path)
        system.step2_collect_fake_samples(args.fake_dir)
        system.step3_prepare_dataset()
        system.step4_train_model()
        print("\n✅ PIPELINE COMPLETE!")
        print("You can now use 'predict' command to authenticate pendant images")
        
    elif args.command == 'render':
        system.step1_generate_3d_renders(args.glb_path)
        
    elif args.command == 'collect-fakes':
        system.step2_collect_fake_samples(args.fake_dir)
        system.step3_prepare_dataset()
        
    elif args.command == 'train':
        system.step4_train_model()
        
    elif args.command == 'predict':
        system.step5_evaluate(args.image)
        
    elif args.command == 'batch-predict':
        model_path = system.dirs['models'] / 'pendant_authenticator.pth'
        if not model_path.exists():
            print("⚠ No trained model found. Please train the model first.")
            return
        
        predictor = PendantPredictor(str(model_path))
        
        # Get all image files in the directory
        image_dir = Path(args.dir)
        image_files = list(image_dir.glob('*.jpg')) + \
                     list(image_dir.glob('*.png')) + \
                     list(image_dir.glob('*.jpeg'))
        
        results = predictor.batch_predict([str(f) for f in image_files])
        
        # 保存结果
        output_file = system.dirs['results'] / f"batch_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Summary
        authentic_count = sum(1 for r in results if r['is_authentic'])
        fake_count = len(results) - authentic_count
        
        print(f"\nProcessed {len(results)} images:")
        print(f"  Authentic: {authentic_count}")
        print(f"  Fake: {fake_count}")
        print(f"\n✓ Results saved to: {output_file}")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()