"""
Production Rendering System Configurations
"""

import os
import json
from pendant_renderer import PendantRenderer
from PIL import Image
import numpy as np
from tqdm import tqdm

# Calibrated perfect parameters
CALIBRATED_PARAMS = {
    "distance": 3.0,
    "azimuth": 0,
    "elevation": 5,
    "offset": [0, -1.0, 0],
    "fov": 60
}

class ProductionRenderer:
    """Production Renderer with Calibrated Parameters"""
    
    def __init__(self, glb_path):
        """
        initialize the Production Renderer
        Args:
            glb_path: GLB file path
        """
        self.glb_path = glb_path
        self.base_params = CALIBRATED_PARAMS.copy()
        self.renderer = PendantRenderer(glb_path, self.base_params)
        
        print("="*60)
        print(" Production Renderer Initialized ")
        print(" Base Parameters:")
        for key, value in self.base_params.items():
            print(f"  {key}: {value}")
    
    def generate_authentic_dataset(self, output_dir="./dataset/authentic", 
                                  num_views=200, 
                                  var_config=None):
        """
        Generate authentic dataset with realistic variations
        Args:
            output_dir: Output directory
            num_views: Number of views to generate
            var_config: Configuration for variations
        """
        if var_config is None:
            var_config = {
                'distance_range': (2.75, 3.75),  # range for distance
                'azimuth_std': 10,             # Standard deviation for horizontal rotation
                'elevation_std': 10,           # Standard deviation for vertical rotation
                'offset_std': 0.2,            # Standard deviation for center offset
                'lighting_variations': True    # Lighting variations
            }
        
        os.makedirs(output_dir, exist_ok=True)
        metadata = []
        
        print(f"\nGenerated {num_views} product views...")
        print(f"Output Dir: {output_dir}")
        
        for i in tqdm(range(num_views), desc="rendering views progress"):
            params = self.base_params.copy()
            
            # Distance, uniform distribution
            params['distance'] = np.random.uniform(
                var_config['distance_range'][0], var_config['distance_range'][1]
            )
            
            # Horizontal rotation (normal distribution, centered around 0)
            params['azimuth'] = np.random.normal(0, var_config['azimuth_std'])
            params['azimuth'] = np.clip(params['azimuth'], -22.5, 22.5)
            
            # Vertical rotation (normal distribution, centered around 5)
            params['elevation'] = np.random.normal(5, var_config['elevation_std'])
            params['elevation'] = np.clip(params['elevation'], -22.5, 22.5)
            
            # Center Offset, based on the -1.0 Y offset
            params['offset'] = [
                np.random.normal(0, var_config['offset_std']),
                -1.0 + np.random.normal(0, var_config['offset_std']),
                np.random.normal(0, var_config['offset_std'] * 0.5)
            ]
            
            # Render view
            image = self.renderer.render_view_with_params(params)
            
            # Apply realistic augmentations
            if var_config['lighting_variations']:
                image = self.apply_realistic_augmentation(image, i)
            
            # Save image
            filename = f'authentic_{i:05d}.jpg'
            filepath = os.path.join(output_dir, filename)
            Image.fromarray(image).save(filepath, quality=95)
            
            # Record metadata
            metadata.append({
                'filename': filename,
                'label': 'authentic',
                'params': {
                    'distance': float(params['distance']),
                    'azimuth': float(params['azimuth']),
                    'elevation': float(params['elevation']),
                    'offset': params['offset']
                }
            })
        
        # Save metadata
        metadata_file = os.path.join(output_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Finished generating authentic dataset.")
        print(f"  - Number of views generated: {num_views}")
        print(f"  - Metadata: {metadata_file}")
        
        return metadata
    
    def apply_realistic_augmentation(self, image, seed=None):
        """
        Mimic realistic photo variations:
        - Lighting changes (brightness, contrast), color shifts
        - Slight blur (focus imperfections)
        """
        if seed is not None:
            np.random.seed(seed)
        
        import cv2
        
        image = np.array(image, copy=True)
        # Convert RGBA to RGB if needed
        if image.shape[2] == 4:
            alpha = image[:, :, 3:4] / 255.0
            white_bg = np.ones_like(image[:, :, :3]) * 255
            image = (image[:, :, :3] * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        
        # 1. Lighting variations
        if np.random.random() > 0.7:
            # Brightness adjustment
            brightness = np.random.uniform(0.7, 1.3)
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
            
            # Contrast adjustment
            contrast = np.random.uniform(0.8, 1.2)
            mean = np.mean(image) # type: ignore
            image = cv2.convertScaleAbs(image, alpha=contrast, beta=(1-contrast)*mean)
        
        # 2. Slight Perspective Warp to simulate angle imperfections
        if np.random.random() > 0.4 and np.random.random() < 0.5:
            h, w = image.shape[:2]
            # A small random perspective transform
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) # type: ignore
            pts2 = np.float32([
                [np.random.uniform(0, w*0.02), np.random.uniform(0, h*0.02)],
                [np.random.uniform(w*(1-0.02), w), np.random.uniform(0, h*0.02)],
                [np.random.uniform(0, w*0.02), np.random.uniform(h*(1-0.02), h)],
                [np.random.uniform(w*(1-0.02), w), np.random.uniform(h*(1-0.02), h)]
            ]) # type: ignore
            matrix = cv2.getPerspectiveTransform(pts1, pts2) # type: ignore
            image = cv2.warpPerspective(image, matrix, (w, h), borderValue=(255, 255, 255))
        
        return image
    
    def generate_test_views(self, output_dir="./test_views"):
        """
        Generate test views to verify calibration
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n Generating test views...")
        
        test_configs = [
            ("calibrated_perfect", self.base_params.copy()),
            ("front_close", {**self.base_params, 'distance': 2.7}),
            ("front_far", {**self.base_params, 'distance': 3.5}),
            ("slight_left", {**self.base_params, 'azimuth': -10}),
            ("slight_right", {**self.base_params, 'azimuth': 10}),
            ("from_above", {**self.base_params, 'elevation': 10}),
            ("from_level", {**self.base_params, 'elevation': 0}),
        ]
        
        for name, params in test_configs:
            image = self.renderer.render_view_with_params(params)
            Image.fromarray(image).save(f"{output_dir}/{name}.jpg")
            print(f"  ✓ {name}.jpg")
        
        print(f"\nTest images are saved to: {output_dir}")
    
    def validate_calibration(self):
        """
        Validate calibration by creating a multi-angle preview grid
        """
        print("\n Validating calibration...")
        
        # A 3x3 grid of views around the calibrated view
        angles = [
            (-15, 10),  (0, 10),  (15, 10),   # Row 1
            (-15,  5),  (0,  5),  (15,  5),   # Row 2 (the calibrated view is here at center)
            (-15, -5),  (0, -5),  (15, -5)    # Row 3
        ]
        
        grid_img = Image.new('RGB', (900, 900), (240, 240, 240))
        
        for i, (azi, ele) in enumerate(angles):
            params = self.base_params.copy()
            params['azimuth'] = azi
            params['elevation'] = ele
            
            # Render and resize
            image = self.renderer.render_view_with_params(params)
            img = Image.fromarray(image)
            img = img.resize((290, 290), Image.Resampling.LANCZOS)

            # Paste into grid
            grid_img.paste(img, (i % 3 * 300 + 5, i // 3 * 300 + 5))
        
        # Save grid image
        grid_img.save("calibration_validation.jpg")
        print("✓ Calibration Validation Grid Image: calibration_validation.jpg")
        print("  The image in the center should match your calibrated view.")

# Main function to run different modes
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="The Production Rendering System")
    parser.add_argument('glb_path')
    parser.add_argument('--mode', choices=['generate', 'test', 'validate'],
                       default='generate', help='The mode to run')
    parser.add_argument('--num-views', type=int, default=200)
    parser.add_argument('--output-dir', default='./dataset/authentic')
    
    args = parser.parse_args()
    
    # Initialize renderer
    renderer = ProductionRenderer(args.glb_path)
    
    if args.mode == 'generate':
        # Generate authentic dataset
        renderer.generate_authentic_dataset(
            output_dir=args.output_dir, num_views=args.num_views
        )
    elif args.mode == 'test':
        # Generate test views
        renderer.generate_test_views()
    elif args.mode == 'validate':
        # The validate calibration mode
        renderer.validate_calibration()

if __name__ == "__main__":
    main()