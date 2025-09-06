"""
A quick start script to test rendering and generate a small dataset.
Run: python quick_start.py bulgari_pendant.glb
"""

import os
import sys
from pathlib import Path
from pendant_renderer import PendantRenderer
from PIL import Image
import numpy as np
from tqdm import tqdm

# The perfect calibration parameters
PERFECT_PARAMS = {
    "distance": 3.0,
    "azimuth": 0,
    "elevation": 5,
    "offset": [0, -1.0, 0],
    "fov": 60
}

def quick_test(glb_path):
    """A quick test rendering with perfect params"""
    print("Quick test with perfect params")
    
    # Initialize renderer
    renderer = PendantRenderer(glb_path, PERFECT_PARAMS)
    
    # Generate perfect front view
    print("\n1. Generate perfect front view...")
    perfect_view = renderer.render_view_with_params(PERFECT_PARAMS)
    Image.fromarray(perfect_view).save("perfect_front_view.jpg")
    print("   ✓ perfect_front_view.jpg is saved.")
    
    # Generate multi-angle test grid
    print("\n2. Generate multi-angle test grid...")
    
    images = []
    for angle in [-15, 0, 15]:
        params = PERFECT_PARAMS.copy()
        params['azimuth'] = angle
        img = renderer.render_view_with_params(params)
        images.append(Image.fromarray(img))
    
    # 创建网格
    grid = Image.new('RGB', (800*3, 800), (255, 255, 255))
    for i, img in enumerate(images):
        grid.paste(img, (i*800, 0))
    grid = grid.resize((1200, 400), Image.Resampling.LANCZOS)
    grid.save("angle_test_grid.jpg")
    print("   ✓ angle_test_grid.jpg is saved.")
    
    print("\n ✓✓Test completed. Check the generated images.")

def generate_small_dataset(glb_path, num_samples=100):
    """Generate a small dataset for quick inspection"""
    print("\n" + "="*60)
    print(f"A small dataset with {num_samples} samples will be generated.")
    
    # Create output directory
    output_dir = Path("quick_dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize renderer
    renderer = PendantRenderer(glb_path, PERFECT_PARAMS)
    
    print(f"\n {num_samples} samples will be generated with slight random variations around perfect params.")
    for i in tqdm(range(num_samples)):
        # Add small random variations around perfect params
        params = PERFECT_PARAMS.copy()
        
        # tweak parameters slightly
        params['distance'] = np.random.uniform(2.8, 3.3)
        params['azimuth'] = np.random.normal(0, 12)
        params['elevation'] = np.random.normal(1, 8)
        params['offset'] = [np.random.normal(0, 0.05), -1.0 + np.random.normal(0, 0.05), 0]
        
        # Render
        image = renderer.render_view_with_params(params)
        
        # Save image
        Image.fromarray(image).save(output_dir / f"sample_{i:04d}.jpg")
    
    print(f"\n✅ Dataset generated and saved at {output_dir} ")
    print(f"   With {num_samples} samples for quick inspection.")

def main():
    """main function"""
    if len(sys.argv) < 2:
        print("Syntax: python quick_start.py <glb_file> [command]")
        print("\nCommands:")
        print("  test     - Quick test rendering (default)")
        print("  dataset  - Generate a small dataset")
        print("\nExample:")
        print("  python quick_start.py bulgari_pendant.glb")
        print("  python quick_start.py bulgari_pendant.glb dataset")
        sys.exit(1)
    
    glb_path = sys.argv[1]
    command = sys.argv[2] if len(sys.argv) > 2 else "test"
    
    # Check if file exists
    if not os.path.exists(glb_path):
        print(f"Error: File not found {glb_path}")
        sys.exit(1)
    
    print("\n" + "="*30)
    print("  Quick Start Script ")
    print(f"\nGLB file: {glb_path}")
    print("\nWith starting param:")
    for key, value in PERFECT_PARAMS.items():
        print(f"  {key}: {value}")
    print("-"*60)
    
    if command == "test":
        quick_test(glb_path)
    elif command == "dataset":
        generate_small_dataset(glb_path)
    else:
        print(f"Unknown command: {command}")
    
    print("\n Finish Quick Start!")
    
    print("\nThe next steps:")
    print("\n1. If satisfied with the rendering, generate a full dataset for training:")
    print("   python production_config.py bulgari_pendant.glb --num-views 500")
    print("\n2. Run the full workflow with counterfeit images:")
    print("   python workflow.py --glb-path bulgari_pendant.glb --fake-dir ./fake_images")

if __name__ == "__main__":
    main()