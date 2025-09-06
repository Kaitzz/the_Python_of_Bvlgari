"""
Test script to verify the fix for data augmentation issues in pendant_renderer.py.
"""

import numpy as np
from pendant_renderer import PendantRenderer
from production_config import ProductionRenderer
from PIL import Image
from pathlib import Path

def test_augmentation():
    """Test data augmentation fix"""
    print("Testing data augmentation...")
    
    # Initialize renderer
    renderer = ProductionRenderer(Path("./bulgari_pendant.glb"))
    
    # Generate a test image
    test_image = renderer.renderer.render_view_with_params(renderer.base_params)
    print(f"✓ The test image: shape={test_image.shape}, dtype={test_image.dtype}")
    print(f"  Read only? - {not test_image.flags.writeable}")
    
    # Test 1: Apply augmentation to a single image
    try:
        augmented = renderer.apply_realistic_augmentation(test_image, seed=42)
        print(f"✓ Test1: single image augmentation passed: shape={augmented.shape}, dtype={augmented.dtype}")
        
        # Save images for visual inspection
        Image.fromarray(test_image).save("test_original.jpg")
        Image.fromarray(augmented).save("test_augmented.jpg")
        print("✓ Saved test images for visual inspection:")
        print("  - test_original.jpg : base image")
        print("  - test_augmented.jpg : augmented image")
        
        return True
    except Exception as e:
        print(f"✗ Test1: single image augmentation failed: {e}")
        return False

def test_batch_generation():
    """Test batch dataset generation"""
    print("\n Testing batch dataset generation With 10 views...")
    
    renderer = ProductionRenderer(Path("bulgari_pendant.glb"))
    
    try:
        renderer.generate_authentic_dataset(
            output_dir="./test_dataset", num_views=10
        )
        print("✓ Test2: batch generation passed. Check ./test_dataset for results.")
        return True
    except Exception as e:
        print(f"✗ Test2: batch generation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing data augmentation fix in pendant_renderer.py")
    
    # Tests
    test1_passed = test_augmentation()
    test2_passed = test_batch_generation()
    
    if test1_passed and test2_passed:
        print("\n🎉 Both tests passed!")
        print("\n You can now generate a full dataset with:")
        print("python production_config.py bulgari_pendant.glb --num-views 200")
    else:
        print("\n !Please fix the issues above before proceeding.")