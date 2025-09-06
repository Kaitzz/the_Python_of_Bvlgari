import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
import sys
from pendant_renderer import PendantRenderer

class CameraCalibrator:
    """A tool for interactively calibrating camera parameters."""
    
    def __init__(self, glb_path):
        """
        Initialize the Calibrator
        Args:
            glb_path: GLB file path
        """
        self.glb_path = glb_path
        
        # Base camera parameters
        self.camera_params = {
            'distance': 3.0,      # Distance from the object
            'azimuth': 0,         # Horizontal rotation
            'elevation': 0,       # Vertical rotation
            'offset': [0, 0, 0],  # Center offset (x, y, z)
            'fov': 60             # FOC
        }
        
        # Initialize renderer
        self.renderer = PendantRenderer(glb_path, self.camera_params)
        
    def interactive_calibrate(self):
        """Interactive Calibration Loop"""
        print("\n" + "="*60)
        print("This tool helps you calibrate camera parameters interactively.")
        print("✓ Parameters are independent of each other:")
        print("✓ Zoom does not affect angle")
        print("✓ Rotation does not affect zoom or center")
        print("\n How to Use:")
        print("  [Q/A] - Distance ±0.5")
        print("  [W/S] - Elevation ±5°")
        print("  [E/D] - Azimuth ±5°")
        print("  [R/F] - FOV ±5°")
        print("  [T/G] - X-axis Offset ±0.5")
        print("  [Y/H] - Y-axis Offset ±0.5")
        print("  [U/J] - Z-axis Offset ±0.5")
        print("  test  - Run independence test")
        print("  save  - Save parameters and generate usage code")
        print("  reset - Reset parameters to default")
        print("  help  - Help")
        print("  quit  - Exit")
        
        os.makedirs("./calibration", exist_ok=True)
        
        while True:
            # Generate and show preview
            self.generate_preview()
            
            # Display current parameters
            self.display_params()
            
            # Get user input
            command = input("\nInput: ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'save':
                self.save_parameters()
            elif command == 'reset':
                self.reset_parameters()
            elif command == 'test':
                self.test_independence()
            elif command == 'help':
                self.show_help()
            else:
                self.process_command(command)
    
    def display_params(self):
        """Show Current Parameters"""
        print("\nCurrent Camera Parameters:")
        print(f"  Distance:  {self.camera_params['distance']:.1f}")
        print(f"  Azimuth | Left-Right:   {self.camera_params['azimuth']:.0f}°")
        print(f"  Elevation | Pitching: {self.camera_params['elevation']:.0f}°")
        print(f"  Center Offset:    [{self.camera_params['offset'][0]:.2f}, "
              f"{self.camera_params['offset'][1]:.2f}, "
              f"{self.camera_params['offset'][2]:.2f}]")
        print(f"  FOV:     {self.camera_params['fov']:.0f}°")
    
    def process_command(self, command):
        """Process User Command"""
        # Distance
        if command == 'q':
            self.camera_params['distance'] += 0.5
            print(f"→ Distance = {self.camera_params['distance']:.1f} Zoom In")
        elif command == 'a':
            self.camera_params['distance'] = max(1.0, 
                self.camera_params['distance'] - 0.5)
            print(f"→ Distance = {self.camera_params['distance']:.1f} Zoom Out")
        
        # Elevation
        elif command == 'w':
            self.camera_params['elevation'] += 5
            print(f"→ Elevation = {self.camera_params['elevation']:.0f}° (向上)")
        elif command == 's':
            self.camera_params['elevation'] -= 5
            print(f"→ Elevation = {self.camera_params['elevation']:.0f}° Facing Downward")
        
        # Horizontal Rotation
        elif command == 'e':
            self.camera_params['azimuth'] += 5
            print(f"→ Azimuth = {self.camera_params['azimuth']:.0f}° Rotate Right")
        elif command == 'd':
            self.camera_params['azimuth'] -= 5
            print(f"→ Azimuth = {self.camera_params['azimuth']:.0f}°  Rotate Left")
        
        # FOV
        elif command == 'r':
            self.camera_params['fov'] = min(120, self.camera_params['fov'] + 5)
            print(f"→ FOV = {self.camera_params['fov']:.0f}° Increase")
        elif command == 'f':
            self.camera_params['fov'] = max(20, self.camera_params['fov'] - 5)
            print(f"→ FOV = {self.camera_params['fov']:.0f}° Decrease")
        
        # X-axis Offset
        elif command == 't':
            self.camera_params['offset'][0] += 0.1
            print(f"→ X Offset = {self.camera_params['offset'][0]:.2f} Shift Right")
        elif command == 'g':
            self.camera_params['offset'][0] -= 0.1
            print(f"→ X Offset = {self.camera_params['offset'][0]:.2f} Shift Left")
        
        # Y-axis Offset
        elif command == 'y':
            self.camera_params['offset'][1] += 0.1
            print(f"→ Y Offset = {self.camera_params['offset'][1]:.2f} Shift Up")
        elif command == 'h':
            self.camera_params['offset'][1] -= 0.1
            print(f"→ Y Offset = {self.camera_params['offset'][1]:.2f} Shift Down")
        
        # Z-axis Offset
        elif command == 'u':
            self.camera_params['offset'][2] += 0.1
            print(f"→ Z Offset = {self.camera_params['offset'][2]:.2f} Inward")
        elif command == 'j':
            self.camera_params['offset'][2] -= 0.1
            print(f"→ Z Offset = {self.camera_params['offset'][2]:.2f}  Backward")
        
        # Or, set specific parameter with param value input
        elif ' ' in command:
            parts = command.split()
            if len(parts) == 2:
                param, value = parts
                try:
                    value = float(value)
                    if param == 'dist' or param == 'distance':
                        self.camera_params['distance'] = value
                    elif param == 'azi' or param == 'azimuth':
                        self.camera_params['azimuth'] = value
                    elif param == 'ele' or param == 'elevation':
                        self.camera_params['elevation'] = value
                    elif param == 'fov':
                        self.camera_params['fov'] = value
                    print(f"✓ Set {param} = {value}")
                except ValueError:
                    print("✗ Invalid number format")
        else:
            print(f"✗ Unknown Command: {command}")
    
    def generate_preview(self):
        """Generate Preview Image"""
        print("\nPreview generating...")
        
        # The main rendered image
        main_image = self.renderer.render_view_with_params(self.camera_params)
        
        # Preview of the main rendered image with parameters
        preview = Image.new('RGB', (800, 600), (240, 240, 240))
        
        main_img = Image.fromarray(main_image)
        main_img = main_img.resize((500, 500), Image.Resampling.LANCZOS)
        preview.paste(main_img, (50, 50))
        
        # Add text overlay
        draw = ImageDraw.Draw(preview)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Title
        draw.text((360, 20), "Camera Calibration Preview", 
                 fill=(0, 0, 0), font=font, anchor="mm")
        
        # Parameters text
        params_text = [
            f"Distance: {self.camera_params['distance']:.1f}",
            f"Azimuth: {self.camera_params['azimuth']:.0f}°",
            f"Elevation: {self.camera_params['elevation']:.0f}°",
            f"Center X: {self.camera_params['offset'][0]:.2f}",
            f"Center Y: {self.camera_params['offset'][1]:.2f}",
            f"Center Z: {self.camera_params['offset'][2]:.2f}",
            f"FOV: {self.camera_params['fov']:.0f}°"
        ]
        
        for i, text in enumerate(params_text):
            draw.text((560, 100 + i * 30), text, 
                     fill=(0, 0, 0), font=small_font)
        
        # Save preview images
        preview_path = "./calibration/preview.png"
        preview.save(preview_path)
        print(f"✓ The priveiw is saved to {preview_path}")
        pure_path = "./calibration/current_view.png"
        Image.fromarray(main_image).save(pure_path)
    
    def test_independence(self):
        """Independence Test on Parameters"""
        self.renderer.camera_params = self.camera_params
        # Test1：Only change zoom (keep rotation unchanged)
        for dist in [2, 3, 4, 5]:
            params = {'distance': dist}
            img = self.renderer.render_view_with_params(params)
            Image.fromarray(img).save(f"./calibration/zoom_{dist}.png")
            print(f"✓ zoom_{dist}.png generated")

        # Test2：Only change rotation (keep zoom unchanged)
        for angle in [-15, 0, 15]:
            params = {'distance': 3, 'azimuth': angle}
            img = self.renderer.render_view_with_params(params)
            Image.fromarray(img).save(f"./calibration/rotate_{angle}.png")
            print(f"✓ rotate_{angle}.png generated")

        print("\n View the generated images to verify the indepency of params.")
    
    def reset_parameters(self):
        """Reset the Parameters"""
        self.camera_params = {
            'distance': 3.0,
            'azimuth': 0,
            'elevation': 0,
            'offset': [0, 0, 0],
            'fov': 60
        }
        print("✓ Parameters reset to default values.")
    
    def save_parameters(self):
        """Save Current Parameters"""
        # Save as JSON
        params_file = "./calibration/camera_params.json"
        with open(params_file, 'w') as f:
            json.dump(self.camera_params, f, indent=2)
        print(f"✓ Params saved to: {params_file}")
        
        # Code snippet for usage
        code = f"""
# Parameters generated by calibration_tool.py
from pendant_renderer import PendantRenderer

camera_params = {{
    'distance': {self.camera_params['distance']},
    'azimuth': {self.camera_params['azimuth']},
    'elevation': {self.camera_params['elevation']},
    'offset': {self.camera_params['offset']},
    'fov': {self.camera_params['fov']}
}}

# Apply parameters to renderer
renderer = PendantRenderer("bulgari_pendant.glb", camera_params)

# 生成训练数据
renderer.generate_training_data(
    output_dir="./training_data", num_views=200
)
"""
        
        code_file = "./calibration/use_params.py"
        with open(code_file, 'w') as f:
            f.write(code)
        print(f"✓ Code saved to: {code_file}")
    
    def show_help(self):
        """Show Help Information"""
        print("\n" + "="*50)
        print("Parameter Descriptions:")
        print("  Distance: The distance from the camera to the object")
        print("  Azimuth: The horizontal rotation angle (negative=left, positive=right)")
        print("  Elevation: The vertical rotation angle (negative=down, positive=up)")
        print("  Center Offset: The XYZ offset to center the object")
        print("  FOV: The camera's field of view angle, affecting zoom level")
        print("\nCalibration Tips:")
        print("  - Adjust Distance to fit the object size")
        print("  - Then, adjust Azimuth and Elevation to get the desired angle")
        print("  - Finally, adjust the Center Offset to perfectly center the object")

def main():
    """Main Function"""
    if len(sys.argv) < 2:
        print("Syntax: python calibration_tool.py <glb_file>")
        sys.exit(1)
    
    glb_file = sys.argv[1]
    
    if not os.path.exists(glb_file):
        print(f"Error: the file {glb_file} does not exist.")
        sys.exit(1)
    
    # Create calibrator instance
    calibrator = CameraCalibrator(glb_file)
    
    # Start interactive calibration
    calibrator.interactive_calibrate()

if __name__ == "__main__":
    main()