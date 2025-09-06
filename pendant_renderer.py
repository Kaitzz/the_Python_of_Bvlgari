"""
This script renders a 3D model of a Bulgari pendant to generate training data.
"""

import numpy as np
import trimesh
import pyrender
import os
from PIL import Image

class PendantRenderer:
    """Renderer for Bulgari pendant 3D model"""
    
    def __init__(self, glb_path, camera_params=None):
        """
        Initialize the renderer with a GLB model and camera parameters.
        Args:
            glb_path: GLB File path
            camera_params:  {
                'distance': The distance of the camera from the model (controls zoom),
                'azimuth': The horizontal rotation angle,
                'elevation': The vertical rotation angle,
                'offset': [x, y, z] offset to adjust the look-at point,
                'fov': Field of view angle in degrees
            }
        """
        # Load the 3D model
        self.scene = trimesh.load(glb_path, force='scene')
        
        # Get model bounds and center
        self.bounds = self.scene.bounds
        self.original_center = self.bounds.mean(axis=0)
        self.scale = np.linalg.norm(self.bounds[1] - self.bounds[0])
        
        # Default camera parameters
        self.default_camera_params = {
            'distance': 3.0,
            'azimuth': 0,
            'elevation': 5,
            'offset': [0, -1, 0],
            'fov': 60  # FOV
        }
        
        # Override with provided parameters
        self.camera_params = self.default_camera_params.copy()
        if camera_params:
            self.camera_params.update(camera_params)
        
        # Set up the pyrender scene
        self.setup_renderer()
        
        print(f"Model loaded from: {glb_path}")
        print(f"The bounds: {self.bounds}")
        print(f"Center point: {self.original_center}")
        print(f"Scale: {self.scale}")
    
    def setup_renderer(self):
        """Set up the pyrender scene with lights and camera"""
        # Initialize pyrender scene
        self.render_scene = pyrender.Scene(ambient_light=[0.6, 0.6, 0.6])
        
        # Convert trimesh geometries to pyrender meshes and add to scene
        for name, mesh in self.scene.geometry.items(): # type: ignore
            if hasattr(mesh, 'vertices'):
                pymesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
                self.render_scene.add(pymesh)
        
        # Add lights
        self.add_lights()
        
        # Initialize offscreen renderer
        self.renderer = pyrender.OffscreenRenderer(800, 800)
    
    def add_lights(self):
        """Add multiple lights to the scene for more realistic rendering"""
        # The 3-point lighting setup
        key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        self.render_scene.add(key_light, pose=self.create_look_at_matrix(
            [1, 1, 1], [0, 0, 0]
        ))
        
        # Fill light
        fill_light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=1.0)
        self.render_scene.add(fill_light, pose=self.create_look_at_matrix(
            [-1, 0.5, 1], [0, 0, 0]
        ))
        
        # Rim light
        rim_light = pyrender.DirectionalLight(color=[0.6, 0.6, 0.7], intensity=0.6)
        self.render_scene.add(rim_light, pose=self.create_look_at_matrix(
            [0, -1, 0.5], [0, 0, 0]
        ))
    
    @staticmethod
    def create_look_at_matrix(eye, target, up=[0, 1, 0]):
        """Create a look-at view matrix."""
        eye = np.array(eye, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)
        
        z = eye - target
        if np.linalg.norm(z) > 0:
            z = z / np.linalg.norm(z)
        else:
            z = np.array([0, 0, 1])
            
        x = np.cross(up, z)
        if np.linalg.norm(x) > 0:
            x = x / np.linalg.norm(x)
        else:
            x = np.array([1, 0, 0])
        
        matrix = np.eye(4)
        matrix[:3, 0] = x
        matrix[:3, 1] = np.cross(z, x)
        matrix[:3, 2] = z
        matrix[:3, 3] = eye
        
        return matrix
    
    def get_camera_position(self, azimuth, elevation, distance, offset):
        """
        Get camera position in Cartesian coordinates from spherical coordinates.
        Args:
            distance, elevation, azimuth, offset
        Returns:
            camera position as a numpy array
        """
        # Convert degrees to radians
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)
        
        # Convert spherical to Cartesian coordinates
        x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        y = distance * np.sin(elevation_rad)
        z = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        
        camera_pos = np.array([x, y, z]) + self.original_center + np.array(offset)
        
        return camera_pos
    
    def render_view_with_params(self, params=None):
        """
        Render the model with specified camera parameters.
        Args:
            params: camera_params; if None, use self.camera_params
        Returns:
            Rendered image as a numpy array
        """
        if params is None:
            params = self.camera_params
        
        # Get parameters
        distance = params.get('distance', 3.0)
        azimuth = params.get('azimuth', 0)
        elevation = params.get('elevation', 5)
        offset = params.get('offset', [0, -1, 0])
        
        # Calculate camera position
        camera_pos = self.get_camera_position(azimuth, elevation, distance, offset)
        
        # Calculate look-at target
        look_at_target = self.original_center + np.array(offset)
        
        # Create camera
        camera_pose = self.create_look_at_matrix(camera_pos, look_at_target)
        yfov = np.radians(params.get('fov', 60))
        camera = pyrender.PerspectiveCamera(yfov=yfov)
        cam_node = self.render_scene.add(camera, pose=camera_pose)
        
        # Render the scene
        color, depth = self.renderer.render(self.render_scene) # type: ignore
        
        # Remove camera
        self.render_scene.remove_node(cam_node)
        
        return color
    

# 使用示例
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        sys.exit(1)
    
    glb_file = sys.argv[1]
    
    # Initialize renderer
    print("\nInitializing renderer...")
    renderer = PendantRenderer(glb_file)
    
    # Render and save test views
    print("\nGenerating test views...")
    os.makedirs("./test_views", exist_ok=True)
    
    # Some test configurations
    test_configs = [
        ("default", {}),
        ("zoomed_out", {'distance': 4}),
        ("zoomed_in", {'distance': 2.5}),
        ("rotated_left", {'azimuth': -10}),
        ("rotated_right", {'azimuth': 10}),
        ("from_above", {'elevation': 10}),
    ]
    
    for name, params_update in test_configs:
        params = renderer.camera_params.copy()
        params.update(params_update)
        image = renderer.render_view_with_params(params)
        Image.fromarray(image).save(f"./test_views/{name}.png")
        print(f"  ✓ {name}.png")
    
    print("\n完成！请查看: test_views/ (各种视角测试)")