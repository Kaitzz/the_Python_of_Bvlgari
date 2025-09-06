#!/usr/bin/env python
"""
GitHub Repository Setup Helper
"""

import os
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

class GitHubSetup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.required_files = {
            '.gitignore': self.project_root / '.gitignore',
            'LICENSE': self.project_root / 'LICENSE',
            'requirements.txt': self.project_root / 'requirements.txt'
        }
        
    def run_command(self, cmd):
        """Run shell command and return output"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except Exception as e:
            return False, str(e)
    
    def check_git_installed(self):
        """Check if git is installed"""
        success, _ = self.run_command("git --version")
        return success
    
    def check_existing_repo(self):
        """Check if already a git repository"""
        return (self.project_root / '.git').exists()
    
    def create_assets_folder(self):
        """Create assets folder with sample images"""
        assets_dir = self.project_root / 'assets'
        assets_dir.mkdir(exist_ok=True)
        
        # Copy sample images if they exist
        sample_images = [
            'perfect_front_view.jpg',
            'angle_test_grid.jpg',
            'parameter_test.png',
            'calibration_validation.jpg'
        ]
        
        copied = 0
        for img in sample_images:
            if Path(img).exists():
                shutil.copy2(img, assets_dir / img)
                copied += 1
        
        if copied > 0:
            print(f"✓ Copied {copied} sample images to assets/")
        
        # Create a placeholder if no images
        if copied == 0:
            readme_path = assets_dir / 'README.md'
            with open(readme_path, 'w') as f:
                f.write("# Assets\nSample images will be added here.\n")
    
    def check_file_sizes(self):
        """Check for files that might be too large for GitHub"""
        large_files = []
        for file_path in Path('.').rglob('*'):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > 50:  # Warn for files > 50MB
                    large_files.append((file_path, size_mb))
        
        if large_files:
            print("\n!Large files detected (>50MB):")
            for path, size in large_files:
                print(f"   {path}: {size:.1f}MB")
            print("   Consider adding these to .gitignore or using Git LFS")
    
    def setup_repository(self):
        """Main setup process"""
        print("\n" + "GitHub Repository Setup Helper")
        
        # Check git
        if not self.check_git_installed():
            print(" Git is not installed. Download from: https://git-scm.com/downloads")
            return
        
        # Check existing repo
        if self.check_existing_repo():
            print("⚠️  This is already a Git repository")
            response = input("Do you want to continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Create assets folder
        self.create_assets_folder()
        
        # Check file sizes
        self.check_file_sizes()
        
        # Initialize git if needed
        if not self.check_existing_repo():
            print("\nInitializing Git repository...")
            success, output = self.run_command("git init")
            if success:
                print("✓ Git repository initialized")
            else:
                print(f"! Failed to initialize: {output}")
                return
        
        # Add files
        print("\nAdding files to Git...")
        success, output = self.run_command("git add .")
        if success:
            print("✓ Files added to staging")
        
        # Show status
        print("\nGit Status:")
        success, output = self.run_command("git status --short")
        if success:
            print(output)
        
        print("✅ Project is ready for GitHub!")
        print("\nNext steps:")
        print("1. Review the files that will be committed")
        print("2. Run: git commit -m \"Initial commit\"")
        print("3. Create a new repository on GitHub.com")
        print("4. Follow GitHub's instructions to push your code")
        
        # Offer to create initial commit
        print("\nCreate the initial commit now?")
        response = input("(y/n): ")
        if response.lower() == 'y':
            commit_msg = '''
            Initial commit: Bulgari pendant authentication system

- 3D model rendering with calibrated camera parameters
- Deep learning architecture for authentication
- Complete data generation pipeline
- Workflow automation tools
- Camera calibration completed (elevation: 5°, center offset: -1.0)'''
            
            success, output = self.run_command(f'git commit -m "{commit_msg}"')
            if success:
                print("✓ Initial commit created!")
                print("\nNow: Follow the instructions to push an existing repository")
            else:
                print(f"❌ Commit failed: {output}")

if __name__ == "__main__":
    setup = GitHubSetup()
    setup.setup_repository()