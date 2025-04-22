#!/usr/bin/env python
"""
Diagnostic script to create a visible output folder and test file permissions,
with special handling for Docker environments.
"""

import os
import datetime
import shutil
import subprocess
import sys

def create_test_folder():
    """Create a test folder with various permission levels to diagnose visibility issues."""
    # Check if running in Docker and set appropriate paths
    in_docker = os.path.exists('/.dockerenv')
    
    # Use both Docker and host paths to ensure visibility from both environments
    if in_docker:
        base_dir = "/workspace/results"
        host_base_dir = "/home/santhi/Documents/SurgLatentGraph/results"
    else:
        base_dir = "/home/santhi/Documents/SurgLatentGraph/results"
        host_base_dir = base_dir
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join(base_dir, f"visibility_test_{timestamp}")
    host_test_dir = os.path.join(host_base_dir, f"visibility_test_{timestamp}")
    
    print(f"Creating test directory: {test_dir}")
    os.makedirs(test_dir, exist_ok=True)
    
    # Create subfolders with different permission levels
    perm_levels = [
        ("0777_full_access", 0o777),
        ("0755_standard", 0o755), 
        ("0666_all_read_write", 0o666),
        ("0644_standard_file", 0o644)
    ]
    
    for folder_name, permission in perm_levels:
        folder_path = os.path.join(test_dir, folder_name)
        print(f"Creating {folder_path} with permission {oct(permission)}")
        os.makedirs(folder_path, exist_ok=True)
        os.chmod(folder_path, permission)
        
        # Create a test file in each folder
        test_file = os.path.join(folder_path, "test_file.txt")
        with open(test_file, 'w') as f:
            f.write(f"Test file created at {datetime.datetime.now()}\n")
            f.write(f"This folder has permissions: {oct(permission)}\n")
            f.write(f"Docker environment: {in_docker}\n")
        
        # Set the same permission on the file
        os.chmod(test_file, permission)
    
    # Create a special permissions file at the root
    root_test_file = os.path.join(test_dir, "READ_ME_FIRST.txt")
    with open(root_test_file, 'w') as f:
        f.write("===== VISIBILITY TEST =====\n\n")
        f.write(f"Created at: {datetime.datetime.now()}\n")
        f.write(f"Running inside Docker: {in_docker}\n")
        f.write(f"If you can see this file, basic file visibility is working.\n\n")
        f.write("Check if you can see the following folders and their contents:\n")
        for folder_name, permission in perm_levels:
            f.write(f"- {folder_name} (permission {oct(permission)})\n")
        
        f.write("\nSystem information:\n")
        f.write(f"User ID: {os.getuid()}\n")
        f.write(f"Group ID: {os.getgid()}\n")
        f.write(f"Current user: {os.environ.get('USER', 'unknown')}\n")
        f.write(f"Python version: {sys.version}\n")
        
        # Add Docker-specific information
        if in_docker:
            f.write("\nDocker environment information:\n")
            f.write(f"Docker path: {test_dir}\n")
            f.write(f"Host path: {host_test_dir}\n")
    
    # Make the root test file readable by all
    os.chmod(root_test_file, 0o666)
    
    # Set permission on the test directory itself
    os.chmod(test_dir, 0o777)
    
    # Return the paths and Docker flag
    return test_dir, root_test_file, in_docker, host_test_dir if in_docker else None

def print_diagnostic_info(test_dir, in_docker, host_test_dir=None):
    """Print diagnostic information about the test directory."""
    print("\n===== DIAGNOSTIC INFORMATION =====")
    print(f"Running inside Docker: {in_docker}")
    print(f"Test directory: {test_dir}")
    if host_test_dir:
        print(f"Host system path: {host_test_dir}")
    print(f"Owner UID: {os.stat(test_dir).st_uid}")
    print(f"Owner GID: {os.stat(test_dir).st_gid}")
    print(f"Current permissions: {oct(os.stat(test_dir).st_mode)}")
    
    # Try to get more detailed information
    try:
        ls_output = subprocess.check_output(['ls', '-la', test_dir], text=True)
        print("\nDirectory listing:")
        print(ls_output)
    except Exception as e:
        print(f"Could not run ls command: {e}")
    
    # Print recommended next steps based on environment
    print("\n===== RECOMMENDED NEXT STEPS =====")
    if in_docker:
        print("DOCKER-SPECIFIC INSTRUCTIONS:")
        print("1. Inside Docker container, files should be visible at:")
        print(f"   {test_dir}")
        print("2. From host system, files should be visible at:")
        print(f"   {host_test_dir}")
        print("3. If files are not visible in Docker, check:")
        print("   - Docker volume mounting in your docker-compose or docker run command")
        print("   - User permissions inside Docker container")
        print("4. Commands to run inside Docker:")
        print(f"   ls -la {test_dir}")
        print("5. To copy files out of Docker if needed:")
        print(f"   docker cp <container_id>:{test_dir} /host/destination/")
    else:
        print("HOST SYSTEM INSTRUCTIONS:")
        print("1. To run this script inside Docker:")
        print("   docker exec <container_id> python /workspace/first_trial.py")
        print("2. Check the test directory from your host system:")
        print(f"   ls -la {test_dir}")
        print("3. If you can't see the files, try:")
        print(f"   sudo chmod -R 777 {test_dir}")

if __name__ == "__main__":
    test_dir, readme_file, in_docker, host_test_dir = create_test_folder()
    print_diagnostic_info(test_dir, in_docker, host_test_dir)
    print(f"\nTest folder created at: {test_dir}")
    if host_test_dir:
        print(f"Host system path: {host_test_dir}")
    print(f"Please check if you can see the README file: {readme_file}")
    
    if in_docker:
        print("\nDocker-specific instructions:")
        print("- To make this file accessible to the host, ensure proper volume mounting")
        print("- Check the Docker run command or docker-compose.yml used to start this container")
        print("- Typical volume mount: -v /home/santhi/Documents/SurgLatentGraph:/workspace")