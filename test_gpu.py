import subprocess
import os
import sys

def check_cuda():
    print("🔍 Checking CUDA installation...")
    
    # التحقق من nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA Drivers installed")
            print(result.stdout.split('\n')[0])  # أول سطر فقط
        else:
            print("❌ NVIDIA Drivers not found")
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA drivers not installed")
    
    # التحقق من nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA Toolkit installed")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'release' in line.lower():
                    print(f"   {line.strip()}")
        else:
            print("❌ CUDA Toolkit not found")
    except FileNotFoundError:
        print("❌ nvcc not found - CUDA Toolkit not installed")
    
    # التحقق من مجلدات CUDA
    cuda_paths = [
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit",
        "C:\\CUDA"
    ]
    
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"✅ CUDA folder found: {path}")
            # عرض المجلدات الفرعية
            try:
                subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
                print(f"   Subfolders: {subfolders}")
            except:
                pass
            break
    else:
        print("❌ No CUDA folders found in standard locations")

# تشغيل الفحص
check_cuda()