#!/usr/bin/env python3
"""
Test script to verify model detection and system health
Run this on RunPod after downloading models to verify everything is working
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def check_models():
    """Check which models are available"""
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models"

    print("=" * 60)
    print("COMIFY MODEL DETECTION TEST")
    print("=" * 60)
    print()

    print(f"Models directory: {models_dir}")
    print(f"Models directory exists: {models_dir.exists()}")
    print()

    if not models_dir.exists():
        print("ERROR: Models directory does not exist!")
        print("Please run the model downloader first:")
        print("  python -c \"from installer.model_downloader import ModelDownloader; d = ModelDownloader('./models'); d.download_for_tryon()\"")
        return False

    # Check InsightFace
    print("Checking InsightFace (face detection)...")
    insightface_dir = models_dir / "insightface"
    if insightface_dir.exists():
        onnx_files = list(insightface_dir.glob("**/*.onnx"))
        if onnx_files:
            print(f"  FOUND: {len(onnx_files)} ONNX files")
            for f in onnx_files[:5]:  # Show first 5
                print(f"    - {f.relative_to(models_dir)}")
        else:
            print("  NOT FOUND: No .onnx files in insightface directory")
    else:
        print("  NOT FOUND: insightface directory does not exist")
    print()

    # Check DWPose
    print("Checking DWPose (body pose)...")
    dwpose_dir = models_dir / "dwpose"
    if dwpose_dir.exists():
        onnx_files = list(dwpose_dir.glob("*.onnx"))
        if onnx_files:
            print(f"  FOUND: {len(onnx_files)} ONNX files")
            for f in onnx_files:
                print(f"    - {f.name}")
        else:
            print("  NOT FOUND: No .onnx files in dwpose directory")
    else:
        print("  NOT FOUND: dwpose directory does not exist")
    print()

    # Check Checkpoints
    print("Checking Base Models (checkpoints)...")
    checkpoints_dir = models_dir / "checkpoints"
    if checkpoints_dir.exists():
        safetensor_files = list(checkpoints_dir.glob("*.safetensors"))
        ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
        all_models = safetensor_files + ckpt_files
        if all_models:
            print(f"  FOUND: {len(all_models)} model files")
            for f in all_models:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    - {f.name} ({size_mb:.1f} MB)")
        else:
            print("  NOT FOUND: No .safetensors or .ckpt files")
    else:
        print("  NOT FOUND: checkpoints directory does not exist")
    print()

    # Check VAE
    print("Checking VAE...")
    vae_dir = models_dir / "vae"
    if vae_dir.exists():
        vae_files = list(vae_dir.glob("*.safetensors")) + list(vae_dir.glob("*.ckpt"))
        if vae_files:
            print(f"  FOUND: {len(vae_files)} VAE files")
            for f in vae_files:
                print(f"    - {f.name}")
        else:
            print("  NOT FOUND: No VAE files")
    else:
        print("  NOT FOUND: vae directory does not exist")
    print()

    # List all directories in models
    print("All directories in models/:")
    if models_dir.exists():
        for item in sorted(models_dir.iterdir()):
            if item.is_dir():
                file_count = len(list(item.rglob("*")))
                print(f"  - {item.name}/ ({file_count} files)")
    print()

    # Now test the actual backend function
    print("=" * 60)
    print("TESTING BACKEND MODEL DETECTION")
    print("=" * 60)
    print()

    try:
        # Import and run the backend check
        from main import check_v2_models_available, get_available_models

        v2_status = check_v2_models_available()
        print("V2 Models Status:")
        for key, value in v2_status.items():
            print(f"  {key}: {value}")
        print()

        models = get_available_models()
        print(f"Available Models ({len(models)} total):")
        for model in models:
            print(f"  - {model['name']} ({model['type']}, source: {model['source']})")
        print()

        # Determine final status
        if v2_status["available"]:
            print("STATUS: HEALTHY - V2 engine is available!")
            print("The 'Generate Try-On' button should be ENABLED.")
            return True
        else:
            print("STATUS: DEGRADED - No models found")
            print("The 'Generate Try-On' button will be DISABLED.")
            print()
            print("To fix, download models:")
            print("  source venv/bin/activate")
            print("  python -c \"from installer.model_downloader import ModelDownloader; d = ModelDownloader('./models'); d.download_for_tryon()\"")
            return False

    except Exception as e:
        print(f"ERROR testing backend: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = check_models()
    print()
    print("=" * 60)
    print(f"RESULT: {'PASS' if success else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if success else 1)
