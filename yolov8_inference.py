#!/usr/bin/env python3
"""
YOLO -> NCNN Full Pipeline
Primary:   .pt -> TorchScript -> pnnx -> NCNN
Fallback:  .pt -> ONNX -> onnx2ncnn -> NCNN

Usage:
    python export_pipeline.py
    python export_pipeline.py --model best.pt --data kiwi_disease.yaml
    python export_pipeline.py --model yolov8n.pt --use-onnx-fallback
    python export_pipeline.py --auto-install-system
"""

import subprocess
import sys
import os
import time
import re
import argparse
import shutil
from pathlib import Path

# ---- Deferred import: ensure ultralytics is installed ----
try:
    from ultralytics import YOLO
    from ultralytics.utils.benchmarks import benchmark
except ImportError:
    print("❌ Fatal error: ultralytics not installed.")
    print("   Please run: pip install ultralytics")
    sys.exit(1)

# ---- Default configuration ----
MODEL = "yolov8n.pt"
DATA  = "coco8.yaml"
IMGSZ = 640
PYPIMIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
NCNN = os.path.expanduser("~/ML/ncnn")

REQUIRED_PACKAGES = [
    "onnx>=1.12.0,<2.0.0",
    "onnxruntime",
    "onnxslim>=0.1.82",
    "tf_keras<=2.19.0",
    "sng4onnx>=1.0.1",
    "onnx_graphsurgeon>=0.3.26",
]

OPTIONAL_PACKAGES = [
    "tensorflow>=2.0.0,<=2.19.0",
    "tensorflowjs",
    "openvino>=2024.0.0",
    "coremltools>=9.0"
]


# ==================== Dependency Management ====================

def run_pip_command_with_progress(cmd_list):
    """Install a pip package and show real-time progress bar."""
    full_cmd = [sys.executable, "-m", "pip", "install"] + cmd_list + ["-i", PYPIMIRROR]
    pkg_display_name = cmd_list[-1].split(">=")[0].split("<")[0].split(",")[0]
    
    print(f"  ⏳ Installing {pkg_display_name} ...")
    
    try:
        process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        progress_pattern = re.compile(r'\s+[━╸╺]+\s+\d+')
        
        for line in process.stdout:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            if progress_pattern.search(line_stripped) or "MB" in line_stripped or "kB" in line_stripped:
                sys.stdout.write(f"\r  {line_stripped[:100]}")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\r{' ' * 100}\r")
                if line_stripped.startswith(("Requirement", "Collecting", "Downloading", "Installing", "Successfully")):
                    print(f"  {line_stripped}")
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\n  ❌ Installation failed for {pkg_display_name}")
            return False
        else:
            print(f"  ✅ {pkg_display_name} installed successfully")
            return True
            
    except Exception as e:
        print(f"\n  ❌ Installation error for {pkg_display_name}: {e}")
        return False


def auto_install_dependencies(package_spec_list, optional=False):
    """Install dependencies one by one. If optional=True, skip on failure silently."""
    label = "optional" if optional else "core"
    print("=" * 50)
    print(f"📦 Checking {label} Python dependencies...")
    print("=" * 50)
    
    failed_packages = []

    for spec in package_spec_list:
        pkg_name = spec.split(">=")[0].split("<")[0].split(",")[0]
        try:
            __import__(pkg_name)
            print(f"  ✅ {pkg_name} already installed")
        except ImportError:
            print(f"  ⚠️ {pkg_name} missing")
            if not run_pip_command_with_progress([spec]):
                if optional:
                    print(f"  ⏭️  Skipping optional package: {pkg_name}")
                else:
                    failed_packages.append(spec)
    
    print("=" * 50)
    
    if failed_packages:
        print(f"⚠️  {len(failed_packages)} required package(s) failed to install:")
        for pkg in failed_packages:
            print(f"    • pip install {pkg}")
        return False
    else:
        print(f"✅ All {label} dependencies ready!\n")
        return True


# ==================== System Dependency Check ====================

def check_system_dependencies(auto_install=False):
    """Check for cmake, protobuf, and git. Optionally auto-install."""
    print("=" * 50)
    print("🔧 Checking system dependencies...")
    print("=" * 50)
    
    missing = []
    
    for tool in ["cmake", "git", "protoc"]:
        if shutil.which(tool) is None:
            missing.append(tool)
            print(f"  ❌ {tool} not found")
        else:
            print(f"  ✅ {tool} found")
    
    if not missing:
        print("=" * 50)
        print("✅ All system dependencies ready!\n")
        return True
    
    pkg_map = {
        "cmake": "cmake",
        "git": "git",
        "protoc": "libprotobuf-dev protobuf-compiler",
    }
    missing_pkgs = []
    for tool in missing:
        missing_pkgs.extend(pkg_map[tool].split())
    missing_pkgs = list(set(missing_pkgs))
    missing_pkgs.append("build-essential")
    
    install_cmd = f"sudo apt-get install -y {' '.join(missing_pkgs)}"
    
    if auto_install:
        print(f"\n📥 Auto-installing missing packages ...")
        print(f"   Command: {install_cmd}")
        
        if os.geteuid() != 0:
            print("   ⚠️ Sudo password may be required.")
        
        result = subprocess.run(
            install_cmd.split(),
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"\n❌ Auto-install failed. Please run manually:")
            print(f"    {install_cmd}")
            return False
        
        for tool in missing:
            if shutil.which(tool) is None:
                print(f"  ❌ {tool} still missing after install")
                return False
            else:
                print(f"  ✅ {tool} now available")
        
        print("=" * 50)
        print("✅ All system dependencies ready!\n")
        return True
    else:
        print(f"\n⚠️  Missing system packages. Install with:")
        print(f"    {install_cmd}")
        print("   Or run this script with --auto-install-system to install automatically.")
        return False


# ==================== NCNN Build ====================

def build_ncnn_tools():
    """Clone and compile NCNN including pnnx, onnx2ncnn, and ncnnoptimize.
    Returns: (pnnx_ok, onnx_ok, opt_ok)
    """
    print("=" * 50)
    print("🔨 Building NCNN tools (pnnx + onnx2ncnn + ncnnoptimize)...")
    print("=" * 50)
    
    os.makedirs(os.path.dirname(NCNN), exist_ok=True)
    ncnn_path = Path(NCNN)
    
    if not ncnn_path.exists():
        print(f"  📥 Cloning NCNN to {NCNN} ...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/Tencent/ncnn.git", NCNN],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ❌ Failed to clone NCNN: {result.stderr}")
            return False, False, False
        print("  ✅ Clone complete")
        
        subprocess.run(
            ["git", "submodule", "update", "--init"],
            cwd=NCNN, capture_output=True, text=True
        )
    else:
        print(f"  ✅ NCNN source already exists at {NCNN}")
    
    build_dir = ncnn_path / "build"
    build_dir.mkdir(exist_ok=True)
    
    # cmake
    print("  🔧 Running cmake ...")
    cmake_result = subprocess.run(
        ["cmake", "-DNCNN_BUILD_TOOLS=ON", "-DNCNN_BUILD_EXAMPLES=OFF", ".."],
        cwd=str(build_dir), capture_output=True, text=True
    )
    if cmake_result.returncode != 0:
        print(f"  ❌ cmake failed: {cmake_result.stderr[-500:]}")
        return False, False, False
    print("  ✅ cmake complete")
    
    # Build each tool independently
    tools = {
        "pnnx": "pnnx",
        "onnx2ncnn": "onnx2ncnn",
        "ncnnoptimize": "ncnnoptimize",
    }
    results = {}
    
    for tool_name, make_target in tools.items():
        print(f"  🔨 Compiling {tool_name} ...")
        make_result = subprocess.run(
            ["make", make_target, "-j$(nproc)"],
            cwd=str(build_dir), capture_output=True, text=True, shell=True
        )
        if make_result.returncode != 0:
            print(f"  ⚠️ {tool_name} compilation failed: {make_result.stderr[-200:]}")
            results[tool_name] = False
        else:
            print(f"  ✅ {tool_name} compiled")
            results[tool_name] = True
    
    pnnx_ok = results.get("pnnx", False)
    onnx_ok = results.get("onnx2ncnn", False)
    opt_ok = results.get("ncnnoptimize", False)
    
    print("=" * 50)
    if pnnx_ok:
        print("✅ NCNN tools ready (pnnx path available)!")
    elif onnx_ok:
        print("⚠️  pnnx not available, will use ONNX fallback path.")
    else:
        print("❌ No conversion tool available. Check build errors above.")
    print(f"   pnnx: {'✅' if pnnx_ok else '❌'}  "
          f"onnx2ncnn: {'✅' if onnx_ok else '❌'}  "
          f"ncnnoptimize: {'✅' if opt_ok else '❌'}")
    print("=" * 50 + "\n")
    
    return pnnx_ok, onnx_ok, opt_ok


# ==================== Path 1: pnnx (Recommended) ====================

def pt_to_torchscript(model_path, output_dir):
    """Export .pt model to TorchScript for pnnx."""
    print("📤 Exporting to TorchScript ...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ts_path = output_dir / f"{Path(model_path).stem}.torchscript"
    
    try:
        model = YOLO(model_path)
        export_result = model.export(format="torchscript", imgsz=IMGSZ)
        actual_ts = Path(str(export_result))
        
        if actual_ts != ts_path:
            shutil.copy(actual_ts, ts_path)
        
        # Verify
        if ts_path.exists() and ts_path.stat().st_size > 1024:
            print(f"  ✅ TorchScript saved: {ts_path} ({ts_path.stat().st_size / 1024:.0f} KB)")
            return str(ts_path)
        else:
            print(f"  ❌ TorchScript file is missing or too small")
            return None
    except Exception as e:
        print(f"  ❌ TorchScript export failed: {e}")
        return None


def pnnx_convert(ts_path, output_dir):
    """Convert TorchScript to NCNN using pnnx."""
    print("\n🔄 Running pnnx: TorchScript → NCNN ...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 多路径查找 pnnx
    pnnx_candidates = [
        Path(NCNN) / "tools" / "pnnx" / "build" / "pnnx",
        Path(NCNN) / "build" / "tools" / "pnnx" / "pnnx",
    ]
    #pnnx_bin = Path(NCNN) / "build" / "tools" / "pnnx" / "pnnx"

    pnnx_bin = None
    for candidate in pnnx_candidates:
        if candidate.exists():
            pnnx_bin = str(candidate)
            break

    if not pnnx_bin.exists():
        print(f"  ❌ pnnx not found at {pnnx_bin}")
        return None, None
    
    model_name = Path(ts_path).stem
    
    cmd = [
        str(pnnx_bin),
        ts_path,
        f"inputshape=1,3,{IMGSZ},{IMGSZ}",
    ]
    
    result = subprocess.run(
        cmd,
        cwd=str(output_dir),
        capture_output=True,
        text=True
    )
    
    # Print pnnx output (usually useful info)
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            print(f"  {line.strip()}")
    
    if result.returncode != 0:
        print(f"  ❌ pnnx failed: {result.stderr}")
        return None, None
    
    # pnnx generates files with various naming patterns
    param_candidates = (
        list(output_dir.glob("*.ncnn.param")) +
        list(output_dir.glob(f"{model_name}.ncnn.param")) +
        list(output_dir.glob(f"{model_name}.param")) +
        list(output_dir.glob("*.param"))
    )
    bin_candidates = (
        list(output_dir.glob("*.ncnn.bin")) +
        list(output_dir.glob(f"{model_name}.ncnn.bin")) +
        list(output_dir.glob(f"{model_name}.bin")) +
        list(output_dir.glob("*.bin"))
    )
    
    # Deduplicate
    param_files = list(set(param_candidates))
    bin_files = list(set(bin_candidates))
    
    if param_files and bin_files:
        param_path = str(param_files[0])
        bin_path = str(bin_files[0])
        print(f"  ✅ pnnx conversion complete:")
        print(f"     • {param_path}")
        print(f"     • {bin_path}")
        return param_path, bin_path
    else:
        print(f"  ❌ pnnx output files not found in {output_dir}")
        print(f"     Contents: {list(output_dir.iterdir())}")
        return None, None


# ==================== Path 2: ONNX → onnx2ncnn (Fallback) ====================

def simplify_onnx(onnx_path, simplified_path):
    """Simplify ONNX model for NCNN compatibility."""
    print("🪄 Simplifying ONNX model ...")
    try:
        import onnx
        model = onnx.load(onnx_path)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, simplified_path)
        print(f"  ✅ Simplified ONNX saved: {simplified_path}")
        return True
    except Exception as e:
        print(f"  ⚠️ ONNX simplify failed: {e}")
        print("  → Using original ONNX file instead")
        return False


def onnx_to_ncnn(onnx_path, output_dir, opt_ok=True):
    """Convert ONNX model to NCNN using onnx2ncnn."""
    print("\n🔄 Running onnx2ncnn: ONNX → NCNN (fallback path) ...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = Path(onnx_path).stem
    param_path = output_dir / f"{model_name}.param"
    bin_path = output_dir / f"{model_name}.bin"
    
    onnx2ncnn_bin = Path(NCNN) / "build" / "tools" / "onnx" / "onnx2ncnn"
    
    if not onnx2ncnn_bin.exists():
        print(f"  ❌ onnx2ncnn not found at {onnx2ncnn_bin}")
        return None, None
    
    # Step 1: Simplify ONNX
    simplified_onnx = output_dir / f"{model_name}-sim.onnx"
    if not simplify_onnx(onnx_path, str(simplified_onnx)):
        simplified_onnx = onnx_path
    
    # Step 2: Convert
    print(f"  🔄 Running onnx2ncnn ...")
    result = subprocess.run(
        [str(onnx2ncnn_bin), str(simplified_onnx), str(param_path), str(bin_path)],
        capture_output=True, text=True
    )
    
    if result.returncode != 0 or not param_path.exists():
        print(f"  ❌ Conversion failed:")
        print(f"  {result.stdout}")
        print(f"  {result.stderr}")
        return None, None
    
    param_size = param_path.stat().st_size
    bin_size = bin_path.stat().st_size
    print(f"  ✅ NCNN model generated:")
    print(f"     • {param_path} ({param_size / 1024:.1f} KB)")
    print(f"     • {bin_path} ({bin_size / 1024:.1f} KB)")
    
    # Step 3: Optimize
    if opt_ok:
        ncnnoptimize_bin = Path(NCNN) / "build" / "tools" / "ncnnoptimize"
        opt_param = output_dir / f"{model_name}-opt.param"
        opt_bin = output_dir / f"{model_name}-opt.bin"
        
        if ncnnoptimize_bin.exists():
            print(f"  ⚡ Running ncnnoptimize ...")
            opt_result = subprocess.run(
                [str(ncnnoptimize_bin), str(param_path), str(bin_path),
                 str(opt_param), str(opt_bin), "0"],
                capture_output=True, text=True
            )
            if opt_result.returncode == 0:
                opt_param_size = opt_param.stat().st_size
                opt_bin_size = opt_bin.stat().st_size
                print(f"  ✅ Optimized NCNN model:")
                print(f"     • {opt_param} ({opt_param_size / 1024:.1f} KB)")
                print(f"     • {opt_bin} ({opt_bin_size / 1024:.1f} KB)")
                param_path, bin_path = opt_param, opt_bin
            else:
                print(f"  ⚠️ Optimization failed, using unoptimized model")
    
    return str(param_path), str(bin_path)


# ==================== NCNN Optimize (standalone) ====================

def optimize_ncnn(param_path, bin_path, output_dir):
    """Run ncnnoptimize on existing NCNN model."""
    print("⚡ Optimizing NCNN model ...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = Path(param_path).stem
    # Remove suffixes to get clean name
    for suffix in [".ncnn", "-opt", ".param"]:
        model_name = model_name.replace(suffix, "")
    
    opt_param = output_dir / f"{model_name}-opt.param"
    opt_bin = output_dir / f"{model_name}-opt.bin"
    
    ncnnoptimize_bin = Path(NCNN) / "build" / "tools" / "ncnnoptimize"
    
    if not ncnnoptimize_bin.exists():
        print("  ⚠️ ncnnoptimize not found, skipping")
        return param_path, bin_path
    
    result = subprocess.run(
        [str(ncnnoptimize_bin), param_path, bin_path, str(opt_param), str(opt_bin), "0"],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print(f"  ✅ Optimized model: {opt_param}, {opt_bin}")
        return str(opt_param), str(opt_bin)
    else:
        print(f"  ⚠️ Optimization failed, using original")
        return param_path, bin_path


# ==================== Utility ====================

def verify_file(path, label=""):
    """Check if a file exists and has reasonable size."""
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        status = "⚠️ (small)" if size_kb < 1 else f"✅ ({size_kb:.1f} KB)"
        print(f"  {status} {label}: {path}")
        return True
    else:
        print(f"  ❌ Not found: {path}")
        return False


# ==================== Main Pipeline ====================

def main():
    parser = argparse.ArgumentParser(description="YOLO → NCNN Full Pipeline")
    parser.add_argument("--model", type=str, default=MODEL, help="Path to .pt model")
    parser.add_argument("--data", type=str, default=DATA, help="Path to dataset YAML")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="Input image size")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark step")
    parser.add_argument("--skip-ncnn", action="store_true", help="Skip NCNN conversion entirely")
    parser.add_argument("--use-onnx-fallback", action="store_true",
                        help="Force ONNX→onnx2ncnn path instead of pnnx")
    parser.add_argument("--auto-install-system", action="store_true",
                        help="Auto-install missing system packages (requires sudo)")
    parser.add_argument("--install-optional", action="store_true",
                        help="Also install optional large packages (TensorFlow, OpenVINO, CoreML)")
    args = parser.parse_args()
    
    start_time = time.time()
    conversion_path_used = "skipped"
    
    # ---- Step 1: Install Python dependencies ----
    print("\n" + "=" * 60)
    print("STEP 1: Python Dependencies")
    print("=" * 60)
    
    if not auto_install_dependencies(REQUIRED_PACKAGES, optional=False):
        print("❌ Core dependencies missing. Please install them manually and re-run.")
        sys.exit(1)
    
    if args.install_optional:
        auto_install_dependencies(OPTIONAL_PACKAGES, optional=True)
    
    # ---- Step 2: Check system dependencies ----
    if not args.skip_ncnn:
        print("=" * 60)
        print("STEP 2: System Dependencies")
        print("=" * 60)
        
        if not check_system_dependencies(auto_install=args.auto_install_system):
            print("⚠️ System dependencies missing, NCNN conversion will be skipped.")
            print("   Re-run with --auto-install-system to install automatically.\n")
            args.skip_ncnn = True
    
    # ---- Step 3: Load model ----
    print("=" * 60)
    print("STEP 3: Load Model")
    print("=" * 60)
    
    if not os.path.exists(args.model):
        model_name = Path(args.model).name
        is_official = bool(re.match(r'yolo(v|w)\d+[nsmlx]\.pt$', model_name))
        
        if is_official:
            print(f"📥 Model '{model_name}' not found locally, downloading from Ultralytics...")
            try:
                _ = YOLO(model_name)
                print(f"  ✅ Downloaded: {model_name}")
            except Exception as e:
                print(f"  ❌ Failed to download {model_name}: {e}")
                print("  Please check your network or download manually.")
                sys.exit(1)
        else:
            print(f"❌ Model file not found: {args.model}")
            print("   If this is a custom model, check the path.")
            print("   If this is an official YOLO model, ensure the filename is correct (e.g. yolov8n.pt).")
            sys.exit(1)
    
    print(f"📥 Loading: {args.model}")
    model = YOLO(args.model)
    
    # ---- Step 4: Validation ----
    print("\n" + "=" * 60)
    print("STEP 4: Validation")
    print("=" * 60)
    
    print("🧪 Running validation...")
    model.val(data=args.data, imgsz=args.imgsz)
    
    # ---- Step 5: Benchmark (optional) ----
    if not args.skip_benchmark:
        print("\n" + "=" * 60)
        print("STEP 5: Benchmark")
        print("=" * 60)
        
        print("⏱️ Running benchmark...")
        benchmark(model, data=args.data, imgsz=args.imgsz, half=False, device=None)
    
    # ---- Step 6: NCNN Conversion ----
    print("\n" + "=" * 60)
    print("STEP 6: NCNN Conversion")
    print("=" * 60)
    
    if args.skip_ncnn:
        print("⏭️  Skipping NCNN conversion (--skip-ncnn).")
    else:
        pnnx_ok, onnx_ok, opt_ok = build_ncnn_tools()
        
        if not pnnx_ok and not onnx_ok:
            print("❌ No NCNN conversion tool available.")
            print("   Check build errors above. Try:")
            print("   sudo apt-get install -y cmake git libprotobuf-dev protobuf-compiler build-essential")
            args.skip_ncnn = True
        
        if not args.skip_ncnn:
            ncnn_output_dir = Path(args.model).parent / "ncnn_output"
            param_path, bin_path = None, None
            
            # ---- Try pnnx path first ----
            if pnnx_ok and not args.use_onnx_fallback:
                print("🛤️  Path: PRIMARY (.pt → TorchScript → pnnx → NCNN)")
                print()
                
                ts_path = pt_to_torchscript(args.model, str(ncnn_output_dir))
                if ts_path:
                    param_path, bin_path = pnnx_convert(ts_path, str(ncnn_output_dir))
                    
                    if param_path and bin_path and opt_ok:
                        param_path, bin_path = optimize_ncnn(param_path, bin_path, str(ncnn_output_dir))
                    
                    if param_path and bin_path:
                        conversion_path_used = "pnnx"
                    else:
                        print("⚠️  pnnx path failed, trying ONNX fallback...\n")
                        args.use_onnx_fallback = True
                else:
                    print("⚠️  TorchScript export failed, trying ONNX fallback...\n")
                    args.use_onnx_fallback = True
            
            # ---- Fallback: ONNX path ----
            if not pnnx_ok or args.use_onnx_fallback:
                if not onnx_ok:
                    print("❌ ONNX fallback path also unavailable.")
                else:
                    print("🛤️  Path: FALLBACK (.pt → ONNX → onnx2ncnn → NCNN)")
                    print()
                    
                    try:
                        print(f"🚀 Exporting to ONNX (imgsz={args.imgsz})...")
                        export_result = model.export(format="onnx", imgsz=args.imgsz)
                        onnx_path = str(export_result)
                        verify_file(onnx_path, "ONNX model")
                        
                        param_path, bin_path = onnx_to_ncnn(onnx_path, str(ncnn_output_dir), opt_ok)
                        if param_path and bin_path:
                            conversion_path_used = "onnx2ncnn"
                    except Exception as e:
                        print(f"❌ ONNX export error: {e}")
            
            # ---- Final output ----
            print("\n" + "=" * 60)
            if param_path and bin_path:
                print("📱 Deployment files ready for Android!")
                print("=" * 60)
                print(f"   Param: {param_path}")
                print(f"   Bin:   {bin_path}")
                print()
                print("   Next steps:")
                print("   1. Copy these files to your Android project's assets/ folder")
                print("   2. Add ncnn-android library to your project")
                print("   3. Load model in your Kotlin/Java code with NCNN API")
            else:
                print("⚠️ NCNN conversion could not be completed.")
                print("   The ONNX file can still be used with other frameworks:")
                print(f"   {ncnn_output_dir}")
            print("=" * 60)
    
    # ---- Summary ----
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"📊 Pipeline Summary")
    print(f"{'=' * 60}")
    print(f"   Total time:    {elapsed:.1f}s")
    print(f"   Model:         {args.model}")
    print(f"   Image size:    {args.imgsz}")
    print(f"   Validation:    ✅")
    print(f"   Benchmark:     {'✅' if not args.skip_benchmark else '⏭️ Skipped'}")
    print(f"   NCNN convert:  {conversion_path_used if conversion_path_used != 'skipped' else '⏭️ Skipped'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
