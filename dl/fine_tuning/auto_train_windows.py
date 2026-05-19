import os
import shutil
from pathlib import Path
import yaml
import zipfile
from roboflow import Roboflow
from ultralytics import YOLO

# =====================================================================
# 🛠️ USER CONFIGURATION
# =====================================================================
# Download your datasets from Roboflow Universe as ZIP files (YOLOv8 format).
# Extract them all into the 'dataset_downloads' folder next to this script.

# Unified ATCS Class Map. All foreign classes will be mapped to these IDs.
# If a class is not in this map, the bounding box will be discarded.
UNIVERSAL_CLASS_MAP = {
    0: "car",
    1: "bus",
    2: "truck",
    3: "motorcycle",
    4: "auto_rickshaw",
    5: "ambulance",
    6: "fire_truck"
}


# Directories
BASE_DIR = Path(__file__).parent.resolve()
DOWNLOAD_DIR = BASE_DIR / "dataset_downloads"
MERGED_DIR = BASE_DIR / "merged_universe"
RUNS_DIR = BASE_DIR / "runs"

# =====================================================================
# 2. MERGE & REMAP CLASSES
# =====================================================================
def create_class_mapping(foreign_classes):
    """Maps the foreign dataset's classes to our universal ATCS IDs."""
    mapping = {}
    
    # Reverse lookup for universal map
    universal_name_to_id = {v: k for k, v in UNIVERSAL_CLASS_MAP.items()}
    
    for foreign_id, foreign_name in enumerate(foreign_classes):
        name = foreign_name.lower().replace("-", "_").replace(" ", "_")
        
        # Heuristics to catch common variations
        if "car" in name or "vehicle" in name: target = "car"
        elif "bus" in name: target = "bus"
        elif "truck" in name or "lorry" in name: target = "truck"
        elif "bike" in name or "motorcycle" in name or "two-wheeler" in name: target = "motorcycle"
        elif "auto" in name or "rickshaw" in name: target = "auto_rickshaw"
        elif "ambulance" in name: target = "ambulance"
        elif "fire" in name: target = "fire_truck"
        else:
            continue # Unknown class, discard
            
        if target in universal_name_to_id:
            mapping[foreign_id] = universal_name_to_id[target]
            
    return mapping

def merge_datasets():
    print("\n[PHASE 2] Merging Datasets & Remapping Classes...")
    
    # Setup merged directory structure
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    
    for split in ["train", "valid", "test"]:
        (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
        
    global_img_counter = 0
    total_boxes = 0
    kept_boxes = 0

    if not DOWNLOAD_DIR.exists() or not any(DOWNLOAD_DIR.iterdir()):
        print(f"[ERROR] No datasets found in {DOWNLOAD_DIR}!")
        print("Please download the ZIP files from Roboflow and extract them into that folder.")
        exit(1)

    # 1. Auto-Extract any ZIP files found
    for file in DOWNLOAD_DIR.iterdir():
        if file.suffix.lower() == ".zip":
            extract_path = DOWNLOAD_DIR / file.stem
            if not extract_path.exists():
                print(f"\n[PHASE 1] Auto-extracting {file.name}...")
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            # We don't delete the zip just in case the user wants to keep it

    # 2. Process all dataset folders
    for dataset_path in DOWNLOAD_DIR.iterdir():
        if not dataset_path.is_dir(): continue
        
        project_name = dataset_path.name
        yaml_file = dataset_path / "data.yaml"
        
        if not yaml_file.exists():
            continue
            
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
            
        foreign_classes = data.get("names", [])
        if isinstance(foreign_classes, dict):
            foreign_classes = [foreign_classes[i] for i in range(len(foreign_classes))]
            
        mapping = create_class_mapping(foreign_classes)
        print(f"  [>] Processing {project_name} | Class Map: {mapping}")
        
        for split in ["train", "valid", "test"]:
            img_dir = dataset_path / split / "images"
            lbl_dir = dataset_path / split / "labels"
            
            if not img_dir.exists() or not lbl_dir.exists():
                continue
                
            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]: continue
                
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if not lbl_file.exists(): continue
                
                # Parse and remap labels
                new_labels = []
                with open(lbl_file, "r") as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        
                        foreign_id = int(parts[0])
                        total_boxes += 1
                        
                        if foreign_id in mapping:
                            new_id = mapping[foreign_id]
                            new_labels.append(f"{new_id} {' '.join(parts[1:])}\n")
                            kept_boxes += 1
                
                # Only copy image if it has valid mapped bounding boxes
                if new_labels:
                    new_img_name = f"merged_{global_img_counter}{img_file.suffix}"
                    new_lbl_name = f"merged_{global_img_counter}.txt"
                    
                    shutil.copy(img_file, MERGED_DIR / split / "images" / new_img_name)
                    with open(MERGED_DIR / split / "labels" / new_lbl_name, "w") as f:
                        f.writelines(new_labels)
                        
                    global_img_counter += 1

    # Write unified data.yaml
    yaml_content = {
        "train": str(MERGED_DIR / "train" / "images"),
        "val": str(MERGED_DIR / "valid" / "images"),
        "test": str(MERGED_DIR / "test" / "images"),
        "nc": len(UNIVERSAL_CLASS_MAP),
        "names": [UNIVERSAL_CLASS_MAP[i] for i in range(len(UNIVERSAL_CLASS_MAP))]
    }
    
    with open(MERGED_DIR / "data.yaml", "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"  [✓] Merged {global_img_counter} images.")
    print(f"  [✓] Kept {kept_boxes}/{total_boxes} relevant ATCS bounding boxes.")

# =====================================================================
# 3. TRAIN YOLOv8
# =====================================================================
def train_model():
    print("\n[PHASE 3] Initiating YOLOv8 Training (GPU)...")
    
    model = YOLO("yolov8n.pt") # Automatically downloads the base model
    
    # Using device=0 assumes a Windows machine with an Nvidia GPU.
    model.train(
        data=str(MERGED_DIR / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,       
        patience=10,    # early stopping
        augment=True,   
        device=0,       # Windows CUDA GPU (change to 'cpu' if no GPU)
        cache="disk",   
        workers=8,      
        project=str(RUNS_DIR),
        name="windows_atcs_fine_tuned",
        exist_ok=True,
    )
    
    return RUNS_DIR / "windows_atcs_fine_tuned" / "weights" / "best.pt"

# =====================================================================
# 4. EXPORT ONNX
# =====================================================================
def export_onnx(best_pt_path):
    print(f"\n[PHASE 4] Exporting Final Model to ONNX...")
    if not best_pt_path.exists():
        print(f"  [X] Could not find {best_pt_path}")
        return
        
    model = YOLO(best_pt_path)
    onnx_path = model.export(format="onnx")
    
    # Auto-rename and move to the main dl/ folder so it's ready to use
    final_onnx = BASE_DIR.parent / "yolov8n_traffic_v2.onnx"
    shutil.copy(onnx_path, final_onnx)
    
    print(f"\n=======================================================")
    print(f"🚀 SUCCESS! OVERNIGHT PIPELINE COMPLETE.")
    print(f"🎯 Final ONNX model generated at: {final_onnx}")
    print(f"Copy this 'yolov8n_traffic_v2.onnx' file back to your Mac for the Live Demo!")
    print(f"=======================================================")

# =====================================================================
# 5. AUTOMATED DISK CLEANUP
# =====================================================================
def cleanup_space():
    print(f"\n[PHASE 5] Deep Cleaning Disk Space...")
    
    # 1. Delete merged universe
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR, ignore_errors=True)
        print("  [✓] Deleted heavy merged dataset folder.")
        
    # 2. Delete raw downloads
    if DOWNLOAD_DIR.exists():
        shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
        print("  [✓] Deleted raw dataset ZIPs and extracted folders.")
        
    # 3. Delete PyTorch weights from ALL run folders (keep the graphs!)
    if RUNS_DIR.exists():
        for run_folder in RUNS_DIR.iterdir():
            if run_folder.is_dir():
                weights_dir = run_folder / "weights"
                if weights_dir.exists():
                    shutil.rmtree(weights_dir, ignore_errors=True)
                    print(f"  [✓] Deleted massive .pt weights from {run_folder.name}")

if __name__ == "__main__":
    merge_datasets()
    best_weights = train_model()
    export_onnx(best_weights)
    cleanup_space()
