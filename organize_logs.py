#!/usr/bin/env python3
"""
Organize existing log files into stage-specific directories.

Moves log files from logs/ to logs/stage{N}/ based on their naming patterns.
"""

import shutil
from pathlib import Path
import re

def organize_logs(project_root: Path):
    """Organize log files into stage-specific directories."""
    logs_dir = project_root / "logs"
    
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        return
    
    # Stage patterns and their target directories
    stage_patterns = {
        "stage1": {
            "dir": logs_dir / "stage1",
            "patterns": [
                r"^stage1_augmentation_\d+\.log$",
                r"^stage1[a-d]_augmentation_\d+\.log$",
                r"^stage1[a-d]_aug-\d+\.(out|err|log)$",
            ]
        },
        "stage2": {
            "dir": logs_dir / "stage2",
            "patterns": [
                r"^stage2_features_\d+\.log$",
                r"^stage2_feat-\d+.*\.(out|err|log)$",
                r"^stage2_features_combined.*\.log$",
            ]
        },
        "stage3": {
            "dir": logs_dir / "stage3",
            "patterns": [
                r"^stage3_scaling_\d+\.log$",
                r"^stage3[a-h]_scaling_\d+\.log$",
                r"^stage3[a-h]_scaling-\d+\.(out|err|log)$",
                r"^stage3_(down|coord)-\d+.*\.(out|err|log)$",
            ]
        },
        "stage4": {
            "dir": logs_dir / "stage4",
            "patterns": [
                r"^stage4_scaled_features_\d+\.log$",
                r"^stage4[a-c]_feat.*\.(out|err|log)$",
                r"^stage4_coord-.*\.(out|err|log)$",
            ]
        },
        "stage5": {
            "dir": logs_dir / "stage5",
            "patterns": [
                r"^stage5_training_\d+\.log$",
                r"^stage5[a-z]+_\d+\.log$",
                r"^stage5_coord-.*\.(out|err|log)$",
            ]
        },
        "validation": {
            "dir": logs_dir / "validation",
            "patterns": [
                r"^validate_stage5_imports_\d+\.log$",
            ]
        },
        "repairs": {
            "dir": logs_dir / "repairs",
            "patterns": [
                r"^repair_stage\d+.*\.(out|err|log)$",
            ]
        },
        "sanity_checks": {
            "dir": logs_dir / "sanity_checks",
            "patterns": [
                r"^sanity_check_.*\.log$",
            ]
        }
    }
    
    # Create stage directories
    for stage_info in stage_patterns.values():
        stage_info["dir"].mkdir(parents=True, exist_ok=True)
    
    # Track moved files
    moved_count = {stage: 0 for stage in stage_patterns.keys()}
    moved_count["other"] = 0
    
    # Process all files in logs directory (not subdirectories)
    for log_file in logs_dir.iterdir():
        if not log_file.is_file():
            continue
        
        file_name = log_file.name
        moved = False
        
        # Try to match each stage pattern
        for stage_name, stage_info in stage_patterns.items():
            if moved:
                break
            for pattern in stage_info["patterns"]:
                if re.match(pattern, file_name):
                    target = stage_info["dir"] / file_name
                    if target.exists():
                        # File already exists in target, skip
                        print(f"⚠ Skipping {file_name} (already exists in {stage_info['dir'].name}/)")
                        moved = True
                        break
                    try:
                        shutil.move(str(log_file), str(target))
                        print(f"✓ Moved {file_name} → {stage_info['dir'].name}/")
                        moved_count[stage_name] += 1
                        moved = True
                        break
                    except Exception as e:
                        print(f"✗ Error moving {file_name}: {e}")
                        moved = True  # Mark as handled even if failed
                        break
                if moved:
                    break
        
        if not moved:
            # Check if it's a special file
            if file_name == ".gitkeep" or file_name.startswith("."):
                # Keep hidden/system files in root
                print(f"⊘ Keeping {file_name} in root (system file)")
                moved_count["other"] += 1
            else:
                print(f"⚠ No pattern matched for {file_name}")
                moved_count["other"] += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Organization Summary:")
    print("=" * 60)
    for stage, count in moved_count.items():
        if count > 0:
            print(f"  {stage}: {count} file(s)")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1]).resolve()
    
    print("Organizing log files into stage-specific directories...")
    print(f"Project root: {project_root}")
    print()
    organize_logs(project_root)
    print("\n✓ Log organization complete!")

