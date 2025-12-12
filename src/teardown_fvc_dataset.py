#!/usr/bin/env python3
"""
Teardown script for FVC dataset.
Removes folders created by setup_fvc_dataset.py (FVC1, FVC2, FVC3, Metadata).
Preserves data/ output folder and archive/ source files.
"""
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
VIDEOS_DIR = PROJECT_ROOT / "videos"
FOLDERS_TO_REMOVE = ["FVC1", "FVC2", "FVC3", "Metadata"]


def remove_folder(folder_path, folder_name):
    """Remove a folder if it exists"""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"  {folder_name}/ does not exist, skipping...")
        return False
    
    if not folder_path.is_dir():
        print(f"  Warning: {folder_name} exists but is not a directory, skipping...")
        return False
    
    try:
        print(f"  Removing {folder_name}/...")
        shutil.rmtree(folder_path)
        print(f"  ✓ Removed {folder_name}/")
        return True
    except PermissionError:
        print(f"  ✗ Error: Permission denied removing {folder_name}/")
        return False
    except Exception as e:
        print(f"  ✗ Error removing {folder_name}/: {e}")
        return False


def main():
    """Main teardown function"""
    print("=" * 70)
    print("FVC Dataset Teardown Script")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print()
    print("This will remove the following folders from videos/:")
    for folder in FOLDERS_TO_REMOVE:
        folder_path = VIDEOS_DIR / folder
        if folder_path.exists():
            # Calculate size
            try:
                total_size = sum(f.stat().st_size for f in folder_path.rglob('*') if f.is_file())
                size_mb = total_size / (1024 * 1024)
                print(f"  - {folder}/ ({size_mb:.1f} MB)")
            except (OSError, PermissionError, FileNotFoundError):
                print(f"  - {folder}/")
        else:
            print(f"  - {folder}/ (does not exist)")
    
    print()
    print("Note: This will NOT delete:")
    print("  - data/ folder (output manifests)")
    print("  - archive/ folder (source zip files)")
    print("  - videos/ folder itself (only contents)")
    print("  - lib/ package")
    print("  - src/ folder (setup/teardown scripts)")
    print()
    
    # Ask for confirmation
    response = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Teardown cancelled.")
        sys.exit(0)
    
    print()
    print("Removing folders...")
    print("-" * 70)
    
    removed = []
    for folder_name in FOLDERS_TO_REMOVE:
        folder_path = VIDEOS_DIR / folder_name
        if remove_folder(folder_path, folder_name):
            removed.append(folder_name)
    
    print("-" * 70)
    print()
    
    if removed:
        print(f"✓ Successfully removed {len(removed)} folder(s): {', '.join(removed)}")
    else:
        print("No folders were removed.")
    
    # Check what remains
    remaining = [f for f in FOLDERS_TO_REMOVE if (VIDEOS_DIR / f).exists()]
    if remaining:
        print(f"\n⚠ Warning: {len(remaining)} folder(s) still exist: {', '.join(remaining)}")
    
    print()
    print("=" * 70)
    print("✓ Teardown completed")
    print("=" * 70)
    print()
    print("To rebuild, run: python3 src/setup_fvc_dataset.py")


if __name__ == "__main__":
    main()

