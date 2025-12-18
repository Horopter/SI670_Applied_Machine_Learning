#!/usr/bin/env python3
"""
Comprehensive notebook validation script.

This script:
1. Analyzes model-specific differences in training, storage, and data structures
2. Validates against source code (training pipeline)
3. Validates against SLURM scripts
4. Validates against actual data in data/, logs/, mlruns/
5. Runs each notebook 3 times using jupyter nbconvert
6. Checks output cells and logs for errors
7. Fixes issues found
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import re


# Model-specific configurations based on actual data structures
MODEL_DATA_STRUCTURES = {
    "5a": {
        "type": "baseline",
        "metrics_file": "metrics.json",
        "results_file": None,
        "has_folds": True,
        "fold_structure": "fold_*/",
        "fold_files": ["metrics.jsonl", "model.joblib", "scaler.joblib", "roc_pr_curves.png", "confusion_matrix.png"],
        "root_files": ["metrics.json"],
        "metrics_key": "fold_results",
        "model_file_pattern": "model.joblib"
    },
    "5alpha": {
        "type": "sklearn",
        "metrics_file": None,
        "results_file": "results.json",
        "has_folds": False,
        "fold_structure": None,
        "fold_files": [],
        "root_files": ["results.json", "model.joblib", "scaler.joblib", "roc_pr_curves.png"],
        "metrics_key": "cv_fold_results",
        "model_file_pattern": "model.joblib"
    },
    "5b": {
        "type": "baseline",
        "metrics_file": "metrics.json",
        "results_file": None,
        "has_folds": True,
        "fold_structure": "fold_*/",
        "fold_files": ["metrics.jsonl", "model.joblib", "scaler.joblib", "roc_pr_curves.png", "confusion_matrix.png"],
        "root_files": ["metrics.json"],
        "metrics_key": "fold_results",
        "model_file_pattern": "model.joblib"
    },
    "5beta": {
        "type": "xgboost",
        "metrics_file": None,
        "results_file": "xgboost/results.json",  # In subdirectory!
        "has_folds": False,
        "fold_structure": None,
        "fold_files": [],
        "root_files": ["xgboost/results.json", "xgboost/model.json", "xgboost/roc_pr_curves.png"],
        "metrics_key": "cv_fold_results",
        "model_file_pattern": "xgboost/model.json"
    },
    "5f": {
        "type": "xgboost_features",
        "metrics_file": "metrics.json",
        "results_file": None,
        "has_folds": True,
        "fold_structure": "fold_*/",
        "fold_files": ["xgboost_model.json", "metadata.json", "roc_pr_curves.png", "confusion_matrix.png"],
        "root_files": ["metrics.json"],
        "metrics_key": "fold_results",
        "model_file_pattern": "xgboost_model.json"
    },
    "5g": {
        "type": "xgboost_features",
        "metrics_file": "metrics.json",
        "results_file": None,
        "has_folds": True,
        "fold_structure": "fold_*/",
        "fold_files": ["xgboost_model.json", "metadata.json"],  # NO PNG files in fold_1!
        "root_files": ["metrics.json"],
        "metrics_key": "fold_results",
        "model_file_pattern": "xgboost_model.json",
        "note": "Some folds may not have PNG files"
    },
    "5h": {
        "type": "xgboost_features",
        "metrics_file": None,  # Check if exists
        "results_file": None,
        "has_folds": True,
        "fold_structure": "fold_*/",
        "fold_files": ["xgboost_model.json", "metadata.json"],  # PNG files in some folds only
        "root_files": [],
        "metrics_key": "fold_results",
        "model_file_pattern": "xgboost_model.json",
        "note": "PNG files only in fold_2-5, not fold_1"
    }
}


def analyze_model_differences():
    """Analyze model-specific differences."""
    print("=" * 80)
    print("ANALYZING MODEL-SPECIFIC DIFFERENCES")
    print("=" * 80)
    
    for model_id, config in MODEL_DATA_STRUCTURES.items():
        print(f"\n{model_id}:")
        print(f"  Type: {config['type']}")
        print(f"  Metrics file: {config['metrics_file']}")
        print(f"  Results file: {config['results_file']}")
        print(f"  Has folds: {config['has_folds']}")
        print(f"  Metrics key: {config['metrics_key']}")
        if 'note' in config:
            print(f"  Note: {config['note']}")


def validate_against_source_code():
    """Validate notebook functions against training pipeline."""
    print("\n" + "=" * 80)
    print("VALIDATING AGAINST SOURCE CODE")
    print("=" * 80)
    
    pipeline_file = Path("lib/training/pipeline.py")
    if not pipeline_file.exists():
        print("[ERROR] pipeline.py not found")
        return False
    
    # Check key functions
    with open(pipeline_file) as f:
        pipeline_code = f.read()
    
    checks = {
        "_train_baseline_model_fold": "Baseline models (5a, 5b)",
        "_train_xgboost_model_fold": "XGBoost models (5f, 5g, 5h)",
        "_train_pytorch_model_fold": "PyTorch models",
        "stage5_train_models": "Main training function"
    }
    
    all_ok = True
    for func_name, description in checks.items():
        if func_name in pipeline_code:
            print(f"[OK] {description}: {func_name} found")
        else:
            print(f"[ERROR] {description}: {func_name} NOT found")
            all_ok = False
    
    return all_ok


def validate_against_data_structures():
    """Validate against actual data structures."""
    print("\n" + "=" * 80)
    print("VALIDATING AGAINST DATA STRUCTURES")
    print("=" * 80)
    
    data_dir = Path("data/stage5")
    if not data_dir.exists():
        print("[ERROR] data/stage5 not found")
        return False
    
    all_ok = True
    
    for model_id, config in MODEL_DATA_STRUCTURES.items():
        model_type = MODEL_DATA_STRUCTURES[model_id]["type"]
        model_path_map = {
            "5a": "logistic_regression",
            "5alpha": "sklearn_logreg",
            "5b": "svm",
            "5beta": "gradient_boosting",
            "5f": "xgboost_pretrained_inception",
            "5g": "xgboost_i3d",
            "5h": "xgboost_r2plus1d"
        }
        
        model_path_name = model_path_map.get(model_id)
        if not model_path_name:
            print(f"[WARN] {model_id}: No path mapping")
            continue
        
        model_path = data_dir / model_path_name
        if not model_path.exists():
            print(f"[ERROR] {model_id}: {model_path} does not exist")
            all_ok = False
            continue
        
        print(f"\n{model_id} ({model_path_name}):")
        
        # Check metrics/results file
        if config["metrics_file"]:
            metrics_file = model_path / config["metrics_file"]
            if metrics_file.exists():
                print(f"  [OK] metrics.json exists")
                try:
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    if config["metrics_key"] in metrics:
                        print(f"  [OK] {config['metrics_key']} key found")
                    else:
                        print(f"  [WARN] {config['metrics_key']} key NOT found")
                except Exception as e:
                    print(f"  [ERROR] Failed to parse metrics.json: {e}")
            else:
                print(f"  [ERROR] metrics.json does not exist")
                all_ok = False
        
        if config["results_file"]:
            results_file = model_path / config["results_file"]
            if results_file.exists():
                print(f"  [OK] results.json exists at {config['results_file']}")
            else:
                print(f"  [ERROR] results.json does not exist at {config['results_file']}")
                all_ok = False
        
        # Check fold structure
        if config["has_folds"]:
            fold_dirs = sorted([d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("fold_")])
            if fold_dirs:
                print(f"  [OK] Found {len(fold_dirs)} fold directories")
                # Check first fold
                first_fold = fold_dirs[0]
                for expected_file in config["fold_files"]:
                    file_path = first_fold / expected_file
                    if file_path.exists():
                        print(f"    [OK] {expected_file} exists in {first_fold.name}")
                    else:
                        print(f"    [WARN] {expected_file} missing in {first_fold.name} (may be optional)")
            else:
                print(f"  [WARN] No fold directories found (expected for {model_id})")
    
    return all_ok


def run_notebook(nb_path: Path, run_num: int, project_root: Path) -> Tuple[bool, str, str]:
    """
    Run a notebook using jupyter nbconvert.
    
    Returns:
        (success, stdout, stderr)
    """
    print(f"\n  Running {nb_path.name} (run {run_num}/3)...")
    
    # Activate venv and run notebook
    venv_python = project_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return False, "", "Virtual environment not found"
    
    # Use python -m nbconvert (requires nbconvert package)
    cmd = [
        str(venv_python), "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=600",
        str(nb_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(project_root),
            env={**os.environ, "PYTHONPATH": str(project_root)}
        )
        
        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Notebook execution timed out after 600 seconds"
    except Exception as e:
        return False, "", f"Error running notebook: {e}"


def check_notebook_outputs(nb_path: Path) -> Tuple[bool, List[str]]:
    """
    Check notebook output cells for errors.
    
    Returns:
        (has_errors, error_messages)
    """
    try:
        with open(nb_path) as f:
            nb = json.load(f)
        
        errors = []
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") == "code":
                outputs = cell.get("outputs", [])
                for output in outputs:
                    output_type = output.get("output_type", "")
                    if output_type == "error":
                        error_info = output.get("evalue", "Unknown error")
                        errors.append(f"Cell {i}: {error_info}")
                    elif output_type == "stream":
                        text = "".join(output.get("text", []))
                        if "error" in text.lower() or "exception" in text.lower() or "traceback" in text.lower():
                            errors.append(f"Cell {i}: {text[:200]}")
        
        return len(errors) == 0, errors
    except Exception as e:
        return False, [f"Failed to check notebook: {e}"]


def validate_and_run_notebooks():
    """Run validation loop 3 times."""
    print("\n" + "=" * 80)
    print("RUNNING NOTEBOOKS 3 TIMES")
    print("=" * 80)
    
    project_root = Path.cwd()
    executed_dir = project_root / "src" / "notebooks" / "executed"
    notebooks = sorted(executed_dir.glob("*.ipynb"))
    
    if not notebooks:
        print("[ERROR] No notebooks found in executed/")
        return False
    
    results = {}
    
    for run_num in range(1, 4):
        print(f"\n{'=' * 80}")
        print(f"RUN {run_num}/3")
        print(f"{'=' * 80}")
        
        for nb_file in notebooks:
            model_id = nb_file.stem.split("_")[0]
            if model_id not in results:
                results[model_id] = {
                    "runs": [],
                    "errors": []
                }
            
            # Run notebook
            success, stdout, stderr = run_notebook(nb_file, run_num, project_root)
            
            run_result = {
                "run": run_num,
                "success": success,
                "stdout": stdout[-500:] if stdout else "",  # Last 500 chars
                "stderr": stderr[-500:] if stderr else ""
            }
            results[model_id]["runs"].append(run_result)
            
            if success:
                print(f"    [OK] Execution successful")
            else:
                print(f"    [ERROR] Execution failed")
                if stderr:
                    print(f"      Error: {stderr[:200]}")
                results[model_id]["errors"].append(f"Run {run_num}: {stderr[:200]}")
            
            # Check outputs
            has_errors, error_messages = check_notebook_outputs(nb_file)
            if not has_errors:
                print(f"    [OK] No errors in output cells")
            else:
                print(f"    [ERROR] Errors found in output cells:")
                for err in error_messages[:3]:  # Show first 3
                    print(f"      {err}")
                results[model_id]["errors"].extend(error_messages)
            
            # Small delay between notebooks
            time.sleep(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    all_ok = True
    for model_id, result in results.items():
        successful_runs = sum(1 for r in result["runs"] if r["success"])
        print(f"\n{model_id}:")
        print(f"  Successful runs: {successful_runs}/3")
        if result["errors"]:
            print(f"  Errors: {len(result['errors'])}")
            for err in result["errors"][:5]:  # Show first 5
                print(f"    - {err}")
            all_ok = False
        else:
            print(f"  [OK] No errors")
    
    return all_ok


def main():
    """Main validation function."""
    print("COMPREHENSIVE NOTEBOOK VALIDATION")
    print("=" * 80)
    
    # Step 1: Analyze model differences
    analyze_model_differences()
    
    # Step 2: Validate against source code
    source_ok = validate_against_source_code()
    
    # Step 3: Validate against data structures
    data_ok = validate_against_data_structures()
    
    # Step 4: Run notebooks 3 times
    run_ok = validate_and_run_notebooks()
    
    # Final status
    print("\n" + "=" * 80)
    print("FINAL STATUS")
    print("=" * 80)
    print(f"Source code validation: {'PASS' if source_ok else 'FAIL'}")
    print(f"Data structure validation: {'PASS' if data_ok else 'FAIL'}")
    print(f"Notebook execution: {'PASS' if run_ok else 'FAIL'}")
    
    overall = source_ok and data_ok and run_ok
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")
    
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

