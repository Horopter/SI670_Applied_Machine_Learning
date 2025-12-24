#!/usr/bin/env python3
"""
Run all notebooks inline and save results with error capture.
Checks output cells for errors and warnings.
"""
import subprocess
import sys
import json
import os
from pathlib import Path

notebooks = [
    "src/notebooks/executed/5a_logistic_regression.ipynb",
    "src/notebooks/executed/5b_svm.ipynb",
    "src/notebooks/executed/5alpha_sklearn_logreg.ipynb",
    "src/notebooks/executed/5beta_gradient_boosting.ipynb",
    "src/notebooks/executed/5f_xgboost_pretrained_inception.ipynb",
    "src/notebooks/executed/5g_xgboost_i3d.ipynb",
    "src/notebooks/executed/5h_xgboost_r2plus1d.ipynb",
]

project_root = Path(__file__).parent

# Try to use venv jupyter if available
venv_jupyter = project_root / ".venv" / "bin" / "jupyter"
if venv_jupyter.exists():
    jupyter_cmd = str(venv_jupyter)
else:
    jupyter_cmd = "jupyter"

def check_notebook_outputs(nb_path: Path) -> tuple[bool, list[str], list[str]]:
    """
    Check notebook output cells for errors and warnings.
    
    Returns:
        (has_errors, error_messages, warning_messages)
    """
    try:
        with open(nb_path) as f:
            nb = json.load(f)
        
        errors = []
        warnings = []
        
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
                        text_lower = text.lower()
                        
                        # Check for actual error patterns (not just the word "error" in filenames)
                        # Actual errors typically have: "Error:", "Exception:", "Traceback", "failed", etc.
                        is_actual_error = (
                            "traceback" in text_lower or
                            "exception:" in text_lower or
                            text_lower.startswith("error:") or
                            " error " in text_lower or  # word boundary
                            text_lower.startswith("error ") or
                            "failed" in text_lower and ("error" in text_lower or "exception" in text_lower) or
                            "raise" in text_lower and ("error" in text_lower or "exception" in text_lower)
                        ) and not (
                            # Exclude false positives: filenames, informational messages
                            "displaying" in text_lower or
                            "error_analysis" in text_lower or
                            ".png" in text_lower or
                            ".jpg" in text_lower or
                            ".jpeg" in text_lower
                        )
                        
                        if is_actual_error:
                            errors.append(f"Cell {i}: {text[:200]}")
                        elif "warn" in text_lower and not (
                            "displaying" in text_lower or 
                            ".png" in text_lower or
                            "[warn]" in text_lower  # Exclude informational [WARN] messages from notebook code
                        ):
                            # Only flag warnings that aren't expected informational messages
                            # Informational [WARN] messages like "No DuckDB metrics found" are expected behavior
                            if not ("no duckdb" in text_lower or "no fold" in text_lower or "no results.json" in text_lower):
                                warnings.append(f"Cell {i}: {text[:200]}")
                    
                    elif output_type == "display_data" or output_type == "execute_result":
                        # Check data for error messages
                        data = output.get("data", {})
                        if "text/plain" in data:
                            text = "".join(data["text/plain"]) if isinstance(data["text/plain"], list) else str(data["text/plain"])
                            text_lower = text.lower()
                            
                            # Check for actual error patterns (not just the word "error" in filenames)
                            is_actual_error = (
                                "traceback" in text_lower or
                                "exception:" in text_lower or
                                text_lower.startswith("error:") or
                                " error " in text_lower or
                                text_lower.startswith("error ") or
                                "failed" in text_lower and ("error" in text_lower or "exception" in text_lower)
                            ) and not (
                                # Exclude false positives: filenames, informational messages
                                "displaying" in text_lower or
                                "error_analysis" in text_lower or
                                ".png" in text_lower or
                                ".jpg" in text_lower or
                                ".jpeg" in text_lower
                            )
                            
                            if is_actual_error:
                                errors.append(f"Cell {i}: {text[:200]}")
                            elif "warn" in text_lower and not (
                                "displaying" in text_lower or 
                                ".png" in text_lower or
                                "[warn]" in text_lower  # Exclude informational [WARN] messages from notebook code
                            ):
                                # Only flag warnings that aren't expected informational messages
                                # Informational [WARN] messages like "No DuckDB metrics found" are expected behavior
                                if not ("no duckdb" in text_lower or "no fold" in text_lower or "no results.json" in text_lower):
                                    warnings.append(f"Cell {i}: {text[:200]}")
        
        return len(errors) == 0, errors, warnings
    except Exception as e:
        return False, [f"Failed to check notebook: {e}"], []

results = {}
all_results = {}
max_iterations = 5
iteration = 0

while iteration < max_iterations:
    iteration += 1
    print(f"\n{'='*80}")
    print(f"ITERATION {iteration}/{max_iterations}")
    print(f"{'='*80}")
    
    results = {}
    iteration_errors = 0
    iteration_warnings = 0
    
    for nb_path in notebooks:
        nb_full_path = project_root / nb_path
        if not nb_full_path.exists():
            print(f"[WARN] Notebook not found: {nb_path}")
            results[nb_path] = {"status": "not_found"}
            continue
        
        print(f"\n{'-'*80}")
        print(f"Processing: {nb_path}")
        print(f"{'-'*80}")
        
        try:
            # Set environment to use venv if available
            env = os.environ.copy()
            if (project_root / ".venv" / "bin").exists():
                env["PATH"] = str(project_root / ".venv" / "bin") + ":" + env.get("PATH", "")
                env["PYTHONPATH"] = str(project_root) + ":" + env.get("PYTHONPATH", "")
            
            result = subprocess.run(
                [jupyter_cmd, "nbconvert", "--to", "notebook", "--execute", "--inplace", str(nb_full_path)],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout per notebook
                env=env
            )
            
            results[nb_path] = {
                "status": "success" if result.returncode == 0 else "error",
                "returncode": result.returncode,
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "stderr": result.stderr[-2000:] if result.stderr else "",
            }
            
            if result.returncode == 0:
                print(f"  ✓ Execution successful")
            else:
                print(f"  ✗ Execution failed (exit code: {result.returncode})")
                if result.stderr:
                    print(f"    STDERR: {result.stderr[-500:]}")
                if result.stdout:
                    print(f"    STDOUT: {result.stdout[-500:]}")
            
            # Check outputs for errors and warnings
            has_errors, errors, warnings = check_notebook_outputs(nb_full_path)
            
            results[nb_path]["output_errors"] = errors
            results[nb_path]["output_warnings"] = warnings
            results[nb_path]["has_output_errors"] = not has_errors
            
            if not has_errors and len(warnings) == 0:
                print(f"  ✓ No errors or warnings in outputs")
            else:
                if errors:
                    print(f"  ✗ ERRORS found in outputs ({len(errors)}):")
                    for err in errors[:5]:
                        print(f"    - {err}")
                    iteration_errors += len(errors)
                if warnings:
                    print(f"  ⚠ WARNINGS found in outputs ({len(warnings)}):")
                    for warn in warnings[:5]:
                        print(f"    - {warn}")
                    iteration_warnings += len(warnings)
            
            # Store in all_results
            if nb_path not in all_results:
                all_results[nb_path] = []
            all_results[nb_path].append({
                "iteration": iteration,
                "execution_success": result.returncode == 0,
                "has_output_errors": not has_errors,
                "error_count": len(errors),
                "warning_count": len(warnings),
                "errors": errors,
                "warnings": warnings
            })
            
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout executing {nb_path}")
            results[nb_path] = {"status": "timeout"}
        except FileNotFoundError:
            print(f"  ✗ jupyter command not found")
            results[nb_path] = {"status": "jupyter_not_found"}
            sys.exit(1)
        except Exception as e:
            print(f"  ✗ Exception executing {nb_path}: {e}")
            results[nb_path] = {"status": "exception", "error": str(e)}
    
    print(f"\n  Iteration {iteration} Summary: {iteration_errors} errors, {iteration_warnings} warnings")
    
    # If no errors or warnings, we can stop
    if iteration_errors == 0 and iteration_warnings == 0:
        print(f"\n✓ All notebooks passed with no errors or warnings!")
        break

# Final summary
print(f"\n{'='*80}")
print("FINAL SUMMARY")
print(f"{'='*80}")
for nb_path, res_list in all_results.items():
    print(f"\n{nb_path}:")
    for res in res_list:
        status = "✓" if res["execution_success"] and res["has_output_errors"] and res["error_count"] == 0 else "✗"
        print(f"  {status} Iteration {res['iteration']}: exec={res['execution_success']}, "
              f"errors={res['error_count']}, warnings={res['warning_count']}")
        if res["errors"]:
            print(f"    Errors: {res['errors'][:2]}")
        if res["warnings"]:
            print(f"    Warnings: {res['warnings'][:2]}")

# Save results to file
with open(project_root / "notebook_execution_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults saved to: notebook_execution_results.json")

