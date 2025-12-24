#!/usr/bin/env python3
"""
Analyze Stage 5 Models: Extract metrics, time logs, and generate summary reports.

This script:
1. Extracts metrics from all trained models
2. Parses time logs from SLURM and Python logs
3. Generates summary reports
4. Creates comparison visualizations
5. Exports data for DuckDB analytics
"""

import sys
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
matplotlib.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

STAGE5_DIR = project_root / "data" / "stage5"
LOGS_DIR = project_root / "logs" / "stage5"
TRAINING_LOGS_DIR = project_root / "logs" / "stage5"


def parse_time_string(time_str: str) -> Optional[float]:
    """Parse time string to seconds."""
    # Format: "1234s (20 minutes)" or "1234s" or "20 minutes"
    if not time_str:
        return None
    
    # Extract seconds
    seconds_match = re.search(r'(\d+)s', time_str)
    if seconds_match:
        return float(seconds_match.group(1))
    
    # Extract minutes
    minutes_match = re.search(r'(\d+)\s*minutes?', time_str)
    if minutes_match:
        return float(minutes_match.group(1)) * 60
    
    # Extract hours
    hours_match = re.search(r'(\d+)\s*hours?', time_str)
    if hours_match:
        return float(hours_match.group(1)) * 3600
    
    return None


def extract_slurm_times(model_type: str) -> List[Dict]:
    """Extract execution times from SLURM log files."""
    times = []
    
    # Pattern: stage5{model}-{JOB_ID}.out
    pattern = f"stage5{model_type.replace('_', '')}-*.out"
    log_files = list(LOGS_DIR.glob(pattern))
    
    if not log_files:
        # Try alternative pattern
        pattern = f"stage5{model_type[0]}-*.out"
        log_files = list(LOGS_DIR.glob(pattern))
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract execution time
            time_match = re.search(
                r'Execution time:\s*(\d+)s\s*\((\d+)\s*minutes?\)',
                content
            )
            if time_match:
                seconds = int(time_match.group(1))
                minutes = int(time_match.group(2))
                times.append({
                    'model_type': model_type,
                    'log_file': str(log_file),
                    'duration_seconds': seconds,
                    'duration_minutes': minutes,
                    'source': 'slurm'
                })
            
            # Extract job ID
            job_id_match = re.search(r'SLURM_JOBID:\s*(\d+)', content)
            if job_id_match:
                times[-1]['job_id'] = job_id_match.group(1)
        except Exception as e:
            logger.debug(f"Error reading {log_file}: {e}")
    
    return times


def extract_python_times() -> List[Dict]:
    """Extract execution times from Python training logs."""
    times = []
    
    log_files = list(TRAINING_LOGS_DIR.glob("stage5_training_*.log"))
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract execution time
            time_match = re.search(
                r'Execution time:\s*([\d.]+)\s*seconds\s*\(([\d.]+)\s*minutes?\)',
                content
            )
            if time_match:
                seconds = float(time_match.group(1))
                minutes = float(time_match.group(2))
                
                # Extract model types from log
                model_types_match = re.search(r'Models trained:\s*\[(.*?)\]', content)
                model_types = []
                if model_types_match:
                    model_types = [
                        m.strip().strip("'\"") 
                        for m in model_types_match.group(1).split(',')
                    ]
                
                for model_type in model_types:
                    times.append({
                        'model_type': model_type,
                        'log_file': str(log_file),
                        'duration_seconds': seconds,
                        'duration_minutes': minutes,
                        'source': 'python'
                    })
        except Exception as e:
            logger.debug(f"Error reading {log_file}: {e}")
    
    return times


def load_model_metrics(model_type: str) -> Optional[Dict]:
    """Load aggregated metrics for a model."""
    metrics_file = STAGE5_DIR / model_type / "metrics.json"
    
    if not metrics_file.exists():
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"Error loading metrics for {model_type}: {e}")
        return None


def load_fold_metrics(model_type: str, fold: int) -> Optional[Dict]:
    """Load metrics for a specific fold."""
    fold_dir = STAGE5_DIR / model_type / f"fold_{fold}"
    
    # Try metadata.json first
    metadata_file = fold_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Error loading metadata for {model_type} fold {fold}: {e}")
    
    return None


def analyze_all_models() -> Dict:
    """Analyze all Stage 5 models."""
    results = {
        'models': {},
        'times': [],
        'summary': {}
    }
    
    # Get all model directories
    model_dirs = [d for d in STAGE5_DIR.iterdir() 
                  if d.is_dir() and not d.name.startswith('.')]
    
    logger.info(f"Found {len(model_dirs)} model directories")
    
    for model_dir in model_dirs:
        model_type = model_dir.name
        logger.info(f"Analyzing {model_type}...")
        
        model_data = {
            'model_type': model_type,
            'folds': {},
            'aggregated_metrics': None,
            'has_best_model': False,
            'has_plots': False,
            'fold_count': 0
        }
        
        # Load aggregated metrics
        aggregated = load_model_metrics(model_type)
        if aggregated:
            model_data['aggregated_metrics'] = aggregated
        
        # Check for folds
        fold_dirs = [d for d in model_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('fold_')]
        model_data['fold_count'] = len(fold_dirs)
        
        for fold_dir in fold_dirs:
            fold_num = int(fold_dir.name.split('_')[1])
            fold_metrics = load_fold_metrics(model_type, fold_num)
            if fold_metrics:
                model_data['folds'][fold_num] = fold_metrics
        
        # Check for best model
        best_model_dir = model_dir / "best_model"
        if best_model_dir.exists():
            model_data['has_best_model'] = True
        
        # Check for plots
        plots_dir = model_dir / "plots"
        if plots_dir.exists() and list(plots_dir.glob("*.png")):
            model_data['has_plots'] = True
        
        results['models'][model_type] = model_data
        
        # Extract times
        slurm_times = extract_slurm_times(model_type)
        results['times'].extend(slurm_times)
    
    # Extract Python training times
    python_times = extract_python_times()
    results['times'].extend(python_times)
    
    # Generate summary
    results['summary'] = generate_summary(results)
    
    return results


def generate_summary(results: Dict) -> Dict:
    """Generate summary statistics."""
    summary = {
        'total_models': len(results['models']),
        'models_with_folds': 0,
        'models_with_metrics': 0,
        'models_with_plots': 0,
        'models_with_best_model': 0,
        'total_folds': 0,
        'model_types': []
    }
    
    for model_type, model_data in results['models'].items():
        summary['model_types'].append(model_type)
        
        if model_data['fold_count'] > 0:
            summary['models_with_folds'] += 1
            summary['total_folds'] += model_data['fold_count']
        
        if model_data['aggregated_metrics']:
            summary['models_with_metrics'] += 1
        
        if model_data['has_plots']:
            summary['models_with_plots'] += 1
        
        if model_data['has_best_model']:
            summary['models_with_best_model'] += 1
    
    # Time statistics
    if results['times']:
        times_df = pd.DataFrame(results['times'])
        summary['time_stats'] = {
            'total_executions': len(times_df),
            'avg_duration_seconds': times_df['duration_seconds'].mean(),
            'avg_duration_minutes': times_df['duration_minutes'].mean(),
            'total_duration_seconds': times_df['duration_seconds'].sum(),
            'total_duration_hours': times_df['duration_seconds'].sum() / 3600
        }
    
    return summary


def create_metrics_comparison_plot(results: Dict, output_path: Path):
    """Create comparison plot of model metrics."""
    metrics_data = []
    
    for model_type, model_data in results['models'].items():
        if not model_data['aggregated_metrics']:
            continue
        
        metrics = model_data['aggregated_metrics']
        
        # Extract mean metrics
        if 'mean_val_acc' in metrics:
            metrics_data.append({
                'model_type': model_type,
                'val_acc': metrics.get('mean_val_acc', 0),
                'val_f1': metrics.get('mean_val_f1', 0),
                'val_precision': metrics.get('mean_val_precision', 0),
                'val_recall': metrics.get('mean_val_recall', 0),
            })
    
    if not metrics_data:
        logger.warning("No metrics data available for comparison plot")
        return
    
    df = pd.DataFrame(metrics_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy comparison
    axes[0, 0].barh(df['model_type'], df['val_acc'])
    axes[0, 0].set_xlabel('Validation Accuracy', fontweight='bold')
    axes[0, 0].set_title('Model Accuracy Comparison', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # F1 Score comparison
    axes[0, 1].barh(df['model_type'], df['val_f1'])
    axes[0, 1].set_xlabel('Validation F1 Score', fontweight='bold')
    axes[0, 1].set_title('Model F1 Score Comparison', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Precision comparison
    axes[1, 0].barh(df['model_type'], df['val_precision'])
    axes[1, 0].set_xlabel('Validation Precision', fontweight='bold')
    axes[1, 0].set_title('Model Precision Comparison', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Recall comparison
    axes[1, 1].barh(df['model_type'], df['val_recall'])
    axes[1, 1].set_xlabel('Validation Recall', fontweight='bold')
    axes[1, 1].set_title('Model Recall Comparison', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved metrics comparison plot: {output_path}")


def create_time_analysis_plot(results: Dict, output_path: Path):
    """Create time analysis plot."""
    if not results['times']:
        logger.warning("No time data available for time analysis plot")
        return
    
    times_df = pd.DataFrame(results['times'])
    
    # Group by model type
    model_times = times_df.groupby('model_type')['duration_minutes'].agg(['mean', 'std', 'count'])
    model_times = model_times.sort_values('mean', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bar plot of average times
    axes[0].barh(model_times.index, model_times['mean'])
    axes[0].set_xlabel('Average Duration (minutes)', fontweight='bold')
    axes[0].set_title('Average Training Time by Model', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Box plot of all times
    if len(times_df) > 0:
        times_df_pivot = times_df.pivot_table(
            values='duration_minutes',
            index='model_type',
            aggfunc='mean'
        )
        axes[1].barh(times_df_pivot.index, times_df_pivot['duration_minutes'])
        axes[1].set_xlabel('Duration (minutes)', fontweight='bold')
        axes[1].set_title('Training Time Distribution', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved time analysis plot: {output_path}")


def export_for_duckdb(results: Dict, output_path: Path):
    """Export results in format suitable for DuckDB."""
    # Create metrics table
    metrics_rows = []
    for model_type, model_data in results['models'].items():
        if model_data['aggregated_metrics']:
            metrics = model_data['aggregated_metrics']
            metrics_rows.append({
                'model_type': model_type,
                'mean_val_acc': metrics.get('mean_val_acc'),
                'std_val_acc': metrics.get('std_val_acc'),
                'mean_val_f1': metrics.get('mean_val_f1'),
                'std_val_f1': metrics.get('std_val_f1'),
                'mean_val_precision': metrics.get('mean_val_precision'),
                'std_val_precision': metrics.get('std_val_precision'),
                'mean_val_recall': metrics.get('mean_val_recall'),
                'std_val_recall': metrics.get('std_val_recall'),
                'n_folds': model_data['fold_count']
            })
    
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_parquet = output_path / "stage5_metrics.parquet"
        metrics_df.to_parquet(metrics_parquet, index=False)
        logger.info(f"Exported metrics to: {metrics_parquet}")
    
    # Create times table
    if results['times']:
        times_df = pd.DataFrame(results['times'])
        times_parquet = output_path / "stage5_times.parquet"
        times_df.to_parquet(times_parquet, index=False)
        logger.info(f"Exported times to: {times_parquet}")


def generate_report(results: Dict, output_path: Path):
    """Generate text report."""
    report_lines = [
        "=" * 80,
        "Stage 5 Models: Comprehensive Analysis Report",
        "=" * 80,
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "SUMMARY",
        "-" * 80,
        f"Total models found: {results['summary']['total_models']}",
        f"Models with folds: {results['summary']['models_with_folds']}",
        f"Models with metrics: {results['summary']['models_with_metrics']}",
        f"Models with plots: {results['summary']['models_with_plots']}",
        f"Models with best_model: {results['summary']['models_with_best_model']}",
        f"Total folds: {results['summary']['total_folds']}",
        "",
    ]
    
    if 'time_stats' in results['summary']:
        time_stats = results['summary']['time_stats']
        report_lines.extend([
            "TIME STATISTICS",
            "-" * 80,
            f"Total executions logged: {time_stats['total_executions']}",
            f"Average duration: {time_stats['avg_duration_minutes']:.2f} minutes",
            f"Total duration: {time_stats['total_duration_hours']:.2f} hours",
            "",
        ])
    
    report_lines.extend([
        "MODEL DETAILS",
        "-" * 80,
    ])
    
    for model_type, model_data in sorted(results['models'].items()):
        report_lines.extend([
            f"\n{model_type}:",
            f"  Folds: {model_data['fold_count']}",
            f"  Has metrics: {model_data['aggregated_metrics'] is not None}",
            f"  Has plots: {model_data['has_plots']}",
            f"  Has best_model: {model_data['has_best_model']}",
        ])
        
        if model_data['aggregated_metrics']:
            metrics = model_data['aggregated_metrics']
            report_lines.extend([
                f"  Mean Val Acc: {metrics.get('mean_val_acc', 'N/A'):.4f}" if isinstance(metrics.get('mean_val_acc'), (int, float)) else f"  Mean Val Acc: N/A",
                f"  Mean Val F1: {metrics.get('mean_val_f1', 'N/A'):.4f}" if isinstance(metrics.get('mean_val_f1'), (int, float)) else f"  Mean Val F1: N/A",
            ])
    
    report_lines.append("\n" + "=" * 80)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Generated report: {output_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Stage 5 models")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/stage5_analysis",
        help="Output directory for analysis results"
    )
    
    args = parser.parse_args()
    
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Stage 5 Models Analysis")
    logger.info("=" * 80)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Stage 5 directory: {STAGE5_DIR}")
    logger.info(f"Output directory: {output_dir}")
    
    # Analyze all models
    logger.info("\nAnalyzing all models...")
    results = analyze_all_models()
    
    # Save results as JSON
    results_json = output_dir / "analysis_results.json"
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved analysis results: {results_json}")
    
    # Generate report
    report_file = output_dir / "analysis_report.txt"
    generate_report(results, report_file)
    
    # Create plots
    if results['models']:
        metrics_plot = output_dir / "metrics_comparison.png"
        create_metrics_comparison_plot(results, metrics_plot)
        
        time_plot = output_dir / "time_analysis.png"
        create_time_analysis_plot(results, time_plot)
    
    # Export for DuckDB
    export_for_duckdb(results, output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

