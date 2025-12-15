"""
Airflow DAG for AURA pipeline orchestration.

This DAG automates the execution of the 5-stage AURA pipeline:
1. Video Augmentation
2. Feature Extraction
3. Video Scaling
4. Scaled Feature Extraction
5. Model Training
"""

from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Default arguments
default_args = {
    'owner': 'fvc_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'fvc_pipeline',
    default_args=default_args,
    description='FVC Binary Classifier Pipeline',
    schedule_interval=None,  # Manual trigger or set to specific schedule
    start_date=days_ago(1),
    catchup=False,
    tags=['fvc', 'video-classification', 'ml'],
)


def stage1_augmentation(**context):
    """Stage 1: Video Augmentation"""
    import sys
    import os
    from pathlib import Path
    
    # Get project root from Airflow variables or default
    project_root = Path(context.get('dag_run').conf.get('project_root', os.getcwd()))
    sys.path.insert(0, str(project_root))
    
    from lib.augmentation import stage1_augment_videos
    
    stage1_augment_videos(
        project_root=str(project_root),
        num_augmentations=10,
        output_dir="data/augmented_videos"
    )


def stage2_features(**context):
    """Stage 2: Feature Extraction"""
    import sys
    import os
    from pathlib import Path
    
    project_root = Path(context.get('dag_run').conf.get('project_root', os.getcwd()))
    sys.path.insert(0, str(project_root))
    
    from lib.features import stage2_extract_features
    
    stage2_extract_features(
        project_root=str(project_root),
        augmented_metadata_path="data/augmented_videos/augmented_metadata.arrow",
        output_dir="data/features_stage2"
    )


def stage3_scaling(**context):
    """Stage 3: Video Scaling"""
    import sys
    import os
    from pathlib import Path
    
    project_root = Path(context.get('dag_run').conf.get('project_root', os.getcwd()))
    sys.path.insert(0, str(project_root))
    
    from lib.scaling import stage3_scale_videos
    
    stage3_scale_videos(
        project_root=str(project_root),
        augmented_metadata_path="data/augmented_videos/augmented_metadata.arrow",
        output_dir="data/scaled_videos"
    )


def stage4_scaled_features(**context):
    """Stage 4: Scaled Feature Extraction"""
    import sys
    import os
    from pathlib import Path
    
    project_root = Path(context.get('dag_run').conf.get('project_root', os.getcwd()))
    sys.path.insert(0, str(project_root))
    
    from lib.features import stage4_extract_scaled_features
    
    stage4_extract_scaled_features(
        project_root=str(project_root),
        scaled_metadata_path="data/scaled_videos/scaled_metadata.arrow",
        output_dir="data/features_stage4"
    )


def stage5_training(**context):
    """Stage 5: Model Training"""
    import sys
    import os
    from pathlib import Path
    
    project_root = Path(context.get('dag_run').conf.get('project_root', os.getcwd()))
    sys.path.insert(0, str(project_root))
    
    from lib.training import stage5_train_models
    
    stage5_train_models(
        project_root=str(project_root),
        scaled_metadata_path="data/scaled_videos/scaled_metadata.arrow",
        features_stage2_path="data/features_stage2/features_metadata.arrow",
        features_stage4_path="data/features_stage4/features_metadata.arrow",
        model_types=["logistic_regression", "svm", "i3d"],
        n_splits=5,
        output_dir="data/training_results"
    )


# Define tasks
task_stage1 = PythonOperator(
    task_id='stage1_augmentation',
    python_callable=stage1_augmentation,
    dag=dag,
)

task_stage2 = PythonOperator(
    task_id='stage2_features',
    python_callable=stage2_features,
    dag=dag,
)

task_stage3 = PythonOperator(
    task_id='stage3_scaling',
    python_callable=stage3_scaling,
    dag=dag,
)

task_stage4 = PythonOperator(
    task_id='stage4_scaled_features',
    python_callable=stage4_scaled_features,
    dag=dag,
)

task_stage5 = PythonOperator(
    task_id='stage5_training',
    python_callable=stage5_training,
    dag=dag,
)

# Define task dependencies
task_stage1 >> task_stage2 >> [task_stage3, task_stage4] >> task_stage5

