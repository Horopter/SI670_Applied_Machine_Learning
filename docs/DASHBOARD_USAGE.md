# AURA Training Results Dashboard

A Streamlit-based interactive dashboard for visualizing and analyzing training results from the 5-stage FVC pipeline.

## Features

- **📊 Overview Tab**: Summary statistics and model performance overview
- **📈 Model Comparison Tab**: Visual comparison of all trained models
- **🔍 Model Details Tab**: Detailed analysis of individual models with K-fold distributions
- **📋 Dataset Info Tab**: Dataset statistics and class distribution

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have completed Stage 5 training to generate results in `data/training_results/`

## Usage

### Running the Dashboard

```bash
# From project root
streamlit run src/dashboard_results.py
```

Or with custom port:
```bash
streamlit run src/dashboard_results.py --server.port 8502
```

### Accessing the Dashboard

Once running, the dashboard will be available at:
- Local: `http://localhost:8501`
- Network: `http://<your-ip>:8501`

## Dashboard Sections

### 1. Overview Tab
- Dataset statistics (total videos, real/fake counts)
- Model performance summary table with:
  - Mean accuracy
  - Standard deviation
  - Min/Max accuracy across folds
  - Number of folds

### 2. Model Comparison Tab
- **Accuracy Comparison**: Bar chart comparing validation accuracy across all models with error bars
- **Multi-Metric Comparison**: Grouped bar chart showing accuracy, precision, recall, and F1 scores

### 3. Model Details Tab
- **Fold-by-Fold Results**: Detailed table showing results for each fold
- **K-Fold Distribution**: Box plot showing the distribution of accuracy across folds
- **Summary Statistics**: Mean and standard deviation for all metrics

### 4. Dataset Info Tab
- Total number of videos
- Real vs Fake video counts
- Class distribution pie chart

## Results Structure

The dashboard expects results in the following structure:

```
data/training_results/
├── {model_type}/
│   ├── fold_results.csv          # K-fold results (preferred)
│   └── fold_{n}/                 # Individual fold directories
│       └── metrics.jsonl         # Alternative: metrics from tracker
```

### Expected CSV Format (fold_results.csv)

```csv
fold,val_acc,val_loss,accuracy,precision,recall,f1
1,0.85,0.32,0.85,0.82,0.88,0.85
2,0.87,0.30,0.87,0.84,0.90,0.87
...
```

## Customization

### Changing Project Root

Use the sidebar input field to specify a custom project root directory.

### Adding Custom Metrics

To add custom metrics to the dashboard:

1. Ensure your `fold_results.csv` includes the metric columns
2. Update the `plot_metrics_comparison()` function to include your metric
3. Add the metric to the `metrics_data` dictionary

## Tips for Presentations

1. **Pre-load Results**: Run the dashboard once to cache results (faster subsequent loads)
2. **Full Screen Mode**: Press `F` in the browser for full-screen presentation
3. **Sidebar Toggle**: Press `>` to toggle sidebar for more screen space
4. **Export Charts**: Click the camera icon on any Plotly chart to download as PNG

## Troubleshooting

### No Results Found

- Ensure Stage 5 training has completed
- Check that `data/training_results/` directory exists
- Verify that model directories contain `fold_results.csv` or `fold_*/metrics.jsonl`

### Missing Metrics

- Some models may not have all metrics (precision, recall, F1)
- The dashboard will show available metrics only
- Check that your training pipeline saves the required metrics

### Performance Issues

- Results are cached after first load
- For large datasets, consider filtering results before loading
- Use `@st.cache_data` decorator for expensive computations

## Example Workflow

1. **Run Training**:
   ```bash
   python src/run_new_pipeline.py --only-stage 5
   ```

2. **Launch Dashboard**:
   ```bash
   streamlit run src/dashboard_results.py
   ```

3. **Navigate Tabs**:
   - Start with Overview for high-level summary
   - Use Model Comparison to see which model performs best
   - Dive into Model Details for specific analysis
   - Check Dataset Info for data statistics

## Future Enhancements

Potential additions:
- Sample prediction viewer with video thumbnails
- Confusion matrix visualization
- ROC curve plots
- Training curve visualization (loss/accuracy over epochs)
- Feature importance analysis
- Export functionality for reports

