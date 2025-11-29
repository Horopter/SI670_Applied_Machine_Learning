# Git Repository Setup

## Repository Information

- **GitHub Repository**: [SI670_Applied_Machine_Learning](https://github.com/Horopter/SI670_Applied_Machine_Learning)
- **Project Location**: `Final_Project/` folder within the repository
- **Branch**: `main`
- **Author**: Horopter (santoshdesai12@hotmail.com)

## Repository Structure

```
SI670_Applied_Machine_Learning/
├── README.md                    # Repository-level README
└── Final_Project/               # FVC Binary Video Classifier project
    ├── README.md                # Project README
    ├── .gitignore              # Git ignore rules
    ├── requirements.txt        # Python dependencies
    ├── archive/                # Original dataset archives
    ├── data/                   # Processed metadata
    ├── docs/                   # Project documentation
    │   ├── PROJECT_OVERVIEW.md
    │   ├── CHANGELOG.md
    │   ├── MLOPS_OPTIMIZATIONS.md
    │   ├── OOM_AND_KFOLD_OPTIMIZATIONS.md
    │   └── GIT_SETUP.md
    ├── lib/                    # Core library modules
    ├── src/                    # Scripts and notebooks
    └── ...
```

## Commits

1. **Initial repository setup** - Created repository with basic README
2. **Add Final_Project: FVC Binary Video Classifier** - Added all project files
3. **Add lib/ module files (fix .gitignore)** - Added library modules and fixed .gitignore

## Files Excluded from Git

The following files/folders are excluded via `.gitignore`:
- `__pycache__/` - Python cache files
- `venv/`, `env/` - Virtual environments
- `*.pt`, `*.pth` - Model checkpoints (except in specific locations)
- `videos/` - Extracted video files (too large)
- `runs/` - Experiment outputs
- `logs/*.log`, `*.out`, `*.err` - Log files
- `archive/*.zip` - Archive files (too large)
- `data/*.csv`, `data/*.json` - Large data files (structure preserved with `.gitkeep`)

## Cloning the Repository

To clone and work with the project:

```bash
# Clone the repository
git clone https://github.com/Horopter/SI670_Applied_Machine_Learning.git
cd SI670_Applied_Machine_Learning/Final_Project

# Set up environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Pushing Updates

To push updates to the repository:

```bash
cd /path/to/SI670_Applied_Machine_Learning/Final_Project

# Make changes, then:
git add .
git commit -m "Description of changes"
git push origin main
```

## Notes

- The `archive/` folder structure is preserved but zip files are excluded (too large for Git)
- Large data files (CSV, JSON) are excluded but directory structure is maintained
- Model checkpoints and experiment outputs are excluded but can be added selectively if needed
- All source code, documentation, and configuration files are tracked

