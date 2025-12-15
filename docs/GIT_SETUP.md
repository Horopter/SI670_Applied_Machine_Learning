# Git Repository Setup

## Repository Information

- **GitHub Repository**: [AURA](https://github.com/Horopter/AURA)
- **Project Location**: Root of the repository
- **Branch**: `main`
- **Author**: Horopter (santoshdesai12@hotmail.com)

## Repository Structure

```
AURA/
├── README.md                    # Project README
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
├── archive/                    # Original dataset archives
├── data/                      # Processed metadata
├── docs/                      # Project documentation
│   ├── PROJECT_OVERVIEW.md
│   ├── CHANGELOG.md
│   ├── WORKFLOW.md
│   ├── METHODOLOGY.md
│   ├── OPTIMIZATIONS.md
│   └── GIT_SETUP.md
├── lib/                        # Core library modules
├── src/                        # Scripts and notebooks
└── ...
```

## Commits

1. **Initial repository setup** - Created AURA repository
2. **Project implementation** - Added all project files and modules
3. **Code quality overhaul** - Batman tasks: documentation, PEP8, comments refactoring

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
git clone https://github.com/Horopter/AURA.git
cd AURA

# Set up environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Pushing Updates

To push updates to the repository:

```bash
cd /path/to/AURA

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

