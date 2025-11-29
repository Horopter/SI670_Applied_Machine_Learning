"""Configuration for FVC data preparation"""
import os
from dataclasses import dataclass


@dataclass
class FVCConfig:
    """Configuration for FVC dataset preparation"""
    
    # Root of the fvc data on the Mac
    root_dir: str = os.path.expanduser("~/Downloads/fvc")
    
    # Subsets available
    subsets = ("FVC1", "FVC2", "FVC3")
    
    # Where videos are stored (FVC1, FVC2, FVC3 folders)
    videos_dir: str = None  # will be derived from root_dir
    
    # Where metadata CSVs live (only used locally)
    metadata_dir: str = None  # will be derived from root_dir
    
    # Where to write dataset manifests
    data_dir: str = None   # will be derived from root_dir
    
    # Main metadata filename (you can adjust to actual name)
    main_metadata_filename: str = "FVC.csv"
    
    # Duplicates CSV filename
    dup_metadata_filename: str = "FVC_dup.csv"
    
    def __post_init__(self):
        """Initialize derived paths and create directories"""
        # Videos directory
        self.videos_dir = os.path.join(self.root_dir, "videos")
        
        # Use "Metadata" (capital M) if it exists in videos/, otherwise "metadata"
        metadata_capital = os.path.join(self.videos_dir, "Metadata")
        metadata_lower = os.path.join(self.videos_dir, "metadata")
        if os.path.exists(metadata_capital):
            self.metadata_dir = metadata_capital
        else:
            self.metadata_dir = metadata_lower
            os.makedirs(self.metadata_dir, exist_ok=True)
        
        self.data_dir = os.path.join(self.root_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

