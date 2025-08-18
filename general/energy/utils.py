"""
Utilities for standardizing file paths, directory creation, and data output operations for profiling analysis.
"""
import re
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import os


class PathManager:
    """
    Manages standard paths across the profiling analysis modules.
    """
    
    BASE_DIR = Path(__file__).parent 
    _output_dir = None  
    DATA_DIR = BASE_DIR.parent.parent / "raw/tegra"
    
    @classmethod
    def set_output_dir(cls, output_dir: str) -> None:
        """Set the output directory dynamically."""
        cls._output_dir = Path(output_dir)
    
    @classmethod
    def get_output_dir(cls) -> Path:
        """Get the current output directory."""
        if cls._output_dir is None:
            cls._output_dir = cls.BASE_DIR.parent / "energy_results"
        return cls._output_dir
    
    @classmethod
    def ensure_dirs(cls) -> None:
        """Create all standard directories if they don't exist."""
        cls.get_output_dir().mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        (cls.get_output_dir() / "insight_charts").mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_data_path(cls, filename: str) -> Path:
        """Get path for a data file in the standard data directory."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        return cls.DATA_DIR / filename
    
    @classmethod
    def get_output_path(cls, filename: str) -> Path:
        """Get path for an output file in the output directory."""
        output_dir = cls.get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename
    
    @classmethod
    def get_chart_path(cls, filename: str, chart_type: str = "general") -> Path:
        """
        Get path for a chart file in the insights charts directory.
        
        Args:
            filename: Name of the chart file
            chart_type: Type of chart (ignored, all go to insight_charts)
            
        Returns:
            Path object for the chart file
        """
        chart_dir = cls.get_output_dir() / "insight_charts"
        chart_dir.mkdir(parents=True, exist_ok=True)
        return chart_dir / filename


def save_dataframe(
    df: pd.DataFrame, 
    filename: str, 
    format_type: str = "excel", 
    sheet_name: str = "Sheet1", 
    index: bool = False
) -> Path:
    """
    Save a DataFrame to a file in the output directory or absolute path.
    
    Args:
        df: DataFrame to save
        filename: Name for the output file or absolute path
        format_type: "excel" or "csv"
        sheet_name: Sheet name (for Excel only)
        index: Whether to include index in output
    
    Returns:
        Path to the saved file
    """
    # Handle absolute paths vs relative paths
    if os.path.isabs(filename):
        file_path = Path(filename)
    else:
        if not filename.endswith((".xlsx", ".csv")):
            ext = ".xlsx" if format_type == "excel" else ".csv"
            filename = f"{filename}{ext}"
        # Use output path instead of data path for energy results
        file_path = PathManager.get_output_path(filename)
    
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if str(file_path).endswith(".xlsx"):
        df.to_excel(file_path, sheet_name=sheet_name, index=index)
    else:
        df.to_csv(file_path, index=index)
    
    return file_path


def load_dataframe(filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load a DataFrame from a file in the data directory.
    
    Args:
        filename: Name of the file to load
        sheet_name: Sheet name (for Excel only)
    
    Returns:
        Loaded DataFrame
    """
    file_path = PathManager.get_data_path(filename)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix == ".xlsx":
        return pd.read_excel(file_path, sheet_name=sheet_name or 0)
    else:
        return pd.read_csv(file_path)


def save_figure(figure, filename: str, chart_type: str = "general", dpi: int = 300) -> Path:
    """
    Save pdf figure to a file in the insight charts directory.
    

    """

    
    file_path = PathManager.get_chart_path(filename, chart_type)
    figure.savefig(file_path, dpi=dpi, bbox_inches="tight")
    
    return file_path


def collect_energy_files(
    base_dir: str,
    file_prefix: str = "energy_base_",
    file_suffix: str = ".csv"
) -> "Dict[str, Dict[str, str]]":
    """
    Recursively collect all energy CSV files under base_dir.
    Returns a nested dict: {model_name: {subject_key: csv_path}}
    where subject_key uniquely identifies each subject
    
    Handles directory structure:
    - base_all_subjects_TIMESTAMP_MODEL_NAME/subject/energy_base_*.csv
    """
    import re
    from pathlib import Path
    energy_files = {}
    base_path = Path(base_dir)
    
    def parse_filename_for_subject_key(filepath):
        """Parse filename to create unique subject key"""
        filename = Path(filepath).stem
        
        # Pattern: energy_base_SUBJECT_TIMESTAMP
        pattern = r'energy_base_(.+?)_\d{8}_\d{6}'
        match = re.match(pattern, filename)
        
        if match:
            subject = match.group(1)
            return subject, subject
        
        # Fallback: use parent directory name as subject
        parent_dir = Path(filepath).parent.name
        return parent_dir, parent_dir
    
    for csv_path in base_path.rglob(f"{file_prefix}*{file_suffix}"):
        try:
            # Parse filename to get subject key
            subject_key, subject = parse_filename_for_subject_key(csv_path)
            
            # Determine model name from directory structure
            # Structure: base_all_subjects_TIMESTAMP_MODEL_NAME/subject/energy_base_*.csv
            model_dir = csv_path.parent.parent.name
            match = re.search(r'base_all_subjects_\d{8}_\d{6}_(.+)', model_dir)
            if match:
                model_name = match.group(1)
            else:
                model_name = model_dir
            
            if not model_name:
                print(f"Skipping {csv_path}: could not determine model name.")
                continue
                
            if model_name not in energy_files:
                energy_files[model_name] = {}
                
            # Use subject as the key
            if subject_key in energy_files[model_name]:
                print(f"Warning: Duplicate subject '{subject_key}' for model '{model_name}'. Skipping {csv_path}.")
                continue
                
            energy_files[model_name][subject_key] = str(csv_path)
            
        except Exception as e:
            print(f"Skipping {csv_path}: {e}")
    
    return energy_files
