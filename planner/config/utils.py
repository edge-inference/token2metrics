"""Directory utilities."""

from pathlib import Path


def ensure_directories():
    from .settings import PATHS
    
    # Output directories
    for key in ["output_dir", "plots_dir", "merged_results_dir", "scored_results_dir"]:
        if key in PATHS:
            Path(PATHS[key]).mkdir(parents=True, exist_ok=True)


def ensure_output_dirs(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dirs = {
        "output_dir": output_dir,
        "plots_dir": output_dir / "plots",
        "insights_dir": output_dir / "insights",
        "reports_dir": output_dir / "reports",
    }
    
    # Create all subdirectories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs



