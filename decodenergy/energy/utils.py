# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Shared utilities for standardizing file paths, directory creation, and data output operations across energy analysis modules.
"""
import re
from pathlib import Path
from typing import Optional
import pandas as pd
import os


def sort_models_by_size(model_names):
    """
    Sort model names by their size (1.5B, 8B, 14B) for consistent ordering across modules.
    """
    size_order = {'1.5B': 1, '1_5B': 1, '8B': 2, '14B': 3}
    def get_sort_key(model_name):
        for size, order in size_order.items():
            if size in model_name:
                return order
        return 999
    return sorted(model_names, key=get_sort_key)


def extract_token_length(filename_or_subject):
    """
    Extract token length information from filename/subject.
    """
    pattern = r'in(\d+).*?out(\d+)'
    match = re.search(pattern, str(filename_or_subject))
    
    if match:
        input_tokens = match.group(1)
        output_tokens = match.group(2)
        return f"in_{input_tokens}_out_{output_tokens}"
    
    return "unknown"


def _resolve_existing_dir(path_str: str) -> str:
    """
    Resolve directory path, checking parent directories if needed.
    """
    candidates = [Path(path_str)]
    if not path_str.startswith('/'):
        candidates.append(Path('..') / path_str.lstrip('./'))
        candidates.append(Path('..') / '..' / path_str.lstrip('./'))
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return path_str


class PathManager:
    """
    Manages standard paths across the profiling analysis modules.
    """
    
    BASE_DIR = Path(__file__).parent 
    _output_dir = None  
    _data_dir = None
    
    @classmethod
    def set_output_dir(cls, output_dir: str) -> None:
        """Set the output directory dynamically."""
        cls._output_dir = Path(output_dir)
    
    @classmethod
    def set_data_dir(cls, data_dir: str) -> None:
        """Set the data directory dynamically."""
        cls._data_dir = Path(data_dir)
    
    @classmethod
    def get_output_dir(cls) -> Path:
        """Get the current output directory."""
        if cls._output_dir is None:
            repo_root = cls.BASE_DIR.resolve().parents[3] 
            cls._output_dir = repo_root / "outputs" / "decode"
            cls._output_dir.mkdir(parents=True, exist_ok=True)
        return cls._output_dir
    
    @classmethod
    def ensure_dirs(cls) -> None:
        """Create all standard directories if they don't exist."""
        cls.get_output_dir().mkdir(parents=True, exist_ok=True)
        if cls._data_dir:
            cls._data_dir.mkdir(parents=True, exist_ok=True)
        (cls.get_output_dir() / "charts").mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_data_path(cls, filename: str) -> Path:
        """Get path for a data file in the standard data directory."""
        if cls._data_dir is None:
            cls._data_dir = cls.BASE_DIR.parent
        cls._data_dir.mkdir(parents=True, exist_ok=True)
        return cls._data_dir / filename
    
    @classmethod
    def get_output_path(cls, filename: str) -> Path:
        """Get path for an output file in the output directory."""
        output_dir = cls.get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename
    
    @classmethod
    def get_chart_path(cls, filename: str, chart_type: str = "general") -> Path:
        """
        Get path for a chart file in the charts directory.
        
        Args:
            filename: Name of the chart file
            
        Returns:
            Path object for the chart file
        """
        chart_dir = cls.get_output_dir() / "charts"
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
    """
    if os.path.isabs(filename):
        file_path = Path(filename)
    else:
        if not filename.endswith((".xlsx", ".csv")):
            ext = ".xlsx" if format_type == "excel" else ".csv"
            filename = f"{filename}{ext}"
        file_path = PathManager.get_output_path(filename)
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if str(file_path).endswith(".xlsx"):
        df.to_excel(file_path, sheet_name=sheet_name, index=index)
    else:
        df.to_csv(file_path, index=index)
    
    return file_path


def load_dataframe(filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load a DataFrame from a file in the data directory.
    

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
    file_prefix: str = "energy_",
    file_suffix: str = ".csv"
) -> dict:
    """
    Recursively collect all energy CSV files under base_dir.
    Returns a nested dict: {model_name: {question_key: csv_path}}
    where question_key uniquely identifies each question (subject_qID_inTOKENS_outTOKENS)

    """
    import re
    from pathlib import Path
    energy_files = {}
    base_path = Path(base_dir)
    
    def parse_filename_for_question_key(filepath):
        """Parse filename to create unique question key"""
        filename = Path(filepath).stem
        
        # CPU decode pattern: energy_decode_synthetic_SUBJECT_qNUMBER_inNUMBER_outNUMBER_TIMESTAMP
        cpu_pattern = r'energy_decode_synthetic_(.+?)_(q\d+)_in(\d+)_(\d+tokens)_\d{8}_\d{6}'
        cpu_match = re.match(cpu_pattern, filename)
        
        if cpu_match:
            subject = cpu_match.group(1)
            question_id = cpu_match.group(2)
            input_tokens = cpu_match.group(3)
            output_tokens = cpu_match.group(4)
            return f"{subject}_{question_id}_in{input_tokens}_{output_tokens}", subject
        
        # Tegra pattern: energy_SUBJECT_qNUMBER_inNUMBER_outNUMBER_TIMESTAMP
        tegra_pattern = r'energy_(.+?)_(q\d+)_in(\d+)_out(\d+)_\d{8}_\d{6}'
        tegra_match = re.match(tegra_pattern, filename)
        
        if tegra_match:
            subject = tegra_match.group(1)
            question_id = tegra_match.group(2)
            input_tokens = tegra_match.group(3)
            output_tokens = tegra_match.group(4)
            return f"{subject}_{question_id}_in{input_tokens}_out{output_tokens}", subject
        
        return filename, "unknown"
    
    def read_model_name_from_json(csv_path):
        """Read clean model name from results JSON file in the same directory"""
        import json
        
        # Look for results_*.json files in the same directory as the CSV
        json_files = list(csv_path.parent.glob('results_*.json'))
        if json_files:
            try:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)
                    return data.get('model_name', None)
            except (json.JSONDecodeError, FileNotFoundError, KeyError):
                pass
        return None

    for csv_path in base_path.rglob(f"{file_prefix}*{file_suffix}"):
        try:
            question_key, subject = parse_filename_for_question_key(csv_path)
            model_name = read_model_name_from_json(csv_path)
            if not model_name:
                if csv_path.parent.parent.name in ['figure3', 'figure3']:
                    model_name = csv_path.parent.name
                elif csv_path.parent.name in ['figure3', 'figure3'] and len(csv_path.parts) >= 2:
                    model_name = csv_path.name.split('_')[1] if '_' in csv_path.name else csv_path.parent.name
                elif csv_path.parent.name.startswith('decode_synthetic_'):
                    dir_name = csv_path.parent.name
                    parts = dir_name.split('_')
                    if len(parts) >= 6:
                        model_parts = []
                        for i in range(3, len(parts)):
                            if parts[i].startswith('in') and parts[i][2:].isdigit():
                                break
                            model_parts.append(parts[i])
                        model_name = '_'.join(model_parts) if model_parts else "unknown_cpu_model"
                    else:
                        model_name = dir_name
                else:
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
            if question_key in energy_files[model_name]:
                print(f"Warning: Duplicate question '{question_key}' for model '{model_name}'. Skipping {csv_path}.")
                continue
            energy_files[model_name][question_key] = str(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path}: {e}")
    return energy_files
