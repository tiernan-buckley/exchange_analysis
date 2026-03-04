"""
Project: European Electricity Exchange Analysis
Author: Tiernan Buckley
Year: 2026
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Source: https://github.com/INATECH-CIG/exchange_analysis

Description:
Central Configuration Class for the Exchange Analysis Pipeline.
Manages temporal boundaries, execution flags, I/O routing, and spatial (zonal) constraints.
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from mappings_alt import NEIGHBOURS

class PipelineConfig:
    def __init__(
        self, 
        date_range: Tuple[str, str],
        key_file: str = "keys.yaml", 
        run_flags: Optional[Dict[str, bool]] = None,
        target_zones: Optional[List[str]] = None,
        data_types: Optional[Dict[str, bool]] = None,
        io_settings: Optional[Dict[str, Any]] = None,
        analysis_flags: Optional[Dict[str, bool]] = None
    ):
        # ==========================================
        # DIRECTORY MAPPING
        # ==========================================
        # Path(__file__).parent is the 'src' directory.
        # .parent.parent goes up one level to the main project root.
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "outputs"
        self.input_dir = self.project_root / "inputs"
        
        # ==========================================
        # TEMPORAL BOUNDARIES
        # ==========================================
        self.start = pd.Timestamp(date_range[0], tz="UTC")
        raw_end = pd.Timestamp(date_range[1], tz="UTC")
        
        # --- TIME BOUNDARY CORRECTION ---
        # Shifts exact midnight end-dates back by 1 minute to prevent dangling 
        # indices that fall outside the API's inclusive hourly blocks.
        if raw_end.hour == 0 and raw_end.minute == 0:
            self.end = raw_end - pd.Timedelta(minutes=1)
            print(f"[Config] Adjusted end date from {raw_end} to {self.end} for inclusive indexing.")
        else:
            self.end = raw_end
        
        self.time_index = pd.date_range(start=self.start, end=self.end, freq="1h")
        self.year = self.start.year 
        
        # ==========================================
        # PIPELINE ORCHESTRATION FLAGS
        # ==========================================
        self.run_phases = {
            "download": True, 
            "process": True, 
            "analysis": True, 
            "post_processing": True
        }
        if run_flags: self.run_phases.update(run_flags)

        self.analysis_flags = {
            "zone_to_gen_type_analysis": True,
            "ac_flow_tracing_analysis": True,
            "dc_flow_tracing_analysis": True,
            "pooling_analysis": True,
        }
        if analysis_flags: self.analysis_flags.update(analysis_flags)

        # ==========================================
        # DATA I/O ROUTING
        # ==========================================
        self.save_csv = True
        self.save_db = True
        self.load_source = 'csv' # Options: 'csv' or 'db'
        
        if io_settings:
            self.save_csv = io_settings.get("save_csv", self.save_csv)
            self.save_db = io_settings.get("save_db", self.save_db)
            self.load_source = io_settings.get("load_source", self.load_source)

        # ==========================================
        # API DOWNLOAD FILTERS
        # ==========================================
        self.data_types = {
            "generation": True,
            "flows_commercial_total": True,
            "flows_commercial_dayahead": True,
            "flows_physical": True,
            "metrics": True
        }
        if data_types: self.data_types.update(data_types)

        # ==========================================
        # SPATIAL CONFIGURATION (ZONES & TOPOLOGY)
        # ==========================================
        self.all_zones = list(NEIGHBOURS.copy().keys())
        self.neighbours_map = NEIGHBOURS.copy()
        self._filter_zones()

        if target_zones:
            self.target_zones = [z for z in target_zones if z in self.all_zones]
            print(f"Configured for subset of zones: {self.target_zones}")
        else:
            self.target_zones = self.all_zones

        # ==========================================
        # CREDENTIALS & METADATA
        # ==========================================
        # Looks for keys.yaml in the main project root
        with open(self.project_root / key_file, "r") as f:
            self.api_key = yaml.safe_load(f)["entsoe-key"]
            
        self.gen_types_df = pd.read_csv(
            self.input_dir / "generation_data/gen_types_and_emission_factors.csv"
        )
        self.gen_types_list = self.gen_types_df["entsoe"].tolist()

    def _filter_zones(self):
        """Removes predefined structural anomalies from the active topology."""
        to_remove = ["DE_AT_LU", "IE_SEM", "IE", "NIE", "MT", 
                     "IT", "IT_BRNN", "IT_ROSN", "IT_FOGN"]
        for z in to_remove:
            if z in self.all_zones: self.all_zones.remove(z)
            if z in self.neighbours_map: del self.neighbours_map[z]

    @property
    def zones(self):
        return self.all_zones

    # ==========================================
    # PATH GENERATORS
    # ==========================================
    def get_output_path(self, subfolder: str) -> Path:
        """Constructs and guarantees the existence of temporal-partitioned (yearly) output directories."""
        path = self.output_dir / subfolder / str(self.year)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_gaps_path(self, subfolder: str) -> Path:
        """Constructs and guarantees the existence of directories for data quality audit logs."""
        path = self.output_dir / subfolder / str(self.year) / "gaps"
        path.mkdir(parents=True, exist_ok=True)
        return path