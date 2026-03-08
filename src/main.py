"""
Project: European Electricity Exchange Analysis
Author: Tiernan Buckley
Year: 2026
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Source: https://github.com/INATECH-CIG/exchange_analysis

Description:
Orchestrates the execution of the entire data pipeline, acting as the main control panel 
to trigger downloading, processing, analyzing, and aggregating the grid data.
"""

import sys
from datetime import datetime
from entsoe import EntsoePandasClient
from config import PipelineConfig
from utils import start_logging

# --- MODULE IMPORTS ---
from download_data import (
    download_generation_demand,
    process_generation_demand,
    download_flows,
    process_flows,
    balance_flows_symmetry,
    fetch_simple_metrics
)
from data_analysis import (
    perform_decomposition_analysis, 
    perform_aggregated_flow_tracing,
    perform_direct_flow_tracing,
    perform_pooling_analysis,
    perform_post_processing_aggregation
)

def main():
    # ==========================================
    # CONTROL PANEL
    # ==========================================
    
    # 1. Execution Flags (True = Run this step)
    my_run_flags = {
        "download": False,
        "process": False,
        "analysis": True,
        "post_processing": True,
    }

    analysis_subset = {
        "zone_to_gen_type_analysis": True,
        "ac_flow_tracing_analysis": True,
        "dc_flow_tracing_analysis": True,
        "pooling_analysis": True,
    }
    
    # 2. Define Period (UTC)
    period = ("2026-03-04 00:00", "2026-03-04 23:59") 

    # 3. Define I/O Settings (Storage & Loading)
    my_io_settings = {
        "save_csv": False,       # Save outputs locally as CSVs
        "save_db": True,        # Push outputs to the TimescaleDB server
        "load_source": "db"     # 'csv' or 'db' - where to read data from in Process/Analysis steps
    }

    # 4. Optional: Download only Subsets of Data (Uncomment to use)
    # -------------------------------------------------------
    # selected_bzs = ["DE_LU", "FR", "GB"] 
    #
    # selected_data_types = {
    #     "generation": True,
    #     "flows_commercial_total": True,
    #     "flows_commercial_dayahead": True, # Download only
    #     "flows_physical": True,
    #     "metrics": False
    # }
    # -------------------------------------------------------

    # 5. Initialize Config
    config = PipelineConfig(
        date_range=period,
        run_flags=my_run_flags,
        io_settings=my_io_settings,
        analysis_flags=analysis_subset,
        # Uncomment below to apply subsets:
        # target_zones=selected_bzs,
        # data_types=selected_data_types
    )
    
    # 6. Setup Logging
    timestamp = datetime.now().strftime("%Y-%m-%d")
    timestamp_detailed = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    start_logging(config.project_root / "logs" / f"log_{timestamp}" / f"log_{timestamp_detailed}.log")

    # ==========================================
    # PIPELINE EXECUTION
    # ==========================================

    # --- PHASE 1: DOWNLOAD ---
    if config.run_phases["download"]:
        print(f"=== STARTING DOWNLOAD ({config.start} to {config.end}) ===")
        client = EntsoePandasClient(api_key=config.api_key)
        
        download_generation_demand(client, config)
        download_flows(client, config, "commercial", dayahead=False)
        download_flows(client, config, "commercial", dayahead=True)
        download_flows(client, config, "physical")
        fetch_simple_metrics(client, config)

    # --- PHASE 2: PROCESS ---
    gen_data, final_comm, final_phys = None, None, None
    if config.run_phases["process"]:
        print("\n=== STARTING PROCESSING ===")
        
        # A. Generation & Demand (Clean, Resample, Patch Gaps)
        gen_data = process_generation_demand(config)
        
        # B. Commercial Flows (Total) -> Balance & Keep
        raw_comm = process_flows(config, "commercial", dayahead=False)
        final_comm = balance_flows_symmetry(raw_comm, config, "commercial", dayahead=False)
        
        # C. Day Ahead Flows -> Balance & Save (discard memory)
        raw_da = process_flows(config, "commercial", dayahead=True)
        balance_flows_symmetry(raw_da, config, "commercial", dayahead=True)
        
        # D. Physical Flows -> Balance & Keep
        raw_phys = process_flows(config, "physical")
        final_phys = balance_flows_symmetry(raw_phys, config, "physical")

    # --- PHASE 3: ANALYSIS ---
    if config.run_phases["analysis"]:
        print("\n=== STARTING ANALYSIS ===")

        # 1. Neighbor Decomposition (Import Mix based on neighbors)
        if config.analysis_flags["zone_to_gen_type_analysis"]:
            perform_decomposition_analysis(config, gen_dfs=gen_data, comm_dfs=final_comm)
        
        # 2. Flow Tracing (Matrix Inversion)
        if config.analysis_flags["ac_flow_tracing_analysis"]:
            perform_aggregated_flow_tracing(config, gen_dfs=gen_data, phys_flow_dfs=final_phys)
        if config.analysis_flags["dc_flow_tracing_analysis"]:
            perform_direct_flow_tracing(config, gen_dfs=gen_data, phys_flow_dfs=final_phys)
        
        # 3. Pooling (European Mix)
        if config.analysis_flags["pooling_analysis"]:
            perform_pooling_analysis(config, gen_dfs=gen_data, comm_dfs=final_comm, phys_flow_dfs=final_phys)
    
    if config.run_phases["post_processing"]:
        # 4. Aggregation (Annual Totals)
        perform_post_processing_aggregation(config)

if __name__ == "__main__":
    main()