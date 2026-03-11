"""
Project: European Electricity Exchange Analysis
Author: Tiernan Buckley
Year: 2026
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Source: https://github.com/INATECH-CIG/exchange_analysis

Description:
Connects to the ENTSO-E API to download, enforce structural symmetry on, 
and standardize raw generation, demand, and cross-border flow data.
"""

import pandas as pd
import requests
import logging
from io import StringIO
from typing import Dict, Optional, Any
from entsoe import EntsoePandasClient
from config import PipelineConfig
from utils import DataIO, safe_query, fill_gaps_wrapper, correct_zero_values

# Initialize logger
logger = logging.getLogger(__name__)

TIMEOUT = 60
GB_GENERATION_TYPES = [
    "Biomass", "Fossil Gas", "Fossil Hard coal", "Fossil Oil", 
    "Hydro Pumped Storage", "Hydro Run-of-river and poundage", 
    "Nuclear", "Other", "Solar", "Wind Offshore", "Wind Onshore"
]

# ==========================================
# GENERATION & DEMAND
# ==========================================
def download_generation_demand(client: EntsoePandasClient, config: PipelineConfig, io: DataIO) -> None:
    if not config.data_types["generation"]: return
    raw_dir = config.get_output_path("generation_demand_data_bidding_zones") / "raw"
    
    for bz in config.target_zones:
        logger.info(f"[Download] Gen/Demand for {bz}...")
        gen_df: Optional[pd.DataFrame] = None
        load_df: Optional[pd.DataFrame] = None

        if bz == "GB":
            try:
                gen_df = download_GB_per_type_data(config.start, config.end)
                load_df = download_GB_demand_data(config.start, config.end)
            except Exception as e: 
                logger.error(f"[Error] Failed to download GB data: {e}", exc_info=config.debug_mode)
        else:
            gen_df = safe_query(client.query_generation, context=f"Generation {bz}", country_code=bz, start=config.start, end=config.end, nett=True)
            load_df = safe_query(client.query_load, context=f"Load {bz}", country_code=bz, start=config.start, end=config.end)

        # Save raw downloaded data
        io.save(gen_df, raw_dir / f"{bz}_raw_generation.csv", "raw_generation", config, bz=bz)
        io.save(load_df, raw_dir / f"{bz}_raw_load.csv", "raw_load", config, bz=bz)

def process_generation_demand(config: PipelineConfig, io: DataIO) -> Dict[str, pd.DataFrame]:
    raw_dir = config.get_output_path("generation_demand_data_bidding_zones") / "raw"
    out_dir = config.get_output_path("generation_demand_data_bidding_zones")
    gaps_dir = config.get_gaps_path("generation_demand_data_bidding_zones")
    gen_storage_dict: Dict[str, pd.DataFrame] = {}

    for bz in config.zones:
        gen_path = raw_dir / f"{bz}_raw_generation.csv"
        load_path = raw_dir / f"{bz}_raw_load.csv"
        
        gen_df: Optional[pd.DataFrame] = io.load(gen_path, "raw_generation", config, bz=bz)
        load_df: Optional[pd.DataFrame] = io.load(load_path, "raw_load", config, bz=bz)
        
        if gen_df is None and load_df is None: continue

        if gen_df is not None:
            gen_df = gen_df.loc[:, ~gen_df.columns.duplicated()].apply(pd.to_numeric, errors='coerce').resample("1h").mean()
            gen_df = fill_gaps_wrapper(gen_df, gaps_dir, f"{bz}_gen", config=config, io=io, bz=bz)
            storage_cols = [c for c in ["Hydro Pumped Storage", "Energy storage"] if c in gen_df.columns]
            
            if storage_cols:
                storage_series = gen_df[storage_cols].fillna(0).sum(axis=1)
                gen_df = gen_df.drop(columns=storage_cols)
            else: storage_series = None

            gen_df = gen_df.clip(lower=0.0)
            gen_df["Generation"] = gen_df.sum(axis=1)
            gen_df["Storage Discharge"] = storage_series.clip(lower=0.0) if storage_series is not None else 0.0
            gen_df["Storage Charge"] = storage_series.clip(upper=0.0).abs() if storage_series is not None else 0.0

        if load_df is not None:
            load_df = load_df.apply(pd.to_numeric, errors='coerce').resample("1h").mean()
            load_df = fill_gaps_wrapper(load_df, gaps_dir, f"{bz}_load", config=config, io=io, bz=bz)

        if gen_df is not None:
            if load_df is not None:
                for k in ["Actual Load", "Load"]:
                    if k in load_df.columns: gen_df["Demand"] = load_df[k]
                gen_df["Total Generation"] = gen_df["Generation"] + gen_df["Storage Discharge"]
                gen_df["Total Load"] = gen_df.get("Demand", 0) + gen_df["Storage Charge"]
                gen_df["Net Export"] = gen_df["Total Generation"] - gen_df["Total Load"]
            
            gen_df = correct_zero_values(gen_df, gaps_dir, bz, config)
            
            # Save processed generation and demand dataset
            io.save(gen_df, out_dir / f"{bz}_generation_demand_data_bidding_zones.csv", "processed_generation", config, bz=bz)
            gen_storage_dict[bz] = gen_df

    return gen_storage_dict

# ==========================================
# FLOWS
# ==========================================
def download_flows(client: EntsoePandasClient, config: PipelineConfig, io: DataIO, flow_type: str = "commercial", dayahead: bool = False) -> None:
    if flow_type == "commercial" and not config.data_types.get(f"flows_commercial{'_dayahead' if dayahead else '_total'}"): return
    if flow_type == "physical" and not config.data_types.get("flows_physical"): return

    folder = "physical_flow_data_bidding_zones" if flow_type == "physical" else f"comm_flow_{'dayahead' if dayahead else 'total'}_bidding_zones"
    raw_dir = config.get_output_path(folder) / "raw"

    for bz in config.target_zones:
        logger.info(f"[Download] {flow_type} flows for {bz} (Dayahead={dayahead})...")

        flow_df: Optional[pd.DataFrame] = None
        for n in [z for z in config.neighbours_map[bz] if z in config.zones]:
            if flow_type == "commercial":
                f_out = safe_query(client.query_scheduled_exchanges, context=f"{bz}->{n}", country_code_from=bz, country_code_to=n, start=config.start, end=config.end, dayahead=dayahead)
                f_in = safe_query(client.query_scheduled_exchanges, context=f"{n}->{bz}", country_code_from=n, country_code_to=bz, start=config.start, end=config.end, dayahead=dayahead)
            else:
                f_out = safe_query(client.query_crossborder_flows, context=f"{bz}->{n}", country_code_from=bz, country_code_to=n, start=config.start, end=config.end)
                f_in = safe_query(client.query_crossborder_flows, context=f"{n}->{bz}", country_code_from=n, country_code_to=bz, start=config.start, end=config.end)

            if f_out is not None: flow_df = pd.concat([flow_df, f_out.loc[~f_out.index.duplicated()].to_frame(name=f"{bz}_{n}")], axis=1)
            if f_in is not None: flow_df = pd.concat([flow_df, f_in.loc[~f_in.index.duplicated()].to_frame(name=f"{n}_{bz}")], axis=1)

        if flow_df is not None: 
            table_name = f"raw_{flow_type}_flows" + ("_da" if dayahead else "")
            io.save(flow_df.loc[~flow_df.index.duplicated()], raw_dir / f"{bz}_raw_flows.csv", table_name, config, bz=bz)

def process_flows(config: PipelineConfig, io: DataIO, flow_type: str = "commercial", dayahead: bool = False) -> Dict[str, pd.DataFrame]:
    folder = "physical_flow_data_bidding_zones" if flow_type == "physical" else f"comm_flow_{'dayahead' if dayahead else 'total'}_bidding_zones"
    raw_dir, out_dir, gaps_dir = config.get_output_path(folder) / "raw", config.get_output_path(folder), config.get_gaps_path(folder)
    flow_dict: Dict[str, pd.DataFrame] = {}

    for bz in config.zones:
        table_name = f"raw_{flow_type}_flows" + ("_da" if dayahead else "")
        df: Optional[pd.DataFrame] = io.load(raw_dir / f"{bz}_raw_flows.csv", table_name, config, bz=bz)
        if df is None: continue

        df = fill_gaps_wrapper(df.resample("1h").mean(), gaps_dir, f"{bz}_flows", config=config, io=io, bz=bz, flow_type=flow_type, dayahead=dayahead)
        
        logger.info(f"[Process] {flow_type} flows for {bz}...")

        net_df = pd.DataFrame(index=df.index)
        for n in [z for z in config.neighbours_map[bz] if z in config.zones]:
            if f"{bz}_{n}" in df.columns and f"{n}_{bz}" in df.columns:
                net_df[f"{bz}_{n}_net_export"] = df[f"{bz}_{n}"] - df[f"{n}_{bz}"]
        
        net_df["Net Export"] = net_df.sum(axis=1)
        final_df = correct_zero_values(pd.concat([df, net_df], axis=1), gaps_dir, bz, config)
        
        filename = f"{bz}_comm_flow_{'dayahead' if dayahead else 'total'}_bidding_zones.csv" if flow_type == "commercial" else f"{bz}_physical_flow_data_bidding_zones.csv"
        processed_table = f"processed_{flow_type}_flows" + ("_da" if dayahead else "")
        io.save(final_df, out_dir / filename, processed_table, config, bz=bz)
        flow_dict[bz] = final_df

    return flow_dict

def balance_flows_symmetry(data_dict: Dict[str, pd.DataFrame], config: PipelineConfig, io: DataIO, flow_type: str = "commercial", dayahead: bool = False) -> Dict[str, pd.DataFrame]:
    logger.info(f"[Balance] Ensuring symmetry for {flow_type} flows...")
    folder = "physical_flow_data_bidding_zones" if flow_type == "physical" else f"comm_flow_{'dayahead' if dayahead else 'total'}_bidding_zones"
    gaps_dir, out_dir = config.get_gaps_path(folder), config.get_output_path(folder)

    for bz in data_dict:
        bz_trusted = (gaps_dir / f"{bz}_zeros.csv").exists()
        for n in config.neighbours_map[bz]:
            if n not in data_dict: continue
            
            col_out = f"{bz}_{n}"
            col_in = f"{n}_{bz}"
            
            # --- 1. RESOLVE DATA ASYMMETRY ---
            if col_out in data_dict[bz] and col_in in data_dict[bz] and col_out in data_dict[n] and col_in in data_dict[n]:
                if (data_dict[bz][col_out] - data_dict[n][col_out]).abs().sum() > 1e-3 or \
                   (data_dict[bz][col_in] - data_dict[n][col_in]).abs().sum() > 1e-3:
                    if bz_trusted:
                        data_dict[n][col_out] = data_dict[bz][col_out].values
                        data_dict[n][col_in] = data_dict[bz][col_in].values
                    else:
                        data_dict[bz][col_out] = data_dict[n][col_out].values
                        data_dict[bz][col_in] = data_dict[n][col_in].values
            
            # --- 2. CALCULATE NET EXPORTS ---
            if col_out in data_dict[bz] and col_in in data_dict[bz]:
                data_dict[bz][f"{col_out}_net_export"] = data_dict[bz][col_out] - data_dict[bz][col_in]
            if col_out in data_dict[n] and col_in in data_dict[n]:
                data_dict[n][f"{col_in}_net_export"] = data_dict[n][col_in] - data_dict[n][col_out]

    # --- 3. RECALCULATE TOTAL NET EXPORT ---
    for bz, df in data_dict.items():
        net_cols = [c for c in df.columns if "net_export" in c and c != "Net Export"]
        if net_cols: df["Net Export"] = df[net_cols].sum(axis=1)
        
        suffix = f"_{folder}.csv"
        processed_table = f"balanced_{flow_type}_flows" + ("_da" if dayahead else "")
        io.save(df, out_dir / f"{bz}{suffix}", processed_table, config, bz=bz)

    return data_dict

# ==========================================
# GB HELPERS (BMRS API)
# ==========================================
def download_GB_per_type_data(start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Fetches GB generation mix from BMRS."""
    all_days = pd.date_range(start=start.tz_convert("UTC"), end=end.tz_convert("UTC"), normalize=True, freq='D', tz="UTC")
    df_list = []
    for date in all_days:
        try:
            df_new = _download_GB_per_type_data(date)
            if df_new is not None: df_list.append(df_new)
        except Exception as e:
            logger.warning(f"Failed to fetch GB Gen for {date.date()}: {e}")
    return pd.concat(df_list).loc[start:end] if df_list else None

def download_GB_demand_data(start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Fetches GB demand from BMRS."""
    all_days = pd.date_range(start=start.tz_convert("UTC"), end=end.tz_convert("UTC"), normalize=True, freq='D', tz="UTC")
    df_list = []
    for date in all_days:
        try:
            df_new = _download_GB_demand_data(date)
            if df_new is not None: df_list.append(df_new)
        except Exception as e:
            logger.warning(f"Failed to fetch GB Demand for {date.date()}: {e}")
    return pd.concat(df_list).loc[start:end] if df_list else None

def _download_GB_per_type_data(date: pd.Timestamp) -> Optional[pd.DataFrame]:
    range_start = pd.Timestamp(f"{date.strftime('%Y-%m-%d')} 00:00", tz="UTC")
    range_end = pd.Timestamp(f"{date.strftime('%Y-%m-%d')} 23:30", tz="UTC")
    date_range = pd.date_range(start=range_start, end=range_end, freq="30min")

    df = pd.DataFrame(index=date_range, columns=GB_GENERATION_TYPES)
    url = f"https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type?from={date.strftime('%Y-%m-%d')}T00:00&to={date.strftime('%Y-%m-%d')}T23:30&format=json"

    response = requests.get(url, timeout=TIMEOUT)
    if response.status_code == 200:
        response_json = response.json()
        if len(response_json["data"]) > 0:
            data = pd.DataFrame(response_json["data"])
            data.set_index("startTime", inplace=True)
            data.index = pd.to_datetime(data.index)
            
            for i, ttype in enumerate(sorted(GB_GENERATION_TYPES)):
                data[ttype] = [sorted(data["data"].values[x], key=lambda d: d['psrType'])
                            [i]["quantity"] for x in range(len(data["data"].values))]
                df.loc[data.index, ttype] = data[ttype].values
    return df

def _download_GB_demand_data(date: pd.Timestamp) -> Optional[pd.DataFrame]:
    range_start = pd.Timestamp(f"{date.strftime('%Y-%m-%d')} 00:00", tz="UTC")
    range_end = pd.Timestamp(f"{date.strftime('%Y-%m-%d')} 23:30", tz="UTC")
    date_range = pd.date_range(start=range_start, end=range_end, freq="30min")

    df = pd.DataFrame(index=date_range, columns=["Actual Load"])
    url = f"https://data.elexon.co.uk/bmrs/api/v1/demand/actual/total?from={date.strftime('%Y-%m-%d')}T00:00&to={date.strftime('%Y-%m-%d')}T23:30&format=csv"
    
    response = requests.get(url, timeout=TIMEOUT)
    if response.status_code == 200:
        data = pd.read_csv(StringIO(response.text))
        data.sort_values(by=['StartTime'], inplace=True)
        data.set_index(["StartTime"], inplace=True)
        data.index = pd.to_datetime(data.index)
        df.loc[data.index, "Actual Load"] = data["Quantity"].values
    return df

def fetch_simple_metrics(client: EntsoePandasClient, config: PipelineConfig, io: DataIO) -> None:
    """Fetches Prices and Net Positions for Target Zones."""
    if not config.data_types["metrics"]: return
    
    for name, method, kwargs in [
        ("net_positions_dayahead", client.query_net_position, {"dayahead":True}), 
        ("market_price_dayahead", client.query_day_ahead_prices, {})
    ]:
        out_dir = config.get_output_path(name)
        
        for bz in config.target_zones:
            logger.info(f"[Download] Fetching {name} for {bz}...")
            df: Optional[pd.DataFrame] = safe_query(method, context=f"{name} {bz}", country_code=bz, start=config.start, end=config.end, **kwargs)
            
            if df is not None:
                if isinstance(df, pd.Series): 
                    df = df.to_frame(name="Value")
                
                df.index = pd.to_datetime(df.index, utc=True)
                
                if name == "net_positions_dayahead" and bz.startswith("IT"):
                    mask_2025 = df.index.year == 2025
                    if mask_2025.any():
                        logger.info(f"  -> Adjusting sign convention for {bz} 2025 Net Positions.")
                        df.loc[mask_2025] = df.loc[mask_2025] * -1
                
                # Note: Currently saving to CSV direct, but could be passed to io.save
                df.resample("1h").mean().to_csv(out_dir / f"{bz}_{name}.csv")