"""
Project: European Electricity Exchange Analysis
Author: Tiernan Buckley
Year: 2026
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Source: https://github.com/INATECH-CIG/exchange_analysis

Description:
Manages robust database and CSV file I/O operations, handles system logging,
and executes heuristics-based gap filling for missing or anomalous time-series
data.
"""

import time
import pandas as pd
import numpy as np
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Callable, Union
from sqlalchemy import create_engine, text, inspect

from config import get_db_engine

# Initialize the logger for this specific module
logger = logging.getLogger(__name__)

# ==========================================
# DATA I/O HANDLER
# ==========================================
class DataIO:
    """
    Centralized Data Input/Output Handler.
    Orchestrates dual-writing to local flat CSV files and a TimescaleDB
    instance.
    Features idempotent database writes and dynamic schema evolution.
    """
    def __init__(self, config: Any) -> None:
        """
        Initialize the engine only if the configuration requires database access.
        """
        self.save_db = getattr(config, 'save_db', False)
        self.load_source = getattr(config, 'load_source', 'csv')

        # Only create the engine if we actually need it
        if self.save_db or self.load_source == 'db':
            self.engine = get_db_engine()
        else:
            self.engine = None
            logger.info("[IO] Running in CSV-only mode. No database engine initialized.")

    def save(self, df: Optional[Union[pd.DataFrame, pd.Series]], filepath: Path, table_name: str, config: Any, bz: Optional[str] = None) -> None:
        """
        Persists dataframe to configured storage backends.
        Automatically injects bidding zone identifiers and evolves DB schema
        if new columns are detected.
        """
        if df is None or df.empty: return

        # 1. Local Flat File Storage (CSV)
        if getattr(config, 'save_csv', True):
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath)

        # 2. Relational Database Storage (TimescaleDB / PostgreSQL)
        if getattr(config, 'save_db', True):
            clean_table = table_name.lower().replace("-", "_").replace(" ", "_")[:63]
            df_db = df.to_frame() if isinstance(df, pd.Series) else df.copy()

            if bz is not None:
                df_db["bidding_zone"] = bz

                # --- IDEMPOTENT WRITE LOGIC ---
                # Prevents duplicate rows by deleting overlapping time ranges for
                # the specific bidding zone before inserting the freshly processed
                # data.
                if isinstance(df.index, pd.DatetimeIndex):
                    min_time = df.index.min().strftime('%Y-%m-%d %H:%M:%S%z')
                    max_time = df.index.max().strftime('%Y-%m-%d %H:%M:%S%z')
                    index_col = df.index.name or 'index'

                    delete_query = text(f"""
                        DELETE FROM {clean_table}
                        WHERE bidding_zone = '{bz}'
                        AND "{index_col}" >= '{min_time}'
                        AND "{index_col}" <= '{max_time}'
                    """)
                else:
                    # Handles non-time-series aggregated tables (e.g. annual totals)
                    delete_query = text(f"""
                        DELETE FROM {clean_table}
                        WHERE bidding_zone = '{bz}'
                    """)

                with self.engine.begin() as conn:
                    try:
                        conn.execute(delete_query)
                    except Exception:
                        pass # Table doesn't exist yet; safe to proceed to insertion

            # --- DYNAMIC SCHEMA EVOLUTION ---
            # Automatically detects and appends new columns to the SQL table if
            # the data structure changes
            try:
                inspector = inspect(self.engine)
                if inspector.has_table(clean_table):
                    existing_cols = [col['name'] for col in inspector.get_columns(clean_table)]
                    new_cols = [c for c in df_db.columns if c not in existing_cols]

                    if new_cols:
                        with self.engine.begin() as conn:
                            for c in new_cols:
                                col_type = "TEXT" if c == "bidding_zone" else "DOUBLE PRECISION"
                                conn.execute(text(f'ALTER TABLE {clean_table} ADD COLUMN "{c}" {col_type}'))
            except Exception as e:
                logger.warning(f"[DB Schema Warning] Could not auto-evolve schema for {clean_table}: {e}")

            # Append new data to table
            try:
                df_db.to_sql(clean_table, self.engine, if_exists="append")
            except Exception as e:
                logger.error(f"[DB Error] Failed to save {clean_table} to database: {e}")

    def load(self, filepath: Path, table_name: str, config: Any, bz: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Retrieves data from the configured primary source (DB or CSV).
        Automatically slices the retrieved data to match the strict time
        boundaries defined in the config.
        """
        source = getattr(config, 'load_source', 'csv')

        start_str = config.start.strftime('%Y-%m-%d %H:%M:%S%z')
        end_str = config.end.strftime('%Y-%m-%d %H:%M:%S%z')

        # Primary Load: Database
        if source == 'db':
            clean_table = table_name.lower().replace("-", "_").replace(" ", "_")[:63]
            try:
                base_query = f'SELECT * FROM {clean_table} WHERE "index" >= \'{start_str}\' AND "index" <= \'{end_str}\''
                query = f"{base_query} AND bidding_zone = '{bz}'" if bz is not None else base_query

                df = pd.read_sql(text(query), self.engine)

                if df.empty:
                    raise ValueError(f"No data found in DB for {clean_table} (bz={bz})")

                # Reconstruct standardized datetime index
                index_col = str(df.columns[0])
                df.set_index(index_col, inplace=True)
                df.index = pd.to_datetime(df.index, utc=True)
                df.index.name = None

                if bz is not None:
                    if "bidding_zone" in df.columns:
                        df = df.drop(columns=["bidding_zone"])
                else:
                    # If bz is None (Global Load), we keep the column
                    pass

                df.dropna(axis=1, how='all', inplace=True)
                return df

            except Exception as e:
                logger.warning(f"[DB Warning] Falling back to CSV for {clean_table} (bz={bz}). Reason: {e}")

        # Fallback Load: CSV
        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)

            # Enforce time range boundaries
            mask = (df.index >= config.start) & (df.index <= config.end)
            return df.loc[mask]

        return None

# ==========================================
# LOGGING UTILS
# ==========================================

def setup_logging(log_file_path: Path, log_level_str: str, debug_mode: bool) -> None:
    """
    Configures the root logger. 
    Routes logs to both the console (stdout) and a persistent log file.
    """
    # 1. Safely convert string "INFO" to the numeric logging.INFO
    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # 2. Define how the logs look
    # Debug mode gets exact line numbers and module names.
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s' if not debug_mode 
        else '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s'
    )

    # 3. Configure the Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to prevent duplicate printouts
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 4. Console Handler (Terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 5. File Handler (.log file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # We use root_logger explicitly here because this is the setup function itself
    root_logger.info(f"Logging initialized. Level: {log_level_str} | Debug Mode: {debug_mode}")
    
# ==========================================
# API UTILS
# ==========================================
def safe_query(func: Callable, max_retries: int = 3, delay: int = 2, context: Optional[str] = None, **kwargs: Any) -> Any:
    """Wrapper to handle intermittent API failures and timeouts smoothly."""
    for attempt in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            # 1. Forensic Extraction: Try to get the real error message
            error_msg = str(e)
            
            # If standard string conversion is blank, fallback to the raw object representation
            if not error_msg.strip():
                error_msg = repr(e)
                
            # If the error contains an HTTP response (common with ENTSO-E timeouts or 400s)
            if hasattr(e, 'response') and e.response is not None:
                # Append the raw XML/JSON returned by the ENTSO-E server
                error_msg += f" | API Response: {e.response.text}"

            # 2. Format the warning message
            msg = f"[Attempt {attempt + 1}/{max_retries}] Failed"
            if context: msg += f" for {context}"
            msg += f": {error_msg}"
            
            # Use WARNING for retries. 
            logger.warning(msg)

            # 3. Handle known empty data responses gracefully
            if "NoMatchingDataError" in error_msg: 
                logger.warning(f"Data gap detected for {context}: Source returned empty.")
                return None

            # 4. Retry Logic
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                # Use ERROR if it completely fails after all retries, and include the full stack trace!
                logger.error(
                    f"CRITICAL FAILURE: Skipping {context if context else 'query'} after max retries.", 
                    exc_info=True 
                )
                return None
    return None

# ==========================================
# GAP FILLING ENGINE
# ==========================================

def default_rules(series: pd.Series, gaps: pd.DataFrame, inferred_freq: pd.Timedelta) -> None:
    """
    Defines heuristics for time-series imputation based on gap duration and
    location.
    - Gaps <= 3 hours: Linearly interpolated or forward/backward filled at
      edges.
    - Gaps <= 1 week: Replaced with exact historical data from the previous
      week to preserve daily profiles.
    - Unmatched: Defaults to ZERO.
    """
    gaps["method"] = "ZERO"

    # Heuristic 1: Historical Profile Substitution (Week-over-week)
    MAX_WEEK_BEFORE = pd.Timedelta(weeks=1)
    gaps.loc[
        (gaps["type"] == "nan") & (gaps["duration"] * inferred_freq <= MAX_WEEK_BEFORE) &
        (gaps["start"] - series.index[0] >= MAX_WEEK_BEFORE), "method",
    ] = "WEEK_BEFORE"

    # Heuristic 2: Short-duration edge and internal interpolation
    MAX_LINEAR = pd.Timedelta(hours=3)
    gaps.loc[
        (gaps["type"] == "nan") & (gaps["duration"] * inferred_freq <= MAX_LINEAR) &
        (gaps["start"] > series.index[0]) & (gaps["end"] < series.index[-1]), "method",
    ] = "LINEAR"

    gaps.loc[
        (gaps["type"] == "nan") & (gaps["duration"] * inferred_freq <= MAX_LINEAR) &
        (gaps["start"] > series.index[0]) & (gaps["end"] == series.index[-1]), "method",
    ] = "FORWARD_FILL"

    gaps.loc[
        (gaps["type"] == "nan") & (gaps["duration"] * inferred_freq <= MAX_LINEAR) &
        (gaps["start"] == series.index[0]) & (gaps["end"] < series.index[-1]), "method",
    ] = "BACKWARD_FILL"

def fill_gaps_series(series: pd.Series, gaps: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Executes the imputation strategies defined by the gap filling heuristics."""
    gaps["success"] = False
    gaps["filled_values"] = 0
    gaps["filled_quantity"] = 0.0

    for i, gap in gaps.iterrows():
        start, end, duration, method = gap["start"], gap["end"], gap["duration"], gap["method"]
        if method == "ZERO":
            series.loc[start:end] = 0
        elif method == "LINEAR":
            pos_start = series.index.get_loc(start)
            series.loc[start:end] = np.linspace(series.iloc[pos_start - 1], series.iloc[pos_start + duration], duration + 2)[1:-1]
        elif method == "FORWARD_FILL":
            series.loc[start:end] = series.iloc[series.index.get_loc(start) - 1]
        elif method == "BACKWARD_FILL":
            series.loc[start:end] = series.iloc[series.index.get_loc(start) + duration]
        elif method == "WEEK_BEFORE":
            one_week = pd.Timedelta(weeks=1)
            series.loc[start:end] = series.loc[(start - one_week):(end - one_week)].values

        gaps.loc[i, "success"] = series.loc[start:end].count() > 0
        gaps.loc[i, "filled_values"] = series.loc[start:end].count()
        gaps.loc[i, "filled_quantity"] = series.loc[start:end].sum()

    return series, gaps

def find_gaps_series(
    series: pd.Series,
    output_dict: Optional[Dict[str, pd.DataFrame]] = None,
    check_negatives: bool = False,
    allow_negatives: Optional[List[str]] = None,
    fill_gaps: bool = False,
    gap_filling_rules: Optional[Callable] = None
) -> pd.Series:
    """Identifies continuous blocks of missing or anomalous data in a single time-series vector."""
    if allow_negatives is None: allow_negatives = []

    series = series.where(series < 100000, np.nan) # Filter unrealistic physical outliers
    is_nan = series.isna()
    gap_starts = is_nan & (~is_nan.shift(1, fill_value=False))
    gap_ends = is_nan & (~is_nan.shift(-1, fill_value=False))

    gaps = pd.DataFrame({"start": series[gap_starts].index, "end": series[gap_ends].index})
    gaps["duration"] = gaps.apply(lambda row: is_nan[row["start"] : row["end"]].sum(), axis=1, result_type="reduce").astype("int")
    gaps["value"], gaps["type"] = np.nan, "nan"

    if check_negatives and (str(series.name) not in allow_negatives):
        is_neg = series < 0
        neg_starts = is_neg & (~is_neg.shift(1, fill_value=False))
        neg_ends = is_neg & (~is_neg.shift(-1, fill_value=False))
        negs = pd.DataFrame({"start": series[neg_starts].index, "end": series[neg_ends].index})
        negs["duration"] = negs.apply(lambda row: is_neg[row["start"] : row["end"]].sum(), axis=1, result_type="reduce").astype("int")
        negs["value"] = negs.apply(lambda row: series[row["start"] : row["end"]].sum(), axis=1, result_type="reduce")
        negs["type"] = "negative"
        gaps = pd.concat([gaps, negs]).sort_values(by="start").reset_index(drop=True)

    inferred_freq = pd.infer_freq(series.index[:3])
    if (inferred_freq is not None) and (len(inferred_freq) == 1): inferred_freq = "1" + inferred_freq
    freq_td = pd.to_timedelta(inferred_freq) if inferred_freq else pd.Timedelta(hours=1)
    gaps["method"] = "UNDEFINED"

    if gap_filling_rules is not None: gap_filling_rules(series, gaps, freq_td)
    if fill_gaps: series, gaps = fill_gaps_series(series, gaps)
    if output_dict is not None: output_dict[str(series.name)] = gaps
    return series

def find_gaps(
    df: pd.DataFrame,
    check_negatives: bool = False,
    allow_negatives: Optional[List[str]] = None,
    fill_gaps: bool = False,
    gap_filling_rules: Callable = default_rules
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Applies the gap finding and filling routines across an entire DataFrame."""
    if allow_negatives is None: allow_negatives = []
    output_dict: Dict[str, pd.DataFrame] = {}
    df_result = df.apply(find_gaps_series, axis=0, output_dict=output_dict, check_negatives=check_negatives,
                         allow_negatives=allow_negatives, fill_gaps=fill_gaps, gap_filling_rules=gap_filling_rules)
    return df_result, output_dict

def patch_gaps_with_dayahead(
    flow_df: pd.DataFrame,
    gap_dict: Dict[str, pd.DataFrame],
    bz: str,
    neighbour: str,
    config: Any, 
    io: DataIO,
    min_gap_length: pd.Timedelta = pd.Timedelta(weeks=1)
) -> pd.DataFrame:
    """
    Data-source bridging logic:
    For massive gaps in realized Physical/Commercial flows, falls back to
    Day-Ahead market schedules via the io utility (supporting Server/DB/CSV)
    to preserve matrix continuity.
    """
    # 1. Identify which columns have gaps exceeding the threshold
    long_gaps: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for col in [f"{bz}_{neighbour}", f"{neighbour}_{bz}"]:
        if col in gap_dict:
            for _, row in gap_dict[col].iterrows():
                if (row["end"] - row["start"]) > min_gap_length:
                    long_gaps.append((col, row["start"], row["end"]))

    if not long_gaps:
        return flow_df

    # 2. Define the metadata for the Day-Ahead proxy data
    # These match the folder and table naming logic in process_flows()
    folder = "comm_flow_dayahead_bidding_zones"
    filename = f"{bz}_comm_flow_dayahead_bidding_zones.csv"
    path = config.get_output_path(folder) / filename
    table_name = "processed_commercial_flows_da"

    # 3. Load via unified IO (this handles Server vs CSV automatically)
    da_df = io.load(path, table_name, config, bz=bz)

    if da_df is None or da_df.empty:
        return flow_df

    # 4. Patch the gaps
    patched_count = 0
    for col, start, end in long_gaps:
        # Check if the specific neighbor column exists in the proxy data
        if col in da_df.columns:
            # Slice the proxy data for the gap duration
            replacement = da_df.loc[start:end, col]

            if not (replacement.empty or replacement.isna().all()):
                flow_df.loc[start:end, col] = replacement
                patched_count += 1

    if patched_count > 0:
        logger.info(f"   -> [Patch] Used {table_name} to fill {patched_count} long-duration gaps for {bz}.")

    return flow_df

def fill_gaps_wrapper(
    df: pd.DataFrame,
    gaps_dir: Optional[Path],
    prefix: str,
    config: Optional[Any] = None,
    io: DataIO = None,
    bz: Optional[str] = None,
    flow_type: Optional[str] = None,
    dayahead: bool = False
) -> pd.DataFrame:
    """Orchestrates the entire anomaly detection and gap imputation lifecycle."""
    if df.empty: return df
    _, gaps_dict = find_gaps(df, check_negatives=False, fill_gaps=False)

    if config and bz and (flow_type == "commercial") and (not dayahead):
        if hasattr(config, 'neighbours_map') and bz in config.neighbours_map:
            for neighbour in [n for n in config.neighbours_map[bz] if f"{bz}_{n}" in df.columns]:
                df = patch_gaps_with_dayahead(df, gaps_dict, bz, neighbour, config, io)

    df_filled, _ = find_gaps(df, check_negatives=False, fill_gaps=True, gap_filling_rules=default_rules)

    if gaps_dir:
        for key, gap_df in gaps_dict.items():
            file_path = gaps_dir / f"{prefix}_{str(key).replace('/', '_').replace(' ', '_')}_gaps.csv"
            if not gap_df.empty:
                gap_df.to_csv(file_path)
            else:
                # Maintain state consistency by cleaning up obsolete audit logs
                # from previous runs if the incoming data quality has improved.
                if file_path.exists():
                    file_path.unlink()

    return df_filled

def correct_zero_values(df: pd.DataFrame, gaps_dir: Path, bz: str, config: Any) -> pd.DataFrame:
    """
    Corrects completely 'dead' rows (e.g., API returns exactly 0.0 for total
    generation) by substituting the generation profile from exactly one week
    prior.
    """
    if df.empty: return df

    if "Total Generation" in df.columns:
        zeros_mask = df["Total Generation"] == 0
    else:
        zeros_mask = (df.select_dtypes(include=[np.number]) != 0).sum(axis=1) == 0

    zeros_df = df[zeros_mask]
    file_path = gaps_dir / f"{bz}_zeros.csv"

    if len(zeros_df) > 0:
        one_week = pd.Timedelta(weeks=1)
        range_start = config.start

        for timestamp in zeros_df.index:
            patch_time = timestamp - one_week
            # If the required historical timestamp falls outside the downloaded
            # bounds, look forward instead
            if patch_time < range_start: patch_time = timestamp + one_week
            if patch_time in df.index:
                df.loc[timestamp] = df.loc[patch_time]

        # Log interventions for data-quality auditing
        zeros_df.to_csv(file_path)
    else:
        # Maintain state consistency by cleaning up obsolete audit logs
        # from previous runs if the incoming data quality has improved.
        if file_path.exists():
            file_path.unlink()

    return df