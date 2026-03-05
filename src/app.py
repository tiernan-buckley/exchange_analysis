import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import json
import numpy as np
from pathlib import Path
from entsoe.geo.utils import load_zones

# Direct imports from your project
from mappings_alt import NEIGHBOURS
from utils import io

FLOW_TYPES = {
    "Physical": {
        "table": "processed_physical_flows", 
        "folder": "physical_flow_bidding_zones"
    },
    "Commercial Total": {
        "table": "processed_commercial_flows", 
        "folder": "comm_flow_total_bidding_zones"
    },
    "Commercial Day-Ahead": {
        "table": "processed_commercial_flows_da", 
        "folder": "comm_flow_dayahead_bidding_zones"
    }
}

# ==========================================
# 0. MOCK CONFIG (Optimized for 24h Windows)
# ==========================================
class MockConfig:
    """A minimal object to keep io.load() happy and fast."""
    def __init__(self, selected_date):
        # selected_date is a datetime.date object from the sidebar
        self.year = selected_date.year
        self.output_dir = Path(__file__).parent.parent / "outputs"
        self.load_source = 'db' # Primary source is PostgreSQL
        self.save_db = True
        self.save_csv = True
        
        # Midnight to Midnight (24h window) for the DB Query
        self.start = pd.Timestamp(selected_date).tz_localize("UTC")
        self.end = self.start + pd.Timedelta(hours=23, minutes=59, seconds=59)

def get_clean_zones():
    all_zones = list(NEIGHBOURS.keys())
    to_remove = ["DE_AT_LU", "IE_SEM", "IE", "NIE", "MT", 
                 "IT", "IT_BRNN", "IT_ROSN", "IT_FOGN"]
    return [z for z in all_zones if z not in to_remove]

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="European Grid Analysis", layout="wide")
st.title("⚡ European Electricity Flow Tracing")

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_data
def load_geography(active_zones):
    custom_zones = ["GB", "ME", "BA", "MK"]
    entsoe_zones = [z for z in active_zones if z not in custom_zones]
    geo_df = load_zones(entsoe_zones, pd.Timestamp('2024-01-01'))
    
    input_dir = Path(__file__).parent.parent / "inputs"
    for country in custom_zones:
        try:
            zone = gpd.read_file(input_dir / f"zones/{country}.geojson")
            geo_df.loc[country] = zone["geometry"][0]
        except: continue

    geo_df['lon'] = geo_df.geometry.centroid.x
    geo_df['lat'] = geo_df.geometry.centroid.y
    geoj = json.loads(geo_df.to_json())
    return geo_df.drop(["geometry"], axis=1), geoj

@st.cache_data
def load_real_flows(target_bz, selected_date, selected_hour, flow_settings):
    mock_config = MockConfig(selected_date)
    folder = flow_settings["folder"]
    table = flow_settings["table"]
    path = mock_config.output_dir / folder / str(mock_config.year) / f"{target_bz}_{folder}.csv"
    
    df = io.load(path, table, mock_config, bz=target_bz)
    if df is None or df.empty: return []

    target_time = pd.to_datetime(f"{selected_date} {selected_hour:02d}:00:00").tz_localize('UTC')
    if target_time not in df.index: return []
    
    row = df.loc[target_time]
    active_flows = []
    
    for col in df.columns:
        if "_net_export" in col:
            val = row[col]
            if abs(val) > 10: 
                pair = col.replace("_net_export", "")
                
                # Logic: val > 0 is EXPORT from target_bz. val < 0 is IMPORT to target_bz.
                if pair.startswith(f"{target_bz}_"):
                    other_bz = pair.replace(f"{target_bz}_", "")
                    direction = 'import' if val < 0 else 'export'
                    source = other_bz if val < 0 else target_bz
                    target = target_bz if val < 0 else other_bz
                elif pair.endswith(f"_{target_bz}"):
                    other_bz = pair.replace(f"_{target_bz}", "")
                    # Note: order is flipped in column name, so sign logic flips
                    direction = 'export' if val < 0 else 'import'
                    source = target_bz if val < 0 else other_bz
                    target = other_bz if val < 0 else target_bz
                else: continue

                active_flows.append({
                    "source": source, "target": target,
                    "mw": abs(val), 
                    "label": f"{abs(val)/1000:.1f} GW",
                    "flow_type": direction # 'import' or 'export'
                })
    return active_flows

# ==========================================
# 3. UTILS & MAP GENERATOR
# ==========================================
def get_bearing(lon1, lat1, lon2, lat2):
    d_lon = lon2 - lon1
    y = np.sin(np.radians(d_lon)) * np.cos(np.radians(lat2))
    x = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - \
        np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(d_lon))
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

def draw_flow_map(geo_df, geoj, flows):
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        geojson=geoj, locations=geo_df.index, z=[0] * len(geo_df),
        colorscale=[[0, '#f8f9fa'], [1, '#f8f9fa']], showscale=False,
        marker_line_color='#adb5bd', marker_line_width=1
    ))

    # Define our colors
    COLOR_MAP = {
        'export': {'line': 'rgba(40, 167, 69, 0.6)', 'arrow': '#28a745', 'text': '#155724'}, # Green
        'import': {'line': 'rgba(0, 123, 255, 0.6)', 'arrow': '#007bff', 'text': '#004085'}  # Blue
    }

    for flow in flows:
        if flow['source'] in geo_df.index and flow['target'] in geo_df.index:
            s_lon, s_lat = geo_df.loc[flow['source'], 'lon'], geo_df.loc[flow['source'], 'lat']
            t_lon, t_lat = geo_df.loc[flow['target'], 'lon'], geo_df.loc[flow['target'], 'lat']
            
            # Select colors based on flow direction
            colors = COLOR_MAP.get(flow['flow_type'])
            
            width = max(2, flow['mw'] / 1000 * 2.5) 
            angle = get_bearing(s_lon, s_lat, t_lon, t_lat)

            # Flow Line
            fig.add_trace(go.Scattergeo(
                lon=[s_lon, t_lon], lat=[s_lat, t_lat],
                mode='lines', line=dict(width=width, color=colors['line']),
                hoverinfo='none'
            ))

            # Directional Arrow & Label
            mid_lon, mid_lat = (s_lon + t_lon) / 2, (s_lat + t_lat) / 2
            fig.add_trace(go.Scattergeo(
                lon=[mid_lon], lat=[mid_lat],
                mode='markers+text', text=[flow['label']],
                textposition="top center",
                marker=dict(size=12, symbol='triangle-up', color=colors['arrow'], angle=angle),
                textfont=dict(size=13, color=colors['text'], family="Arial Black")
            ))

    fig.update_layout(
        geo=dict(fitbounds="locations", visible=False, projection_type="mercator"),
        margin={"r":0,"t":0,"l":0,"b":0}, height=800, showlegend=False
    )
    return fig

# ==========================================
# 5. RENDER DASHBOARD
# ==========================================
active_zones = get_clean_zones()

st.sidebar.header("Control Panel")
target_bz = st.sidebar.selectbox("Focus Bidding Zone", active_zones, index=active_zones.index("DE_LU"))
date = st.sidebar.date_input("Day", pd.to_datetime("2026-01-10"))
hour = st.sidebar.slider("Hour (UTC)", 0, 23, 12)

# In Step 4: INTERFACE
st.sidebar.header("Data Selection")
flow_choice = st.sidebar.radio("Flow Methodology", list(FLOW_TYPES.keys()))

# Get the specific settings for the chosen type
selected_type = FLOW_TYPES[flow_choice]

# Execution
geo_data, geo_json = load_geography(active_zones)

# Load raw data for metrics
mock_config = MockConfig(date)
path = mock_config.output_dir / selected_type["folder"] / str(mock_config.year) / f"{target_bz}_{selected_type['folder']}.csv"
raw_df = io.load(path, selected_type["table"], mock_config, bz=target_bz)

st.subheader(f"Analysis: {target_bz} @ {date} {hour:02d}:00:00")

if raw_df is not None and not raw_df.empty:
    target_time = pd.to_datetime(f"{date} {hour:02d}:00:00").tz_localize('UTC')
    
    if target_time in raw_df.index:
        current_row = raw_df.loc[target_time]
        
        # Calculate Metrics (assuming columns are 'target_bz_other_bz_net_export')
        # We find all export columns related to our target zone
        export_cols = [c for c in raw_df.columns if "_net_export" in c]
        vals = current_row[export_cols]
        
        # Total Exports (Sum of positive values) / 1000 for GW
        total_exp = vals[vals > 0].sum() / 1000
        # Total Imports (Sum of negative values) / 1000 for GW
        total_imp = abs(vals[vals < 0].sum()) / 1000
        # Net Balance
        net_pos = (total_exp - total_imp)
        
        # --- METRIC CARDS ---
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Total Exports", f"{total_exp:.2f} GW")
        col2.metric("Total Imports", f"{total_imp:.2f} GW")
        
        # Net Position with dynamic color (Green for surplus, Red/Blue for deficit)
        status = "Export Surplus" if net_pos >= 0 else "Import Deficit"
        col3.metric(
            label="Net Position", 
            value=f"{net_pos:.2f} GW", 
            delta=status,
            delta_color="normal" if net_pos >= 0 else "inverse" 
        )
        st.divider()

# --- NEW: DATA INSPECTOR ---
with st.expander("🔍 Inspect Raw Database Data"):
    if raw_df is None or raw_df.empty:
        st.error(f"No data returned from DB for {target_bz}. Check if the table 'processed_physical_flows' exists.")
    else:
        st.write(f"Total rows retrieved for the day: {len(raw_df)}")
        
        # Filter for the specific hour row
        target_time = pd.to_datetime(f"{date} {hour:02d}:00:00").tz_localize('UTC')
        
        if target_time in raw_df.index:
            st.success(f"Found data for {hour:02d}:00!")
            current_row = raw_df.loc[[target_time]]
            
            # Show only columns that have non-zero values to clear clutter
            non_zero_cols = current_row.loc[:, (current_row != 0).any(axis=0)]
            if not non_zero_cols.empty:
                st.write("Non-zero values for this hour:")
                st.dataframe(non_zero_cols)
            else:
                st.warning("All values for this specific hour are exactly 0.0.")
        else:
            st.warning(f"Timestamp {target_time} not found in the returned data.")
            st.write("Available timestamps in DB (first 5):")
            st.write(raw_df.index[:5].tolist())

# --- RENDER MAP ---
flow_data = load_real_flows(target_bz, date, hour, selected_type)

if not flow_data:
    st.info("The map is empty because no neighbors with flow > 10MW were found for this hour.")
else:
    m = draw_flow_map(geo_data, geo_json, flow_data)
    st.plotly_chart(m, width='stretch')