"""
Project: European Electricity Exchange Analysis
Author: Tiernan Buckley
Year: 2026
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Source: https://github.com/INATECH-CIG/exchange_analysis

Description:
Interactive Streamlit dashboard for visualizing the European Electricity Market 
Exchange Analysis. This module renders dynamic geographic flow maps, tracks 
bidding zone net positions, and plots high-resolution hourly internal generation 
and imported fuel mixes using Plotly and GeoPandas.
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import json
import numpy as np
from pathlib import Path
from entsoe.geo.utils import load_zones

from mappings_alt import NEIGHBOURS
from utils import io

# ==========================================
# 0. CONFIG & CONSTANTS
# ==========================================
# Maps UI selection to internal data tables and file paths
FLOW_TYPES = {
    "Physical": {"table": "processed_physical_flows", "agg_table": "analysis_cft_netted_type", "folder": "physical_flow_bidding_zones", "type": "standard"},
    "Commercial Total": {"table": "processed_commercial_flows", "agg_table": "analysis_cft_total_type", "folder": "comm_flow_total_bidding_zones", "type": "standard"},
    "Commercial Day-Ahead": {"table": "processed_commercial_flows_da", "agg_table": None, "folder": "comm_flow_dayahead_bidding_zones", "type": "standard"},
    "Agg. Coupling Flow Tracing": {"table": "tracing_agg_coupling_bz", "agg_table": "tracing_agg_coupling_type", "folder": "import_flow_tracing_bidding_zones/agg_coupling", "type": "tracing"},
    "Direct Coupling Flow Tracing": {"table": "tracing_direct_coupling_bz", "agg_table": "tracing_direct_coupling_type", "folder": "import_flow_tracing_bidding_zones/direct_coupling", "type": "tracing"},
    "Net Pooled CFT": {"table": "pool_commercial_net_pos_bz", "agg_table": "pool_commercial_net_pos_type", "folder": "pooling/commercial_net_pos", "type": "tracing"}
}

# Standardized color palette for consistent visualization across all charts
GEN_COLORS = {
    # Renewables
    "Solar": "#f1c40f",
    "Wind Onshore": "#3498db",
    "Wind Offshore": "#2980b9",
    "Biomass": "#27ae60",
    "Hydro Water Reservoir": "#1abc9c",
    "Hydro Run-of-river and poundage": "#16a085",
    "Geothermal": "#d35400",
    "Marine": "#16a085", 
    "Other renewable": "#2ecc71", 
    
    # Fossils & Nuclear
    "Nuclear": "#9b59b6",
    "Fossil Gas": "#e67e22",
    "Fossil Hard coal": "#34495e", # Dark grey/blue
    "Fossil Oil": "#2c3e50",
    "Waste": "#7f8c8d",
    "Fossil Brown coal/Lignite": "#795548", 
    "Fossil Coal-derived gas": "#5d4037",
    "Fossil Oil shale": "#4e342e",
    "Fossil Peat": "#3e2723",
    
    # Storage (Distinct Modern Palette)
    "Storage Discharge": "#546e7a",          # Blue-Grey / Slate (Neutral but distinct)
    "Storage Charge": "#78909c",             # Lighter Blue-Grey for charging (negative)
    "Storage": "#546e7a",        
    
    # Others
    "Other": "#bdc3c7"
}

# Minimal config class to interface seamlessly with the existing `io.load` utility
class MockConfig:
    def __init__(self, selected_date):
        self.year = selected_date.year
        self.output_dir = Path(__file__).parent.parent / "outputs"
        self.load_source, self.save_db, self.save_csv = 'db', True, True
        self.start = pd.Timestamp(selected_date).tz_localize("UTC")
        self.end = self.start + pd.Timedelta(hours=23, minutes=59, seconds=59)

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def get_clean_zones():
    """Filters out non-standard or sub-bidding zones for a cleaner map."""
    to_remove = ["DE_AT_LU", "IE_SEM", "IE", "NIE", "MT", "IT", "IT_BRNN", "IT_ROSN", "IT_FOGN"]
    return sorted([z for z in NEIGHBOURS.keys() if z not in to_remove])

def get_bearing(lon1, lat1, lon2, lat2):
    """Calculates the angle for arrow markers to point from Source to Target."""
    d_lon = lon2 - lon1
    y = np.sin(np.radians(d_lon)) * np.cos(np.radians(lat2))
    x = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - \
        np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(d_lon))
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

def get_curve(p1, p2, num_points=20):
    """Generates coordinates for curved lines to prevent visual overlapping of straight lines."""
    lons, lats = np.linspace(p1[0], p2[0], num_points), np.linspace(p1[1], p2[1], num_points)
    dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    offset = dist * 0.15 
    shift = offset * np.sin(np.linspace(0, np.pi, num_points))
    return lons + shift*0.2, lats + shift

# ==========================================
# 2. DATA LOADING (Cached for performance)
# ==========================================
@st.cache_data
def load_geography(active_zones):
    """Loads and combines standard ENTSO-E zone geometry with custom local geojson files."""
    custom_zones = ["GB", "ME", "BA", "MK"]
    entsoe_zones = [z for z in active_zones if z not in custom_zones]
    geo_df = load_zones(entsoe_zones, pd.Timestamp('2024-01-01'))
    input_dir = Path(__file__).parent.parent / "inputs"
    for country in custom_zones:
        try:
            zone = gpd.read_file(input_dir / f"zones/{country}.geojson")
            geo_df.loc[country] = zone["geometry"][0]
        except: continue
    geo_df['lon'], geo_df['lat'] = geo_df.geometry.centroid.x, geo_df.geometry.centroid.y
    return geo_df.drop(["geometry"], axis=1), json.loads(geo_df.to_json())

@st.cache_data
def load_full_day_data(selected_date, flow_settings):
    """Loads flow matrix data for the selected day and methodology."""
    mock_config = MockConfig(selected_date)
    is_comm = "comm_flow" in flow_settings["folder"]
    subfolder = "per_bidding_zone" if flow_settings["type"] == "tracing" else "results/per_bidding_zone" if is_comm else ""
    path = mock_config.output_dir / flow_settings["folder"] / str(mock_config.year) / subfolder / "daily_dump.csv"
    return io.load(path, flow_settings["table"], mock_config, bz=None)

@st.cache_data
def load_generation_data(selected_date, target_bz):
    """Loads internal generation mix and demand for a specific bidding zone."""
    mock_config = MockConfig(selected_date)
    path = mock_config.output_dir / "generation_demand_data_bidding_zones" / str(mock_config.year) / f"{target_bz}_gen.csv"
    return io.load(path, "processed_generation", mock_config, bz=target_bz)

@st.cache_data
def load_import_mix(selected_date, target_bz, flow_settings):
    """Loads fuel decomposition data (import mix) based on the selected flow methodology."""
    table = flow_settings.get("agg_table")
    if not table: return None
    mock_config = MockConfig(selected_date)
    path = mock_config.output_dir / "placeholder.csv"
    df = io.load(path, table, mock_config, bz=target_bz)
    if df is not None and not df.empty:
        rename_map = {col: col.split("_")[-1] for col in df.columns if "_" in col}
        df = df.rename(columns=rename_map)
        return df.T.groupby(level=0).sum().T
    return df

def extract_arrow_flows(target_bz, hourly_all, active_zones, flow_settings):
    """Parses raw flow matrices to build Source-Target pairs > 10MW for map plotting."""
    active_flows = []
    if flow_settings.get("type") == "tracing":
        target_row = hourly_all[hourly_all['bidding_zone'] == target_bz]
        for source_bz in active_zones:
            if source_bz == target_bz: continue
            if not target_row.empty and source_bz in target_row.columns:
                val = target_row[source_bz].iloc[0]
                if val > 10: active_flows.append({"Source": source_bz, "Target": target_bz, "MW": val, "Type": "Import"})
            source_row = hourly_all[hourly_all['bidding_zone'] == source_bz]
            if not source_row.empty and target_bz in source_row.columns:
                val = source_row[target_bz].iloc[0]
                if val > 10: active_flows.append({"Source": target_bz, "Target": source_bz, "MW": val, "Type": "Export"})
    else:
        focus_row = hourly_all[hourly_all['bidding_zone'] == target_bz]
        if focus_row.empty: return []
        for col in focus_row.columns:
            if "_net_export" in col:
                val = focus_row[col].iloc[0]
                if abs(val) > 10:
                    pair = col.replace("_net_export", "")
                    if pair.startswith(f"{target_bz}_"):
                        other = pair.replace(f"{target_bz}_", ""); direction = 'Import' if val < 0 else 'Export'
                        s, t = (other, target_bz) if val < 0 else (target_bz, other)
                    elif pair.endswith(f"_{target_bz}"):
                        other = pair.replace(f"_{target_bz}", ""); direction = 'Export' if val < 0 else 'Import'
                        s, t = (target_bz, other) if val < 0 else (other, target_bz)
                    active_flows.append({"Source": s, "Target": t, "MW": abs(val), "Type": direction})
    return active_flows

# ==========================================
# 3. MAP GENERATOR
# ==========================================
def draw_flow_map(geo_df, geoj, flows, hourly_all, target_bz, flow_type_meta):
    """Constructs the Plotly Map: Base Choropleth (zones) + Scattergeo (curved arrows & values)."""
    fig = go.Figure()
    hover_labels, z_values, b_colors, b_widths = [], [], [], []
    bz_cols = [c for c in hourly_all.columns if c in geo_df.index]
    relevant_lons, relevant_lats = [geo_df.loc[target_bz, 'lon']], [geo_df.loc[target_bz, 'lat']]

    # 1. Build Base Map Data (Zone styling & tooltips)
    for zone in geo_df.index:
        val_gw = 0.0
        if flow_type_meta == "tracing":
            exports = hourly_all[zone].sum() if zone in hourly_all.columns else 0
            imports = hourly_all[hourly_all['bidding_zone'] == zone][bz_cols].sum(axis=1).sum()
            val_gw = (exports - imports) / 1000
        else:
            zone_row = hourly_all[hourly_all['bidding_zone'] == zone]
            if not zone_row.empty and 'Net Export' in zone_row.columns: val_gw = zone_row['Net Export'].iloc[0] / 1000
        
        if zone == target_bz:
            b_colors.append('#000000'); b_widths.append(2.0); z_values.append(1 if val_gw >= 0 else -1)
        else:
            b_colors.append('#adb5bd'); b_widths.append(0.8); z_values.append(0)

        hover_labels.append(f"<b>{zone}</b><br>{'Exporting' if val_gw >= 0 else 'Importing'}: {abs(val_gw):.2f} GW")

    # Add Choropleth Trace (Background zones)
    fig.add_trace(go.Choropleth(
        geojson=geoj, locations=geo_df.index, z=z_values, text=hover_labels, hoverinfo="text",
        colorscale=[[0, '#e3f2fd'], [0.5, '#ffffff'], [1, '#e8f5e9']], zmin=-1, zmax=1, showscale=False, 
        marker_line_color=b_colors, marker_line_width=b_widths
    ))

    COLOR_MAP = {'Export': {'l': 'rgba(40, 167, 69, 0.4)', 'a': '#28a745', 't': '#1e7e34'},
                 'Import': {'l': 'rgba(0, 123, 255, 0.4)', 'a': '#007bff', 't': '#0056b3'}}
    
    # 2. Add Flow Markers (Curved lines and directional arrows)
    for flow in flows:
        p1, p2 = (geo_df.loc[flow['Source'], 'lon'], geo_df.loc[flow['Source'], 'lat']), (geo_df.loc[flow['Target'], 'lon'], geo_df.loc[flow['Target'], 'lat'])
        relevant_lons.extend([p1[0], p2[0]]); relevant_lats.extend([p1[1], p2[1]])
        c = COLOR_MAP.get(flow['Type'])
        cLons, cLats = get_curve(p1, p2)
        
        # Draw Line
        fig.add_trace(go.Scattergeo(lon=cLons, lat=cLats, mode='lines', line=dict(width=max(1.5, flow['MW']/500), color=c['l']), hoverinfo='none'))
        # Draw Arrow & Text
        mid = len(cLons)//2
        fig.add_trace(go.Scattergeo(lon=[cLons[mid]], lat=[cLats[mid]], mode='markers+text', text=[f"<b>{flow['MW']/1000:.1f}</b>"], 
                                    textposition="top center", marker=dict(size=12, symbol='triangle-up', color=c['a'], angle=get_bearing(p1[0], p1[1], p2[0], p2[1]), line=dict(color='white', width=1)),
                                    textfont=dict(size=12, color=c['t'], family="Arial Black"), hoverinfo='none'))

    pad = 2.5
    fig.update_layout(
        geo=dict(projection_type="mercator", lonaxis_range=[min(relevant_lons)-pad, max(relevant_lons)+pad], lataxis_range=[min(relevant_lats)-pad, max(relevant_lats)+pad], visible=False), 
        margin={"r":0,"t":0,"l":0,"b":0}, height=650, showlegend=False, clickmode='event+select', hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial")
    )
    return fig

# ==========================================
# 4. INTERFACE SETUP
# ==========================================
st.set_page_config(page_title="European Grid Analysis", layout="wide")
# Custom CSS for spacing and hiding unnecessary Streamlit metric artifacts
st.markdown("<style>.main > div {padding-left: 2rem; padding-right: 2rem; max-width: 100%;} [data-testid='stMetricDelta'] svg { display: none !important; }</style>", unsafe_allow_html=True)

st.title("⚡ European Electricity Market Exchange Analysis")
active_zones = get_clean_zones()

# Initialize session state variables for interactivity
if "target_bz" not in st.session_state: st.session_state.target_bz = "DE_LU"
if "hour_val" not in st.session_state: st.session_state.hour_val = 12
if "flow_method" not in st.session_state: st.session_state.flow_method = "Physical"

# Sidebar Input Controls
st.sidebar.header("Select Data:")
selected_bz = st.sidebar.selectbox("Bidding Zone", active_zones, index=active_zones.index(st.session_state.target_bz))
if selected_bz != st.session_state.target_bz: st.session_state.target_bz = selected_bz; st.rerun()

date = st.sidebar.date_input("Day", pd.to_datetime("2026-03-04"))
flow_options = list(FLOW_TYPES.keys())
selected_method = st.sidebar.radio("Flow Methodology", flow_options, index=flow_options.index(st.session_state.flow_method))
if selected_method != st.session_state.flow_method: st.session_state.flow_method = selected_method; st.rerun()

selected_type = FLOW_TYPES[st.session_state.flow_method]
hour = st.sidebar.slider("Hour (UTC)", 0, 23, value=st.session_state.hour_val)
st.session_state.hour_val = hour 

# ==========================================
# 5. RENDER DASHBOARD
# ==========================================
# Fetch data for the selected day
geo_data, geo_json = load_geography(active_zones)
full_day_df = load_full_day_data(date, selected_type)
gen_df = load_generation_data(date, st.session_state.target_bz)
import_mix_df = load_import_mix(date, st.session_state.target_bz, selected_type)

if full_day_df is not None and not full_day_df.empty:
    target_time = pd.to_datetime(f"{date} {st.session_state.hour_val:02d}:00:00").tz_localize('UTC')
    hourly_all = full_day_df[full_day_df.index == target_time]
    
    # Create two-column layout: Map (left) and Charts (right)
    col_map, col_analysis = st.columns([65, 35], gap="large")

    with col_map:
        active_flows = extract_arrow_flows(st.session_state.target_bz, hourly_all, active_zones, selected_type)
        map_event = st.plotly_chart(draw_flow_map(geo_data, geo_json, active_flows, hourly_all, st.session_state.target_bz, selected_type["type"]), width="stretch", on_select="rerun")
        # Handle map clicks to change focus zone
        if map_event and 'selection' in map_event and map_event['selection']['points']:
            clicked = map_event['selection']['points'][0].get('location')
            if clicked in active_zones and clicked != st.session_state.target_bz: st.session_state.target_bz = clicked; st.rerun()
        with st.expander(f"📊 {st.session_state.flow_method} Flow Details"):
            if active_flows: st.dataframe(pd.DataFrame(active_flows).sort_values(by="MW", ascending=False), width="stretch", hide_index=True)

    with col_analysis:
        # Calculate and display Net Position
        if selected_type["type"] == "tracing":
            bz_cols = [c for c in full_day_df.columns if c in active_zones]
            h_imp = full_day_df[full_day_df['bidding_zone'] == st.session_state.target_bz].groupby(level=0)[bz_cols].sum().sum(axis=1)
            h_exp = full_day_df.groupby(level=0)[st.session_state.target_bz].sum()
            daily_trend = (h_exp - h_imp) / 1000
        else: daily_trend = full_day_df[full_day_df['bidding_zone'] == st.session_state.target_bz]['Net Export'] / 1000
        
        net_val = daily_trend.loc[target_time] if target_time in daily_trend.index else 0
        color = "#28a745" if net_val >= 0 else "#007bff"
        st.markdown(f"<style>[data-testid='stMetricDelta'] > div {{ color: {color} !important; }}</style>", unsafe_allow_html=True)
        st.metric(f"{st.session_state.target_bz} Net Position", f"{net_val:.2f} GW", f"{'↑' if net_val >= 0 else '↓'} Net {'Exporting' if net_val >= 0 else 'Importing'}")
        
        # Plot Net Position Trend (Bar Chart)
        trend_fig = go.Figure(go.Bar(x=daily_trend.index.hour, y=daily_trend, marker_color=['#28a745' if v >= 0 else '#007bff' for v in daily_trend]))
        trend_fig.add_vline(x=st.session_state.hour_val, line_width=2, line_dash="dash", line_color="#343a40")
        trend_fig.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(range=[-0.5, 23.5]), plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Net Position (GW)")
        st.plotly_chart(trend_fig, width="stretch")

        # Plot Internal Generation vs. Demand (Stacked Area Chart)
        if gen_df is not None:
            st.caption(f"🏠 {st.session_state.target_bz} Generation Mix & Demand")
            fig = go.Figure()
            
            # Standard Generation (Positive Stack)
            pos_cols = [c for c in gen_df.columns if c in GEN_COLORS.keys() and c not in ['Storage Charge', 'Total Load', 'Demand']]
            for c in pos_cols: 
                fig.add_trace(go.Scatter(x=gen_df.index.hour, y=gen_df[c]/1000, name=c, mode='lines', stackgroup='pos', line=dict(width=0, color=GEN_COLORS[c])))
            
            # Storage Charging (Negative Stack, below X-axis)
            if 'Storage Charge' in gen_df.columns:
                charge_vals = -np.abs(gen_df['Storage Charge']) / 1000
                fig.add_trace(go.Scatter(
                    x=gen_df.index.hour, y=charge_vals, name='Storage Charge', mode='lines', stackgroup='neg', 
                    line=dict(width=0, color=GEN_COLORS['Storage Charge']), hovertemplate="%{customdata} GW", customdata=np.abs(charge_vals)
                ))
            
            # Demand Overlay (Dotted Line)
            if 'Demand' in gen_df.columns:
                fig.add_trace(go.Scatter(x=gen_df.index.hour, y=gen_df['Demand']/1000, name='Demand', line=dict(color='#2c3e50', width=3, dash='dot')))
            elif 'Total Load' in gen_df.columns: 
                fig.add_trace(go.Scatter(x=gen_df.index.hour, y=gen_df['Total Load']/1000, name='Total Load', line=dict(color='#2c3e50', width=3, dash='dot')))
            
            fig.add_vline(x=st.session_state.hour_val, line_width=2, line_dash="dash", line_color="white")
            fig.update_layout(
                height=220, margin=dict(l=0, r=0, t=5, b=0), xaxis=dict(range=[-0.5, 23.5]), 
                yaxis=dict(title="GW", zeroline=True, zerolinecolor='black', zerolinewidth=1.5), 
                legend=dict(orientation="h", y=-0.5), hovermode="x unified" 
            )
            st.plotly_chart(fig, width="stretch")

        st.write("---")
        
        # Plot Imported Energy Mix (Stacked Area Chart)
        if import_mix_df is not None and not import_mix_df.empty:
            st.caption(f"🌍 Imported Energy Mix")
            fig = go.Figure()
            for c in [x for x in import_mix_df.columns if x in GEN_COLORS.keys()]:
                fig.add_trace(go.Scatter(x=import_mix_df.index.hour, y=import_mix_df[c]/1000, name=c, mode='lines', stackgroup='one', line=dict(width=0, color=GEN_COLORS.get(c, "#95a5a6"))))
            fig.add_vline(x=st.session_state.hour_val, line_width=2, line_dash="dash", line_color="white")
            fig.update_layout(
                height=220, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(range=[-0.5, 23.5]), 
                yaxis=dict(title="GW", zeroline=True, zerolinecolor='black', zerolinewidth=1), 
                legend=dict(orientation="h", y=-0.5), hovermode="x unified" 
            )
            st.plotly_chart(fig, width="stretch")
        else: 
            st.caption(f"🌍 Imported Energy Mix (N/A)")
            st.info(f"Breakdown not calculated for {st.session_state.flow_method}.")
            st.empty() 

else: st.error(f"No data for {date}")