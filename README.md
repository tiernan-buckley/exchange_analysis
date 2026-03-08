# Exchange Analysis

This project analyses data available through the ENTSO-E API using multiple methods (e.g. flow tracing, pooling), in order to determine the import sources and export sinks in the European electricity market on a per bidding zone and per type basis, for each bidding zone in the network.

In addition to generating local CSV outputs, this project includes an **optional** fully containerised **TimescaleDB (PostgreSQL) and Grafana stack** for high-performance querying and interactive time-series visualisation, as well as an optional **Streamlit** web app for interactive visualisation.

Import/export results for the following methods can be calculated:

1. **Commercial Flows Total (CFT):** Take incoming/outgoing line (for import/export) as the exchange value from/to a neighbouring bidding zone.
2. **Netted Commercial Flows Total (Netted CFT):** Net over incoming and outgoing line as the exchange value from/to a neighbouring bidding zone.
3. **Pooled Net CFT:** A bidding zone's net import is proportionally supplied by all net exporters in the network at each timepoint, a "copper plate" grid model without transmission constraints. Vice versa for net export.
4. **Pooled Net Phys.:** Similar to Pooled Net CFT, however uses physical flow net position values instead of CFT values.
5. **Direct Coupling (DC) Flow Tracing:** Takes into account the potential for flows to continue beyond immediate neighbours and for transit flows. Each zone's generation, load and exchanges with neighbours are elements of the network.
6. **Aggregated Coupling (AC) Flow Tracing:** Similar to Direct Coupling, however the net position of a zone is now its contribution to the network instead of both its generation and load.

## 🛠️ Prerequisites

* **Python:** 3.10 or higher recommended.
* **ENTSO-E API Key:** Required for downloading data. Obtain a free API key by registering on the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) and requesting "Restful API Access" in your account settings.
* **Docker Desktop** *(Optional)*: Only required if you want to run the local TimescaleDB database and Grafana dashboard.
* **Geospatial Libraries** *(Optional)*: For running the **Streamlit** web app (Step 4), **geopandas** requires system-level dependencies for GIS data (**GDAL, PROJ, GEOS**).
    * **Windows/Mac:** Using **Conda** is strongly recommended as it handles these dependencies automatically.
    * **Linux:** You must install system headers first: `sudo apt install libgdal-dev`.



---

## 🚀 Setup Instructions

### 1. Clone or Download the Project

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to avoid conflicts.

**Option A: Using Conda (Recommended for Dashboard users)**

This handles complex geospatial dependencies automatically.

```bash
# Create the environment
conda create -n exchange-analysis python=3.10

# Activate it
conda activate exchange-analysis

# Install geospatial dependencies via conda
conda install -c conda-forge geopandas

```

**Option B: Using venv (Standard Python)**

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

Regardless of the option chosen above, run this command inside your activated environment to ensure all UI and data tools are present:

```bash
pip install -r requirements.txt

```

---

## 🔑 Configuration

You must create a `keys.yaml` file in the **root directory** of the project to store your API key. 

1. Create a file named `keys.yaml`.
2. Paste the following content into it:

```yaml
entsoe-key: "YOUR-UUID-API-KEY-HERE"

```

---

## ▶️ Usage

### Step 1: Configure the Run

Open `main.py` to adjust the **Control Panel** section:

* **Output Destinations:** Choose where to save your processed data (e.g., save to flat CSV files, push directly to the PostgreSQL/TimescaleDB database, or both).
* **Run Flags:** Set steps to `True` or `False` (e.g., set `download: False` if you already have the data).
* **Period:** Set your start and end dates (e.g., `"2024-01-01 00:00"`, `"2024-12-31 23:59"`). Output directories are organised by years, so the best practice is to use a full year.
* **Subsets:** Uncomment `selected_bzs`/`target_zones` if you only want to download data for specific bidding zones (e.g., `["DE_LU", "FR"]`), and `selected_data_types`/`data_types` to download specific types of data.

### Step 2: (Optional) Start the Database & Dashboard Stack

If you chose to save your outputs to the database in the Control Panel, you need to spin up the local server *before* running the Python script.

Ensure **Docker Desktop** is running on your machine, then execute:

```bash
docker compose up -d

```

*(This starts a TimescaleDB instance on port 5433 and a Grafana instance on port 3001. To shut them down later, run `docker compose down`)*.

### Step 3: Run the Pipeline

Ensure your environment is activated (`conda activate exchange-analysis`) and execute the main script:

```bash
python src/main.py

```

Real-time logs will appear in your terminal, and detailed logs are saved to `logs/log_{TIMESTAMP}.log`.

### Step 4: Launch the Interactive Dashboard (Optional)

Visualise the flows and import mix using the **Streamlit** app:

```bash
streamlit run src/app.py

```

**Dashboard Features:**

* **Methodology Comparison:** Visualises how standard commercial/physical flows and advanced flow tracing algorithms alter the perceived cross-border exchanges and imported fuel mixes for any given hour.
* **Interactive Flow Mapping:** Dynamic, curved arrows scaled by MW volume represent real-time exchanges. The map is fully interactive—users can click any bidding zone to instantly update the dashboard focus.
* **Net Position Tracking:** Highlights the selected bidding zone using thermal tinting (Green for Export, Blue for Import) and features an hourly trend bar chart to monitor intra-day market shifts.
* **Generation Mix & Demand:** Displays internal generation stacked by fuel type—explicitly handling storage charging as negative generation below the zero-axis—overlaid with a total demand line to clearly identify import requirements.
* **Import Fuel Decomposition:** Traces the specific fuel types (e.g., Wind, Nuclear, Lignite) imported from the broader European network, calculated dynamically based on the selected flow methodology.

---

## 📂 Flat Files Directory

Import results are saved on a per bidding zone, per type, and "per bidding zone per type" basis as CSV time series in the `outputs/` directory, organised by year. Exports are only saved as totals, as export time series would be essentially a duplicate of import time series results. The following folders can be generated:

* **`generation_demand_data.../`**: Hourly generation and load used for determining per type mixes of import/export.
* **`comm_flow_total.../`**: Hourly commercial flow total exchanges, used as input to pooling approach.
* **`physical_flow_data.../`**: Physical cross-border flows, used as input to pooling and flow tracing.
* **`import_flow_tracing.../`**: Import source values as a result of flow tracing analysis.
* **`annual_totals_per_method/`**: Final aggregated TWh import and export totals.

---

## ✍️ Author

**Tiernan Buckley**

* **Project:** Master's Thesis in Sustainable Systems Engineering at the University of Freiburg
* **Contact:** tbuckle@tcd.ie
* **GitHub:** [@tiernan-buckley](https://github.com/tiernan-buckley)

## 📄 Licence

This project is licensed under the **Creative Commons Attribution 4.0 International Licence (CC BY 4.0)**.
