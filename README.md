# Exchange Analysis

This project analyses data available through the ENTSO-E API using multiple methods (e.g. flow tracing, pooling), in order to determine the import sources and export sinks in the European electricity market on a per bidding zone and per type basis, for each bidding zone in the network.

In addition to generating local CSV outputs, this project includes an **optional** fully containerized **TimescaleDB (PostgreSQL) and Grafana stack** for high-performance querying and interactive time-series visualization.

Import/export results for the following methods can be calculated:

1. **Commercial Flows Total (CFT):** Take incoming/outgoing line (for import/export) as the exchange value from/to a neighbouring bidding zone.
2. **Netted Commercial Flows Total (Netted CFT):** Net over incoming and outgoing line as the exchange value from/to a neighbouring bidding zone.
3. **Pooled Net CFT:** A bidding zone's net import is proportionally supplied by all net exporters in the network at each timepoint, a "copper plate" grid model without transmission constraints. Vice versa for net export.
4. **Pooled Net Phys.:** Similar to Pooled Net CFT, however uses physical flow net position values instead of CFT values.
5. **Direct Coupling (DC) Flow Tracing:** Takes into account the potential for flows to continue beyond immediate neighbours and for transit flows. Each zone's generation, load and exchanges with neighbours are elements of the network.
6. **Aggregated Coupling (AC) Flow Tracing:** Similar to Direct Coupling, however the net position of a zone is now its contribution to the network instead of both its generation and load.

## 🛠️ Prerequisites

* **Python:** 3.10 or higher recommended.
* **Anaconda** (Recommended) or standard Python installation.
* **ENTSO-E API Key:** Required for downloading data. You can obtain a free API key by registering on the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) and requesting "Restful API Access" in your account settings.
* **Docker Desktop** *(Optional)*: Only required if you want to run the local TimescaleDB database and Grafana dashboard.

---

## 🚀 Setup Instructions

### 1. Clone or Download the Project

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to avoid conflicts.

**Option A: Using Conda (Recommended)**

```bash
# Create the environment
conda create -n exchange-analysis python=3.10

# Activate it
conda activate exchange-analysis

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

Run this command inside your activated environment:

```bash
pip install -r requirements.txt

```

---

## 🔑 Configuration

You must create a `keys.yaml` file in the **root directory** of the project to store your API key. This file is ignored by Git to keep your secrets safe.

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
* **Period:** Set your start and end dates (e.g., `"2024-01-01 00:00"`, `"2024-12-31 23:59"`). Output directories are organized by years, so the best practice is to use a full year.
* **Subsets:** Uncomment `selected_bzs`/`target_zones` if you only want to download data for specific bidding zones (e.g., `["DE_LU", "FR"]`), and `selected_data_types`/`data_types` to download specific types of data.

### Step 2: (Optional) Start the Database & Dashboard Stack

If you chose to save your outputs to the database in the Control Panel, you need to spin up the local server *before* running the Python script.

Ensure Docker is running on your machine, then execute:

```bash
docker compose up -d

```

*(This starts a TimescaleDB instance on port 5433 and a Grafana instance on port 3001. To shut them down later, run `docker compose down`)*.

### Step 3: Run the Pipeline

Ensure your environment is activated (`conda activate exchange-analysis`) and execute the main script:

```bash
python main.py

```

Real-time logs will appear in your terminal, and detailed logs are saved to `logs/log_{TIMESTAMP}.log`.

### Step 4: View the Results

Depending on the outputs selected in the Control Panel:

* **Flat Files:** Check the `outputs/` folder for the generated CSV time series.
* **Interactive Dashboards:** Open your browser to **`http://localhost:3001`** (Login: `admin` / `admin`). Navigate to **Connections > Data Sources** to verify TimescaleDB is connected (click `Start & test`), and view all matrices mapped over time!

---

## 📂 Flat Files Directory

Import results are saved on a per bidding zone, per type, and "per bidding zone per type" basis as CSV time series in the `outputs/` directory, organized by year. Exports are only saved as totals, as export time series would be essentially a duplicate of import time series results. The following folders can be generated:

* **`generation_demand_data.../`**: Hourly generation and load used for determining per type mixes of import/export, and used in direct flow tracing.
* **`comm_flow_total.../`**: Hourly commercial flow total exchanges, determines import results using in-coming line values or by netting over both directions, used as input to pooling approach.
* **`physical_flow_data.../`**: Physical cross-border flows, used as input to pooling approach and flow tracing.
* **`import_flow_tracing.../`**: Import source values as a result of flow tracing analysis.
* **`pooling/`**: Import source values as a result of flow tracing analysis.
* **`annual_totals_per_method/`**: Final aggregated TWh import and export totals.