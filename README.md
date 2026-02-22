# COMP 3610 Assignment 1: Data Pipeline & Visualization Dashboard

**NYC Yellow Taxi Trip Analysis - January 2024**

This assignment implements a complete data engineering pipeline for analyzing NYC Yellow Taxi data, including programmatic data downloading, SQL-based analytics, and an interactive Streamlit dashboard.

---

## 📋 Overview

**Objective:** Build an end-to-end big data pipeline that ingests, transforms, analyzes, and visualizes NYC Yellow Taxi trip records.

**Dataset:** 
- ~3 million taxi trip records from January 2024
- Time series data with pickup/dropoff times, locations, fares, and payment methods

**Key Features:**
- ✅ Programmatic data download (Parquet + CSV)
- ✅ Data validation and quality checks
- ✅ Cleaning pipeline (4 step validation)
- ✅ Feature engineering (4 derived columns)
- ✅ SQL analytics (5 analytical queries via DuckDB)
- ✅ Interactive dashboard with filters and visualizations
- ✅ Deployment on Streamlit Community Cloud

---

## 📁 Project Structure

```
Assignment #1/
├── app.py                              # Streamlit dashboard application
├── assignment1.ipynb                   # Jupyter notebook (Parts 1-3)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── .gitignore                          # Version control ignore rules
├── data/
│   ├── raw/                            # Downloaded raw data (not committed)
│   │   ├── yellow_tripdata_2024-01.parquet
│   │   └── taxi_zone_lookup.csv
│   └── summary.json                    # Cleaning summary metadata
└── __pycache__/                        # Python bytecode (ignored)
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- ~5 GB disk space (for raw data files)
- Internet connection (for programmatic downloads)

### Step 1: Clone Repository
```bash
git clone https://github.com/nielconstance2004/COMP3610Assignment1.git 
cd COMP3610Assignment1
code .
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Jupyter Notebook (Optional)
The notebook demonstrates the entire pipeline in Parts 1-3:
```bash
# Using Jupyter Lab
jupyter lab assignment1.ipynb

# Or using VS Code
code assignment1.ipynb
```

### Step 5: Run Dashboard Locally
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 📊 Dashboard Features

### Overview Tab
- **Summary Metrics:** Total trips, average fare, total revenue, average distance, average duration
- **Data Processing Summary:** Shows how many rows were removed at each cleaning step
- **Filter Status:** Current filter selections

### Visualizations Tab
1. **Top 10 Pickup Zones (Bar Chart)**
   - Shows busiest taxi pickup locations
   - Identifies geographic demand concentration

2. **Average Fare by Hour (Line Chart)**
   - Hourly fare patterns (0-23 hours)
   - Reveals time-of-day pricing dynamics

3. **Trip Distance Distribution (Histogram)**
   - Distribution with 50 bins
   - Shows prevalence of short vs long trips

4. **Payment Type Breakdown (Pie Chart)**
   - Split of credit card, cash, and other payment methods
   - Indicates digital payment adoption

5. **Demand Heatmap (Heatmap)**
   - Trips by day of week and hour
   - Reveals weekly and hourly patterns

### Interactive Filters
- 📅 **Date Range:** Select dates within January 2024
- ⏰ **Pickup Hour Range:** Filter by hour of day (0-23)
- 💳 **Payment Types:** Multi-select payment methods
- All visualizations update in real-time as filters change

---

## 📓 Notebook Structure (assignment1.ipynb)

### Part 1: Data Ingestion (20 marks)
- Programmatic download of Parquet and CSV files
- Data validation (columns, datetime types, row counts)
- .gitignore setup for data directory

### Part 2: Data Transformation & Analysis (30 marks)
- **Data Cleaning:** Removes nulls, invalid fares, invalid distances, temporal anomalies
- **Feature Engineering:** Creates 4 derived columns:
  1. `trip_duration_minutes` - Duration in minutes
  2. `trip_speed_mph` - Distance / duration with zero-handling
  3. `pickup_hour` - Hour of day (0-23)
  4. `pickup_day_of_week` - Day name (Monday-Sunday)
- **SQL Queries (5 total):** All implemented using DuckDB
  1. Top 10 pickup zones by trip count
  2. Average fare by hour
  3. Payment type percentages
  4. Average tip percentage by day (card payments only)
  5. Top 5 pickup-dropoff zone pairs

### Part 3: Dashboard Development
- Visualization prototypes using Plotly
- Interactive filter functions
- References app.py for deployed implementation

### Part 4: Documentation
- Complete markdown documentation
- Code comments explaining non-obvious logic
- AI tools disclosure

---

## 🌐 Deployment

1. **Dashboard URL**
   ```
   https://comp3610assignment1-sfk7p2vr9cvdpbwydkkwrc.streamlit.app/
   ```

### Note on Data Files
- Data files are **NOT committed** to repository (per .gitignore)
- Files download automatically on first run (~10-30 seconds)
- Subsequent runs use cached local files

---

## 🔍 Testing

### Unit Tests Included in Notebook
- Feature engineering validation (5-minute trip calculation)
- Datetime parsing verification
- Hour and day-of-week extraction tests

# Check specific queries
# (Run SQL queries from Part 2 of notebook)
```

---

## 📈 Performance Notes

- **Initial Load:** 30-60 seconds (download ~300MB data)
- **Subsequent Loads:** <5 seconds (cached data)
- **Dashboard Filtering:** Real-time (<1 second) with 3M rows
- **Memory Usage:** ~2GB when fully loaded

Streamlit's `@st.cache_data` decorator ensures efficient use of resources.

---

## 📝 Code Quality

- ✅ Comprehensive docstrings for all functions
- ✅ Type hints for function parameters and returns
- ✅ Error handling for network and parsing failures
- ✅ Clear comments explaining non-obvious logic
- ✅ Organized into logical sections and functions
- ✅ Follows PEP 8 style guidelines

---

## 📚 Academic Integrity & AI Disclosure

**AI Tools Used:**
- Deepseek and VSCode Autocompletions
- All code reviewed and adjusted by student author
- Original understanding and implementation by student

---

## 👤 Author Information

**Student:** Nie-l Constance  
**Course:** COMP 3610 — Big Data Analytics  
**Semester:** II, 2025-2026  


