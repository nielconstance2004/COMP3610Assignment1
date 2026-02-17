"""
COMP 3610 Assignment 1: NYC Yellow Taxi Dashboard
==================================================

This Streamlit application implements an interactive visualization dashboard for the NYC Yellow Taxi
trip data (January 2024). It includes:

- Programmatic data download and automatic caching
- Data validation and cleaning
- Feature engineering (trip duration, speed, pickup hour/day)
- Interactive filtering (date range, hour range, payment type)
- 5 required visualizations with analytical insights
- Summary metrics display

All data files are downloaded programmatically at runtime and cached for performance.
"""

import os
import sys
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Tuple, Dict, Any

# ============================================================================
# CONFIGURATION: Data URLs and Paths
# ============================================================================
TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
RAW_DIR = "data/raw"
TRIP_PATH = os.path.join(RAW_DIR, "yellow_tripdata_2024-01.parquet")
ZONE_PATH = os.path.join(RAW_DIR, "taxi_zone_lookup.csv")

# Payment type mapping (per TLC specification)
PAYMENT_TYPE_MAP = {
    1: "Credit Card",
    2: "Cash",
    3: "No Charge",
    4: "Dispute",
    5: "Unknown"
}

# Center Title Utility Function for Plotly Figures
def center_titles(fig: go.Figure) -> go.Figure:
    """
    Utility function to center the title of a Plotly figure.
    
    Args:
        fig: A Plotly Figure object
        
    Returns:
        The same Figure object with the title centered
    """
    fig.update_layout(title_x=0.5)
    return fig

# ============================================================================
# UTILITY FUNCTIONS: Download and Validation
# ============================================================================

def download_if_missing(url: str, dest_path: str, timeout: int = 30) -> None:
    """
    Download a file from `url` to `dest_path` if it doesn't already exist.
    
    Args:
        url: Source URL to download from
        dest_path: Destination file path (directories created if needed)
        timeout: Request timeout in seconds (default: 30s)
        
    Raises:
        RuntimeError: If download fails after retry
        ValueError: If downloaded content is suspiciously small (< 100 bytes)
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if not os.path.exists(dest_path):
        try:
            # Download with timeout to prevent hanging
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()  # Raise HTTPError for bad status codes
            
            # Sanity check: verify downloaded content isn't empty
            if len(resp.content) < 100:
                raise ValueError(f"Downloaded content suspiciously small ({len(resp.content)} bytes)")
                
            with open(dest_path, "wb") as f:
                f.write(resp.content)
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download {url}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error downloading {url}: {e}")

# ============================================================================
# DATA VALIDATION: Verify required columns and datetime formats
# ============================================================================

def validate_trip_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that trip data contains all required columns and datetime fields.
    
    Args:
        df: Raw trip DataFrame to validate
        
    Returns:
        DataFrame with datetime columns converted to proper types
        
    Raises:
        ValueError: If required columns are missing
        TypeError: If datetime columns cannot be parsed
    """
    # Define all required columns per assignment specification
    required_cols = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "PULocationID", "DOLocationID",
        "passenger_count", "trip_distance", "fare_amount",
        "tip_amount", "total_amount", "payment_type"
    ]
    
    # Check for missing columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"VALIDATION FAILED: Missing required columns: {missing}")
    
    # Validate and convert datetime columns
    for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        
        # Check if all values failed to parse
        if df[col].isna().all():
            raise TypeError(f"VALIDATION FAILED: Column '{col}' could not be parsed as datetimes - all values are NaT")
    
    return df

# ============================================================================
# DATA CLEANING: Remove invalid rows and filter outliers
# ============================================================================

def clean_and_feature_engineer(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean trip data and create 4 required feature columns.
    
    Cleaning steps (per assignment):
    1. Remove rows with null values in critical columns
    2. Filter out invalid trips: zero/negative distance, negative/excessive fares
    3. Remove trips where dropoff time is before pickup time
    
    Feature engineering:
    1. trip_duration_minutes: Calculated from pickup/dropoff timestamps
    2. trip_speed_mph: Distance divided by duration (handles division by zero)
    3. pickup_hour: Hour of day (0-23) extracted from pickup timestamp
    4. pickup_day_of_week: Day name (Monday-Sunday) extracted from pickup timestamp
    
    Args:
        df: Cleaned DataFrame (should already have datetimes converted)
        
    Returns:
        Tuple of (cleaned_df, summary_dict) where summary_dict contains row counts at each step
    """
    orig_rows = len(df)
    # NO .copy() - operate directly on input to save memory (more efficient)
    
    # Step 1: Remove rows with nulls in critical columns
    critical_cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID", "fare_amount"]
    df = df.dropna(subset=critical_cols)
    after_null = len(df)
    
    # Step 2: Filter out invalid distance and fare values (memory-efficient chained filtering)
    # Keep only: distance > 0, fare >= 0 and <= $500
    df = df[
        (df["trip_distance"] > 0) & 
        (df["fare_amount"] >= 0) & 
        (df["fare_amount"] <= 500)
    ]
    after_filter = len(df)
    
    # Step 3: Remove trips where dropoff time is before pickup time
    df = df[df["tpep_dropoff_datetime"] >= df["tpep_pickup_datetime"]]
    after_time = len(df)

    # Step 4: Keep only trips from January 2024
    df = df[
        (df["tpep_pickup_datetime"] >= "2024-01-01") &
        (df["tpep_pickup_datetime"] <  "2024-02-01")
    ]
    after_jan = len(df)

    
    # ========== FEATURE ENGINEERING (4 columns) - Memory Efficient ==========
    
    # Feature 1: trip_duration_minutes
    # Calculate time difference and convert to minutes, clip negative values to 0
    df["trip_duration_minutes"] = (
        (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    ).clip(lower=0)
    
    # Feature 2: trip_speed_mph
    # Calculate speed as distance / duration (hours)
    # Handle division by zero by converting inf/NaN to 0
    duration_hours = df["trip_duration_minutes"] / 60.0
    df["trip_speed_mph"] = (
        df["trip_distance"] / duration_hours
    ).replace([float("inf"), -float("inf")], 0).fillna(0)
    
    # Feature 3 & 4: Extract hour and day of week (lightweight datetime operations)
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.day_name()
    
    # Create summary statistics for reporting
    summary = {
        "orig_rows": orig_rows,
        "after_null": after_null,
        "after_filter": after_filter,
        "after_time": after_time,
        "after_jan": after_jan,
        "final_rows": len(df),
        "rows_removed_null": orig_rows - after_null,
        "rows_removed_invalid": after_null - after_filter,
        "rows_removed_time": after_filter - after_time,
        "rows_removed_jan": after_time - after_jan
    }
    
    return df, summary


# ============================================================================
# DATA LOADING: Download, validate, and cache data operations
# ============================================================================

@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Load and prepare taxi trip and zone lookup data.
    
    This function:
    1. Downloads files programmatically if not cached locally
    2. Validates data schema and datetime types
    3. Cleans data according to assignment requirements
    4. Engineers 4 required features
    5. Prepares zone lookup table
    
    The @st.cache_data decorator ensures files are downloaded and processed
    only once, improving dashboard performance for repeated access.
    
    Returns:
        Tuple of (cleaned_trip_df, zones_df, cleaning_summary_dict)
        
    Raises:
        RuntimeError: If download fails
        ValueError: If validation fails
        TypeError: If datetime parsing fails
    """
    try:
        # Download data files if they don't already exist locally
        download_if_missing(TRIP_URL, TRIP_PATH)
        download_if_missing(ZONE_URL, ZONE_PATH)
        
        # Read data files
        trip = pd.read_parquet(TRIP_PATH)
        zones = pd.read_csv(ZONE_PATH)
        
        # Rename 'Zone' column to 'zone' for consistency (CSV has capital Z)
        if "Zone" in zones.columns:
            zones = zones.rename(columns={"Zone": "zone"})
        
        # Validate trip data contains all required columns and datetime format
        trip = validate_trip_df(trip)
        
        # Clean data and engineer features
        trip_clean, summary = clean_and_feature_engineer(trip)
        
        # Ensure consistent data types for joining
        trip_clean["PULocationID"] = trip_clean["PULocationID"].astype(int)
        trip_clean["DOLocationID"] = trip_clean["DOLocationID"].astype(int)
        zones["LocationID"] = zones["LocationID"].astype(int)
        
        return trip_clean, zones, summary
        
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        raise

# ============================================================================
# STREAMLIT APP: Main Dashboard Layout
# ============================================================================

st.set_page_config(page_title="NYC Yellow Taxi Dashboard", layout="wide")

st.title("🚕 NYC Yellow Taxi Dashboard - January 2024")
st.markdown("""
This interactive dashboard analyzes NYC Yellow Taxi trip data for January 2024. 
Filter by date range, pickup hour, and payment type to explore taxi demand patterns, 
pricing dynamics, and operational metrics across New York City.
""")

# ============================================================================
# SECTION 1: LOAD DATA
# ============================================================================

try:
    trip_df, zones_df, summary = load_data()
except Exception as e:
    st.error(f"Failed to load data. Please check your internet connection and try again.")
    st.stop()

# ============================================================================
# SECTION 2: SIDEBAR FILTERS (Interactive)
# ============================================================================

st.sidebar.header("📊 Filter Options")
st.sidebar.markdown("Use these controls to filter all visualizations below.")

# Date range filter
min_date = trip_df["tpep_pickup_datetime"].min().date()
max_date = trip_df["tpep_pickup_datetime"].max().date()
raw_date_range = st.sidebar.date_input(
    "📅 Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date,
    help="Select a date range to filter trips by pickup date"
)

# Handle date range input (can be single date or tuple)
if isinstance(raw_date_range, (list, tuple)):
    if len(raw_date_range) == 2:
        start_date, end_date = raw_date_range
    else:
        start_date = end_date = raw_date_range[0]
else:
    start_date = end_date = raw_date_range

# Hour range filter
hour_range = st.sidebar.slider(
    "⏰ Pickup Hour Range",
    0, 23, (0, 23),
    help="Filter trips by the hour of day they were picked up (0=midnight, 23=11pm)"
)

# Payment type filter - FIXED MAPPING LOGIC
st.sidebar.markdown("---")
payment_options = sorted(trip_df["payment_type"].dropna().unique().tolist())

# Create proper mapping from payment code to label
payment_labels_list = []
payment_code_to_label = {}
for code in payment_options:
    label = PAYMENT_TYPE_MAP.get(code, f"Code {code}")
    payment_labels_list.append(label)
    payment_code_to_label[int(code)] = label

# Reverse mapping from label back to code
payment_label_to_code = {v: k for k, v in payment_code_to_label.items()}

# Multi-select widget for payment types
selected_payment_labels = st.sidebar.multiselect(
    "💳 Payment Types",
    payment_labels_list,
    default=payment_labels_list,
    help="Select one or more payment methods to include in the analysis"
)

# Convert selected labels back to payment codes
selected_payment_codes = [payment_label_to_code[label] for label in selected_payment_labels]

# Fallback: if nothing selected, show all
if not selected_payment_codes:
    st.sidebar.warning("⚠️ No payment types selected - showing all payment methods")
    selected_payment_codes = payment_options

# ============================================================================
# SECTION 3: APPLY FILTERS TO DATA
# ============================================================================

# Create boolean mask for all filter conditions
mask = (
    (trip_df["tpep_pickup_datetime"].dt.date >= start_date) &
    (trip_df["tpep_pickup_datetime"].dt.date <= end_date) &
    (trip_df["pickup_hour"] >= hour_range[0]) &
    (trip_df["pickup_hour"] <= hour_range[1]) &
    (trip_df["payment_type"].isin(selected_payment_codes))
)

# Apply all filters to get filtered dataset (NO .copy() to save memory)
filtered = trip_df.loc[mask]

# Add zone names to filtered data using left join with fallback
zones_lookup = zones_df.set_index("LocationID")["zone"].to_dict()

# Map zone IDs to names with fallback for missing zones (more memory efficient)
filtered = filtered.assign(
    PU_zone=filtered["PULocationID"].map(lambda x: zones_lookup.get(x, f"Unknown Zone {x}")),
    DO_zone=filtered["DOLocationID"].map(lambda x: zones_lookup.get(x, f"Unknown Zone {x}"))
)

# ============================================================================
# SECTION 4: DISPLAY OVERVIEW METRICS
# ============================================================================

# Organize dashboard with tabs
overview_tab, viz_tab = st.tabs(["📈 Overview & Metrics", "📊 Visualizations & Insights"])

with overview_tab:
    st.markdown("## Summary Statistics")
    
    # Calculate key metrics (compute directly without intermediate storage)
    total_trips = len(filtered)
    avg_fare = filtered["fare_amount"].mean() if total_trips > 0 else 0.0
    total_revenue = filtered["total_amount"].sum() if total_trips > 0 else 0.0
    avg_distance = filtered["trip_distance"].mean() if total_trips > 0 else 0.0
    avg_duration = filtered["trip_duration_minutes"].mean() if total_trips > 0 else 0.0
    
    # Display metrics in a 5-column layout
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric(
        "Total Trips",
        f"{total_trips:,}",
        help=f"Number of taxi trips in selected date/hour/payment filter"
    )
    col2.metric(
        "Average Fare",
        f"${avg_fare:.2f}",
        help="Mean fare amount (meter distance and time only, excl. tips)"
    )
    col3.metric(
        "Total Revenue",
        f"${total_revenue:,.0f}",
        help="Sum of all trip charges (meter fare + surcharge)"
    )
    col4.metric(
        "Avg Distance",
        f"{avg_distance:.2f} mi",
        help="Mean trip distance in miles"
    )
    col5.metric(
        "Avg Duration",
        f"{avg_duration:.1f} min",
        help="Mean trip duration in minutes"
    )
    
    # Display data processing summary
    st.markdown("---")
    st.markdown("### Data Processing Summary")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.info(f"""
        **Dataset:** NYC Yellow Taxi - January 2024  
        **Total rows after cleaning:** {summary['final_rows']:,}  
        **Rows removed (null values):** {summary['rows_removed_null']:,}  
        **Rows removed (invalid fare/distance):** {summary['rows_removed_invalid']:,}  
        **Rows removed (invalid timestamps):** {summary['rows_removed_time']:,}  
        **Rows removed (outside January 2024):** {summary.get('rows_removed_jan', 0):,}
        """)
    
    with col_b:
        st.success(f"""
        **Current filter applied:** ✓  
        **Filtered rows displayed:** {total_trips:,}  
        **Date range:** {start_date} to {end_date}  
        **Hour range:** {hour_range[0]}:00 - {hour_range[1]}:59  
        **Payment methods:** {len(selected_payment_codes)}/{len(payment_options)}
        """)

# ============================================================================
# SECTION 5: DISPLAY VISUALIZATIONS WITH INSIGHTS
# ============================================================================

with viz_tab:
    if filtered.empty:
        st.warning(
            "⚠️ No trips match the current filters. Try expanding your date range, hour range, or payment type selections."
        )
    else:
        # Visualization 1: Top 10 Pickup Zones (Bar Chart)
        st.markdown("## 1. Top 10 Pickup Zones by Trip Count")
        st.markdown("Which pickup zones generate the most taxi demand?")
        
        busy_zones = (
            filtered.groupby("PU_zone")
            .size()
            .reset_index(name="trips")
            .sort_values("trips", ascending=False)
            .head(10)
        )
        
        fig1 = center_titles(
            px.bar(
            busy_zones,
            x="PU_zone",
            y="trips",
            title="Top 10 Pickup Zones",
            labels={"PU_zone": "Pickup Zone", "trips": "Number of Trips"},
            color="trips",
            color_continuous_scale="Viridis"
            )
        )
        fig1.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Insights for Visualization 1
        if not busy_zones.empty:
            top_zone = busy_zones.iloc[0]["PU_zone"]
            top_trips = int(busy_zones.iloc[0]["trips"])
            top_share = 100.0 * top_trips / total_trips
            top10_share = 100.0 * busy_zones["trips"].sum() / total_trips
            
            st.markdown(f"""
            **Insights:**
            - **{top_zone}** is the busiest pickup zone with **{top_trips:,}** trips ({top_share:.1f}% of all filtered trips)
            - The top 10 zones account for **{top10_share:.1f}%** of total demand, indicating strong geographic concentration
            - This concentration suggests targeted driver dispatch and marketing efforts could be highly effective
            """)
        
        st.markdown("---")
        
        # Visualization 2: Average Fare by Hour of Day (Line Chart)
        st.markdown("## 2. Average Fare by Pickup Hour")
        st.markdown("How does the average fare vary throughout the day?")
        
        hourly_fare = (
            filtered.groupby("pickup_hour")["fare_amount"]
            .mean()
            .reset_index()
            .sort_values("pickup_hour")
        )
        
        fig2 = center_titles(
            px.line(
            hourly_fare,
            x="pickup_hour",
            y="fare_amount",
            title="Average Fare by Hour of Day",
            markers=True,
            labels={"pickup_hour": "Hour of Day", "fare_amount": "Average Fare ($)"}
            )
        )
        fig2.update_xaxes(range=[0, 23])  # Set x-axis range to cover all hours
        fig2.update_xaxes(dtick=1)
        fig2.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Insights for Visualization 2
        if not hourly_fare.empty:
            peak_idx = hourly_fare["fare_amount"].idxmax()
            low_idx = hourly_fare["fare_amount"].idxmin()
            peak_hour = int(hourly_fare.loc[peak_idx, "pickup_hour"])
            peak_fare = hourly_fare.loc[peak_idx, "fare_amount"]
            low_hour = int(hourly_fare.loc[low_idx, "pickup_hour"])
            low_fare = hourly_fare.loc[low_idx, "fare_amount"]
            fare_range = peak_fare - low_fare
            
            st.markdown(f"""
            **Insights:**
            - **Highest average fare:** {peak_hour}:00 (${peak_fare:.2f}) - likely peak business hours with longer trips
            - **Lowest average fare:** {low_hour}:00 (${low_fare:.2f}) - typically off-peak periods with shorter distances
            - **Fare variance:** ${fare_range:.2f} range ({100*fare_range/low_fare:.0f}% swing) suggests strong time-of-day pricing dynamics
            """)
        
        st.markdown("---")
        
        # Visualization 3: Trip Distance Distribution (Histogram)
        st.markdown("## 3. Distribution of Trip Distances")
        st.markdown("What are typical trip lengths in NYC taxi service?")
        
        fig3 = center_titles(
            px.histogram(
            filtered,
            x="trip_distance",
            nbins=50,
            title="Trip Distance Distribution",
            labels={"trip_distance": "Trip Distance (miles)"},
            color_discrete_sequence=["#636EFA"]
            )
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Insights for Visualization 3
        median_dist = filtered["trip_distance"].median()
        p75_dist = filtered["trip_distance"].quantile(0.75)
        p90_dist = filtered["trip_distance"].quantile(0.90)
        p95_dist = filtered["trip_distance"].quantile(0.95)
        
        st.markdown(f"""
        **Insights:**
        - **Median trip distance:** {median_dist:.2f} miles
        - **75th percentile:** {p75_dist:.2f} miles | **90th percentile:** {p90_dist:.2f} miles | **95th percentile:** {p95_dist:.2f} miles
        - Most trips are short urban journeys (under {p90_dist:.1f} miles), with a long tail of longer intercity/airport trips
        - The distribution shape indicates NYC taxis serve primarily short-haul commuting and local transportation
        """)
        
        st.markdown("---")
        
        # Visualization 4: Payment Type Breakdown (Pie Chart)
        st.markdown("## 4. Payment Type Distribution")
        st.markdown("How do passengers prefer to pay for trips?")
        
        payment_breakdown = (
            filtered["payment_type"]
            .map(PAYMENT_TYPE_MAP)
            .fillna("Other")
            .value_counts()
            .reset_index(name="count")
        )
        payment_breakdown.columns = ["payment_type", "count"]
        
        fig4 = center_titles(
            px.pie(
            payment_breakdown,
            names="payment_type",
            values="count",
            title="Payment Type Distribution",
            hole=0.3
            )
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
        
        # Insights for Visualization 4
        if not payment_breakdown.empty:
            top_payment = payment_breakdown.iloc[0]["payment_type"]
            top_payment_count = int(payment_breakdown.iloc[0]["count"])
            top_payment_pct = 100.0 * top_payment_count / payment_breakdown["count"].sum()
            
            st.markdown(f"""
            **Insights:**
            - **{top_payment}** dominates at **{top_payment_pct:.1f}%** of trips - digital payments strongly preferred in NYC
            - Card payment visibility enables better tip capture and tip analysis
            - Payment type distribution informs mobile app design and terminal infrastructure investments
            """)
        
        st.markdown("---")
        
        # Visualization 5: Heatmap - Trips by Day and Hour
        st.markdown("## 5. Demand Heatmap: Trips by Day of Week & Hour")
        st.markdown("When is a demand for taxi service highest? (Weekly and hourly patterns)")
        
        heatmap_data = (
            filtered.groupby(["pickup_day_of_week", "pickup_hour"])
            .size()
            .reset_index(name="trips")
        )
        
        # Enforce day order (Monday through Sunday)
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap_data["pickup_day_of_week"] = pd.Categorical(
            heatmap_data["pickup_day_of_week"],
            categories=day_order,
            ordered=True
        )
        
        heatmap_pivot = heatmap_data.pivot(
            index="pickup_day_of_week",
            columns="pickup_hour",
            values="trips"
        ).fillna(0)
        
        fig5 = center_titles(
            px.imshow(
                heatmap_pivot,
                labels=dict(x="Hour of Day", y="Day of Week", color="Trip Count"),
                title="Demand Heatmap: Trips by Day and Hour",
                color_continuous_scale="YlOrRd",
                aspect="auto"
            )
        )
        fig5.update_xaxes(dtick=1) # Show every hour on x-axis
        fig5.update_layout(height=400)
        st.plotly_chart(fig5, use_container_width=True)
        
        # Insights for Visualization 5
        flat_heatmap = heatmap_pivot.stack()
        if not flat_heatmap.empty:
            max_key = flat_heatmap.idxmax()
            max_trips = int(flat_heatmap.max())
            busiest_day, busiest_hour = max_key
            
            st.markdown(f"""
            **Insights:**
            - **Peak demand time:** {busiest_day} around {int(busiest_hour)}:00 with {max_trips:,} trips
            - Weekday peak hours (7-9am, 5-7pm) show urban commute patterns
            - Weekend patterns differ, with more distributed demand throughout daytime hours
            - These insights inform surge pricing strategies and driver incentive programs
            """)

# ============================================================================
# FOOTER: Documentation & Credits
# ============================================================================

st.markdown("---")
st.markdown("""
### About This Dashboard
- **Data Source:** NYC Taxi and Limousine Commission (TLC)
- **Dataset:** Yellow Taxi Trip Records - January 2024 (~3M trips)
- **Processing:** Pandas data cleaning, Plotly visualizations, Streamlit deployment
- **Maintained:** COMP 3610 - Big Data Analytics Assignment #1

**Note:** All data is cleaned according to assignment specifications (removed null values, invalid fares/distances, and temporal anomalies).
""")

st.markdown("""
### AI Disclosure
This dashboard was developed with code suggestions from Deepseek and VScode Autocompletions. 
All code was reviewed and adjusted by the student author to meet assignment requirements.
""")

