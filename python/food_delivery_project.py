"""
Food Delivery Data Analysis
=============================

End-to-end food delivery data analysis using:
- Data Cleaning
- Feature Engineering
- NumPy
- Pandas
- Dataset Export

Author: Deepan Hati
"""

import numpy as np
import pandas as pd


# CONFIGURATION


INPUT_FILE = "Food_Delivery.xlsx"
OUTPUT_FILE = "Food_Delivery_Clean.csv"

RANDOM_SEED = 42
PROFIT_MARGIN = 0.20


# DATA LOADING


def load_dataset(file_name):
    """Load the food delivery dataset from an Excel file."""

    return pd.read_excel(file_name)


# DATA CLEANING


def rename_columns(df):
    """Rename columns using consistent and analysis-friendly names."""

    column_mapping = {
        "Restaurant ID": "Restaurant_ID",
        "Restaurant name": "Restaurant_Name",
        "Order ID": "Order_ID",
        "Order Placed At": "Order_DateTime",
        "Order Status": "Order_Status",
        "Bill subtotal": "Bill_Subtotal",
        "Packaging charges": "Packaging_Charges",
        "Total": "Total_Amount",
        "Customer ID": "Customer_ID",
        "Items in order": "Food_Item",
        "KPT duration (minutes)": "Kitchen_Time",
        "Rider wait time (minutes)": "Rider_Wait_Time",
    }

    return df.rename(columns=column_mapping)


def create_datetime_features(df):
    """Convert order datetime and create useful date/time features."""

    df["Order_DateTime"] = pd.to_datetime(
        df["Order_DateTime"],
        errors="coerce"
    )

    df["Order_Date"] = df["Order_DateTime"].dt.date
    df["Order_Time"] = df["Order_DateTime"].dt.time
    df["Month"] = df["Order_DateTime"].dt.month_name()
    df["Month_Number"] = df["Order_DateTime"].dt.month
    df["Day"] = df["Order_DateTime"].dt.day
    df["Weekday"] = df["Order_DateTime"].dt.day_name()
    df["Hour"] = df["Order_DateTime"].dt.hour

    return df


def clean_distance(df):
    """Convert distance values such as '<1km' and '5km' into numeric values."""

    df["Distance"] = (
        df["Distance"]
        .astype("string")
        .str.strip()
        .replace("<1km", "0.5km")
        .str.replace("km", "", regex=False)
    )

    df["Distance"] = pd.to_numeric(
        df["Distance"],
        errors="coerce"
    )

    return df


def handle_missing_values(df):
    """Handle missing values in selected columns."""

    fill_values = {
        "Rating": 0,
        "Review": "No Review",
        "Instructions": "No Instructions",
        "Discount construct": "No Discount",
        "Cancellation / Rejection reason": "Not Cancelled",
        "Customer complaint tag": "No Complaint",
    }

    for column, value in fill_values.items():
        if column in df.columns:
            df[column] = df[column].fillna(value)

    return df


def clean_dataset(df):
    """Execute the complete data-cleaning workflow."""

    df_clean = df.copy()

    df_clean = rename_columns(df_clean)
    df_clean = create_datetime_features(df_clean)
    df_clean = clean_distance(df_clean)
    df_clean = handle_missing_values(df_clean)

    return df_clean


# FEATURE ENGINEERING


def create_profit(df):
    """
    Create estimated profit using a 20% assumed profit margin.

    Note:
        This is an analytical assumption because actual
        business cost data is not available in the dataset.
    """

    df["Profit"] = (
        df["Total_Amount"] * PROFIT_MARGIN
    )

    return df


def create_restaurant_type(df):
    """Create restaurant type categories from restaurant names."""

    restaurant_type = {
        "Aura Pizzas": "Fast Food",
        "Swaad": "North Indian",
        "The Belgian Waffle Co.": "Cafe",
        "Chinese Wok": "Chinese",
        "BOX8": "Cloud Kitchen",
        "Burger King": "Fast Food",
    }

    df["Restaurant_Type"] = (
        df["Restaurant_Name"]
        .map(restaurant_type)
        .fillna("Other")
    )

    return df


def create_payment_mode(df):
    """
    Generate payment modes using predefined probabilities.

    A fixed random seed is used to make the results reproducible.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    payment_modes = ["UPI", "Card", "Cash"]

    df["Payment_Mode"] = rng.choice(
        payment_modes,
        size=len(df),
        p=[0.60, 0.25, 0.15]
    )

    return df


def create_delivery_time(df):
    """Calculate estimated delivery time."""

    kitchen_time = df["Kitchen_Time"].fillna(15)
    rider_wait_time = df["Rider_Wait_Time"].fillna(5)

    df["Delivery_Time"] = (
        kitchen_time + rider_wait_time
    )

    return df


def engineer_features(df):
    """Execute all feature-engineering operations."""

    df = create_profit(df)
    df = create_restaurant_type(df)
    df = create_payment_mode(df)
    df = create_delivery_time(df)

    return df


# NUMPY ANALYSIS


def perform_numpy_analysis(df):
    """Perform numerical analysis using NumPy."""

    total_amount = (
        df["Total_Amount"]
        .dropna()
        .to_numpy()
    )

    delivery_time = (
        df["Delivery_Time"]
        .dropna()
        .to_numpy()
    )

    rating = (
        df["Rating"]
        .dropna()
        .to_numpy()
    )

    distance = (
        df["Distance"]
        .dropna()
        .to_numpy()
    )

    return {
        "total_revenue": np.sum(total_amount),
        "average_order_value": np.mean(total_amount),
        "maximum_order_value": np.max(total_amount),
        "minimum_order_value": np.min(total_amount),
        "median_order_value": np.median(total_amount),
        "order_value_std": np.std(total_amount),
        "average_delivery_time": np.mean(delivery_time),
        "maximum_delivery_time": np.max(delivery_time),
        "minimum_delivery_time": np.min(delivery_time),
        "average_rating": np.mean(rating),
        "average_distance": np.mean(distance),
        "revenue_25th_percentile": np.percentile(
            total_amount, 25
        ),
        "revenue_50th_percentile": np.percentile(
            total_amount, 50
        ),
        "revenue_75th_percentile": np.percentile(
            total_amount, 75
        ),
    }


# PANDAS ANALYSIS

def perform_pandas_analysis(df):
    """Perform business-focused analysis using Pandas."""

    revenue_by_restaurant = (
        df.groupby("Restaurant_Name")["Total_Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    return {
        "total_revenue": df["Total_Amount"].sum(),

        "average_order_value": (
            df["Total_Amount"].mean()
        ),

        "total_orders": (
            df["Order_ID"].nunique()
        ),

        "revenue_by_city": (
            df.groupby("City")["Total_Amount"]
            .sum()
            .sort_values(ascending=False)
        ),

        "revenue_by_restaurant": revenue_by_restaurant,

        "top_10_restaurants": (
            revenue_by_restaurant.head(10)
        ),

        "orders_by_city": (
            df["City"].value_counts()
        ),

        "orders_by_restaurant": (
            df["Restaurant_Name"].value_counts()
        ),

        "order_status": (
            df["Order_Status"].value_counts()
        ),

        "payment_mode": (
            df["Payment_Mode"].value_counts()
        ),

        "average_rating_by_restaurant": (
            df.groupby("Restaurant_Name")["Rating"]
            .mean()
            .sort_values(ascending=False)
        ),

        "average_delivery_time_by_city": (
            df.groupby("City")["Delivery_Time"]
            .mean()
            .sort_values()
        ),

        "average_profit_by_restaurant": (
            df.groupby("Restaurant_Name")["Profit"]
            .mean()
            .sort_values(ascending=False)
        ),

        "monthly_revenue": (
            df.groupby(
                ["Month_Number", "Month"]
            )["Total_Amount"]
            .sum()
            .sort_index()
        ),

        "weekday_revenue": (
            df.groupby("Weekday")["Total_Amount"]
            .sum()
        ),
    }


# EXPORT CLEAN DATASET

def export_clean_dataset(df):
    """Export the cleaned and feature-engineered dataset."""

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

# MAIN WORKFLOW

def main():
    """Run the complete Food Delivery Data Analysis workflow."""

    # Load dataset
    df = load_dataset(INPUT_FILE)

    # Data cleaning
    df_clean = clean_dataset(df)

    # Feature engineering
    df_clean = engineer_features(df_clean)

    # NumPy analysis
    numpy_results = perform_numpy_analysis(df_clean)

    # Pandas analysis
    pandas_results = perform_pandas_analysis(df_clean)

    # Export cleaned dataset
    export_clean_dataset(df_clean)

    return {
        "cleaned_data": df_clean,
        "numpy_analysis": numpy_results,
        "pandas_analysis": pandas_results,
    }

if __name__ == "__main__":
    results = main()
