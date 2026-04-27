import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Climate Dashboard", layout="wide")

st.title("Climate Trends Dashboard (2015–2026)")

# -----------------------------
# Load Data
# -----------------------------
def load_data():
    files = [
        "notebooks/data/ethiopia_clean.csv",
        "notebooks/data/kenya_clean.csv",
        "notebooks/data/nigeria_clean.csv",
        "notebooks/data/sudan_clean.csv",
        "notebooks/data/tanzania_clean.csv"
    ]

    dfs = []

    for file in files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dfs.append(df)
        else:
            st.warning(f"Missing: {file}")

    if len(dfs) == 0:
        st.error("No datasets found in data/ folder!")
        st.stop()

    return pd.concat(dfs, ignore_index=True)

# -----------------------------
# Load dataset
# -----------------------------
df = load_data()

# -----------------------------
# FIX: Date handling (CRITICAL)
# -----------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Filters")

countries = st.sidebar.multiselect(
    "Select Countries",
    options=df["Country"].unique(),
    default=df["Country"].unique()
)

year_range = st.sidebar.slider(
    "Select Year Range",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (2015, 2026)
)

variable = st.sidebar.selectbox(
    "Select Variable",
    ["T2M", "PRECTOTCORR", "RH2M", "WS2M"]
)

# -----------------------------
# Filter Data
# -----------------------------
filtered_df = df[
    (df["Country"].isin(countries)) &
    (df["Year"] >= year_range[0]) &
    (df["Year"] <= year_range[1])
]

# -----------------------------
# Time Series Plot
# -----------------------------
st.subheader(f" {variable} Trends Over Time")

monthly = filtered_df.groupby(
    ["Country", "Year", "Month"]
)[variable].mean().reset_index()

monthly["Date"] = pd.to_datetime(
    monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str)
)

fig, ax = plt.subplots(figsize=(12, 6))

for country in monthly["Country"].unique():
    subset = monthly[monthly["Country"] == country]
    ax.plot(subset["Date"], subset[variable], label=country)

ax.set_title(f"{variable} Over Time")
ax.set_xlabel("Date")
ax.set_ylabel(variable)
ax.legend()

st.pyplot(fig)

# -----------------------------
# Distribution Plot
# -----------------------------
st.subheader(f" Distribution of {variable}")

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.histplot(
    data=filtered_df,
    x=variable,
    hue="Country",
    bins=40,
    kde=True,
    ax=ax2
)

st.pyplot(fig2)

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.subheader(" Correlation Heatmap")

corr = filtered_df.select_dtypes(include=np.number).corr()

fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax3)

st.pyplot(fig3)

# -----------------------------
# Extreme Heat Events
# -----------------------------
st.subheader(" Extreme Heat Days (>35°C)")

if "T2M_MAX" in filtered_df.columns:
    extreme = filtered_df[filtered_df["T2M_MAX"] > 35]
    extreme_counts = extreme.groupby("Country").size()
    st.bar_chart(extreme_counts)
else:
    st.warning("T2M_MAX column not found")

# -----------------------------
# Summary Stats
# -----------------------------
st.subheader("Summary Statistics")

summary = filtered_df.groupby("Country")[variable].agg(["mean", "median", "std"])
st.dataframe(summary)