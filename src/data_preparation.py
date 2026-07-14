"""
Data preparation pipeline for the Greater Kuala Lumpur housing price dataset.

Takes the raw Kaggle CSV through cleaning, feature engineering, and geospatial
feature derivation to produce a modelling-ready dataset.

Geocoding coordinates and POI data were pre-collected via OpenStreetMap Nominatim
and Overpass APIs (see _archive/ for source files). All other steps are
reproduced from scratch here.
"""

import json
import re
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd

from src.config import (
    DATA_RAW, DATA_PROCESSED, RANDOM_STATE,
    KLCC_LAT, KLCC_LON,
    PRICE_LOWER, PRICE_UPPER_QUANTILE,
    BUILT_UP_LOWER, LAND_AREA_LOWER,
    LANDED_TYPES, LAND_EXCLUSIONS, FURNISHING_MAP,
)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    rlat1, rlon1, rlat2, rlon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def load_raw(path=None):
    path = path or DATA_RAW / "property_listings_kl.csv"
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def remove_land_listings(df):
    mask = df["Property Type"].str.lower().apply(
        lambda x: any(kw in str(x) for kw in [
            "land", "agricultural", "industrial", "commercial"
        ])
    )
    removed = mask.sum()
    df = df[~mask].copy()
    print(f"Removed {removed:,} land listings -> {len(df):,} rows")
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"Removed {before - len(df):,} duplicates -> {len(df):,} rows")
    return df


def parse_price(df):
    df["Price"] = (
        df["Price"]
        .str.replace("RM ", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Price"]).copy()
    print(f"Parsed price, dropped {before - len(df):,} missing -> {len(df):,} rows")
    return df


def parse_size(df):
    df["Built_up_sqft"] = np.nan
    df["Land_Area_sqft"] = np.nan

    size_str = df["Size"].astype(str).str.lower()
    prop_type_str = df["Property Type"].astype(str).str.lower()

    bu_pattern = re.compile(r"built-up\s*:\s*([\d,]+(?:\.\d+)?)")
    la_pattern = re.compile(r"land area\s*:\s*([\d,]+(?:\.\d+)?)")
    dim_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)")
    num_pattern = re.compile(r"^([\d,]+(?:\.\d+)?)\s*sq")

    non_landed_kw = ["condominium", "apartment", "serviced residence", "flat",
                     "penthouse", "duplex", "studio", "soho"]
    landed_kw = ["bungalow", "terrace", "semi-detached", "link house",
                 "townhouse", "cluster"]

    for i in df.index:
        s = size_str.at[i]
        pt = prop_type_str.at[i]
        if s == "nan":
            continue

        bu_m = bu_pattern.search(s)
        la_m = la_pattern.search(s)

        if bu_m:
            try:
                df.at[i, "Built_up_sqft"] = float(bu_m.group(1).replace(",", ""))
            except ValueError:
                pass

        if la_m:
            try:
                df.at[i, "Land_Area_sqft"] = float(la_m.group(1).replace(",", ""))
            except ValueError:
                pass

        dim_m = dim_pattern.search(s)
        if dim_m:
            try:
                area = float(dim_m.group(1)) * float(dim_m.group(2))
                if "land area" in s or (
                    pd.isna(df.at[i, "Land_Area_sqft"])
                    and any(kw in pt for kw in landed_kw)
                ):
                    df.at[i, "Land_Area_sqft"] = area
                elif pd.isna(df.at[i, "Built_up_sqft"]):
                    df.at[i, "Built_up_sqft"] = area
            except ValueError:
                pass

        if pd.isna(df.at[i, "Built_up_sqft"]) and pd.isna(df.at[i, "Land_Area_sqft"]):
            num_m = num_pattern.search(s)
            if num_m:
                try:
                    val = float(num_m.group(1).replace(",", ""))
                    if any(kw in pt for kw in non_landed_kw):
                        df.at[i, "Built_up_sqft"] = val
                    else:
                        df.at[i, "Built_up_sqft"] = val
                except ValueError:
                    pass

        if any(kw in pt for kw in non_landed_kw):
            if pd.isna(df.at[i, "Built_up_sqft"]) and not pd.isna(df.at[i, "Land_Area_sqft"]):
                df.at[i, "Built_up_sqft"] = df.at[i, "Land_Area_sqft"]
                df.at[i, "Land_Area_sqft"] = 0
            elif not pd.isna(df.at[i, "Built_up_sqft"]):
                df.at[i, "Land_Area_sqft"] = 0

    print(f"Parsed size -> Built_up missing: {df['Built_up_sqft'].isna().sum():,}, "
          f"Land_Area missing: {df['Land_Area_sqft'].isna().sum():,}")
    return df


def parse_rooms(df):
    def _parse(val):
        if pd.isna(val):
            return np.nan
        val = str(val).lower().strip()
        if "studio" in val:
            return 1
        if "+" in val:
            try:
                return sum(int(p.strip()) for p in val.split("+") if p.strip().isdigit())
            except Exception:
                return np.nan
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return np.nan

    df["Rooms"] = df["Rooms"].apply(_parse).astype("float64")
    print(f"Parsed rooms -> missing: {df['Rooms'].isna().sum():,}")
    return df


def parse_bathrooms(df):
    df["Bathrooms"] = pd.to_numeric(df["Bathrooms"], errors="coerce")
    return df


def parse_car_parks(df):
    df["Car Parks"] = pd.to_numeric(df["Car Parks"], errors="coerce")
    df.rename(columns={"Car Parks": "Car_Parks"}, inplace=True)
    return df


def simplify_property_type(df):
    mapping = {
        "condominium": "Condominium",
        "serviced residence": "Serviced Residence",
        "apartment": "Apartment",
        "flat": "Flat",
        "townhouse": "Townhouse",
        "terrace/link house": "Terrace/Link House",
        "semi-detached house": "Semi-Detached House",
        "bungalow": "Bungalow",
        "cluster house": "Cluster House",
    }
    extra = {
        "penthouse": "Condominium",
        "duplex": "Condominium",
        "studio": "Apartment",
        "soho": "Apartment",
        "link house": "Terrace/Link House",
        "semi-detached": "Semi-Detached House",
        "terrace": "Terrace/Link House",
        "cluster": "Cluster House",
    }

    def _simplify(pt):
        pt_low = str(pt).lower()
        for key, val in mapping.items():
            if key in pt_low:
                return val
        for key, val in extra.items():
            if key in pt_low:
                return val
        if pt_low == "nan" or pd.isna(pt):
            return "Other"
        return "Other"

    df["Property_Type_Simplified"] = df["Property Type"].apply(_simplify)
    print(f"Simplified property types:\n{df['Property_Type_Simplified'].value_counts()}")
    return df


def clean_furnishing(df):
    def _clean(val):
        if pd.isna(val):
            return "Unknown/Missing"
        val = str(val).lower().strip()
        for key, mapped in FURNISHING_MAP.items():
            if key.lower() in val:
                return mapped
        return "Unknown/Missing"

    df["Furnishing_Cleaned"] = df["Furnishing"].apply(_clean)
    print(f"Cleaned furnishing:\n{df['Furnishing_Cleaned'].value_counts()}")
    return df


def impute_missing(df):
    num_cols = ["Built_up_sqft", "Rooms", "Bathrooms", "Car_Parks"]
    for col in num_cols:
        df[col] = df.groupby("Property_Type_Simplified")[col].transform(
            lambda x: x.fillna(x.median())
        )
        remaining = df[col].isna().sum()
        if remaining > 0:
            df[col] = df[col].fillna(df[col].median())

    is_landed = df["Property_Type_Simplified"].isin(LANDED_TYPES)
    df.loc[is_landed, "Land_Area_sqft"] = (
        df.loc[is_landed]
        .groupby("Property_Type_Simplified")["Land_Area_sqft"]
        .transform(lambda x: x.fillna(x.median()))
    )
    df.loc[~is_landed, "Land_Area_sqft"] = df.loc[~is_landed, "Land_Area_sqft"].fillna(0)

    remaining_land = df["Land_Area_sqft"].isna().sum()
    if remaining_land > 0:
        df["Land_Area_sqft"] = df["Land_Area_sqft"].fillna(0)

    print("Imputed missing values (median by property type)")
    return df


def treat_outliers(df):
    before = len(df)

    df = df[df["Price"] >= PRICE_LOWER].copy()
    upper_price = df["Price"].quantile(PRICE_UPPER_QUANTILE)
    df = df[df["Price"] <= upper_price].copy()

    df = df[df["Built_up_sqft"] >= BUILT_UP_LOWER].copy()
    upper_bu = df["Built_up_sqft"].quantile(PRICE_UPPER_QUANTILE)
    df = df[df["Built_up_sqft"] <= upper_bu].copy()

    landed_mask = df["Property_Type_Simplified"].isin(LANDED_TYPES)
    upper_la = df.loc[landed_mask, "Land_Area_sqft"].quantile(PRICE_UPPER_QUANTILE)
    bad_land = landed_mask & (
        (df["Land_Area_sqft"] < LAND_AREA_LOWER) |
        (df["Land_Area_sqft"] > upper_la)
    )
    df = df[~bad_land].copy()

    df = df[df["Rooms"].between(1, 10)].copy()
    df = df[df["Bathrooms"].between(1, 10)].copy()
    df = df[df["Car_Parks"].between(0, 10)].copy()

    print(f"Outlier treatment removed {before - len(df):,} rows -> {len(df):,} rows")
    return df


def add_log_price(df):
    df["Price_Log"] = np.log1p(df["Price"])
    return df


def add_price_per_sqft(df):
    df["Price_per_SqFt_BuiltUp"] = df["Price"] / df["Built_up_sqft"]
    df.loc[df["Land_Area_sqft"] > 0, "Price_per_SqFt_Land"] = (
        df.loc[df["Land_Area_sqft"] > 0, "Price"] /
        df.loc[df["Land_Area_sqft"] > 0, "Land_Area_sqft"]
    )
    df["Price_per_SqFt_Land"] = df["Price_per_SqFt_Land"].fillna(0)
    return df


def add_geospatial_features(df, geocode_path=None, poi_path=None):
    from pathlib import Path
    archive = Path(__file__).resolve().parent.parent / "_archive"
    geocode_path = geocode_path or archive / "geocoded_gkl_unique_property_locations_combined.csv"
    poi_path = poi_path or archive / "poi_coordinates_gkl.json"

    geo_df = pd.read_csv(geocode_path)
    loc_col = "Location_String_Used_For_Geocoding" if "Location_String_Used_For_Geocoding" in geo_df.columns else "Location"
    geo_lookup = dict(zip(geo_df[loc_col], zip(geo_df["Latitude"], geo_df["Longitude"])))

    df["Latitude"] = df["Location"].map(lambda x: geo_lookup.get(x, (np.nan, np.nan))[0])
    df["Longitude"] = df["Location"].map(lambda x: geo_lookup.get(x, (np.nan, np.nan))[1])

    before = len(df)
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    print(f"Geocoding: dropped {before - len(df):,} rows without coords -> {len(df):,} rows")

    with open(poi_path, "r") as f:
        all_pois = json.load(f)

    def _vectorised_haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        rlat1, rlon1 = np.radians(lat1), np.radians(lon1)
        rlat2, rlon2 = np.radians(lat2), np.radians(lon2)
        dlat = rlat2 - rlat1
        dlon = rlon2 - rlon1
        a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    df["Dist_KLCC"] = _vectorised_haversine(
        df["Latitude"].values, df["Longitude"].values, KLCC_LAT, KLCC_LON
    )

    for category, coords_list in all_pois.items():
        if not coords_list:
            df[f"Dist_{category}"] = np.nan
            continue
        poi_arr = np.array(coords_list)
        prop_lat = df["Latitude"].values[:, np.newaxis]
        prop_lon = df["Longitude"].values[:, np.newaxis]
        poi_lat = poi_arr[:, 0][np.newaxis, :]
        poi_lon = poi_arr[:, 1][np.newaxis, :]
        all_dists = _vectorised_haversine(prop_lat, prop_lon, poi_lat, poi_lon)
        df[f"Dist_{category}"] = all_dists.min(axis=1)

    print("Added geospatial distance features")
    return df


def one_hot_encode(df):
    df = pd.get_dummies(
        df,
        columns=["Property_Type_Simplified", "Furnishing_Cleaned"],
        dtype=int,
    )
    return df


def run_pipeline(raw_path=None, save=True):
    df = load_raw(raw_path)
    df = remove_land_listings(df)
    df = remove_duplicates(df)
    df = parse_price(df)
    df = parse_size(df)
    df = parse_rooms(df)
    df = parse_bathrooms(df)
    df = parse_car_parks(df)
    df = simplify_property_type(df)
    df = clean_furnishing(df)
    df = impute_missing(df)
    df = treat_outliers(df)
    df = add_log_price(df)
    df = add_price_per_sqft(df)
    df = add_geospatial_features(df)

    cols_to_drop = ["Size", "Property Type", "Furnishing"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        eda_path = DATA_PROCESSED / "housing_eda.csv"
        df.to_csv(eda_path, index=False)
        print(f"Saved EDA dataset to {eda_path} ({len(df):,} rows, {df.shape[1]} columns)")

    df_encoded = one_hot_encode(df)
    df_encoded = df_encoded.drop(columns=[c for c in ["Location"] if c in df_encoded.columns])

    if save:
        out_path = DATA_PROCESSED / "housing_model.csv"
        df_encoded.to_csv(out_path, index=False)
        print(f"Saved model dataset to {out_path} ({len(df_encoded):,} rows, {df_encoded.shape[1]} columns)")

    return df, df_encoded


if __name__ == "__main__":
    run_pipeline()
