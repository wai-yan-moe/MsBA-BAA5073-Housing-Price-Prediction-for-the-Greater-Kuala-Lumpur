from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET_RAW = "Price"
TARGET_LOG = "Price_Log"

KLCC_LAT = 3.15687
KLCC_LON = 101.71473
GKL_BBOX = {
    "min_lat": 2.987,
    "min_lon": 101.575,
    "max_lat": 3.275,
    "max_lon": 101.816,
}

PRICE_LOWER = 42_000
PRICE_UPPER_QUANTILE = 0.995
BUILT_UP_LOWER = 250
LAND_AREA_LOWER = 500

PROPERTY_TYPE_MAP = {
    "Condominium": "Condominium",
    "Serviced Residence": "Serviced Residence",
    "Apartment": "Apartment",
    "Flat": "Flat",
    "Townhouse": "Townhouse",
    "Penthouse": "Condominium",
    "Studio": "Apartment",
    "Duplex": "Condominium",
    "1-sty Terrace/Link House": "Terrace/Link House",
    "2-sty Terrace/Link House": "Terrace/Link House",
    "2.5-sty Terrace/Link House": "Terrace/Link House",
    "3-sty Terrace/Link House": "Terrace/Link House",
    "4-sty Terrace/Link House": "Terrace/Link House",
    "Terrace/Link House": "Terrace/Link House",
    "Bungalow": "Bungalow",
    "Semi-detached House": "Semi-Detached House",
    "2-sty Semi-detached House": "Semi-Detached House",
    "2.5-sty Semi-detached House": "Semi-Detached House",
    "3-sty Semi-detached House": "Semi-Detached House",
    "Cluster House": "Cluster House",
    "Cluster Homes": "Cluster House",
}

LANDED_TYPES = {"Terrace/Link House", "Bungalow", "Semi-Detached House", "Cluster House"}

LAND_EXCLUSIONS = {
    "Residential Land", "Bungalow Land", "Agricultural Land",
    "Industrial Land", "Commercial Land",
}

FURNISHING_MAP = {
    "Fully Furnished": "Fully Furnished",
    "Partly Furnished": "Partly Furnished",
    "Unfurnished": "Unfurnished",
    "Not Furnished": "Unfurnished",
}

FEATURES_TO_DROP_BEFORE_MODELLING = [
    "Price", "Price_per_SqFt_BuiltUp", "Price_per_SqFt_Land",
    "Latitude", "Longitude", "Location",
]

# -- Colour palette: clean professional + vibrant accents --
PALETTE = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent1": "#F18F01",
    "accent2": "#C73E1D",
    "accent3": "#3B1F2B",
    "success": "#44AF69",
    "light_bg": "#F7F7F7",
    "grid": "#E0E0E0",
    "text": "#2D2D2D",
}

MODEL_COLOURS = {
    "Ridge": "#90A4AE",
    "LightGBM": "#2E86AB",
    "XGBoost": "#F18F01",
    "CatBoost": "#A23B72",
}

CATEGORICAL_PALETTE = [
    "#2E86AB", "#F18F01", "#A23B72", "#44AF69", "#C73E1D",
    "#6C5B7B", "#F67280", "#355C7D", "#C06C84", "#3B1F2B",
]
