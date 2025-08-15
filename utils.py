
import numpy as np
import xarray as xr
from typing import List, Dict, Any, Union, Optional, Tuple, Literal

# ============================================================================
# Type Definitions
# ============================================================================

OperatorType = Literal[">", ">=", "<", "<=", "==", "!="]
LogicalOperatorType = Literal["and", "or"]
AggregationType = Literal["mean", "max", "min", "sum", "all", "any", "percentile"]
ThresholdType = Literal["absolute", "percentile", "climatology", "anomaly"]


# ============================================================================
# Utility Functions
# ============================================================================

def get_time_dimension(data: xr.DataArray) -> Optional[str]:
    """
    Determine the appropriate time dimension for a DataArray.
    
    Priority order:
    1. lead_time (for forecast data)
    2. valid_time (for historical data)
    3. time (fallback)
    
    Args:
        data: xarray DataArray to check
        
    Returns:
        Time dimension name or None if no time dimension found
    """
    if "lead_time" in data.dims:
        return "lead_time"
    elif "valid_time" in data.dims:
        return "valid_time"
    elif "time" in data.dims:
        return "time"
    else:
        return None

def apply_aggregation(data: xr.DataArray, aggregation: AggregationType, dim: Union[str, List[str]], **kwargs) -> xr.DataArray:
    """Apply aggregation to data."""
    if isinstance(dim, str):
        dim = [dim]

    if aggregation == "mean":
        return data.mean(dim=dim, skipna=True)
    elif aggregation == "max":
        return data.max(dim=dim, skipna=True)
    elif aggregation == "min":
        return data.min(dim=dim, skipna=True)
    elif aggregation == "sum":
        return data.sum(dim=dim, skipna=True)
    elif aggregation == "all":
        return data.all(dim=dim)
    elif aggregation == "any":
        return data.any(dim=dim)
    elif aggregation == "percentile":
        q = kwargs.get('q', 0.5)
        return data.quantile(q, dim=dim, skipna=True)
    else:
        raise ValueError(f"Unknown aggregation type: {aggregation}")

# ============================================================================
# Named Locations Database
# ============================================================================

LOCATION_DATABASE = {
    # Power trading hubs
    "phoenix": {"lat": 33.45, "lon": -112.07, "iso": "WECC", "state": "AZ"},
    "chicago": {"lat": 41.88, "lon": -87.63, "iso": "PJM", "state": "IL"},
    "houston": {"lat": 29.76, "lon": -95.37, "iso": "ERCOT", "state": "TX"},
    "minneapolis": {"lat": 44.98, "lon": -93.27, "iso": "MISO", "state": "MN"},
    "denver": {"lat": 39.74, "lon": -104.99, "iso": "WECC", "state": "CO"},
    "dallas": {"lat": 32.78, "lon": -96.80, "iso": "ERCOT", "state": "TX"},
    "new_york": {"lat": 40.71, "lon": -74.01, "iso": "NYISO", "state": "NY"},
    "los_angeles": {"lat": 34.05, "lon": -118.24, "iso": "CAISO", "state": "CA"},
    "sacramento": {"lat": 38.58, "lon": -121.49, "iso": "CAISO", "state": "CA"},
    "burbank": {"lat": 34.18, "lon": -118.31, "iso": "CAISO", "state": "CA"},
    "boston": {"lat": 42.36, "lon": -71.06, "iso": "ISO-NE", "state": "MA"},
    "atlanta": {"lat": 33.75, "lon": -84.39, "iso": "SERC", "state": "GA"},
    "seattle": {"lat": 47.61, "lon": -122.33, "iso": "WECC", "state": "WA"},
    "miami": {"lat": 25.76, "lon": -80.19, "iso": "FRCC", "state": "FL"},
    
    # Additional key cities
    "washington_dc": {"lat": 38.90, "lon": -77.04, "iso": "PJM", "state": "DC"},
    "san_francisco": {"lat": 37.77, "lon": -122.42, "iso": "CAISO", "state": "CA"},
    "portland": {"lat": 45.52, "lon": -122.68, "iso": "WECC", "state": "OR"},
    "las_vegas": {"lat": 36.17, "lon": -115.14, "iso": "WECC", "state": "NV"},
    "omaha": {"lat": 41.26, "lon": -95.94, "iso": "SPP", "state": "NE"},
    "kansas_city": {"lat": 39.10, "lon": -94.58, "iso": "SPP", "state": "MO"},
}
