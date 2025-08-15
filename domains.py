import numpy as np
import pandas as pd
import xarray as xr
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional, Tuple

from .utils import AggregationType, get_time_dimension, LOCATION_DATABASE, apply_aggregation

# ============================================================================
# Spatial Domain Classes
# ============================================================================

@dataclass
class SpatialDomain:
    """
    Defines spatial scope for event evaluation.
    
    Supports multiple types:
    - point: Single lat/lon coordinate or named location
    - bbox: Bounding box defined by lat/lon min/max
    - radius: Circle around a point with radius in km
    - shapefile: Path to shapefile for complex regions
    - iso: ISO/RTO region (ERCOT, CAISO, etc.)
    """
    type: str  # "point", "bbox", "radius", "shapefile", "iso"
    
    # For point or named location
    lat: Optional[float] = None
    lon: Optional[float] = None
    location: Optional[str] = None  # Named location from database
    
    # For bbox
    lat_min: Optional[float] = None
    lat_max: Optional[float] = None
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None
    
    # For radius (around point)
    radius_km: Optional[float] = None
    
    # For shapefile
    shapefile_path: Optional[str] = None
    
    # For ISO/RTO regions
    iso: Optional[str] = None
    
    def __post_init__(self):
        """Resolve named locations and validate."""
        if self.type == "point" and self.location:
            if self.location.lower() in LOCATION_DATABASE:
                loc_data = LOCATION_DATABASE[self.location.lower()]
                self.lat = loc_data["lat"]
                self.lon = loc_data["lon"]
            else:
                raise ValueError(f"Unknown location: {self.location}")
        self.validate()
    
    def validate(self):
        """Validate spatial domain parameters."""
        if self.type == "point":
            if self.lat is None or self.lon is None:
                raise ValueError("Point domain requires lat and lon")
        elif self.type == "bbox":
            if any(x is None for x in [self.lat_min, self.lat_max, self.lon_min, self.lon_max]):
                raise ValueError("Bbox domain requires lat_min, lat_max, lon_min, lon_max")
            # Only compare if all are not None
            if self.lat_min is not None and self.lat_max is not None and self.lat_min >= self.lat_max:
                raise ValueError("lat_min must be less than lat_max")
            if self.lon_min is not None and self.lon_max is not None and self.lon_min >= self.lon_max:
                raise ValueError("lon_min must be less than lon_max")
        elif self.type == "radius":
            if self.lat is None or self.lon is None or self.radius_km is None:
                raise ValueError("Radius domain requires lat, lon, and radius_km")
            if self.radius_km <= 0:
                raise ValueError("radius_km must be positive")
        elif self.type == "shapefile":
            if self.shapefile_path is None:
                raise ValueError("Shapefile domain requires shapefile_path")
        elif self.type == "iso":
            if self.iso is None:
                raise ValueError("ISO domain requires iso code")
            valid_isos = ["ERCOT", "CAISO", "MISO", "PJM", "NYISO", "ISO-NE", "SPP", "WECC"]
            if self.iso not in valid_isos:
                raise ValueError(f"Invalid ISO: {self.iso}. Must be one of {valid_isos}")
        else:
            raise ValueError(f"Unknown spatial domain type: {self.type}")
    
    def apply_to_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply spatial selection to dataset."""
        if self.type == "point":
            # Find nearest point, keep lat and lon dimensions
            return ds.sel(lat=self.lat, lon=self.lon, method="nearest")
        
        elif self.type == "bbox":
            # Select bounding box
            return ds.sel(
                lat=slice(self.lat_min, self.lat_max),
                lon=slice(self.lon_min, self.lon_max)
            )
        
        elif self.type == "radius":
            # Calculate distances and mask
            return self._apply_radius_mask(ds)
        
        elif self.type == "shapefile":
            # Would integrate with geopandas/rasterio for production
            raise NotImplementedError("Shapefile support requires geopandas integration")
        
        elif self.type == "iso":
            # Would use predefined ISO boundaries
            raise NotImplementedError("ISO region support requires boundary data")
    
    def _apply_radius_mask(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply radius mask to dataset."""
        # Calculate distances using vectorized operations
        lat_rad = np.radians(ds.lat)
        lon_rad = np.radians(ds.lon)
        
        lat_center_rad = np.radians(self.lat)
        lon_center_rad = np.radians(self.lon)
        
        # Haversine formula
        dlat = lat_rad - lat_center_rad
        dlon = lon_rad - lon_center_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat_center_rad) * np.cos(lat_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        distances = 6371 * c  # Earth radius in km
        
        # Create mask
        mask = distances <= self.radius_km
        
        # Apply mask
        return ds.where(mask)


# ============================================================================
# Temporal Domain Classes
# ============================================================================

@dataclass
class TemporalDomain:
    """
    Defines temporal domain operations for continuous data.
    
    Supports:
    - Resampling (e.g., daily max, hourly mean)
    - Rolling windows with aggregation
    - Time-based filtering
    """
    # Window definition
    window: Optional[str] = None  # e.g., "1D", "6h", "7D"
    window_type: str = "rolling"  # "rolling", "resample", "custom"
    aggregation: Optional[AggregationType] = None
    
    # Time filtering
    time_filter: Optional[str] = None  # e.g., "business_hours", "night_only"
    
    def __post_init__(self):
        """Validate and process initialization."""
        pass
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalDomain':
        """Create from dictionary"""
        return cls(**data)
    
    def apply(self, data: xr.DataArray, time_dim: Optional[str] = None) -> xr.DataArray:
        """Apply temporal domain operations to continuous data."""
        # Auto-detect time dimension if not provided
        if time_dim is None:
            time_dim = get_time_dimension(data)
            if time_dim is None:
                # No time dimension found, return unchanged
                return data
        
        if self.window:
            return self._apply_window(data, time_dim)
        elif self.time_filter:
            return self._apply_time_filter(data, time_dim)
        else:
            return data
    
    def _apply_window(self, data: xr.DataArray, time_dim: str) -> xr.DataArray:
        """Apply simple rolling or resampling window (left-aligned)."""
        window_td = pd.Timedelta(self.window)
        if self.window_type == "rolling":
            time_diff = pd.Series(data[time_dim].values).diff().median()
            steps = int(window_td / time_diff)
            # Always left-align: shift by -(steps-1)
            if self.aggregation:
                rolled = data.rolling({time_dim: steps}, center=False)
                result = apply_aggregation(rolled, self.aggregation, dim=time_dim)
            else:
                result = data.rolling({time_dim: steps}, center=False).mean()
            return result.shift({time_dim: -(steps-1)})
        elif self.window_type == "resample":
            # xarray resample is left-aligned by default
            resampled = data.resample({time_dim: self.window})
            if self.aggregation:
                if self.aggregation == "max":
                    return resampled.max()
                elif self.aggregation == "min":
                    return resampled.min()
                elif self.aggregation == "mean":
                    return resampled.mean()
                elif self.aggregation == "sum":
                    return resampled.sum()
            return resampled.mean()
    
    def _apply_time_filter(self, data: xr.DataArray, time_dim: str) -> xr.DataArray:
        """Apply time-based filtering."""
        if self.time_filter == "business_hours":
            # Filter to 9 AM - 5 PM
            hour_mask = (data[time_dim].dt.hour >= 9) & (data[time_dim].dt.hour <= 17)
            return data.where(hour_mask)
        elif self.time_filter == "night_only":
            # Filter to 6 PM - 6 AM
            hour_mask = (data[time_dim].dt.hour >= 18) | (data[time_dim].dt.hour <= 6)
            return data.where(hour_mask)
        else:
            return data


# ============================================================================
# Temporal Pattern Classes
# ============================================================================

@dataclass
class TemporalPattern:
    """
    Handles temporal patterns and threshold logic.
    
    Supports:
    - Simple threshold + duration (e.g., temperature > 95F for 2+ consecutive days)
    - Complex patterns (e.g., 6+ hours < 5 m/s within any 24-hour period)
    - Season filtering
    """
    
    # Simple threshold + duration
    threshold: Optional[float] = None
    operator: str = ">"  # >, <, >=, <=, ==
    duration: Optional[str] = None  # e.g., "2D" for 2 consecutive days
    
    # Complex patterns
    pattern: Optional[str] = None  # e.g., "6h_in_24h"
    pattern_threshold: Optional[float] = None
    pattern_operator: str = ">"
    
    # Season filtering
    season_months: Optional[List[int]] = None  # Limit to specific months
    
    def __post_init__(self):
        """Validate configuration."""
        if self.pattern and self.threshold:
            raise ValueError("Cannot use both pattern and simple threshold")
        if self.pattern and not self.pattern_threshold:
            raise ValueError("Pattern requires pattern_threshold")
    
    def apply(self, data: xr.DataArray) -> xr.DataArray:
        """Apply pattern logic and return binary result. If duration is in days, output is daily; else, keep original resolution."""
        time_dim = get_time_dimension(data)
        if time_dim is None:
            raise ValueError("Cannot apply temporal pattern without time dimension")
        # Apply threshold to create binary data
        if self.pattern:
            binary = self._apply_complex_pattern(data, time_dim)
        else:
            binary = self._apply_simple_threshold(data)
        # Apply duration requirement
        if self.duration:
            binary = self._apply_duration_requirement(binary, time_dim)
        # Apply season filter
        if self.season_months:
            binary = self._apply_season_filter(binary, time_dim)
        # TODO: If/when supporting more complex patterning, consider reintroducing daily resampling logic here.
        return binary
    
    def _apply_simple_threshold(self, data: xr.DataArray) -> xr.DataArray:
        """Apply simple threshold comparison."""
        if self.threshold is None:
            raise ValueError("Simple threshold requires threshold value")
        
        if self.operator == ">":
            return data > self.threshold
        elif self.operator == ">=":
            return data >= self.threshold
        elif self.operator == "<":
            return data < self.threshold
        elif self.operator == "<=":
            return data <= self.threshold
        elif self.operator == "==":
            return np.isclose(data, self.threshold)
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")
    
    def _apply_complex_pattern(self, data: xr.DataArray, time_dim: str) -> xr.DataArray:
        """Apply complex temporal patterns like '6h_in_24h' (left-aligned, run-based)."""
        duration, window = self._parse_pattern()
        time_diff = pd.Series(data[time_dim].values).diff().median()
        window_steps = int(window / time_diff)
        duration_steps = int(duration / time_diff)
        if self.pattern_operator == ">":
            binary_data = data > self.pattern_threshold
        elif self.pattern_operator == ">=":
            binary_data = data >= self.pattern_threshold
        elif self.pattern_operator == "<":
            binary_data = data < self.pattern_threshold
        elif self.pattern_operator == "<=":
            binary_data = data <= self.pattern_threshold
        elif self.pattern_operator == "==":
            binary_data = np.isclose(data, self.pattern_threshold)
        else:
            raise ValueError(f"Unsupported pattern operator: {self.pattern_operator}")
        # Run-based detection: True if any run of duration_steps in window_steps
        def has_run(arr, run_length):
            count = 0
            for x in arr:
                if x:
                    count += 1
                    if count >= run_length:
                        return True
                else:
                    count = 0
            return False
        result = binary_data.rolling({time_dim: window_steps}).reduce(
            lambda x, axis: np.apply_along_axis(has_run, axis, x, duration_steps)
        )
        result = result.shift({time_dim: -(window_steps-1)})
        # If window is day-based, aggregate to daily
        if window.components.hours == 0 and window.components.minutes == 0 and window.components.seconds == 0:
            result = result.resample({time_dim: '1D'}).max()
        return result
    
    def _parse_pattern(self) -> Tuple[pd.Timedelta, pd.Timedelta]:
        """Parse complex pattern strings like '6h_in_24h'. Always use lowercase 'h'."""
        if not self.pattern:
            raise ValueError("No pattern specified")
        parts = self.pattern.split("_in_")
        if len(parts) != 2:
            raise ValueError(f"Invalid pattern format: {self.pattern}")
        duration = pd.Timedelta(parts[0])
        window = pd.Timedelta(parts[1])
        return duration, window
    
    def _apply_duration_requirement(self, binary: xr.DataArray, time_dim: str) -> xr.DataArray:
        """Apply consecutive duration requirements (left-aligned)."""
        if not self.duration:
            return binary

        # Convert the duration string (e.g., "2D") into a Timedelta object
        duration_td = pd.Timedelta(self.duration)
        
        # Automatically detect the time resolution of the data by finding the
        # median time difference between consecutive data points.
        #TODO: This is hacky and will not work for changing time resolution.
        time_diff = pd.Series(binary[time_dim].values).diff().median()

        # Calculate the number of consecutive time steps required to meet the duration
        steps_required = int(round(duration_td / time_diff))
        if steps_required < 1:
            steps_required = 1
        
        # Use a rolling sum to efficiently check for consecutive 'True' values.
        # If the sum over the window equals the number of steps, it means all
        # values in that window were 'True'. This is more performant than reduce.
        consecutive = (binary.rolling({time_dim: steps_required}).sum() == steps_required)
        
        # The rolling operation is right-aligned by default. Shift the result
        # to be left-aligned, so the timestamp reflects the start of the window.
        return consecutive.shift({time_dim: -(steps_required - 1)})
    
    def _apply_season_filter(self, binary: xr.DataArray, time_dim: str) -> xr.DataArray:
        """Filter to specific months."""
        if self.season_months is None:
            return binary
        
        month_mask = binary[time_dim].dt.month.isin(self.season_months)
        return binary.where(month_mask, False)


# ============================================================================
# Stage 3: Temporal Analysis (Post-Threshold)
# ============================================================================

@dataclass
class TemporalAnalysis:
    """
    Performs analysis on boolean data (post-threshold).
    
    This is Stage 3 of the temporal processing pipeline. It takes the True/False
    output from the thresholding stage and calculates frequencies, counts, and
    other persistence metrics over a specified window.
    """
    
    window: Optional[str] = None  # e.g., "1D", "6h", "7D"
    window_type: str = "rolling"  # "rolling", "resample"
    aggregation: Optional[AggregationType] = None  # mean (frequency), sum (count)
    
    def apply(self, binary_data: xr.DataArray) -> xr.DataArray:
        """Apply temporal operations to binary data."""
        if not self.window:
            return binary_data
        
        time_dim = get_time_dimension(binary_data)
        if time_dim is None:
            raise ValueError("Cannot apply temporal domain without time dimension")
        
        if self.window_type == "rolling":
            return self._apply_rolling_window(binary_data, time_dim)
        elif self.window_type == "resample":
            return self._apply_resample(binary_data, time_dim)
        else:
            raise ValueError(f"Unsupported window type: {self.window_type}")
    
    def _apply_rolling_window(self, binary_data: xr.DataArray, time_dim: str) -> xr.DataArray:
        """Apply rolling window to binary data (left-aligned). Shift before any thresholding or further logic."""
        window_td = pd.Timedelta(self.window)
        time_diff = pd.Series(binary_data[time_dim].values).diff().median()
        steps = int(window_td / time_diff)
        # Always left-align: shift by -(steps-1) BEFORE any thresholding
        if self.aggregation == "mean":
            result = binary_data.rolling({time_dim: steps}).mean()
        elif self.aggregation == "sum":
            result = binary_data.rolling({time_dim: steps}).sum()
        elif self.aggregation == "all":
            result = binary_data.rolling({time_dim: steps}).reduce(
                lambda x, axis: x.all(axis=axis)
            )
        elif self.aggregation == "any":
            result = binary_data.rolling({time_dim: steps}).reduce(
                lambda x, axis: x.any(axis=axis)
            )
        else:
            result = binary_data.rolling({time_dim: steps}).mean()
        result = result.shift({time_dim: -(steps-1)})
        return result
    
    def _apply_resample(self, binary_data: xr.DataArray, time_dim: str) -> xr.DataArray:
        """Apply resampling to binary data."""
        resampled = binary_data.resample({time_dim: self.window})
        
        if self.aggregation == "mean":
            return resampled.mean()  # Frequency
        elif self.aggregation == "sum":
            return resampled.sum()   # Count
        elif self.aggregation == "all":
            return resampled.all()   # All True
        elif self.aggregation == "any":
            return resampled.any()   # Any True
        else:
            return resampled.mean()  # Default to frequency
