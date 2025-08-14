from datetime import datetime
from typing import List, Dict, Any, Union, Optional
import pandas as pd
import xarray as xr

from .events import Event, SimpleEvent, SpreadEvent
from .utils import get_time_dimension

# ============================================================================
# Event Calculator
# ============================================================================

class EventCalculator:
    """
    Main class for calculating event probabilities and climatologies.
    
    Features:
    - Load forecast and historical data from Zarr stores
    - Calculate event probabilities from ensemble forecasts
    - Compute historical climatologies
    - Handle conditional probabilities
    - Integrate with scoring system
    """
    
    def __init__(self, 
                 forecast_zarr_path: Optional[str] = None,
                 historical_zarr_path: Optional[str] = None,
                 station_zarr_path: Optional[str] = None):
        """
        Initialize calculator with paths to zarr stores.
        
        Args:
            forecast_zarr_path: Path to NWP ensemble data
            historical_zarr_path: Path to ERA5 reanalysis data
            station_zarr_path: Path to station observation data
        """
        self.forecast_path = forecast_zarr_path
        self.historical_path = historical_zarr_path
        self.station_path = station_zarr_path
        
        # Cache for loaded data
        self._cache = {}
    
    def load_forecast_data(self,
                          start_time: datetime,
                          end_time: datetime,
                          variables: List[str],
                          lead_times: Optional[List[int]] = None,
                          members: Optional[List[int]] = None,
                          chunks: Optional[Dict[str, int]] = None) -> xr.Dataset:
        """
        Load NWP forecast data.
        
        Args:
            start_time: Initialization time start
            end_time: Initialization time end
            variables: List of variables to load
            lead_times: Specific lead times (hours) to load
            members: Specific ensemble members to load
            chunks: Dask chunk specification
        """
        if not self.forecast_path:
            raise ValueError("No forecast path specified")
        
        # Default chunking optimized for ensemble operations
        if chunks is None:
            chunks = {
                'time': 1,  # One init time
                'lead': 24,  # 1 day of hourly data
                'member': -1,  # Keep members together
                'lat': 100,
                'lon': 100
            }
        
        # Check cache
        cache_key = f"forecast_{start_time}_{end_time}_{','.join(variables)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        ds = xr.open_zarr(self.forecast_path, chunks=chunks)
        
        # Select time range
        ds = ds.sel(time=slice(start_time, end_time))
        
        # Select variables
        if variables:
            ds = ds[variables]
        
        # Select lead times if specified
        if lead_times is not None:
            ds = ds.sel(lead=lead_times)
        
        # Select members if specified
        if members is not None:
            ds = ds.sel(member=members)
        
        self._cache[cache_key] = ds
        return ds
    
    def load_historical_data(self,
                           start_time: datetime,
                           end_time: datetime,
                           variables: List[str],
                           chunks: Optional[Dict[str, int]] = None) -> xr.Dataset:
        """Load ERA5 historical data."""
        if not self.historical_path:
            raise ValueError("No historical path specified")
        
        # Default chunking for climatology
        if chunks is None:
            chunks = {
                'time': 365,  # 1 year
                'lat': 100,
                'lon': 100
            }
        
        ds = xr.open_zarr(self.historical_path, chunks=chunks)
        ds = ds.sel(time=slice(start_time, end_time))
        
        if variables:
            ds = ds[variables]
        
        return ds
    
    def calculate_probability(self,
                            event: Event,
                            ds: xr.Dataset,
                            by_lead_time: bool = False) -> Union[float, xr.DataArray]:
        """
        Calculate probability of event from ensemble forecast.
        
        Args:
            event: Event definition
            ds: Dataset with ensemble dimension
            by_lead_time: If True, return probability by lead time
        
        Returns:
            Probability value or array
        """
        if 'member' not in ds.dims:
            raise ValueError("Dataset must have 'member' dimension")
        
        # Evaluate event (returns probability)
        prob = event.evaluate(ds, preserve_members=False)
        
        if by_lead_time and get_time_dimension(prob) == "lead_time":
            return prob
        else:
            # Average over all dimensions
            return float(prob.mean().compute())
    
    def calculate_climatology(self,
                            event: Event,
                            ds: xr.Dataset,
                            by: Optional[str] = None) -> Union[float, xr.DataArray]:
        """
        Calculate historical frequency of event.
        
        Args:
            event: Event definition
            ds: Historical dataset
            by: Grouping dimension ('year', 'month', 'dayofyear')
        
        Returns:
            Historical frequency
        """
        # Evaluate event
        binary = event.evaluate(ds)
        
        # Get time dimension
        time_dim = get_time_dimension(binary)
        if time_dim is None:
            raise ValueError("Cannot calculate climatology without time dimension")
        
        if by == "year":
            return binary.groupby(f'{time_dim}.year').mean()
        elif by == "month":
            return binary.groupby(f'{time_dim}.month').mean()
        elif by == "dayofyear":
            return binary.groupby(f'{time_dim}.dayofyear').mean()
        else:
            return float(binary.mean().compute())
    
    def calculate_conditional_probability(self,
                                        event1: Event,
                                        event2: Event,
                                        ds: xr.Dataset) -> Dict[str, float]:
        """Calculate conditional probabilities between events."""
        # Evaluate both events preserving members
        binary1 = event1.evaluate(ds, preserve_members=True)
        binary2 = event2.evaluate(ds, preserve_members=True)
        
        # Calculate probabilities
        p1 = float(binary1.mean().compute())
        p2 = float(binary2.mean().compute())
        p_both = float((binary1 & binary2).mean().compute())
        
        # Conditional probabilities
        p1_given_2 = p_both / p2 if p2 > 0 else 0
        p2_given_1 = p_both / p1 if p1 > 0 else 0
        
        return {
            "P(event1)": p1,
            "P(event2)": p2,
            "P(event1|event2)": p1_given_2,
            "P(event2|event1)": p2_given_1,
            "P(both)": p_both,
            "correlation": float(xr.corr(binary1, binary2).compute())
        }
    
    def create_forecast_dataframe(self,
                                events: List[Event],
                                ds: xr.Dataset,
                                lead_times: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create a DataFrame of event probabilities for multiple events and lead times.
        
        Useful for creating dashboards and reports.
        """
        results = []
        
        # Determine the time dimension
        time_dim = get_time_dimension(ds)
        if time_dim is None:
            raise ValueError("Dataset must have a time dimension")
        
        if lead_times is None:
            lead_times = list(ds[time_dim].values)
        
        for event in events:
            for lead in lead_times:
                ds_lead = ds.sel({time_dim: lead})
                prob = self.calculate_probability(event, ds_lead)
                
                results.append({
                    'event': event.name,
                    'lead_time': lead,
                    'probability': prob,
                    'description': event.description
                })
        
        return pd.DataFrame(results)

def validate_event(event: Event, ds: xr.Dataset) -> List[str]:
    """
    Validate that an event can be evaluated on a dataset.
    
    Returns list of error messages (empty if valid).
    """
    errors = []
    
    # Check required variables
    required_vars = event.get_required_variables()
    missing_vars = [v for v in required_vars if v not in ds]
    if missing_vars:
        errors.append(f"Missing variables: {missing_vars}")
    
    # Check dimensions
    if isinstance(event, (SimpleEvent, SpreadEvent)):
        if "lat" not in ds.dims or "lon" not in ds.dims:
            errors.append("Dataset must have lat/lon dimensions")
    
    # Check time dimension
    if "time" not in ds.dims and "lead" not in ds.dims:
        errors.append("Dataset must have time or lead dimension")
    
    return errors
