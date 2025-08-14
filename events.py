import json
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Union, Optional, Literal

from .utils import (
    OperatorType, LogicalOperatorType, AggregationType, ThresholdType, 
    get_time_dimension, apply_aggregation
)
from .domains import SpatialDomain, TemporalDomain, TemporalPattern, PostThresholdTemporalDomain

# ============================================================================
# Event Base Classes
# ============================================================================

class Event(ABC):
    """Abstract base class for weather events."""
    
    name: str
    description: Optional[str]

    @abstractmethod
    def evaluate(self, ds: xr.Dataset, preserve_members: bool = False) -> xr.DataArray:
        """
        Evaluate event occurrence in dataset.
        
        Args:
            ds: Dataset containing weather data
            preserve_members: If True, keep ensemble member dimension
            
        Returns:
            Binary DataArray (True/False) or probability if ensemble
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        pass
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def get_required_variables(self) -> List[str]:
        """Get list of required variables for this event."""
        if hasattr(self, 'variable'):
            return [self.variable]
        return []

    def _apply_operator(self, data: xr.DataArray, threshold: Union[float, xr.DataArray]) -> xr.DataArray:
        """Apply comparison operator."""
        # Assumes the child class has an 'operator' attribute for comparison.
        operator = getattr(self, 'operator', None)
        if operator == ">":
            return data > threshold
        elif operator == ">=":
            return data >= threshold
        elif operator == "<":
            return data < threshold
        elif operator == "<=":
            return data <= threshold
        elif operator == "==":
            return np.isclose(data, threshold)
        elif operator == "!=":
            return ~np.isclose(data, threshold)
        else:
            raise ValueError(f"Unsupported operator for threshold comparison: {operator}")


# ============================================================================
# Simple Event Implementation
# ============================================================================

@dataclass
class SimpleEvent(Event):
    """
    Simple threshold-based event definition.
    
    Examples:
        - Temperature > 100°F at Phoenix
        - Daily max temperature > 95°F for 2+ consecutive days
        - 6+ hours of wind < 5 m/s within any 24-hour period
    """
    
    # Event identification
    name: str
    variable: str  = "t2m" # Variable name in dataset
    operator: OperatorType = ">"
    threshold_value: float = 0.0
    description: Optional[str] = None
    variable_transform: Optional[str] = None  # e.g., "daily_max", "daily_mean"
    threshold_type: ThresholdType = "absolute"
    threshold_units: Optional[str] = None
    spatial_domain: Optional[SpatialDomain] = None
    spatial_aggregation: Optional[AggregationType] = None
    temporal_domain: Optional[TemporalDomain] = None
    temporal_pattern: Optional[TemporalPattern] = None
    post_threshold_temporal_domain: Optional[PostThresholdTemporalDomain] = None
    
    def __post_init__(self):
        """Validate and process initialization."""
        pass
    
    def evaluate(self, ds: xr.Dataset, preserve_members: bool = False) -> xr.DataArray:
        """Evaluate event occurrence."""
        # Get variable
        if self.variable not in ds:
            raise ValueError(f"Variable '{self.variable}' not found in dataset")
        
        # 1. Apply spatial domain
        if self.spatial_domain:
            ds_spatial = self.spatial_domain.apply_to_dataset(ds)
            data = ds_spatial[self.variable]
        else:
            data = ds[self.variable]

        # 2. Apply variable transform if specified
        if self.variable_transform:
            data = self._apply_variable_transform(data)

        # 3. Apply spatial aggregation
        if self.spatial_aggregation and (not self.spatial_domain or self.spatial_domain.type != "point"):
            spatial_dims = ["lat", "lon"]
            data = apply_aggregation(data, self.spatial_aggregation, spatial_dims)
        
        # 4. Apply temporal domain (for continuous data)
        if self.temporal_domain:
            data = self.temporal_domain.apply(data)
        
        # 5. Apply temporal pattern (creates binary data)
        if self.temporal_pattern:
            binary = self.temporal_pattern.apply(data)
        else:
            # Apply simple threshold
            threshold = self._get_threshold_value(data, ds)
            binary = self._apply_operator(data, threshold)
        
        # 6. Apply post-threshold temporal domain (for binary data)
        if self.post_threshold_temporal_domain:
            binary = self.post_threshold_temporal_domain.apply(binary)
        
        # Assign valid_time if forecast (lead_time present)
        if "lead_time" in binary.dims and "init_time" in binary.dims:
            valid_time = binary["init_time"] + binary["lead_time"]
            binary = binary.assign_coords({"valid_time": valid_time})
        
        # Handle ensemble dimension
        if "member" in binary.dims and not preserve_members:
            return binary.mean(dim="member")
        
        return binary
    
    def _apply_variable_transform(self, data: xr.DataArray) -> xr.DataArray:
        """Apply transformations like daily_max, daily_mean, etc. Handles both forecast (lead_time) and historical (valid_time) data."""
        # Determine which time dimension to use
        time_dim = get_time_dimension(data)
        if time_dim is None:
            # No recognized time dimension, return unchanged
            return data

        if self.variable_transform == "daily_max":
            return data.resample({time_dim: "1D"}).max()
        elif self.variable_transform == "daily_min":
            return data.resample({time_dim: "1D"}).min()
        elif self.variable_transform == "daily_mean":
            return data.resample({time_dim: "1D"}).mean()
        elif self.variable_transform == "daily_sum":
            return data.resample({time_dim: "1D"}).sum()
        else:
            return data
    
    def _get_threshold_value(self, data: xr.DataArray, ds_clim: xr.Dataset = None) -> Union[float, xr.DataArray]:
        """Get threshold value based on threshold type."""
        if self.threshold_type == "absolute":
            return self.threshold_value
        elif self.threshold_type == "percentile":
            # Calculate percentile from data
            time_dim = get_time_dimension(data)
            if time_dim is None:
                raise ValueError("Cannot calculate percentile without time dimension")
            return data.quantile(self.threshold_value, dim=time_dim)
        elif self.threshold_type == "climatology":
            # Would load climatology from separate source like ds_clim
            raise NotImplementedError("Climatology thresholds require external data")
        elif self.threshold_type == "anomaly":
            # Calculate as deviation from mean
            time_dim = get_time_dimension(data)
            if time_dim is None:
                raise ValueError("Cannot calculate anomaly without time dimension")
            return data.mean(dim=time_dim) + self.threshold_value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "type": "simple",
            "name": self.name,
            "description": self.description,
            "variable": self.variable,
            "variable_transform": self.variable_transform,
            "operator": self.operator,
            "threshold_value": self.threshold_value,
            "threshold_type": self.threshold_type,
            "threshold_units": self.threshold_units,
        }
        
        if self.spatial_domain:
            d["spatial_domain"] = asdict(self.spatial_domain)
        if self.spatial_aggregation:
            d["spatial_aggregation"] = self.spatial_aggregation
        if self.temporal_domain:
            d["temporal_domain"] = asdict(self.temporal_domain)
        if self.temporal_pattern:
            d["temporal_pattern"] = asdict(self.temporal_pattern)
        if self.post_threshold_temporal_domain:
            d["post_threshold_temporal_domain"] = asdict(self.post_threshold_temporal_domain)
        
        return {k: v for k, v in d.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleEvent':
        """Create from dictionary (new format only)."""
        spatial_domain = None
        if "spatial_domain" in data:
            spatial_domain = SpatialDomain(**data["spatial_domain"])
        temporal_domain = None
        if "temporal_domain" in data:
            temporal_domain = TemporalDomain.from_dict(data["temporal_domain"])
        temporal_pattern = None
        if "temporal_pattern" in data:
            temporal_pattern = TemporalPattern(**data["temporal_pattern"])
        post_threshold_temporal_domain = None
        if "post_threshold_temporal_domain" in data:
            ptd_data = data["post_threshold_temporal_domain"]
            post_threshold_temporal_domain = PostThresholdTemporalDomain(**ptd_data)
        kwargs = data.copy()
        kwargs.pop("type", None)
        kwargs.pop("spatial_domain", None)
        kwargs.pop("temporal_domain", None)
        kwargs.pop("temporal_pattern", None)
        kwargs.pop("post_threshold_temporal_domain", None)
        return cls(
            spatial_domain=spatial_domain,
            temporal_domain=temporal_domain,
            temporal_pattern=temporal_pattern,
            post_threshold_temporal_domain=post_threshold_temporal_domain,
            **kwargs
        )


# ============================================================================
# Complex Event Implementation
# ============================================================================

@dataclass
class ComplexEvent(Event):
    """
    Complex event combining multiple events with logical operators.
    
    Examples:
        - (Temp > 95°F in Sacramento) AND (Temp > 95°F in Burbank)
        - (High wind in SPP) OR (Low solar in CAISO)
    """
    
    name: str
    events: List[Event]
    operator: LogicalOperatorType = "and"
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate inputs."""
        if not self.events:
            raise ValueError("Complex event must have at least one sub-event")
    
    def evaluate(self, ds: xr.Dataset, preserve_members: bool = False) -> xr.DataArray:
        """Evaluate complex event."""
        # Evaluate all sub-events, preserving members for proper correlation
        results = [event.evaluate(ds, preserve_members=True) for event in self.events]
        
        # Normalize results to handle different data types and drop NaN values
        normalized_results = []
        for result in results:
            # Convert to boolean, dropping NaN values
            if result.dtype == bool:
                normalized = result
            else:
                # For float arrays, drop NaN values and convert to boolean
                normalized = result.dropna(dim=result.dims[-1])  # Drop NaN along time dimension
                normalized = normalized.astype(bool)
            normalized_results.append(normalized)
        
        # Align all results to same coordinates (this will handle different time ranges)
        normalized_results = xr.align(*normalized_results, join='inner')
        
        # Combine with logical operator
        if self.operator == "and":
            combined = normalized_results[0]
            for result in normalized_results[1:]:
                combined = combined & result
        else:  # OR
            combined = normalized_results[0]
            for result in normalized_results[1:]:
                combined = combined | result
        
        # Handle ensemble dimension
        if "member" in combined.dims and not preserve_members:
            return combined.mean(dim="member")
        
        return combined
    
    def get_required_variables(self) -> List[str]:
        """Get all required variables from sub-events."""
        variables = []
        for event in self.events:
            variables.extend(event.get_required_variables())
        return list(set(variables))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "complex",
            "name": self.name,
            "description": self.description,
            "events": [e.to_dict() for e in self.events],
            "operator": self.operator
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplexEvent':
        """Create from dictionary (new format only)."""
        from .factory import create_event_from_dict
        events = [create_event_from_dict(event_data) for event_data in data["events"]]
        return cls(
            name=data["name"],
            events=events,
            operator=data["operator"],
            description=data.get("description")
        )


# ============================================================================
# Spread Event Implementation
# ============================================================================

@dataclass
class SpreadEvent(Event):
    """
    Event based on differential between two locations.
    
    Examples:
        - Temperature difference between Minnesota and Louisiana > 20°F
        - Wind speed difference between two wind farms > 10 m/s
    """
    
    name: str
    variable: str
    location1: SpatialDomain
    location2: SpatialDomain
    operator: OperatorType = ">"
    threshold_value: float = 0.0
    temporal_domain: Optional[TemporalDomain] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate inputs."""
        # Ensure both locations are points
        if self.location1.type != "point" or self.location2.type != "point":
            raise ValueError("Spread events require point locations")
    
    def evaluate(self, ds: xr.Dataset, preserve_members: bool = False) -> xr.DataArray:
        """Evaluate spread event."""
        # Get data at both locations
        data1 = self.location1.apply_to_dataset(ds)[self.variable]
        data2 = self.location2.apply_to_dataset(ds)[self.variable]
        
        # Calculate spread
        spread = data1 - data2
        
        # Apply temporal domain if specified
        if self.temporal_domain:
            spread = self.temporal_domain.apply(spread)
        
        # Apply threshold
        binary = self._apply_operator(spread, self.threshold_value)
        
        # Handle ensemble dimension
        if "member" in binary.dims and not preserve_members:
            return binary.mean(dim="member")
        
        return binary
    
    def get_required_variables(self) -> List[str]:
        """Get required variables."""
        return [self.variable]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "type": "spread",
            "name": self.name,
            "description": self.description,
            "variable": self.variable,
            "location1": asdict(self.location1),
            "location2": asdict(self.location2),
            "operator": self.operator,
            "threshold_value": self.threshold_value,
        }
        
        if self.temporal_domain:
            d["temporal_domain"] = asdict(self.temporal_domain)
        
        return {k: v for k, v in d.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpreadEvent':
        """Create from dictionary (new format only)."""
        temporal_domain = None
        if "temporal_domain" in data:
            td_data = data["temporal_domain"]
            temporal_domain = TemporalDomain(**td_data)
        return cls(
            name=data["name"],
            variable=data["variable"],
            location1=SpatialDomain(**data["location1"]),
            location2=SpatialDomain(**data["location2"]),
            operator=data["operator"],
            threshold_value=data["threshold_value"],
            temporal_domain=temporal_domain,
            description=data.get("description")
        )
