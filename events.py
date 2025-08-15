import json
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Union, Optional, Literal
import warnings

from .utils import (
    OperatorType, LogicalOperatorType, AggregationType, ThresholdType, 
    get_time_dimension, apply_aggregation
)
from .domains import SpatialDomain, TemporalPreprocessor, TemporalPattern, TemporalAnalysis

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
    
    # Threshold specification
    threshold_type: ThresholdType = "absolute"
    threshold_units: Optional[str] = None
    
    # Spatial specification
    spatial_domain: Optional[SpatialDomain] = None
    spatial_aggregation: Optional[AggregationType] = None
    aggregation_order: Literal["temporal_first", "spatial_first"] = "temporal_first"

    # --- Temporal Processing Pipeline ---
    # Stage 1: Pre-processing (Numbers -> Numbers)
    temporal_pre_processing: Optional[TemporalPreprocessor] = None
    # Stage 2: Thresholding & Pattern Logic (Numbers -> Boolean)
    temporal_pattern: Optional[TemporalPattern] = None
    # Stage 3: Post-processing Analysis (Boolean -> Numbers/Boolean)
    temporal_analysis: Optional[TemporalAnalysis] = None
    
    def __post_init__(self):
        """Validate and process initialization."""
        pass
    
    def evaluate(self, ds: xr.Dataset, preserve_members: bool = False, introspection: bool = False) -> Union[xr.DataArray, Dict[str, xr.DataArray]]:
        """
        Evaluate event occurrence using a three-stage temporal pipeline.

        Args:
            ds: The dataset to evaluate.
            preserve_members: If True, keeps the ensemble member dimension.
            introspection: If True, returns a dictionary of intermediate results for debugging.

        Returns:
            The final event result as a DataArray, or a dictionary of intermediate steps if introspection is True.
        """
        print(f"\n--- Evaluating Event: {self.name} ---")
        # --- PREPARATION ---
        if self.variable not in ds:
            raise ValueError(f"Variable '{self.variable}' not found in dataset")
        
        outputs = {}

        # 1. Apply spatial domain to select the geographical area of interest
        if self.spatial_domain:
            print(f"Step 1: Applying spatial domain ({self.spatial_domain.type} at {self.spatial_domain.location or 'custom area'})...")
            data = self.spatial_domain.apply_to_dataset(ds)[self.variable]
        else:
            print("Step 1: No spatial domain specified, using full dataset.")
            data = ds[self.variable]
        if introspection: outputs["1_spatial_selection"] = data.copy()

        # --- TEMPORAL PIPELINE ---
        print(f"Step 2: Following aggregation order: '{self.aggregation_order}'")
        # Handle flexible aggregation order
        if self.aggregation_order == 'temporal_first':
            # STAGE 1: Apply pre-processing (e.g., resampling, rolling averages)
            if self.temporal_pre_processing:
                print("  -> [Stage 1] Applying temporal pre-processing...")
                data = self.temporal_pre_processing.apply(data)
                if introspection: outputs["2a_temporal_pre_processing"] = data.copy()
            
            # Apply spatial aggregation after temporal pre-processing
            if self.spatial_aggregation and (not self.spatial_domain or self.spatial_domain.type != "point"):
                print(f"  -> [Post-Temporal] Applying spatial aggregation ({self.spatial_aggregation})...")
                spatial_dims = ["lat", "lon"]
                data = apply_aggregation(data, self.spatial_aggregation, spatial_dims)
                if introspection: outputs["2b_spatial_aggregation"] = data.copy()

        else: # 'spatial_first'
            # Apply spatial aggregation before temporal pre-processing
            if self.spatial_aggregation and (not self.spatial_domain or self.spatial_domain.type != "point"):
                print(f"  -> [Pre-Temporal] Applying spatial aggregation ({self.spatial_aggregation})...")
                spatial_dims = ["lat", "lon"]
                data = apply_aggregation(data, self.spatial_aggregation, spatial_dims)
                if introspection: outputs["2a_spatial_aggregation"] = data.copy()

            # STAGE 1: Apply pre-processing (e.g., resampling, rolling averages)
            if self.temporal_pre_processing:
                print("  -> [Stage 1] Applying temporal pre-processing...")
                data = self.temporal_pre_processing.apply(data)
                if introspection: outputs["2b_temporal_pre_processing"] = data.copy()

        # STAGE 2: Apply thresholding and pattern logic to get a boolean result
        if self.temporal_pattern:
            print("Step 3: [Stage 2] Applying temporal pattern logic to create boolean data...")
            binary_data = self.temporal_pattern.apply(data)
        else:
            print("Step 3: [Stage 2] Applying simple threshold comparison to create boolean data...")
            threshold = self._get_threshold_value(data, ds)
            binary_data = self._apply_operator(data, threshold)
        if introspection: outputs["3_thresholding_boolean"] = binary_data.copy()
        
        # STAGE 3: Apply post-processing analysis on the boolean data
        if self.temporal_analysis:
            print("Step 4: [Stage 3] Applying temporal analysis...")
            final_data = self.temporal_analysis.apply(binary_data)
        else:
            print("Step 4: [Stage 3] No temporal analysis specified.")
            final_data = binary_data
        if introspection: outputs["4_temporal_analysis"] = final_data.copy()
        
        # --- FINALIZATION ---
        print("Step 5: Finalizing results...")
        # Assign valid_time if forecast (lead_time present)
        if "lead_time" in final_data.dims and "init_time" in final_data.coords:
            print("  -> Assigning 'valid_time' coordinate.")
            valid_time = final_data["init_time"] + final_data["lead_time"]
            final_data = final_data.assign_coords({"valid_time": valid_time})
        
        # Handle ensemble dimension
        if "member" in final_data.dims and not preserve_members:
            print("  -> Calculating ensemble mean probability.")
            result = final_data.mean(dim="member")
            if introspection: 
                outputs["5_ensemble_mean"] = result.copy()
                print("--- Evaluation Complete (Introspection Mode) ---")
                return outputs
            return result

        if introspection:
            print("--- Evaluation Complete (Introspection Mode) ---")
            return outputs
        
        print("--- Evaluation Complete ---")
        return final_data
    
    def _apply_variable_transform(self, data: xr.DataArray) -> xr.DataArray:
        """DEPRECATED: This functionality is now handled by the TemporalPreprocessor class."""
        raise DeprecationWarning("variable_transform is deprecated. Use temporal_pre_processing instead.")

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
            "operator": self.operator,
            "threshold_value": self.threshold_value,
            "threshold_type": self.threshold_type,
            "threshold_units": self.threshold_units,
        }
        
        if self.spatial_domain:
            d["spatial_domain"] = asdict(self.spatial_domain)
        if self.spatial_aggregation:
            d["spatial_aggregation"] = self.spatial_aggregation
            d["aggregation_order"] = self.aggregation_order
        if self.temporal_pre_processing:
            d["temporal_pre_processing"] = asdict(self.temporal_pre_processing)
        if self.temporal_pattern:
            d["temporal_pattern"] = asdict(self.temporal_pattern)
        if self.temporal_analysis:
            d["temporal_analysis"] = asdict(self.temporal_analysis)
        
        return {k: v for k, v in d.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleEvent':
        """Create from dictionary (new format only)."""
        spatial_domain = None
        if "spatial_domain" in data:
            spatial_domain = SpatialDomain(**data["spatial_domain"])
        
        # Handle old format for backward compatibility
        if "temporal_domain" in data:
            warnings.warn(
                "'temporal_domain' is deprecated, use 'temporal_pre_processing'",
                DeprecationWarning
            )
            data["temporal_pre_processing"] = data.pop("temporal_domain")

        if "variable_transform" in data:
             warnings.warn(
                "'variable_transform' is deprecated, use 'temporal_pre_processing'",
                DeprecationWarning
            )
             transform = data.pop("variable_transform")
             if not data.get("temporal_pre_processing"):
                 agg_map = {
                     "daily_max": "max", "daily_min": "min",
                     "daily_mean": "mean", "daily_sum": "sum"
                 }
                 if transform in agg_map:
                     data["temporal_pre_processing"] = {
                         "window_type": "resample", "window": "1D",
                         "aggregation": agg_map[transform]
                     }

        temporal_pre_processing = None
        if "temporal_pre_processing" in data:
            temporal_pre_processing = TemporalPreprocessor.from_dict(data["temporal_pre_processing"])
        
        temporal_pattern = None
        if "temporal_pattern" in data:
            temporal_pattern = TemporalPattern(**data["temporal_pattern"])

        if "post_threshold_temporal_domain" in data:
            warnings.warn(
                "'post_threshold_temporal_domain' is deprecated, use 'temporal_analysis'",
                DeprecationWarning
            )
            data["temporal_analysis"] = data.pop("post_threshold_temporal_domain")

        temporal_analysis = None
        if "temporal_analysis" in data:
            ptd_data = data["temporal_analysis"]
            temporal_analysis = TemporalAnalysis(**ptd_data)
            
        kwargs = data.copy()
        kwargs.pop("type", None)
        kwargs.pop("spatial_domain", None)
        kwargs.pop("temporal_pre_processing", None)
        kwargs.pop("temporal_pattern", None)
        kwargs.pop("temporal_analysis", None)
        
        return cls(
            spatial_domain=spatial_domain,
            temporal_pre_processing=temporal_pre_processing,
            temporal_pattern=temporal_pattern,
            temporal_analysis=temporal_analysis,
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
    description: Optional[str] = None
    
    # --- Temporal Processing Pipeline ---
    temporal_pre_processing: Optional[TemporalPreprocessor] = None
    temporal_pattern: Optional[TemporalPattern] = None
    temporal_analysis: Optional[TemporalAnalysis] = None
    
    def __post_init__(self):
        """Validate inputs."""
        # Ensure both locations are points
        if self.location1.type != "point" or self.location2.type != "point":
            raise ValueError("Spread events require point locations")
    
    def evaluate(self, ds: xr.Dataset, preserve_members: bool = False, introspection: bool = False) -> Union[xr.DataArray, Dict[str, xr.DataArray]]:
        """Evaluate spread event using the standard temporal pipeline."""
        print(f"\n--- Evaluating Spread Event: {self.name} ---")
        outputs = {}

        # 1. Get data at both locations and calculate the spread
        print("Step 1: Calculating spread between two locations...")
        data1 = self.location1.apply_to_dataset(ds)[self.variable]
        data2 = self.location2.apply_to_dataset(ds)[self.variable]
        data = data1 - data2
        if introspection: outputs["1_calculated_spread"] = data.copy()

        # --- TEMPORAL PIPELINE ---
        # STAGE 1: Apply pre-processing to the spread data
        if self.temporal_pre_processing:
            print("Step 2: [Stage 1] Applying temporal pre-processing to the spread...")
            data = self.temporal_pre_processing.apply(data)
            if introspection: outputs["2_temporal_pre_processing"] = data.copy()
        else:
            print("Step 2: [Stage 1] No temporal pre-processing specified.")

        # STAGE 2: Apply thresholding and pattern logic
        if self.temporal_pattern:
            print("Step 3: [Stage 2] Applying temporal pattern logic to create boolean data...")
            binary_data = self.temporal_pattern.apply(data)
        else:
            print("Step 3: [Stage 2] Applying simple threshold comparison to create boolean data...")
            # Spread events always use an absolute threshold value
            binary_data = self._apply_operator(data, self.threshold_value)
        if introspection: outputs["3_thresholding_boolean"] = binary_data.copy()
        
        # STAGE 3: Apply post-processing analysis
        if self.temporal_analysis:
            print("Step 4: [Stage 3] Applying temporal analysis...")
            final_data = self.temporal_analysis.apply(binary_data)
        else:
            print("Step 4: [Stage 3] No temporal analysis specified.")
            final_data = binary_data
        if introspection: outputs["4_temporal_analysis"] = final_data.copy()
        
        # --- FINALIZATION ---
        print("Step 5: Finalizing results...")
        # Handle ensemble dimension
        if "member" in final_data.dims and not preserve_members:
            print("  -> Calculating ensemble mean probability.")
            result = final_data.mean(dim="member")
            if introspection:
                outputs["5_ensemble_mean"] = result.copy()
                print("--- Evaluation Complete (Introspection Mode) ---")
                return outputs
            return result

        if introspection:
            print("--- Evaluation Complete (Introspection Mode) ---")
            return outputs
        
        print("--- Evaluation Complete ---")
        return final_data
    
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
        
        if self.temporal_pre_processing:
            d["temporal_pre_processing"] = asdict(self.temporal_pre_processing)
        if self.temporal_pattern:
            d["temporal_pattern"] = asdict(self.temporal_pattern)
        if self.temporal_analysis:
            d["temporal_analysis"] = asdict(self.temporal_analysis)
        
        return {k: v for k, v in d.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpreadEvent':
        """Create from dictionary."""
        # Handle backward compatibility
        if "temporal_domain" in data:
            warnings.warn(
                "'temporal_domain' is deprecated, use 'temporal_pre_processing'",
                DeprecationWarning
            )
            data["temporal_pre_processing"] = data.pop("temporal_domain")

        temporal_pre_processing = None
        if "temporal_pre_processing" in data:
            temporal_pre_processing = TemporalPreprocessor.from_dict(data["temporal_pre_processing"])

        temporal_pattern = None
        if "temporal_pattern" in data:
            temporal_pattern = TemporalPattern(**data["temporal_pattern"])

        temporal_analysis = None
        if "temporal_analysis" in data:
            temporal_analysis = TemporalAnalysis(**data["temporal_analysis"])
            
        kwargs = data.copy()
        kwargs.pop("type", None)
        kwargs.pop("temporal_pre_processing", None)
        kwargs.pop("temporal_pattern", None)
        kwargs.pop("temporal_analysis", None)
        
        return cls(
            location1=SpatialDomain(**kwargs.pop("location1")),
            location2=SpatialDomain(**kwargs.pop("location2")),
            temporal_pre_processing=temporal_pre_processing,
            temporal_pattern=temporal_pattern,
            temporal_analysis=temporal_analysis,
            **kwargs
        )
