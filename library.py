import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from .factory import create_event_from_dict
from .events import Event, SimpleEvent, ComplexEvent, SpreadEvent
from .domains import SpatialDomain, TemporalPattern
from .utils import AggregationType

# ============================================================================
# Event Library
# ============================================================================

class EventLibrary:
    """
    Centralized storage and management for weather events.
    Features:
    - Save/load events to/from JSON
    - Search by tags, location, variable
    - Version control
    - Team sharing
    """
    
    def __init__(self, library_path: Optional[Path] = None):
        """Initialize event library."""
        self.library_path = library_path or Path("event_library.json")
        self.events: Dict[str, Event] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        if self.library_path.exists():
            self.load()
    
    def add(self, event: Event, tags: Optional[List[str]] = None, 
            author: Optional[str] = None) -> None:
        """Add event to library with metadata."""
        self.events[event.name] = event
        self.metadata[event.name] = {
            "tags": tags or [],
            "author": author,
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "version": 1
        }
    
    def get(self, name: str) -> Optional[Event]:
        """Get event by name."""
        return self.events.get(name)
    
    def search(self, **kwargs) -> List[Event]:
        """
        Search events by various criteria.
        
        Args:
            tags: List of tags to match
            variable: Variable name to match
            location: Location name to match
            author: Author name to match
        """
        results = []
        
        for name, event in self.events.items():
            meta = self.metadata[name]
            
            # Check tags
            if "tags" in kwargs:
                required_tags = set(kwargs["tags"])
                event_tags = set(meta.get("tags", []))
                if not required_tags.issubset(event_tags):
                    continue
            
            # Check variable
            if "variable" in kwargs:
                if kwargs["variable"] not in event.get_required_variables():
                    continue
            
            # Check author
            if "author" in kwargs:
                if meta.get("author") != kwargs["author"]:
                    continue
            
            results.append(event)
        
        return results
    
    def save(self) -> None:
        """Save library to JSON file."""
        data = {
            "events": {f"{name}": event.to_dict() for name, event in self.events.items()},
            "metadata": self.metadata
        }
        
        with open(self.library_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self) -> None:
        """Load library from JSON file."""
        with open(self.library_path, 'r') as f:
            data = json.load(f)
        
        self.events = {}
        for name, event_data in data.get("events", {}).items():
            self.events[name] = create_event_from_dict(event_data)
        
        self.metadata = data.get("metadata", {})


# ============================================================================
# Pre-defined Event Templates
# ============================================================================

class EventTemplates:
    """Collection of pre-defined event templates for common use cases."""
    
    @staticmethod
    def heat_wave(location: str, threshold: float = 95, days: int = 2,
                  threshold_units: str = "F") -> SimpleEvent:
        """Create a heat wave event (consecutive hot days)."""
        return SimpleEvent(
            name=f"{location}_heat_wave_{days}d",
            description=f"{days}-day heat wave in {location} (>{threshold}{threshold_units})",
            variable="t2m",
            variable_transform="daily_max",
            operator=">",
            threshold_value=threshold,
            threshold_units=threshold_units,
            spatial_domain=SpatialDomain(type="point", location=location),
            temporal_pattern=TemporalPattern(
                threshold=threshold,
                operator=">",
                duration=f"{days}D"
            )
        )
    
    @staticmethod
    def cold_snap(location: str, threshold: float = 20, days: int = 3,
                  threshold_units: str = "F") -> SimpleEvent:
        """Create a cold snap event."""
        return SimpleEvent(
            name=f"{location}_cold_snap_{days}d",
            description=f"{days}-day cold snap in {location} (<{threshold}{threshold_units})",
            variable="t2m",
            variable_transform="daily_min",
            operator="<",
            threshold_value=threshold,
            threshold_units=threshold_units,
            spatial_domain=SpatialDomain(type="point", location=location),
            temporal_pattern=TemporalPattern(
                threshold=threshold,
                operator="<",
                duration=f"{days}D"
            )
        )

    @staticmethod
    def low_solar(location: str, threshold: float = 200, threshold_units: str = "W/m2",
                  variable_transform: str = "daily_max") -> SimpleEvent:
        """Create a low solar event (daily max solar radiation below threshold)."""
        return SimpleEvent(
            name=f"{location}_low_solar",
            description=f"Low solar event in {location}: daily max sw_rad_down < {threshold}{threshold_units}",
            variable="sw_rad_down",
            variable_transform=variable_transform,
            operator="<",
            threshold_value=threshold,
            threshold_units=threshold_units,
            spatial_domain=SpatialDomain(type="point", location=location)
        )
    
    @staticmethod
    def high_wind(location: str, threshold: float = 25, hours_in_day: int = 6) -> SimpleEvent:
        """Create high wind event (X hours of high wind in a day)."""
        return SimpleEvent(
            name=f"{location}_high_wind",
            description=f"High wind in {location} ({hours_in_day}h >{threshold}m/s in 24h)",
            variable="wind_speed_100m",
            operator=">",
            threshold_value=threshold,
            spatial_domain=SpatialDomain(type="point", location=location),
            temporal_pattern=TemporalPattern(
                pattern=f"{hours_in_day}h_in_24h",
                pattern_threshold=threshold,
                pattern_operator=">"
            )
        )
    
    @staticmethod
    def low_renewable(region: str, wind_threshold: float = 5, 
                      solar_threshold: float = 100) -> ComplexEvent:
        """Create low renewable generation event."""
        low_wind = SimpleEvent(
            name=f"{region}_low_wind",
            description=f"Low wind in {region}",
            variable="wind_speed_100m",
            operator="<",
            threshold_value=wind_threshold,
            spatial_domain=SpatialDomain(type="iso", iso=region),
            spatial_aggregation="mean"
        )
        
        low_solar = SimpleEvent(
            name=f"{region}_low_solar",
            description=f"Low solar in {region}",
            variable="surface_solar_radiation",
            operator="<",
            threshold_value=solar_threshold,
            spatial_domain=SpatialDomain(type="iso", iso=region),
            spatial_aggregation="mean"
        )
        
        return ComplexEvent(
            name=f"{region}_low_renewable",
            description=f"Low wind AND solar in {region}",
            events=[low_wind, low_solar],
            operator="and"
        )
    
    @staticmethod
    def temperature_spread(location1: str, location2: str, 
                          threshold: float = 20) -> SpreadEvent:
        """Create temperature spread event between two locations."""
        return SpreadEvent(
            name=f"{location1}_{location2}_temp_spread",
            description=f"Temperature spread {location1}-{location2} > {threshold}F",
            variable="t2m",
            location1=SpatialDomain(type="point", location=location1),
            location2=SpatialDomain(type="point", location=location2),
            operator=">",
            threshold_value=threshold
        )


# ============================================================================
# Example Event Database
# ============================================================================

def get_example_event_database() -> Dict[str, Dict[str, Any]]:
    """Get example event definitions matching trader use cases."""
    
    return {
        # ERCOT events
        "ercot_extreme_heat": {
            "type": "simple",
            "name": "ercot_extreme_heat",
            "description": "Extreme heat in ERCOT (daily max > 100F)",
            "variable": "t2m",
            "variable_transform": "daily_max",
            "operator": ">=",
            "threshold_value": 100,
            "threshold_type": "absolute",
            "threshold_units": "F",
            "spatial_domain": {
                "type": "iso",
                "iso": "ERCOT"
            },
            "spatial_aggregation": "mean"
        },
        
        "ercot_heat_low_wind": {
            "type": "complex",
            "name": "ercot_heat_low_wind",
            "description": "High temp with low wind in ERCOT",
            "operator": "and",
            "events": [
                {
                    "type": "simple",
                    "name": "ercot_high_temp",
                    "variable": "t2m",
                    "variable_transform": "daily_max",
                    "operator": ">=",
                    "threshold_value": 95,
                    "spatial_domain": {"type": "iso", "iso": "ERCOT"},
                    "spatial_aggregation": "mean"
                },
                {
                    "type": "simple",
                    "name": "ercot_low_wind",
                    "variable": "wind_speed_100m",
                    "operator": "<",
                    "threshold_value": 5,
                    "spatial_domain": {"type": "iso", "iso": "ERCOT"},
                    "spatial_aggregation": "mean"
                }
            ]
        },
        
        # California events
        "california_heat_wave": {
            "type": "complex",
            "name": "california_heat_wave",
            "description": "2-day heat wave in Sacramento AND Burbank",
            "operator": "and",
            "events": [
                {
                    "type": "simple",
                    "name": "sacramento_heat",
                    "variable": "t2m",
                    "variable_transform": "daily_max",
                    "operator": ">",
                    "threshold_value": 95,
                    "threshold_units": "F",
                    "spatial_domain": {"type": "point", "location": "sacramento"},
                    "temporal_pattern": {
                        "threshold": 95,
                        "operator": ">",
                        "duration": "2D"
                    }
                },
                {
                    "type": "simple",
                    "name": "burbank_heat",
                    "variable": "t2m",
                    "variable_transform": "daily_max",
                    "operator": ">",
                    "threshold_value": 95,
                    "threshold_units": "F",
                    "spatial_domain": {"type": "point", "location": "burbank"},
                    "temporal_pattern": {
                        "threshold": 95,
                        "operator": ">",
                        "duration": "2D"
                    }
                }
            ]
        },
        
        # SPP wind event
        "spp_high_wind": {
            "type": "simple",
            "name": "spp_high_wind",
            "description": "High wind generation in SPP (>30GW equivalent)",
            "variable": "wind_speed_100m",
            "operator": ">",
            "threshold_value": 12,
            "spatial_domain": {
                "type": "bbox",
                "lat_min": 35.0,
                "lat_max": 40.0,
                "lon_min": -102.0,
                "lon_max": -94.0
            },
            "spatial_aggregation": "mean"
        },
        # Low solar in Phoenix event
        "phoenix_low_solar": {
            "type": "simple",
            "name": "phoenix_low_solar",
            "description": "Low solar in Phoenix",
            "variable": "sw_rad_down",
            "operator": "<",
            "threshold_value": 100,
            "spatial_domain": {
                "type": "point",
                "location": "phoenix"
            },
            "spatial_aggregation": "mean"
        },
        
        # Spread events
        "miso_temperature_spread": {
            "type": "spread",
            "name": "miso_temperature_spread",
            "description": "North-South temperature gradient in MISO",
            "variable": "t2m",
            "location1": {"type": "point", "location": "minneapolis"},
            "location2": {"type": "point", "location": "houston"},
            "operator": ">",
            "threshold_value": 30
        },
        
        # Complex temporal pattern
        "intermittent_wind": {
            "type": "simple",
            "name": "intermittent_wind",
            "description": "Intermittent wind (6+ hours <5m/s in any 24h period)",
            "variable": "wind_speed_100m",
            "operator": "<",
            "threshold_value": 5,
            "spatial_domain": {"type": "point", "location": "chicago"},
            "temporal_pattern": {
                "pattern": "6h_in_24h",
                "pattern_threshold": 5,
                "pattern_operator": "<"
            }
        }
    }
