"""
Weather Event Definition and Analysis System
===========================================

A comprehensive system for defining weather events and calculating their occurrence
in gridded weather data (ERA5 historical and NWP ensemble forecasts).
"""

__version__ = "2.0.1"

from .utils import (
    LOCATION_DATABASE,
    get_time_dimension,
    convert_units,
    apply_aggregation
)
from .domains import (
    SpatialDomain,
    TemporalDomain,
    TemporalPattern,
    PostThresholdTemporalDomain,
)
from .events import (
    Event,
    SimpleEvent,
    ComplexEvent,
    SpreadEvent,
)
from .factory import create_event_from_dict
from .library import (
    EventLibrary,
    EventTemplates,
    get_example_event_database,
)
from .parsing import EventParser
from .calculator import EventCalculator, validate_event

__all__ = [
    # Enums
    # Domains
    "SpatialDomain",
    "TemporalDomain",
    "TemporalPattern",
    "PostThresholdTemporalDomain",
    # Events
    "Event",
    "SimpleEvent",
    "ComplexEvent",
    "SpreadEvent",
    # Factory
    "create_event_from_dict",
    # Library & Templates
    "EventLibrary",
    "EventTemplates",
    "get_example_event_database",
    # Parsing
    "EventParser",
    # Calculator
    "EventCalculator",
    "validate_event",
    # Utilities
    "LOCATION_DATABASE",
    "get_time_dimension",
    "convert_units",
    "apply_aggregation"
]
