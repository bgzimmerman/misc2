"""
Weather Event Definition and Analysis System
===========================================

A system for defining weather events and calculating their occurrence
in gridded weather data.
"""

__version__ = "0.0.1"

from .utils import (
    LOCATION_DATABASE,
    get_time_dimension,
    apply_aggregation
)
from .domains import (
    SpatialDomain,
    TemporalPreprocessor,
    TemporalPattern,
    TemporalAnalysis,
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

__all__ = [
    # Enums
    # Domains
    "SpatialDomain",
    "TemporalPreprocessor",
    "TemporalPattern",
    "TemporalAnalysis",
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
    # Utilities
    "LOCATION_DATABASE",
    "get_time_dimension",
    "apply_aggregation"
]
