import re
from typing import Optional

from .events import Event, SimpleEvent, SpreadEvent
from .domains import SpatialDomain
from .library import EventTemplates
from .utils import OperatorType

# ============================================================================
# Natural Language Parser (Basic Implementation)
# ============================================================================

class EventParser:
    """Parse natural language event descriptions into Event objects."""
    
    # Pattern definitions
    PATTERNS = {
        # "Phoenix temperature >= 115F"
        "simple_threshold": r"(\w+)\s+(temperature|temp|wind|precip)\s*(>=?|<=?|>|<|==)\s*([\d.]+)\s*(\w+)?",
        
        # "Chicago - Houston temperature spread > 20F"
        "spread": r"(\w+)\s*-\s*(\w+)\s+(temperature|temp|wind)\s+spread\s*(>=?|<=?|>|<)\s*([\d.]+)\s*(\w+)?",
        
        # "Heat wave in Dallas (3 days > 100F)"
        "heat_wave": r"heat\s+wave\s+in\s+(\w+)\s*\((\d+)\s+days?\s*>\s*([\d.]+)\s*(\w+)?\)",
    }
    
    @classmethod
    def parse(cls, text: str) -> Optional[Event]:
        """Parse text into an Event object."""
        text = text.lower().strip()
        
        # Try each pattern
        
        # Simple threshold pattern
        match = re.match(cls.PATTERNS["simple_threshold"], text)
        if match:
            location, variable, operator, value, units = match.groups()
            
            # Map variable names
            var_map = {
                "temperature": "t2m",
                "temp": "t2m",
                "wind": "wind_speed_100m",
                "precip": "precipitation"
            }
            
            return SimpleEvent(
                _name=f"parsed_{location}_{variable}",
                variable=var_map.get(variable, variable),
                operator=operator,
                threshold_value=float(value),
                threshold_units=units,
                spatial_domain=SpatialDomain(type="point", location=location)
            )
        
        # Spread pattern
        match = re.match(cls.PATTERNS["spread"], text)
        if match:
            loc1, loc2, variable, operator, value, units = match.groups()
            
            var_map = {
                "temperature": "t2m",
                "temp": "t2m",
                "wind": "wind_speed_100m"
            }
            
            return SpreadEvent(
                _name=f"parsed_{loc1}_{loc2}_spread",
                variable=var_map.get(variable, variable),
                location1=SpatialDomain(type="point", location=loc1),
                location2=SpatialDomain(type="point", location=loc2),
                operator=operator,
                threshold_value=float(value)
            )
        
        # Heat wave pattern
        match = re.match(cls.PATTERNS["heat_wave"], text)
        if match:
            location, days, temp, units = match.groups()
            
            return EventTemplates.heat_wave(
                location=location,
                threshold=float(temp),
                days=int(days),
                threshold_units=units or "F"
            )
        
        return None
