from __future__ import annotations
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .events import Event, SimpleEvent, ComplexEvent, SpreadEvent

def create_event_from_dict(data: Dict[str, Any]) -> "Event":
    """Create appropriate event type from dictionary."""
    # Import here to avoid circular dependencies
    from .events import SimpleEvent, ComplexEvent, SpreadEvent

    event_type = data.get("type", "simple")
    if event_type == "simple":
        return SimpleEvent.from_dict(data)
    elif event_type == "complex":
        return ComplexEvent.from_dict(data)
    elif event_type == "spread":
        return SpreadEvent.from_dict(data)
    else:
        raise ValueError(f"Unknown event type: {event_type}")
