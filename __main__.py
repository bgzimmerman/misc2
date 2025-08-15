from .events import SimpleEvent, ComplexEvent
from .library import EventLibrary, EventTemplates
from .parsing import EventParser
from .domains import SpatialDomain, TemporalDomain

def main():
    """Main execution function to demonstrate package capabilities."""
    print("Weather Event System - Example Usage")
    
    # Create some example events
    events = []
    
    # 1. Simple heat event
    phoenix_heat = EventTemplates.heat_wave("phoenix", threshold=115, days=2)
    events.append(phoenix_heat)
    
    # 2. Complex event
    ercot_stress = ComplexEvent(
        name="ercot_grid_stress",
        description="High demand with low renewable generation",
        events=[
            SimpleEvent(
                name="high_temp",
                description="High temperature in ERCOT",
                variable="t2m",
                operator=">=",
                threshold_value=100,
                spatial_domain=SpatialDomain(type="iso", iso="ERCOT"),
                spatial_aggregation="mean",
                temporal_pre_processing=TemporalDomain(
                    window_type="resample", window="1D", aggregation="max"
                )
            ),
            SimpleEvent(
                name="low_wind",
                description="Low wind in ERCOT",
                variable="wind_speed_100m",
                operator="<",
                threshold_value=5,
                spatial_domain=SpatialDomain(type="iso", iso="ERCOT"),
                spatial_aggregation="mean"
            )
        ],
        operator="and"
    )
    events.append(ercot_stress)
    
    # 3. Spread event
    temp_spread = EventTemplates.temperature_spread("minneapolis", "houston", 30)
    events.append(temp_spread)
    
    # Create event library
    library = EventLibrary()
    for event in events:
        library.add(event, tags=["example", "energy"], author="system")
    
    # Save library
    library.save()
    print(f"Saved {len(events)} events to library")
    
    # Example JSON output
    print("\nExample Event JSON:")
    print(phoenix_heat.to_json())
    
    # Parse natural language
    parser = EventParser()
    parsed = parser.parse("phoenix temperature >= 115F")
    if parsed:
        print("\nParsed Event:")
        print(parsed.to_json())
    
    print("Event system initialization complete")


if __name__ == "__main__":
    main()
