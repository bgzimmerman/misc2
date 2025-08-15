# Weather Event System

A flexible Python package for defining, evaluating, and managing weather events based on gridded forecast and historical data (e.g., ERA5, GFS).

## Core Concepts

This system is built around a few key components that allow for the flexible definition of weather phenomena.

### Event Types

There are three primary types of events you can define:

1.  **`SimpleEvent`**: The fundamental building block. It defines an event based on a variable exceeding a threshold at a specific location. It leverages a powerful three-stage temporal pipeline for complex time-series analysis.
2.  **`ComplexEvent`**: Combines multiple sub-events (which can be `Simple`, `Complex`, or `Spread`) using logical operators (`and`/`or`). This is used for defining conditions like "high heat AND low wind."
3.  **`SpreadEvent`**: A specialized event that triggers based on the difference in a variable between two distinct point locations.

### The Three-Stage Temporal Pipeline

For `SimpleEvent` and `SpreadEvent`, a three-stage pipeline allows for sophisticated temporal analysis:

1.  **Stage 1: `TemporalPreprocessor` (Numbers -> Numbers)**
    -   Performs initial processing on the continuous data.
    -   **Examples**: Calculate a daily maximum, a 24-hour rolling average, or resample from hourly to daily data.

2.  **Stage 2: `TemporalPattern` (Numbers -> Boolean)**
    -   Applies thresholding and duration logic to the pre-processed data to produce a boolean (True/False) result.
    -   **Examples**: Check if the daily maximum temperature was above 100°F, or if it stayed above that threshold for 3 consecutive days.

3.  **Stage 3: `TemporalAnalysis` (Boolean -> Numbers/Boolean)**
    -   *(Placeholder for future development)* Intended for performing post-processing on the boolean data.
    -   **Potential Examples**: Count the number of event days per week, or calculate the frequency of windy hours per day.

### `EventLibrary`

The `EventLibrary` is a manager class that allows you to:
-   Load event definitions from dictionaries or JSON files.
-   Save your collection of events to a JSON file.
-   Search for events based on metadata tags.

## Installation

To get started, clone the repository. It is recommended to install the package in a virtual environment.

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install the package in editable mode:**
    This allows you to make changes to the source code and have them immediately reflected in your environment.
    ```bash
    pip install -e .
    ```
    *(Note: You will also need to install dependencies like `xarray` and `pandas` if they are not already in your environment.)*

## Usage

The package is designed to be straightforward to use. The main entrypoint provides a clear demonstration of the intended workflow.

### Running the Example

You can run the built-in example to see the system in action. This script will:
1.  Load a pre-defined database of example events.
2.  Populate an `EventLibrary` with them.
3.  Save the library to `event_library.json`.
4.  Search the library for a specific event and print its definition.

```bash
python -m weathereventsystem
```

### Basic Workflow Example

Here is a programmatic example of how to use the library:

```python
from weathereventsystem import EventLibrary, get_example_event_database, create_event_from_dict

# 1. Get the dictionary of pre-defined example events
example_database = get_example_event_database()

# 2. Create an EventLibrary and populate it
library = EventLibrary()
for name, event_data in example_database.items():
    event = create_event_from_dict(event_data)
    library.add(event, tags=["example"], author="system")

# 3. Save the library to a file
library.save()
print(f"Library saved to '{library.library_path}'")

# 4. Search the library for an event
# This will find the 'phoenix_denver_temp_spread' event
spread_events = library.search(tags=["spread"])

if spread_events:
    print("\nFound Spread Event:")
    print(spread_events[0].to_json())
```

## Event Definition JSON Structure

Events are designed to be serialized to and from JSON for easy storage and sharing. Here is an example of a `SimpleEvent` definition for a 3-day heatwave in Phoenix.

```json
{
  "phoenix_heatwave_3day_gt_105F": {
    "type": "simple",
    "name": "phoenix_heatwave_3day_gt_105F",
    "description": "A 3-day heatwave where the daily maximum temperature in Phoenix exceeds 105F.",
    "variable": "t2m",
    "operator": ">",
    "threshold_value": 105,
    "threshold_type": "absolute",
    "threshold_units": "F",
    "spatial_domain": {
      "type": "point",
      "location": "phoenix"
    },
    "temporal_pre_processing": {
      "window_type": "resample",
      "window": "1D",
      "aggregation": "max"
    },
    "temporal_pattern": {
      "threshold": 105,
      "operator": ">",
      "duration": "3D"
    }
  }
}
```
