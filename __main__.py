from .library import EventLibrary, get_example_event_database
from .factory import create_event_from_dict

def main():
    """Main execution function to demonstrate package capabilities."""
    print("Weather Event System - Example Usage")
    
    # 1. Get the example event database
    print("\nStep 1: Loading example event database...")
    example_database = get_example_event_database()
    print(f" -> Found {len(example_database)} example events.")
    
    # 2. Create an EventLibrary and populate it from the database
    print("\nStep 2: Populating the Event Library...")
    library = EventLibrary()
    for name, event_data in example_database.items():
        event = create_event_from_dict(event_data)
        # In a real scenario, you might derive tags from the data itself
        tags = ["example", event.get_required_variables()[0]]
        if "phoenix" in name:
            tags.append("phoenix")
        library.add(event, tags=tags, author="system_example")
    print(f" -> Library populated with {len(library.events)} events.")

    # 3. Save the library to a file
    library.save()
    print(f"\nStep 3: Library saved to '{library.library_path}'")
    
    # 4. Demonstrate searching the library
    print("\nStep 4: Searching for events tagged with 'spread'...")
    spread_events = library.search(tags=["spread"])
    if spread_events:
        print(f" -> Found {len(spread_events)} spread event(s).")
        # Print the JSON of the first found spread event as an example
        print("\nExample Spread Event JSON:")
        print(spread_events[0].to_json())
    else:
        print(" -> No spread events found.")
    
    print("\nEvent system demonstration complete.")


if __name__ == "__main__":
    main()
