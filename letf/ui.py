import pandas as pd
from typing import List
from letf import config as cfg
from letf.config import clear_all_caches


def get_start_date_interactive() -> tuple:
    """
    Interactive menu to select analysis start AND end dates.
    Uses defaults (full history, latest end) if stdin is not a terminal.

    Returns:
        Tuple of (start_date, end_date) as strings 'YYYY-MM-DD'
    """
    import sys, os

    # Check if running in non-interactive mode
    if not sys.stdin.isatty() or os.getenv('LETF_NON_INTERACTIVE') or os.getenv('LETF_NONINTERACTIVE'):
        selected_start = cfg.START_DATE_OPTIONS[1]['date']
        selected_end = cfg.DATA_END_DATE
        cfg.ANALYSIS_START_DATE = selected_start
        cfg.ANALYSIS_END_DATE = selected_end
        print(f"\n  [Non-interactive mode] Using defaults: {selected_start} to {selected_end}")
        print(f"  Clearing caches for fresh data load...")
        clear_all_caches()
        return selected_start, selected_end

    print(f"\n{'='*80}")
    print("SELECT ANALYSIS DATE RANGE")
    print(f"{'='*80}")

    # Calculate available date range
    earliest = pd.to_datetime(cfg.DATA_START_DATE)
    latest = pd.to_datetime(cfg.DATA_END_DATE)
    years_available = (latest - earliest).days / 365.25

    print(f"\nFull data available: {cfg.DATA_START_DATE} to {cfg.DATA_END_DATE} ({years_available:.2f} years)")

    # ========================================================================
    # START DATE SELECTION
    # ========================================================================
    print("\n" + "-"*60)
    print("STEP 1: Choose START date")
    print("-"*60)

    for num, option in cfg.START_DATE_OPTIONS.items():
        start_dt = pd.to_datetime(option['date'])
        years_from_start = (latest - start_dt).days / 365.25
        print(f"  {num}. {option['date'][:4]} - {option['name']:<15} ({years_from_start:.0f} years of data)")

    print(f"  7. Custom - Enter your own start date")

    while True:
        try:
            choice_input = input("\nEnter START date choice (1-7) [default=1]: ").strip()

            if choice_input == "":
                choice = 1
            else:
                choice = int(choice_input)

            if choice in cfg.START_DATE_OPTIONS:
                selected_start = cfg.START_DATE_OPTIONS[choice]['date']
                break
            elif choice == 7:
                selected_start = get_custom_date("start", cfg.DATA_START_DATE, cfg.DATA_END_DATE)
                break
            else:
                print("  Invalid choice. Please enter 1-7.")
        except ValueError:
            print("  Invalid input. Please enter a number 1-7.")

    cfg.ANALYSIS_START_DATE = selected_start
    start_year = int(selected_start[:4])

    # ========================================================================
    # END DATE SELECTION
    # ========================================================================
    print("\n" + "-"*60)
    print("STEP 2: Choose END date")
    print("-"*60)

    print(f"  1. {cfg.DATA_END_DATE[:4]} - Latest available data (default)")
    print(f"  2. 2023 - Pre-2024 (exclude recent volatility)")
    print(f"  3. 2020 - Pre-COVID")
    print(f"  4. 2019 - Pre-COVID (full year)")
    print(f"  5. 2010 - Pre-TQQQ inception")
    print(f"  6. 2007 - Pre-Financial Crisis")
    print(f"  7. Custom - Enter your own end date")

    end_options = {
        1: cfg.DATA_END_DATE,
        2: "2023-12-31",
        3: "2020-01-01",
        4: "2019-12-31",
        5: "2010-01-01",
        6: "2007-12-31"
    }

    while True:
        try:
            choice_input = input("\nEnter END date choice (1-7) [default=1]: ").strip()

            if choice_input == "":
                choice = 1
            else:
                choice = int(choice_input)

            if choice in end_options:
                selected_end = end_options[choice]
                break
            elif choice == 7:
                selected_end = get_custom_date("end", selected_start, cfg.DATA_END_DATE)
                break
            else:
                print("  Invalid choice. Please enter 1-7.")
        except ValueError:
            print("  Invalid input. Please enter a number 1-7.")

    # Validate end is after start
    if pd.to_datetime(selected_end) <= pd.to_datetime(selected_start):
        print(f"  Warning: End date must be after start date.")
        print(f"  Using latest available: {cfg.DATA_END_DATE}")
        selected_end = cfg.DATA_END_DATE

    cfg.ANALYSIS_END_DATE = selected_end

    # Calculate years in selected range
    start_dt = pd.to_datetime(selected_start)
    end_dt = pd.to_datetime(selected_end)
    years_selected = (end_dt - start_dt).days / 365.25

    # ========================================================================
    # SUMMARY AND CACHE CLEARING
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"[OK] Analysis period: {selected_start} to {selected_end}")
    print(f"  Duration: {years_selected:.2f} years")

    # Show historical events included
    print(f"\n  Historical events in selected period:")
    end_year = int(selected_end[:4])

    events = [
        (1929, 1932, "Great Depression"),
        (1941, 1945, "World War II"),
        (1973, 1974, "Oil Crisis"),
        (1987, 1987, "Black Monday"),
        (2000, 2002, "Dot-com Crash"),
        (2008, 2009, "Financial Crisis"),
        (2020, 2020, "COVID Crash"),
        (2022, 2022, "2022 Bear Market")
    ]

    included = []
    excluded = []
    for event_start, event_end, name in events:
        if start_year <= event_start and end_year >= event_end:
            included.append(name)
        elif start_year > event_end or end_year < event_start:
            excluded.append(name)
        else:
            included.append(f"{name} (partial)")

    for event in included:
        print(f"    [+] {event}")

    if excluded:
        print(f"\n  Events EXCLUDED:")
        for event in excluded:
            print(f"    [-] {event}")

    # Clear caches to ensure fresh data load with new dates
    print(f"\n  Clearing caches for fresh data load...")
    clear_all_caches()

    print(f"{'='*80}\n")

    return selected_start, selected_end


def get_custom_date(date_type: str, min_date: str, max_date: str) -> str:
    """
    Get a custom date from user input.

    Args:
        date_type: "start" or "end"
        min_date: Minimum allowed date
        max_date: Maximum allowed date

    Returns:
        Date string in 'YYYY-MM-DD' format
    """
    print(f"\n  Enter custom {date_type} date:")
    print(f"  (Must be between {min_date} and {max_date})")

    while True:
        try:
            date_input = input(f"  {date_type.title()} date (YYYY-MM-DD or YYYY): ").strip()

            # Handle year-only input
            if len(date_input) == 4 and date_input.isdigit():
                if date_type == "start":
                    date_input = f"{date_input}-01-01"
                else:
                    date_input = f"{date_input}-12-31"

            parsed_date = pd.to_datetime(date_input)
            min_dt = pd.to_datetime(min_date)
            max_dt = pd.to_datetime(max_date)

            if parsed_date < min_dt:
                print(f"  Date too early. Minimum is {min_date}")
                continue

            if parsed_date > max_dt:
                print(f"  Date too late. Maximum is {max_date}")
                continue

            return parsed_date.strftime('%Y-%m-%d')

        except Exception as e:
            print(f"  Invalid date format. Please use YYYY-MM-DD or YYYY")

def get_custom_start_date() -> str:
    """
    Get a custom start date from user input.

    Returns:
        Date string in 'YYYY-MM-DD' format
    """
    print("\n  Enter custom start date:")
    print(f"  (Must be between {cfg.DATA_START_DATE} and {cfg.DATA_END_DATE})")

    while True:
        try:
            date_input = input("  Date (YYYY-MM-DD or YYYY): ").strip()

            # Handle year-only input
            if len(date_input) == 4 and date_input.isdigit():
                date_input = f"{date_input}-01-01"

            # Validate date format
            parsed_date = pd.to_datetime(date_input)

            # Check if within valid range
            earliest = pd.to_datetime(cfg.DATA_START_DATE)
            latest = pd.to_datetime(cfg.DATA_END_DATE)

            if parsed_date < earliest:
                print(f"  Date too early. Minimum is {cfg.DATA_START_DATE}")
                continue

            if parsed_date >= latest:
                print(f"  Date too late. Must be before {cfg.DATA_END_DATE}")
                continue

            return parsed_date.strftime('%Y-%m-%d')

        except Exception as e:
            print(f"  Invalid date format. Please use YYYY-MM-DD or YYYY")


def validate_time_horizons_for_start_date(start_date: str, time_horizons: list) -> list:
    """
    Validate and adjust time horizons based on available data.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(cfg.ANALYSIS_END_DATE)  # Use selected end date, not DATA_END_DATE
    available_years = (end_dt - start_dt).days / 365.25

    valid_horizons = []
    removed_horizons = []

    for horizon in time_horizons:
        if horizon <= available_years:
            valid_horizons.append(horizon)
        else:
            removed_horizons.append(horizon)

    if removed_horizons:
        print(f"\n  Note: Some time horizons removed due to insufficient data:")
        print(f"  Available: {available_years:.2f} years ({start_date} to {cfg.ANALYSIS_END_DATE})")
        print(f"  Removed horizons: {removed_horizons}")
        print(f"  Valid horizons: {valid_horizons}")

    return valid_horizons
