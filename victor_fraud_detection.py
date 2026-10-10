# - Python version used: 3.14.2. 
# - Any recent version of Python 3 should work, but I have not tested it on earlier versions.

# ========================================================================================
# Victor Oshiogwere Pius Fraud Detection Script
# =========================================================================================
# Overall design:
# - The interactive menu is the only entry point. Nothing runs until the user selects an option.
# - Data is held in one DataFrame (df). Load must succeed before Clean, Summarise or Visualise can run.
# - Each step is in a separate function so the menu stays simple and the same code can be used more than once.
# - Required columns are checked during load so a bad file never becomes the working data.
# - Bad values become NaN and are marked "INVALID" instead of being deleted, so the user can still see that the rows existed.
# - Option 3 runs every summary in one go (overall, time, amount, merchant, location).
# - Option 4 still uses a short sub-menu for charts so the user can pick one or run all.
# - Optional columns (Merchant, Location/City) are detected at runtime; if missing, those analyses are skipped with a clear message.
# - Dataset note: typical size is ~100 rows , so merchant/location groups can be very small.
# - Counts and rates are both shown so the user can judge risk more carefully than counts alone.

# ============================================================
# Robust imports
# I used error handling here so that if a required package
# is missing, the user receives clear guidance on how to fix
# the problem instead of an unexplained error.
# ============================================================


try: # Check if pandas is available for working with the data.
    import pandas as pd # Pandas helps me work with the data in the CSV file.
except ImportError: # Show error message if pandas is not available.
    print("ERROR: pandas is not installed.")
    print("Please run: pip install pandas")
    print("Then restart the program.")
    exit()

try:  # Check if matplotlib is available for creating chartss.
    import matplotlib.pyplot as plt # Matplotlib helps me create charts to understand the data
except ImportError: # Show error message if matplotlib is not available.
    print("ERROR: matplotlib is not installed.")
    print("Please run: pip install matplotlib")
    print("Then restart the program.")
    exit()
    

# Keep the filename in onne place so it can be changed easily.
file_name =  "fraud_detection_data.csv"

# Required columns are checked to make sure the dataset has the data needed for analysis.
required_columns = ["Timestamp", "Amount (USD)", "Is Fraud"]

# Use a clear text marker so that records with data-quality problems can be
# identified and investigated rather than silently treated as reliable records.
bad_value_marker = "INVALID"

# Include alternative names to help recognise optional columns in different datasets.
optional_merchant_names = ["Merchant", "merchant", "Merchant Name", "merchant_name"]
optional_location_names = ["Location", "location", "City", "city", "Transaction City", "transaction_city"]


def find_optional_column(df, candidates): # Define a function that accepts a dataset and list of possible column names.
    """Find optional columns even when their names differ between datasets.""" # Explain the purpose of the function.
    for name in candidates: # Go through each possible column name in the list.
        if name in df.columns: # Check whether the current name exists in the dataset.
            return name # Return the name as soon as a matching column is found.
    return None # Return None if no matching column exists.

# ============================================================================================
# Function: load_data
# Description: Loads the fraud detection data from a CSV file into a pandas DataFrame.
#              Also checks that the required columns exist.
# Parameters: file_name (str): - path to the CSV file containing the fraud detection data.
# Returns : pd.DataFrame or None if the file cannot be read / columns are missing.
# ============================================================================================

# I separate loading into a function so that file validation is handled in one place and the menu
# can use the same loading process whenever the user requests it.
def load_data(file_name): # Loads the CSV file so its data can be used for cleaning, summaries and charts.

    try: # Attempts to read the CSV file so the program can work with its data.
        df = pd.read_csv(file_name) # Store the loaded fraud detection data in 'df' so I can use it for cleaning, summaries and charts.
    except FileNotFoundError: # Handles situations where the file cannot be found, preventing the program from crashing.
        print(f"{file_name} not found. Check the file name and the folder.")
        return None # Stops the function because the file could not be loaded.
    except pd.errors.EmptyDataError: # Handles empty file so the program can inform the user instead of crashing.
        print(f"{file_name} is empty. Check the file contents.")
        return None # Stops the function because the file contains no data.
    except Exception as exc: # Catches any other unexpected errors during file reading and informs the user.
        print(f"An error occurred while loading {file_name}: {exc}")
        return None # Return None to show that the file could not be loaded successfully.

    # Checks that the file has all the required columns so the program can work with the data correctly.
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns: # Informs the user which columns are missing.
        print(f"Missing required columns: {', '.join(missing_columns)}. Please check the CSV file.")
        print("Expected columns are:", ", ".join(required_columns)) # Shows the columns the file is expected to contain for clarity.
        return None # Stops the function if the required columns are missing.
    print("File has been found. Required columns are present. Data loaded successfully.")
    print(f"Shape: {df.shape}")
    print(df.head())
    return df

# ============================================================================================
# Function: clean_data
# Description: Fix data types, mark problem rows, and check for duplicates.
#              Does not auto-delete rows because they may need human review.
# Parameters: df (pd.DataFrame) - data from load_data
# Returns: pd.DataFrame - cleaned data 
# ============================================================================================

def clean_data(df): # Cleans the fraud detection data so it is ready for analysis and visualisation.
    # I work on a copy to preserve the original dataset, allowing the records to be checked again
    # if cleaning changes their values or identifies problem.
    data = df.copy() # Keeps the original data safe so it is not affected if cleaning is repeated.

    # I convert timestamps into a consistent date format so that invalid dates can be identified
    # and the data is more suitable for further analysis.
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce") 

    # Count missing timestamps after conversion to identify dates that couldn't be processed.
    unconverted_dates = data["Timestamp"].isnull().sum()

    if unconverted_dates == 0: # Confirms that all timestamps were converted successfully.
        print("All timestamps converted successfully.")
    else: # Informs the user how many timestamps could not be converted.
        print(f"{unconverted_dates} timestamps could not be converted.")

    # I convert amounts to numeric values so that statistical calculations are meaningful
    # and invaid entries can be identified without stopping the cleaning process.
    data["Amount (USD)"] = pd.to_numeric(data["Amount (USD)"], errors="coerce")

    # Count how many Amount (USD) values could not be converted and became NaN.
    unconverted_amount = data["Amount (USD)"].isnull().sum()

    # Check if all Amount (USD) values were converted successfully and inform the user.
    if unconverted_amount == 0:
        print("All Amount (USD) converted successfully.")

    # If some values could not be converted, tell the user how many failed.
    else:
        print(f"{unconverted_amount} Amount (USD) values could not be converted.")

    # I flag zero or negative transaction amounts for review because they may represent
    # data-quality problems and could affect the interpolation of transaction statistics.
    invalid_amounts = (data["Amount (USD)"].notnull()) & (data["Amount (USD)"] <= 0)

    # Count how many invalid amounts were found.
    invalid_amount_count = invalid_amounts.sum()

    # Check if there are no zero or negative amounts.
    if invalid_amount_count == 0:
        print("No zero or negative Amount (USD) values found.")

    # If there are invalid amounts, inform the user how many were found.
    else:
        print(f"{invalid_amount_count} zero or negative Amount (USD) values found. Marked as INVALID.")

    # I standardise different representations of fraud labels so that values such as True, 1, and yes
    # can be treated consistently during counting and comparison.
    def to_boolean(value): # Define a function to convert different values into True or False for the Is Fraud column.
        if pd.isna(value): # Keeps missing values as NA instead of treating them as True or False.
            return pd.NA   # Treat unknown values as missing.
        
        # Converts the value to text, remove extra spaces, and use lowercase 
        # to values like "TRUE", "True" are handled the same way.
        text = str(value).strip().lower()
        if text in ["true", "1", "yes", "y"]: # Treats these values as True for fraud detection.
            return True
        if text in ["false", "0", "no", "n"]: # Treats these values as False for fraud detection.
            return False
        return pd.NA # Treats any other values as missing instead of causing an error.

    # Apply the to_boolean function to every value in the Is Fraud column.
    data["Is Fraud"] = data["Is Fraud"].apply(to_boolean)

    # Count the fraud label(s) that could not be recognized and became missing values (NA).
    unconverted_fraud = data["Is Fraud"].isnull().sum()

    if unconverted_fraud == 0: # Confirms that all Is Fraud labels were converted successfully.
        print("All Is Fraud labels converted successfully.")
    else: # Informs the user how many Is Fraud label(s) could not be converted.
        print(f"{unconverted_fraud} Is Fraud label(s) could not be converted.")

    # Mark every row that has a problem instead of deleting them.
    # This keeps the original data available for the user to inspect later.
    data["Data Status"] = "OK" # Start with all rows marked as OK.

    # I identify problamatic records using the key fields needed for analysis so that incomplete 
    # or unreliable transactions are not mistaken for valid records.
    bad_rows = (data["Timestamp"].isnull() | data["Amount (USD)"].isnull() | data["Is Fraud"].isnull() | invalid_amounts)
    data.loc[bad_rows, "Data Status"] = bad_value_marker # Mark the identified rows as INVALID.

    # Tell the user how many rows were marked as having problems.
    print(f"Rows marked '{bad_value_marker}': {bad_rows.sum()}")


    # I identify duplicate records without automatically deleting them because repeated transactions 
    # may require investigation before a decision is made about removing them.
    duplicate_rows = data.duplicated().sum()

    if duplicate_rows == 0: # Check if there are no duplicate transactions.
        print("No duplicate transactions found.")
    
    else: # If duplicates are found, tell the user how many need to be reviewed.
        print(f"{duplicate_rows} duplicate transactions found. Further investigation is required.")
    
    return data # Return the cleaned data so the other functions can use it.

# Helper function to select rows that contain the essential data for analysis.

def get_valid(df):
    """Keep records with the required values so incomplete data does not affect analysis."""

    return df[  # Select only the rows that meet all three conditions.
        df["Timestamp"].notna()  # Check that the transaction has a timestamp.
        & df["Amount (USD)"].notna()  # Check that the transaction has an amount.
        & df["Is Fraud"].notna()  # Check that the fraud status is available.
    ].copy()  # Create a separate copy of the selected rows.

# ============================================================================================
# Function: summary_overall
# Description: Print the number of INVALID and OK rows, then the total value of fraudulent
#              transactions and the fraud share of transaction count and value.
# Parameters: df (pd.DataFrame): - the cleaned fraud detection data.
# Returns: None
# ============================================================================================
def summary_overall(df):
    """Summarise fraud levels and data quality for an overall view of the dataset."""
    print("\n=== OVERALL SUMMARY ===")

    # Check data quality so users can see how many records may be unreliable.
    if "Data Status" in df.columns:
        invalid_count = (df["Data Status"] == bad_value_marker).sum()
        print(f"INVALID rows (after cleaning): {invalid_count}")
        print(f"OK rows: {(df['Data Status'] == 'OK').sum()}")
    else:
        print("Data Status column not found (run Clean first for INVALID counts).")

    # Exclude rows missing a timestamp, amount or fraud status, since the
    # calculations below cannot use incomplete records.
    valid = get_valid(df)
    if valid.empty:
        print("No valid rows for overall calculations.")
        return

    # Count valid transactions as the denominator for the fraud percentage.
    total_tx = len(valid)
    fraud_tx = (valid["Is Fraud"] == True).sum()

    # Calculate the percentage of transactions marked as fraudulent for easier comparison.
    # The "if total_tx else 0" guards against division by zero.
    fraud_pct_count = (fraud_tx / total_tx * 100) if total_tx else 0

    # Calculate the total transaction value as a baseline for assessing financial impact.
    total_value = valid["Amount (USD)"].sum()

    # Sum the full amount of every fraud-flagged transaction. This assumes none of
    # the money was blocked, refunded or recovered.
    fraud_value = valid.loc[valid["Is Fraud"] == True, "Amount (USD)"].sum()

    # Calculate fraud's share of transaction value to show its potential financial significance.
    # The "if total_value else 0" guards against division by zero.
    fraud_pct_value = (fraud_value / total_value * 100) if total_value else 0

    print(f"\nTotal valid transactions: {total_tx}")
    print(f"Fraudulent transactions:  {fraud_tx}  ({fraud_pct_count:.1f}% of count)")
    print(f"Total transaction value:  ${total_value:,.2f}")
    print(f"Total value of fraudulent transactions: ${fraud_value:,.2f}  ({fraud_pct_value:.1f}% of value)")

    # Highlight that fraud frequency and financial impact can tell different stories.
    print("Note: fraud share of value can differ from share of count when fraud amounts are larger/smaller.")


# ============================================================================================
# TIME summaries
# ============================================================================================

def summary_time(df):
    """Compare fraud patterns across time periods to help identify possible trends."""
    print("\n=== TIME SUMMARY ===")

    # Use complete records so missing essential values do not affect the time analysis.
    valid = get_valid(df)
    if valid.empty:
        print("No valid rows for time analysis.")
        return

    # Establish the time span covered by the dataset to give context to the results.
    earliest = valid["Timestamp"].min()
    latest = valid["Timestamp"].max()
    print(f"Earliest transaction: {earliest}")
    print(f"Latest transaction:   {latest}")

    # Group transactions by month to identify changes in fraud frequency over time.
    valid["YearMonth"] = valid["Timestamp"].dt.to_period("M")
    monthly = valid.groupby("YearMonth").agg(
        Total=("Is Fraud", "count"),
        Fraud_Count=("Is Fraud", "sum"),
    )
    # Calculate the monthly fraud rate so months with different transaction volumes can be compared.
    monthly["Fraud_Rate_%"] = (monthly["Fraud_Count"] / monthly["Total"] * 100).round(1)
    print("\nFraud by Month (count + rate):")
    print(monthly.to_string())

    # Compare years to help reveal longer-term changes in fraud patterns.
    valid["Year"] = valid["Timestamp"].dt.year
    yearly = valid.groupby("Year").agg(
        Total=("Is Fraud", "count"),
        Fraud_Count=("Is Fraud", "sum"),
    )
    # Use a percentage to compare years even when their transaction totals differ.
    yearly["Fraud_Rate_%"] = (yearly["Fraud_Count"] / yearly["Total"] * 100).round(1)
    print("\nFraud by Year (count + rate):")
    print(yearly.to_string())

    # Examine weekdays to see whether fraud rates vary across the week.
    valid["DayOfWeek"] = valid["Timestamp"].dt.day_name()
    # Keep weekdays in calendar order so the output is easier to interpret.
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = valid.groupby("DayOfWeek").agg(
        Total=("Is Fraud", "count"),
        Fraud_Count=("Is Fraud", "sum"),
    )
    # Calculate the fraud rate for each day to account for differences in transaction volume.
    dow["Fraud_Rate_%"] = (dow["Fraud_Count"] / dow["Total"] * 100).round(1)
    # Include only days present in the data while preserving the normal weekly order.
    dow = dow.reindex([d for d in day_order if d in dow.index])
    print("\nFraud by Day of Week (count + rate):")
    print(dow.to_string())

    # Examine transaction hours to identify possible periods when fraud is more common.
    valid["Hour"] = valid["Timestamp"].dt.hour
    hourly = valid.groupby("Hour").agg(
        Total=("Is Fraud", "count"),
        Fraud_Count=("Is Fraud", "sum"),
    )
    # Compare hourly fraud rates rather than counts alone, since transaction volumes may vary.
    hourly["Fraud_Rate_%"] = (hourly["Fraud_Count"] / hourly["Total"] * 100).round(1)
    print("\nFraud by Hour of Day (0=midnight … 23=11pm) — count + rate:")
    print(hourly.to_string())

    # Encourage further investigation of unusual time patterns without assuming they prove fraud.
    print("Tip: higher rates late at night or early morning can be a red flag.")
    
    
# ======================================================================
# Function: plot_fraud_counts
# Description: Creates a bar chart of fraud vs non-fraud counts.
#              Saves the chart as a PNG file (more reliable) and also tries to display it.
# Parameters: df (pd.DataFrame)
# Returns: None 
# =======================================================================

# Define a function to create a chart showing fraud and non-fraud transactions.
def plot_fraud_counts(df):
    # Check how many transactions are marked as fraud and not fraud.
    fraud_counts = df["Is Fraud"].value_counts()
    # Set the size of the chart.
    plt.figure(figsize=(7,5))
    # Use clear labels instead of True and False on the chart\.
    labels = fraud_counts.index.map({True: "Fraud", False: "Not Fraud", pd.NA:"Unknown"})
    # Create a bar chart to compare the transaction types.
    plt.bar(labels, fraud_counts.values, color = ["red", "green", "gray"])
    # Add a title to explain what the chart shows.
    plt.title("Fraud Vs Non-Fraud Transactions")
    # Label the horizontal axis to explain transaction categories.
    plt.xlabel("Transaction Type")
    # Label the vertical axis to show what the numbers represent.
    plt.ylabel("Number of transactions")
    # Adjust the layout so everything fits properly.
    plt.tight_layout()

    # Save a copy of the chart as a PNG file (useful backup).
    filename = "fraud_counts_chart.png"
    plt.savefig(filename)
    print(f"Chart saved as '{filename}' in the current folder.")

    # Try to display the chart window.
    # Some environments cannot open a graphical window, so we catch the error
    # and continue instead of letting the whole program crash.
    try:
        plt.show() # Show the chart to the user.
    except Exception:
        print("Could not open the chart window (this is normal in some environments).")
        print("You can still open the saved file: fraud_counts_chart.png")


# ==============================================================================
# Function: show_menu
# Description: Print the numbered menu options.
# Parameters: none
# Returns: None
# ==============================================================================
def show_menu(): # Define a function to display the available options to the user.
    print("1. Load the fraud detection data") # Display option 1 for loading the fraud detection data.
    print("2. Clean the fraud detection data") # Display option 2 for cleaning the loaded data.
    print("3. Summarise the fraud detection data") # Display option 3 for showing a summary of the data.
    print("4. Visualise the fraud detection data") # Display option 4 for creating charts from the data.
    print("5. Exit") # Display option 5 for closing the program.

# ==============================================================
# Main program - menu is the only entry point
# ==============================================================

df = None # Start with no data so the program knows when a file has not been loaded yet.

while True: # Keeps showing the menu until the user chooses to exit.
    show_menu()
    try: # Ask the user to select an option from the menu.
        choice = input("Enter your choice (1-5): ").strip() # Store the user's choice and remove extra spaces.
    except (KeyboardInterrupt, EOFError): # Handle Ctrl+C or closed input so the program exits instead of getting stuck.
        print("\nExiting.") # Tells the user the program is closing.
        break

    if choice not in [ "1", "2", "3", "4", "5"]: # Check if the user entered a valid menu option before processing their choice.
        print("Invalid choice. Please enter a number from 1 to 5.") # Ask the user to enter a valid option.
        continue # Return to the menu so the user can enter a valid choice.
    if choice == "5": # Check if the user selected option 5 to exit the program.
        print("Goodbye.")
        break # Stop the menu loop and end the program.
    if choice == "1": 
        loaded = load_data(file_name) # Load the data and store the result temporarily.
        if loaded is not None: # Check that the data loaded successfully.
            df = loaded # Replace the working data only when loading succeeds.
        continue # Return to the menu after attempting to load the data.
    if df is None: # Check whether the data has been loaded before using options 2-4.
        print("No data loaded. Please load the data first (Option 1).") # Tell the user that no data is available.
        continue # Return to the menu without running the selected option.
    if choice == "2":
        df = clean_data(df) # Clean the data and save the cleaned result back into df.
    elif choice == "3":
        summarise_data(df) # Display a summary of the data.
    elif choice == "4":
        plot_fraud_counts(df) # Display a chart showing the fraud counts.