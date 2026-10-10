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

    # Use records with the required values so missing timestamps, amounts or fraud labels
    # do not affect the time-based calculations.
    valid = get_valid(df)
    if valid.empty:
        print("No valid rows for time analysis.")
        return

    # Find the earliest and latest transactions to establish the time span covered by the data.
    earliest = valid["Timestamp"].min()
    latest = valid["Timestamp"].max()
    print(f"Earliest transaction: {earliest}")
    print(f"Latest transaction:   {latest}")

    # Convert timestamps into monthly periods so transactions can be compared month by month.
    valid["YearMonth"] = valid["Timestamp"].dt.to_period("M")

    # Group by month to calculate separate totals for each month instead of one overall total.
    # This helps reveal whether fraud patterns change over time.
    monthly = valid.groupby("YearMonth").agg(Total=("Is Fraud", "count"), Fraud_Count=("Is Fraud", "sum"))

    # Calculate the percentage of transactions marked as fraudulent in each month.
    # Rates make months with different transaction volumes easier to compare fairly.
    monthly["Fraud_Rate_%"] = (monthly["Fraud_Count"] / monthly["Total"] * 100).round(1)
    print("\nFraud by Month (count + rate):")
    print(monthly.to_string())

    # Extract the year from each timestamp so transactions can be compared over longer periods.
    valid["Year"] = valid["Timestamp"].dt.year

    # Group by year to compare yearly totals and identify possible longer-term changes in fraud.
    yearly = valid.groupby("Year").agg(Total=("Is Fraud", "count"), Fraud_Count=("Is Fraud", "sum"))

    # Calculate the yearly fraud rate because yearly transaction volumes may differ.
    yearly["Fraud_Rate_%"] = (yearly["Fraud_Count"] / yearly["Total"] * 100).round(1)
    print("\nFraud by Year (count + rate):")
    print(yearly.to_string())

    # Extract the weekday from each timestamp to investigate whether fraud patterns vary across the week.
    valid["DayOfWeek"] = valid["Timestamp"].dt.day_name()

    # Set the normal calendar order because grouped results may otherwise appear alphabetically.
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Group by weekday to calculate fraud totals separately for each day.
    # This makes it possible to compare weekdays and weekends.
    dow = valid.groupby("DayOfWeek").agg(Total=("Is Fraud", "count"), Fraud_Count=("Is Fraud", "sum"))

    # Calculate the fraud rate for each day so days with different transaction volumes can be compared.
    dow["Fraud_Rate_%"] = (dow["Fraud_Count"] / dow["Total"] * 100).round(1)

    # Keep only the weekdays present in the dataset while displaying them in calendar order.
    dow = dow.reindex([d for d in day_order if d in dow.index])
    print("\nFraud by Day of Week (count + rate):")
    print(dow.to_string())

    # Extract the hour from each timestamp to investigate possible differences throughout the day.
    valid["Hour"] = valid["Timestamp"].dt.hour

    # Group by hour to calculate separate fraud statistics for each hour rather than combining
    # all transactions. This helps identify hours that may deserve closer investigation.
    hourly = valid.groupby("Hour").agg(Total=("Is Fraud", "count"), Fraud_Count=("Is Fraud", "sum"))

    # Compare hourly fraud rates instead of counts alone because some hours may have more transactions.
    hourly["Fraud_Rate_%"] = (hourly["Fraud_Count"] / hourly["Total"] * 100).round(1)
    print("\nFraud by Hour of Day (0=midnight … 23=11pm) — count + rate:")
    print(hourly.to_string())

    # Treat unusually high rates at particular times as possible warning signs, not proof of fraud.
    print("Tip: higher rates late at night or early morning can be a red flag.")


# ============================================================================================
# AMOUNT summaries
# ============================================================================================

def summary_amount(df):
    """Compare transaction amounts to identify possible patterns linked to fraud."""
    print("\n=== AMOUNT SUMMARY ===")

    # Use records with the required values so incomplete transactions do not affect the analysis.
    valid = get_valid(df)
    if valid.empty:
        print("No valid rows for amount analysis.")
        return

    # Compare fraudulent and legitimate transactions to see whether their amounts differ.
    print("\nAmount statistics by fraud status:")
    for label, mask in [("Fraudulent", valid["Is Fraud"] == True),
                        ("Legitimate", valid["Is Fraud"] == False)]:

        # Use a Boolean mask to select only the transactions belonging to the current group.
        # This lets the same calculations be repeated for fraud and legitimate transactions.
        subset = valid.loc[mask, "Amount (USD)"]

        # Check for an empty group so the summary does not try to report statistics for missing data.
        if subset.empty:
            print(f"  {label}: No data")
        else:
            print(f"  {label}:")
            print(f"    count  = {len(subset)}")

            # Calculate the mean and median to compare the average and middle transaction amounts.
            print(f"    mean   = ${subset.mean():,.2f}")
            print(f"    median = ${subset.median():,.2f}")

            # Find the minimum and maximum to show the range of transaction amounts in each group.
            print(f"    min    = ${subset.min():,.2f}")
            print(f"    max    = ${subset.max():,.2f}")

    # Define amount ranges so transactions of different sizes can be analysed separately.
    bins = [0, 2500, 5000, 10000, 25000, float("inf")]

    # Assign readable names to the ranges so the results are easier to understand.
    labels = ["Under 2,500", "2,500–5,000", "5,000–10,000", "10,000–25,000", "Over 25,000"]

    # Place each transaction into an amount range to compare fraud patterns across transaction sizes.
    # right=False makes each range include its lower boundary but exclude its upper boundary.
    valid["Amount Band"] = pd.cut(valid["Amount (USD)"], bins=bins, labels=labels, right=False)

    # Group transactions by amount band so each range has its own transaction and fraud counts.
    # This helps reveal whether fraud rates differ between lower-value and higher-value transactions.
    bands = valid.groupby("Amount Band", observed=True).agg(Total=("Is Fraud", "count"), Fraud_Count=("Is Fraud", "sum"))

    # Calculate a percentage for each range so groups with different transaction volumes can be compared fairly.
    bands["Fraud_Rate_%"] = (bands["Fraud_Count"] / bands["Total"] * 100).round(1)
    print("\nFraud rate by Amount Band:")
    print(bands.to_string())

    # Select the five largest transactions to make the highest-value records easier to inspect.
    print("\nTop 5 largest transactions:")
    top5 = valid.nlargest(5, "Amount (USD)")[["Timestamp", "Amount (USD)", "Is Fraud"]]
    print(top5.to_string(index=False))

    # Calculate the mean and standard deviation to establish a threshold for unusually large amounts.
    mean_amt = valid["Amount (USD)"].mean()
    std_amt = valid["Amount (USD)"].std()

    # Only calculate the threshold when the standard deviation exists and is greater than zero.
    if pd.notna(std_amt) and std_amt > 0:

        # Set the threshold three standard deviations above the mean to flag unusually high amounts.
        threshold = mean_amt + 3 * std_amt

        # Select transactions above the threshold so they can be investigated further.
        outliers = valid[valid["Amount (USD)"] > threshold]

        print(f"\nOutliers (amount > mean + 3*std = ${threshold:,.2f}): {len(outliers)} found")

        # Display the flagged records so their amounts and fraud labels can be checked.
        if not outliers.empty:
            print(outliers[["Timestamp", "Amount (USD)", "Is Fraud"]].to_string(index=False))
    else:
        # Explain why an outlier threshold cannot be calculated for this data.
        print("\nCould not compute outlier threshold (insufficient variation).")

        
# ============================================================================================
# MERCHANT summaries
# ============================================================================================

def summary_merchant(df):
    """Top merchants by volume, by fraud count, by fraud rate, total fraud value."""
    print("\n=== MERCHANT SUMMARY ===")

    # Explain the limitation of a small dataset because merchants with few transactions
    # may appear to have unusually high fraud rates based on very little evidence.
    print("Note: with only ~100 rows, many merchants will have very few transactions.")
    print("      Prefer rate + count together; a 100% rate on 1 transaction is weak evidence.")

    # Look for a recognised merchant column name so the analysis can work with datasets
    # that use different names for the same information.
    col = find_optional_column(df, optional_merchant_names)
    if col is None:
        print(f"No Merchant column found. Looked for: {optional_merchant_names}")
        print("Skipping merchant analysis.")
        return

    # Use records with the required values so incomplete transactions do not affect the analysis.
    valid = get_valid(df)
    if valid.empty:
        print("No valid rows for merchant analysis.")
        return

    # Check the merchant column is still available before using it in the calculations.
    if col not in valid.columns:
        print(f"Merchant column '{col}' missing after filtering.")
        return

    # Count transactions for each merchant and show the ten busiest merchants.
    # value_counts() ranks merchants by frequency, making transaction volume easy to compare.
    top_vol = valid[col].value_counts().head(10)
    print(f"\nTop 10 merchants by number of transactions (using column '{col}'):")
    print(top_vol.to_string())

    # Filter to fraudulent transactions so legitimate transactions do not affect the fraud counts.
    fraud_only = valid[valid["Is Fraud"] == True]
    if fraud_only.empty:
        print("\nNo fraudulent transactions to rank merchants by fraud count.")
    else:
        # Count fraud cases for each merchant to identify which merchants have the most flagged transactions.
        top_fraud_count = fraud_only[col].value_counts().head(10)
        print(f"\nTop merchants by fraud count:")
        print(top_fraud_count.to_string())

    # Group transactions by merchant to calculate separate totals and fraud counts for each merchant.
    # This allows merchants to be compared instead of combining all transactions into one result.
    merchant_stats = valid.groupby(col).agg(Total=("Is Fraud", "count"), Fraud_Count=("Is Fraud", "sum"))

    # Calculate each merchant's fraud percentage so merchants with different transaction volumes
    # can be compared more fairly than by fraud count alone.
    merchant_stats["Fraud_Rate_%"] = (merchant_stats["Fraud_Count"] / merchant_stats["Total"] * 100).round(1)

    # Keep only fraudulent transactions and group them by merchant to calculate the total
    # transaction amount associated with fraud for each merchant.
    fraud_value_by_merch = (valid[valid["Is Fraud"] == True].groupby(col)["Amount (USD)"].sum())

    # Add the calculated fraud values to the merchant statistics for comparison.
    merchant_stats["Fraud_Value"] = fraud_value_by_merch

    # Replace missing values with zero for merchants with no flagged transactions,
    # so they can still appear in the merchant summary.
    merchant_stats["Fraud_Value"] = merchant_stats["Fraud_Value"].fillna(0)

    # Only rank merchants with at least two transactions to reduce the risk of treating
    # a 100% fraud rate from a single transaction as strong evidence.
    # Sort from highest to lowest rate and keep the top ten merchants.
    rate_candidates = merchant_stats[merchant_stats["Total"] >= 2].sort_values(
        "Fraud_Rate_%", ascending=False).head(10)

    print("\nTop merchants by fraud rate (min 2 transactions):")
    if rate_candidates.empty:
        print("  Not enough merchants with ≥2 transactions.")
    else:
        print(rate_candidates[["Total", "Fraud_Count", "Fraud_Rate_%", "Fraud_Value"]].to_string())

    # Sort merchants by the total value of fraud-flagged transactions to identify
    # those associated with the largest amounts, then display the top ten.
    print("\nTotal value of fraud per merchant (top 10 by value):")
    top_val = merchant_stats.sort_values("Fraud_Value", ascending=False).head(10)
    print(top_val[["Total", "Fraud_Count", "Fraud_Rate_%", "Fraud_Value"]].to_string())

# ============================================================================================
# LOCATION summaries
# ============================================================================================

def summary_location(df):
    """Locations with most fraud + fraud rate (with min-count filter)."""
    print("\n=== LOCATION SUMMARY ===")

    # Explain that rates based on small numbers of transactions can be misleading,
    # so both the fraud rate and transaction count should be considered.
    print("Note: small sample sizes make rates unstable. Use rate + count together.")

    # Find a recognised location column name so the analysis can work with datasets
    # that may use different names for location information.
    col = find_optional_column(df, optional_location_names)
    if col is None:
        print(f"No Location/City column found. Looked for: {optional_location_names}")
        print("Skipping location analysis.")
        return

    # Use records with the required values so incomplete transactions do not affect the analysis.
    valid = get_valid(df)
    if valid.empty:
        print("No valid rows for location analysis.")
        return

    # Keep only transactions marked as fraudulent so legitimate transactions
    # do not affect the ranking of locations by fraud count.
    fraud_only = valid[valid["Is Fraud"] == True]
    if fraud_only.empty:
        print("No fraudulent transactions.")
    else:
        # Count fraud cases for each location and show the ten highest counts.
        # value_counts() is used to rank locations by how often fraud appears in the data.
        top_fraud_loc = fraud_only[col].value_counts().head(10)
        print(f"\nLocations with the most fraud (count) — column '{col}':")
        print(top_fraud_loc.to_string())

    # Group transactions by location so each location has its own total and fraud count.
    # This makes it possible to compare fraud patterns between locations.
    loc_stats = valid.groupby(col).agg(Total=("Is Fraud", "count"), Fraud_Count=("Is Fraud", "sum"))

    # Calculate the percentage of transactions marked as fraudulent in each location.
    # Rates help compare locations with different numbers of transactions.
    loc_stats["Fraud_Rate_%"] = (loc_stats["Fraud_Count"] / loc_stats["Total"] * 100).round(1)

    # Include locations with at least two transactions before ranking their fraud rates.
    # This reduces the risk of overinterpreting a rate based on just one transaction.
    # Sort from highest to lowest rate and display the top ten locations.
    rate_candidates = loc_stats[loc_stats["Total"] >= 2].sort_values(
        "Fraud_Rate_%", ascending=False).head(10)

    print("\nLocations by fraud rate (min 2 transactions):")
    if rate_candidates.empty:
        print("  Not enough locations with ≥2 transactions.")
    else:
        print(rate_candidates.to_string())


        
# =========================
# CHART functions
# =========================

# ============================================================================================
# Function: _save_and_show
# Description: Save the current chart as a PNG, then try to display it. If no window can
#              open, the saved file can still be viewed.
# Parameters: filename (str): - name of the PNG file to save.
# Returns: None
# ============================================================================================

def _save_and_show(filename):
    """Save figure and attempt to display it."""

    # Adjust spacing so chart labels and titles are less likely to overlap or get cut off.
    plt.tight_layout()

    # Save the chart to a file so the results can still be viewed after the program finishes.
    plt.savefig(filename)
    print(f"Chart saved as '{filename}' in the current folder.")

    # Try to display the chart because some environments support chart windows while others do not.
    try:
        plt.show()

    # Handle display errors so the program can explain the problem instead of stopping unexpectedly.
    except Exception:
        print("Could not open the chart window (normal in some environments).")
        print(f"You can still open the saved file: {filename}")

    # Close the figure whether displaying it succeeds or fails, helping prevent figures
    # from remaining open and interfering with later charts.
    finally:
        plt.close()

# ============================================================================================
# Function: chart_fraud_counts
# Description: Bar chart comparing the number of fraud and non-fraud transactions.
# Parameters: df (pd.DataFrame): - the cleaned fraud detection data.
# Returns: None
# ============================================================================================
def chart_fraud_counts(df):
    """Bar chart of fraud vs non-fraud counts."""

    # Count transactions for each fraud status, including missing values,
    # so unknown statuses are not silently excluded from the summary.
    fraud_counts = df["Is Fraud"].value_counts(dropna=False)

    # Create a figure with a suitable width and height so the chart is easy to read.
    plt.figure(figsize=(7, 5))

    # Replace Boolean values with readable category names for the chart's horizontal axis.
    # The mapping also provides a label for missing fraud statuses.
    labels = fraud_counts.index.map({True: "Fraud", False: "Not Fraud", pd.NA: "Unknown"})

    # Convert labels to strings so unexpected or unmapped values can still be displayed.
    labels = [str(l) for l in labels]

    # Build a colour list so each category can be distinguished visually.
    colors = []
    for lab in labels:
        if lab == "Fraud":
            # Use red to make fraudulent transactions stand out.
            colors.append("red")
        elif lab == "Not Fraud":
            # Use green to distinguish transactions marked as legitimate.
            colors.append("green")
        else:
            # Use grey for unknown or unexpected labels.
            colors.append("gray")

    # Draw a bar for each category, using the transaction counts as bar heights.
    plt.bar(labels, fraud_counts.values, color=colors)

    # Add a title and axis labels so the chart's purpose and values are clear.
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.xlabel("Transaction Type")
    plt.ylabel("Number of transactions")

    # Save and attempt to display the chart using the shared chart-handling function.
    _save_and_show("fraud_counts_chart.png")

# ============================================================================================
# Function: chart_fraud_by_month
# Description: Bar chart of fraudulent transaction counts for each month.
# Parameters: df (pd.DataFrame): - the cleaned fraud detection data.
# Returns: None
# ============================================================================================
def chart_fraud_by_month(df):
    """Bar chart of fraudulent transaction counts by month."""

    # Use records with the required values so incomplete transactions do not affect the chart.
    valid = get_valid(df)
    if valid.empty:
        print("No valid data for monthly chart.")
        return

    # Convert timestamps to year-month labels so transactions can be grouped chronologically by month.
    valid["YearMonth"] = valid["Timestamp"].dt.to_period("M").astype(str)

    # Filter to fraudulent transactions, then group by month and count the rows in each group.
    # groupby() separates the months, while size() counts the fraudulent transactions in each month.
    monthly_fraud = valid[valid["Is Fraud"] == True].groupby("YearMonth").size()

    # Stop if there are no fraudulent transactions, because there would be no bars to display.
    if monthly_fraud.empty:
        print("No fraudulent transactions to plot by month.")
        return

    # Create a wider figure so monthly labels have more room to fit.
    plt.figure(figsize=(10, 5))

    # Draw a bar for each month so the number of fraudulent transactions can be compared visually.
    monthly_fraud.plot(kind="bar", color="crimson")

    # Add a title and axis labels to explain what the chart shows.
    plt.title("Fraudulent Transactions by Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Fraud Transactions")

    # Rotate the month labels and align them to reduce overlap and improve readability.
    plt.xticks(rotation=45, ha="right")

    # Save the chart and attempt to display it using the shared chart-handling function.
    _save_and_show("fraud_by_month_chart.png")

# ============================================================================================
# Function: chart_histogram_amounts
# Description: Histogram showing how transaction amounts are distributed.
# Parameters: df (pd.DataFrame): - the cleaned fraud detection data.
# Returns: None
# ============================================================================================
def chart_histogram_amounts(df):
    """Histogram of transaction amounts."""

    # Use records with the required values so missing amounts do not affect the chart.
    valid = get_valid(df)
    if valid.empty:
        print("No valid data for amount histogram.")
        return

    # Create a figure with enough space for the amount ranges and frequency labels.
    plt.figure(figsize=(8, 5))

    # Use a histogram to group transaction amounts into 20 ranges and show how often
    # amounts fall within each range. The edge colour makes the bars easier to distinguish.
    plt.hist(valid["Amount (USD)"], bins=20, color="steelblue", edgecolor="black")

    # Add a title and axis labels so the chart's information is clear.
    plt.title("Histogram of Transaction Amounts")
    plt.xlabel("Amount (USD)")
    plt.ylabel("Frequency")

    # Save the histogram and attempt to display it using the shared chart-handling function.
    _save_and_show("amount_histogram.png")

# ============================================================================================
# Function: chart_box_amount_by_fraud
# Description: Box plot comparing amounts of fraudulent and legitimate transactions.
# Parameters: df (pd.DataFrame): - the cleaned fraud detection data.
# Returns: None
# ============================================================================================
def chart_box_amount_by_fraud(df):
    """Box plot of amount, fraud vs non-fraud."""

    # Use records with the required values so incomplete transactions do not affect the comparison.
    valid = get_valid(df)
    if valid.empty:
        print("No valid data for box plot.")
        return

    # Separate fraudulent and legitimate transaction amounts so their distributions can be compared.
    # loc selects rows matching each fraud-status condition and returns the amount column.
    fraud_amts = valid.loc[valid["Is Fraud"] == True, "Amount (USD)"]
    legit_amts = valid.loc[valid["Is Fraud"] == False, "Amount (USD)"]

    # Stop if neither group contains data, because there would be nothing to compare in the chart.
    if fraud_amts.empty and legit_amts.empty:
        print("No data for box plot.")
        return

    # Build lists for the box plot so only groups containing data are included.
    data_to_plot = []
    labels = []

    # Add legitimate transaction amounts and their label when that group is available.
    if not legit_amts.empty:
        data_to_plot.append(legit_amts)
        labels.append("Not Fraud")

    # Add fraudulent transaction amounts and their label when that group is available.
    if not fraud_amts.empty:
        data_to_plot.append(fraud_amts)
        labels.append("Fraud")

    # Create a figure with enough space to display both box plots clearly.
    plt.figure(figsize=(7, 5))

    # Draw a box plot for each available group to compare medians, spread and possible outliers.
    # patch_artist=True allows the boxes to be filled with colour.
    plt.boxplot(data_to_plot, labels=labels, patch_artist=True, boxprops=dict(facecolor="lightblue"))

    # Add a title and axis label so the purpose and measurement are clear.
    plt.title("Transaction Amount: Fraud vs Non-Fraud")
    plt.ylabel("Amount (USD)")

    # Save the chart and attempt to display it using the shared chart-handling function.
    _save_and_show("amount_boxplot_fraud.png")

# ============================================================================================
# Function: chart_fraud_by_merchant
# Description: Bar chart of the 10 merchants with the most fraud. Skipped if the dataset
#              has no merchant column.
# Parameters: df (pd.DataFrame): - the cleaned fraud detection data.
# Returns: None
# ============================================================================================
def chart_fraud_by_merchant(df):
    """Bar chart of top merchants by fraud count."""

    # Find the merchant column by checking possible column names,
    # because different datasets may use different names for the same information.
    col = find_optional_column(df, optional_merchant_names)

    # Stop if no merchant column exists, because the chart cannot compare merchants without it.
    if col is None:
        print("No Merchant column available for chart.")
        return

    # Use valid records so transactions missing required values do not affect the analysis.
    valid = get_valid(df)

    # Keep only fraudulent transactions because this chart focuses on which merchants
    # have the highest number of fraud cases, not the number of legitimate transactions.
    fraud_only = valid[valid["Is Fraud"] == True]

    # Stop if there are no fraudulent transactions, because there would be no fraud counts to plot.
    if fraud_only.empty:
        print("No fraudulent transactions for merchant chart.")
        return

    # Count fraud cases for each merchant and keep only the ten highest counts
    # so the chart remains focused and easier to read.
    top = fraud_only[col].value_counts().head(10)

    # Give the chart enough width to display merchant names clearly.
    plt.figure(figsize=(10, 5))

    # Use a bar chart because it makes it easy to compare fraud counts between merchants.
    top.plot(kind="bar", color="darkred")

    # Add a title and axis labels so the reader understands what is being compared.
    plt.title(f"Top Merchants by Fraud Count ({col})")
    plt.xlabel("Merchant")
    plt.ylabel("Fraud Count")

    # Rotate and align the merchant names to reduce overlapping labels.
    plt.xticks(rotation=45, ha="right")

    # Save the chart and attempt to display it using the shared chart-handling function.
    _save_and_show("fraud_by_merchant_chart.png")

# ============================================================================================
# Function: chart_fraud_by_location
# Description: Bar chart of the 10 locations with the most fraud. Skipped if the dataset
#              has no location column.
# Parameters: df (pd.DataFrame): - the cleaned fraud detection data.
# Returns: None
# ============================================================================================
def chart_fraud_by_location(df):
    """Bar chart of top locations by fraud count."""

    # Look for a recognised location column name because different datasets
    # may use different names, such as Location or City.
    col = find_optional_column(df, optional_location_names)

    # Stop if no location column is available, because locations cannot be compared without it.
    if col is None:
        print("No Location/City column available for chart.")
        return

    # Use records with the required values so incomplete transactions do not affect the analysis.
    valid = get_valid(df)

    # Keep only fraudulent transactions because the chart focuses on where
    # fraud cases are recorded, rather than all transactions.
    fraud_only = valid[valid["Is Fraud"] == True]

    # Stop if there are no fraudulent transactions, because there would be no data to plot.
    if fraud_only.empty:
        print("No fraudulent transactions for location chart.")
        return

    # Count fraud cases for each location and keep the ten highest counts
    # to make the chart easier to read and compare.
    top = fraud_only[col].value_counts().head(10)

    # Create a wider figure so location names have more room on the chart.
    plt.figure(figsize=(10, 5))

    # Use a bar chart because it makes differences in fraud counts between locations easy to see.
    top.plot(kind="bar", color="darkorange")

    # Add a descriptive title and axis labels so the chart is easy to interpret.
    plt.title(f"Top Locations by Fraud Count ({col})")
    plt.xlabel("Location")
    plt.ylabel("Fraud Count")

    # Rotate and align the labels to reduce overlap when location names are long.
    plt.xticks(rotation=45, ha="right")

    # Save the chart and attempt to display it using the shared chart-handling function.
    _save_and_show("fraud_by_location_chart.png")
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