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


# ==================================================
# Robust imports
# I used try/except to show a simple error message if a package is missing,
# instead of letting the program stop with an error.
# ==================================================


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
    

# Store the dataset filename in a variable for easy reuse when loading the fraud detection data.
file_name =  "fraud_detection_data.csv"

# Columns the script needs. Checked during load so a mismatch is caught early.
required_columns = ["Timestamp", "Amount (USD)", "Is Fraud"]

# Clear marker for rows that could not be fully cleaned. Chosen so it is
# obvious when inspecting the data later and will not be mistaken for real data.
# This is used instead of deleting rows so the user can see that the row existed and investigate it further.
bad_value_marker = "INVALID"

# ============================================================================================
# Function: load_data
# Description: Loads the fraud detection data from a CSV file into a pandas DataFrame.
#              Also checks that the required columns exist.
# Parameters: file_name (str): - path to the CSV file containing the fraud detection data.
# Returns : pd.DataFrame or None if the file cannot be read / columns are missing.
# ============================================================================================

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
    print(df.shape)
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
    data = df.copy() # Keeps the original data safe so it is not affected if cleaning is repeated.

    # Changes the Timestamp column into dates and times so they can be used for analysis.
    # Use errors="coerce" so incorrect dates become NaT instead of stopping the program.
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce") 

    # Count missing timestamps after conversion to identify dates that couldn't be processed.
    unconverted_dates = data["Timestamp"].isnull().sum()

    if unconverted_dates == 0: # Confirms that all timestamps were converted successfully.
        print("All timestamps converted successfully.")
    else: # Informs the user how many timestamps could not be converted.
        print(f"{unconverted_dates} timestamps could not be converted.")

    # Convert Amount (USD) to numeric values for accurate summaries and charts.
    # Use errors="coerce" so invalid values become NaN instead of stopping the script.
    data["Amount (USD)"] = pd.to_numeric(data["Amount (USD)"], errors="coerce")

    # Count how many Amount (USD) values could not be converted and became NaN.
    unconverted_amount = data["Amount (USD)"].isnull().sum()

    # Check if all Amount (USD) values were converted successfully and inform the user.
    if unconverted_amount == 0:
        print("All Amount (USD) converted successfully.")

    # If some values could not be converted, tell the user how many failed.
    else:
        print(f"{unconverted_amount} Amount (USD) values could not be converted.")

    # Find amounts that are zero or negative (these are usually data errors).
    invalid_amounts = (data["Amount (USD)"].notnull()) and (data["Amount (USD)"] <= 0)

    # Count how many invalid amounts were found.
    invalid_amount_count = invalid_amounts.sum()

    # Check if there are no zero or negative amounts.
    if invalid_amount_count == 0:
        print("No zero or negative Amount (USD) values found.")

    # If there are invalid amounts, inform the user how many were found.
    else:
        print(f"{invalid_amount_count} zero or negative Amount (USD) values found. Marked as INVALID.")

    # Convert different Is Fraud values into True/False/missing.
    # This makes later counting and charting much simpler and safer.
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

    # Find rows where the timestamp, amount, or fraud label is missing,
    # or where the amount is zero or negative, and mark them as INVALID.
    bad_rows = (data["Timestamp"].isnull() | data["Amount (USD)"].isnull() | data["Is Fraud"].isnull() | invalid_amounts)
    data.loc[bad_rows, "Data Status"] = bad_value_marker # Mark the identified rows as INVALID.

    # Tell the user how many rows were marked as having problems.
    print(f"Rows marked '{bad_value_marker}': {bad_rows.sum()}")


    # Check for duplicate transactions without removing them, since repeated 
    # transactions may be genuine and should be investigated before deciding to delete them.
    duplicate_rows = data.duplicated().sum()

    if duplicate_rows == 0: # Check if there are no duplicate transactions.
        print("No duplicate transactions found.")
    
    else: # If duplicates are found, tell the user how many need to be reviewed.
        print(f"{duplicate_rows} duplicate transactions found. Further investigation is required.")
    
    return data # Return the cleaned data so the other functions can use it.

# ============================================================================================
# Function: summarise_data
# Description: Print amount statistics and fraud counts and fraud percentage to give the user a quick overview of the data.
# Parameters: df (pd.DataFrame)
# Returns: None
# ============================================================================================

# Define a function to display a summary of the fraud detection data so the user can quickly understand the key information.
def summarise_data(df):

    # Use only the Amount (USD) column to show clear numerical statistics.
    # Summarising the whole DataFrame would include non-numeric columns and be less clear.
    print("Amount statistics:")

    # Display descriptive statistics for the Amount (USD) column, including count, mean,standard deviation,
    # minimum, maximum, and quartiles for the user to understand the distribution of transaction amounts.
    print(df["Amount (USD)"].describe())

    # Print an empty line to make the output easier to read by separating the amount statistics from the fraud counts.
    print()

    # Display how many transactions are fraudulent and how many are not, so the user can see the balance between the two groups.
    # dropna=False keeps missing fraud labels in the count instead of leaving them out.
    print("Fraud count:")
    print(df["Is Fraud"].value_counts(dropna=False))
    print()

    # Calculate the percentage of transactions marked as fraud.
    # Only use True/False values so missing labels do not affect the percentage calculation.
    valid_labels = df["Is Fraud"].dropna() # Remove missing fraud labels to avoid skewing the percentage calculation.

    # Check if there are no valid fraud labels to calculate a percentage.
    if len(valid_labels) == 0:
        print("No valid fraud labels to calculate a percentage.")

    # If valid labels exist, calculate and display the fraud percentage.
    else:
        fraud_rate = valid_labels.mean() * 100 # Calculate the percentage of fraudulent transactions from the valid labels.
        print(f"Fraud percentage: {fraud_rate:.1f}%") # Display the fraud percentage to one decimal place for easier reading.

    # Compare the average amount of fradulent and legitimate.
    # This helps me see which group has higher transaction values.
    print()
    print("Average amount by fraud status:")

    # Calculate the average transaction amount for each group.
    fraud_avg = df[df["Is Fraud"] == True] ["Amount (USD)"].mean()
    legit_avg = df[df["Is Fraud"] == False] ["Amount (USD)"].mean()

    # Check if an average value was found for fradulent transactions.
    if pd.notna(fraud_avg):
        print(f"Fraudulent transactions: {fraud_avg:.2f}")
    else:
        print("Fraudulent transactions: No data")

    # Check if an average value was found for legitimate transactions.
    if pd.notna(legit_avg):
        print(f"Legitimate transactions: {legit_avg:.2f}")
    else:
        print("Legitimate transactions: No data")
            
    
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

# Save the chart so it can be viewed later.
filename = "fraud_counts_chart.png"
plt.savefig(filename)
print(f"Chart saved as '{filename}' in the current folder.")
# Try to display the chart on the screen.
try:
    plt.show()
except Exception:
    print("Could not open the chart window (this is normal in some environment).")
plt.close() # Close the chart after use.

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