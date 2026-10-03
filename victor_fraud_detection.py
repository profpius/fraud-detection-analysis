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



# Import pandas to read and clean the fraud detection data csv file.
# Using plain Python would require more manual code to handle rows, columns, missing values and data summaries.
# Pandas provides DataFrame and built-in functions to perform these operations efficiently.
# The alias 'pd' provides a shorter name when calling pandas functions repeatedly throughout the script.
# Import matplotlib.pyplot to create charts for comparing fraudulent and legitimate transactions.
# Plain Python does not provide built-in plotting tools, 
# so matplotlib makes it easier to visualise transaction patterns and compare the two groups.
# The alias 'plt' shortens the long module name for repeated use throughout the script.


import pandas as pd
import matplotlib.pyplot as plt

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

df = None # Keeps the current data empty until a file loads successfully.
loaded = load_data(file_name) # Attempts to load the fraud detection dataset.
if loaded is not None: # Replaces the current data only after the new file loads successfully.
    df = loaded

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
    # Use errors="coerce" so incorrect dates becomes NaT instead of stopping the program.
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

    # Check for amounts that are not missing and are zero or negative.
    invalid_amounts = (data["Amount (USD)"].notnull()) & (data["Amount (USD)"] <= 0)

    # Count how many invalid amounts were found.
    invalid_amount_count = invalid_amounts.sum()

    # Check if there are no zero or negative amounts.
    if invalid_amount_count == 0:
        print("No zero or negative Amount (USD) values found.")

    # If there are invalid amounts, inform the user how many were found.
    else:
        print(f"{invalid_amount_count} zero or negative Amount (USD) values found. Marked as INVALID.")

    # Convert different Is Fraud values into True or False.
    # This makes the fraud column easier to use in calculations.
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
    
# ======================================================================
# Function: plot_fraud_counts
# Description: Bar chart of fraud vs non-fraud transaction counts.
# Parameters: df (pd.DataFrame)
# Retuns: None 
# =======================================================================

# Define a function to create a chart showing fraud and non-fraud transactions.
def plot_fraud_counts(df):
    # Check how many transactions are marked as fraud and not fraud.
    fraud_counts = df["Is Fraud"].value_counts()

    # Convert True and False into clear fraud labels so the user can easily understand the chart.
    plt.bar(fraud_counts.index.map({True: "Fraud", False: "Not Fraud"}), fraud_counts.values)
    # Add a title to explain what the chart shows.
    plt.title("Fraud Vs Non-Fraud Transactions")
    # Label the horizontal axis to explain transaction categories.
    plt.xlabel("Transaction Type")
    # Label the vertical axis to show what the numbers represent.
    plt.ylabel("Number of transactions")
    # Display the chart and wait until the chart window is closed before returning to the menu.
    plt.show()


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
        choice = input("Enter your choice (1-5): ")
    except (KeyboardInterrupt, EOFError): # Handle Ctrl+C or closed input so the program exits instead of getting stuck.
        print("\nExiting.") # Tells the user the program is closing.
        break

    if choice not in [ "1", "2", "3", "4", "5"]: # Check if the user entered a valid menu option before processing their choice.
        print("Invalid choice. Please enter a number from 1 to 5.") # Ask the user to enter a valid option.
        continue # Return to the menu so the user can enter a valid choice.
    if choice == "5": # Check if the user selected option 5 to exit the program.
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