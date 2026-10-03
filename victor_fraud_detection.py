# - Python version used: 3.14.2. 
# - Any recent version of Python 3 should work, but I have not tested it on earlier versions.

# ========================================================================================
# Victor's Fraud Detection Script
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
# Plain Python does not provide built in plotting tools, 
# so matplotlib makes it easier to visualise transaction patterns and compare the two groups.
# The alias 'plt' shortens the long module name for repeated use throughout the script.


import pandas as pd
import matplotlib.pyplot as plt

# Store the dataset filename in a variable for easy reuse when loading the fraud detection data.
file_name =  "fraud_detection_data.csv"

# Keep the loading code in one reusable function so the menu can run it
# whenever the user selects the load option.
def load_data(file_name):
    # Use try and except to handle cases where fraud detection CSV file cannot be found.
    # This prevents the script from stopping unexpectedly and displays a helpful message 
    # so the user can check the filename and folder location.
    try:
        # Store the loaded fraud detection data in 'df' so I can use it for cleaning, summaries and charts.
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"{file_name} not found. Check the file name and the folder.")
    else:
        print("File has been found.")
        print(df.shape)
        print(df.head())
        return df

# Save the table in df so the other functions can work with it.
df = load_data("fraud_detection_data.csv")


# Prepare the data by fixing incorrect types and checking for problems before summarising and plotting
def clean_data(df):
    # Convert the Timestamp column to datetime for accurate time-based analysis.
    # Use errors="coerce" to prevent invalid dates from stopping the script.
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Count missing timestamps after conversion to identify dates that couldn't be processed.
    unconverted_dates = df["Timestamp"].isnull().sum()

    # Display a message based on whether any timestamps are missing.
    if unconverted_dates == 0:
        print("All timestamps converted successfully.")
    else:
        print(f"{unconverted_dates} timestamps could not be converted.")

    # Convert Amount (USD) to numeric values for accurate summaries and charts.
    # Use errors="coerce" so invalid values become missing instead of stopping the script.
    df["Amount (USD)"] = pd.to_numeric(df["Amount (USD)"], errors="coerce")

    # Count missing Amount (USD) after conversion to identify amounts that could not be processed.
    unconverted_amount = df["Amount (USD)"].isnull().sum()

    # Display a message based on whether any amounts are missing, 
    # so the user knows if bad values were found, rather than assuming the clean-up worked.
    if unconverted_amount == 0:
        print("All Amount (USD) converted successfully.")
    else:
        print(f"{unconverted_amount} Amount (USD) values could not be converted.")

    # Check for duplicate transactions without removing them, since repeated 
    # transactions may be genuine and should be investigated before deciding to delete them.
    duplicate_rows = df.duplicated().sum()

    if duplicate_rows == 0:
        print("No duplicate transactions found.")
    else:
        print(f"{duplicate_rows} duplicate transactions found. Further investigation is required.")
    # Return the clean data so the other functions can use it.
    return df
# Keep df updated with the cleaned data so other functions use the latest version.
df = clean_data(df)

# Summarise the data so the menu can show the user useful information.
def summarise_data(df):
    # Gives the user a quick summary of the numerical data without reading every row.
    print("Amount statistics:")
    print(df.describe())
    # Check how common fraud is compared with normal transactions.
    print("Fraud count:")
    print(df["Is Fraud"].value_counts())
summarise_data(df)

# Create a chart showing the number of fraud and non-fraud transactions when selected from the menu
def plot_fraud_counts(df):
    fraud_counts = df["Is Fraud"].value_counts()
    # Convert True and False into clear fraud labels so the user can easily understand the chart.
    plt.bar(fraud_counts.index.map({True: "Fraud", False: "Not Fraud"}), fraud_counts.values)
    plt.title("Fraud Vs Non-Fraud Transactions")
    plt.xlabel("Fraud and Not Fraud")
    plt.ylabel("Number of transactions")
    # Display the chart so the user can see it when the script runs.
    plt.show()
plot_fraud_counts(df)

# Keep the menu in one function so it can be displayed again whenever the user needs to make a choice.
def show_menu():
    print("1. Load the fraud detection data")
    print("2. Clean the fraud detection data")
    print("3. Summarise the fraud detection data")
    print("4. Visualise the fraud detection data")
    print("5. Exit")

# Start with no data loaded so the program knows when the user has not selected a file yet.
df = None
# Keeps showing the menu until the user chooses to exit.
while True:
    show_menu()
    choice = input("Enter your choice (1-5): ")
    # Check if the user entered a valid menu option before processing their choice.
    if choice not in [ "1", "2", "3", "4", "5"]:
        print("Invalid choice. Please enter a number from 1 to 5.")
    # Check which option the user selected so the program knows what to do.
    elif choice == "1":
        df = load_data("fraud_detection_data.csv")
    elif choice == "5":
        # Stops the menu loop when the user chooses to exit.
        break
    elif df is None:
        print("Fraud detection data is empty. Please load the data first using option 1")
    elif choice == "2":
        df = clean_data(df) # Cleans the fraud detection data so it is ready for analysis.
    elif choice == "3":
        summarise_data(df) # Summarises the fraud detection data so the user can understand the key information.
    elif choice == "4":
        plot_fraud_counts(df) # Plot fraud counts so the user can easily compare fraudulent and non-fraudulent transactions.
    
    
