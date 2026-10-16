# Fraud Detection Analysis

A menu-driven Python program that loads, cleans, summarises and visualises a fraud detection dataset. Everything runs from one script, `victor_fraud_detection.py`, and nothing happens until the user chooses an option from the menu.

**Author:** Victor Oshiogwere Pius

## What it does

1. **Load** the CSV file and check that the required columns are present.
2. **Clean** the data: fix data types, flag problem rows, detect duplicates, and let the user choose whether to keep or delete them.
3. **Summarise** the data: overall, over time, by amount, by merchant and by location.
4. **Visualise** the data: six charts, each saved as a PNG file.

## Files

| File | Purpose |
| --- | --- |
| `victor_fraud_detection.py` | The main program |
| `fraud_detection_data.csv` | The dataset the program reads |
| `instructions.txt` | Short run instructions |
| `fraud project.ipynb` | Notebook used to prototype the analysis |
| `charts/` | Example chart images produced by the program |

## Requirements

- Python 3 (developed on Python 3.14.2; other recent Python 3 versions should work but have not been tested)
- pandas
- matplotlib

Install the libraries from the terminal:

```
pip install pandas matplotlib
```

## How to run

1. Put `victor_fraud_detection.py` and `fraud_detection_data.csv` in the **same folder**.
2. Open a terminal in that folder.
3. Run:

```
python victor_fraud_detection.py
```

The program looks for the exact file name `fraud_detection_data.csv`. If your file is named differently, rename it or change the `file_name` line near the top of the script.

## Menu

| Option | Action |
| --- | --- |
| 1 | Load the CSV file |
| 2 | Clean the data |
| 3 | Show summaries |
| 4 | Show charts |
| 5 | Exit |

Load must succeed before cleaning, summarising or visualising. If you choose summaries or charts before cleaning, the program cleans the data first.

## Data cleaning

Cleaning does not silently remove anything. It works as follows:

- **Timestamps** are converted to dates. Unreadable or missing values are marked as missing.
- **Amounts** are converted to numbers. Text, blanks and values such as `$1,200.50` that cannot be read are marked as missing.
- **Fraud labels** are converted to True/False. Variants such as `Yes`, `N`, `TRUE`, `1` and `0` are accepted. Unrecognised or blank labels are marked as missing.
- **Problem rows** are flagged as `INVALID` in a `Data Status` column. This covers missing or unreadable values and amounts of zero or less.
- **Duplicates** (exact copies of another row) are counted and reported.

After cleaning, the program asks the user what to do with the flagged rows:

```
Delete the 10 INVALID rows? Type 'keep' or 'delete'
Delete the 2 duplicate rows? Type 'keep' or 'delete'
```

- **Delete** removes the rows from the data held in memory. The original CSV file is never changed.
- **Keep** leaves the rows in the data but they stay flagged. Rows marked `INVALID` are left out of every summary and chart, so a bad value never affects the results.

## Summaries

- Overall: row count, fraud count and fraud rate, plus data quality counts
- Time: fraud by month
- Amount: statistics for transaction amounts
- Merchant: fraud rate by merchant (merchants with at least 2 transactions)
- Location: fraud rate by location

## Charts

Each chart is shown on screen and saved as a PNG file. The program checks that the file exists after saving and prints its full path.

| Chart | File |
| --- | --- |
| Fraud vs non-fraud counts | `fraud_counts_chart.png` |
| Fraud by month | `fraud_by_month_chart.png` |
| Amount histogram | `amount_histogram.png` |
| Amount box plot, fraud vs non-fraud | `amount_boxplot_fraud.png` |
| Fraud by merchant | `fraud_by_merchant_chart.png` |
| Fraud by location | `fraud_by_location_chart.png` |

Option 7 in the chart menu produces all six charts at once.

## Testing with messy data

The program was tested with a deliberately messy copy of the dataset containing:

- unreadable and missing timestamps
- text, blank and currency-formatted amounts
- zero and negative amounts
- fraud label variants, unrecognised labels and blank labels
- missing merchant and location values
- exact duplicate rows

It was also tested with a CSV that is missing the Amount column and with an empty CSV, to check that the error messages appear instead of the program crashing.

## Notes

- The program only reads the CSV. It never writes to it.
