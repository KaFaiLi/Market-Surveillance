# Marking the Close Analysis

Identifies potential "marking the close" trades by flagging executions in the last 15 minutes before market close across global markets. Generates summaries, visualizations, and exports for review.

## Outputs
All generated files are saved under the `output/` folder, including:
- Market severity comparison chart
- Market-specific charts
- Timeline charts
- Flagged deals export (`.xlsx` or `.csv` fallback)

## Requirements
- Python 3.9+
- pandas, numpy, matplotlib
- openpyxl (optional, for Excel export)

## Run
Update the `__main__` section or call `main()` with your dataset.

Example:
- Load a CSV/Excel file by passing `data_path`
- Or pass a prepared DataFrame

```python
results = main(data_path="path/to/your/booking_data.csv")
```

## Notes
- Execution times are stored in Paris time and converted to local market time for close-window checks.
- Output directory is created automatically if missing.
