# COVID-19 Data Analysis Project

## Overview
This project provides a comprehensive analysis of COVID-19 case data from January 2020 through February 2025. It offers tools for data processing, statistical analysis, and visualization of pandemic trends, including wave detection, growth rate calculations, and temporal patterns.

## Features
- **Data Loading & Preprocessing**: Handles COVID-19 time series data with proper date formatting and column renaming
- **Statistical Analysis**: Calculates basic pandemic statistics (peak cases, dates, totals)
- **Wave Detection**: Automatically identifies distinct COVID-19 waves based on customizable thresholds
- **Growth Rate Analysis**: Calculates and visualizes growth rates over specified time periods
- **Visualizations**: Generates multiple plot types:
  - Overall trend visualization
  - Wave identification with peak highlighting
  - Growth rate charts
  - Monthly averages
  - Year-month heatmaps for seasonal pattern analysis
- **Reporting**: Generates comprehensive analysis reports with key findings

## Prerequisites
- Python 3.7+
- Required packages:
  - pandas
  - matplotlib
  - seaborn
  - numpy

## Installation

### Using Conda (Recommended)
```bash
# Create a new conda environment
conda create -n covid_analysis python=3.9

# Activate the environment
conda activate covid_analysis

# Install required packages
conda install pandas matplotlib seaborn numpy
```

### Using pip
```bash
pip install pandas matplotlib seaborn numpy
```

## Project Structure
```
covid_analysis/
│
├── KIAnalyzer.py         # Main analysis script
├── covid_data.csv        # COVID-19 case data
│
└── covid_report/         # Generated reports and visualizations
    ├── covid_report.txt  # Summary analysis report
    ├── overall_trend.png # Overall case trend visualization
    ├── covid_waves.png   # Wave detection visualization
    ├── growth_rates.png  # Growth rate analysis
    ├── monthly_averages.png # Monthly case comparisons
    └── monthly_heatmap.png  # Year-month heatmap
```

## Usage
1. Ensure your COVID-19 data file is in the expected format (see Data Format section below)
2. Run the analysis script:
```bash
python KIAnalyzer.py
```
3. Review the generated report and visualizations in the `covid_report` directory

## Code Structure
The main analysis is performed using the `CovidDataAnalyzer` class which provides the following key methods:

- `__init__(file_path)`: Initializes the analyzer with a path to the data file
- `load_data()`: Loads and preprocesses the COVID data
- `get_basic_stats()`: Calculates basic statistics about the pandemic
- `detect_waves(threshold_multiplier, min_duration, separation_days)`: Detects COVID waves with customizable parameters
- `calculate_growth_rates(window)`: Calculates growth rates over specific time windows
- `plot_overall_trend(save_path)`: Plots the overall COVID case trend
- `plot_waves(waves, save_path)`: Visualizes detected waves
- `plot_growth_rates(growth_df, window, save_path)`: Visualizes growth rates
- `plot_monthly_averages(save_path)`: Shows monthly case averages
- `plot_heatmap(save_path)`: Creates a year-month heatmap of case data
- `generate_full_report(output_dir)`: Generates a complete analysis report

## Data Format
The script expects a CSV file with the following columns:
- `country`: Country/region name
- `date`: Date of observation (format: MM/DD/YYYY)
- `new_cases_per_million`: Daily new cases per million people
- `total_cases_per_million`: Cumulative cases per million
- `new_cases_per_million_7_day_avg_right`: 7-day moving average of new cases (optional, will be calculated if missing)

## Key Findings
The analysis identified 6 distinct COVID-19 waves:

1. **First Wave**: Dec 13, 2020 - Jan 23, 2021 (41 days)
   - Peak: Jan 12, 2021 (91.51 cases per million)

2. **Second Wave**: Apr 10, 2021 - May 22, 2021 (42 days)
   - Peak: May 1, 2021 (103.52 cases per million)

3. **Third Wave**: Aug 7, 2021 - Sep 5, 2021 (29 days)
   - Peak: Aug 20, 2021 (82.45 cases per million)

4. **Fourth Wave**: Dec 16, 2021 - Apr 30, 2022 (135 days)
   - Peak: Jan 26, 2022 (430.95 cases per million)

5. **Fifth Wave**: Jun 26, 2022 - Sep 3, 2022 (69 days)
   - Peak: Jul 29, 2022 (133.17 cases per million)

6. **Sixth Wave**: Dec 11, 2022 - Jan 17, 2023 (37 days)
   - Peak: Dec 26, 2022 (801.37 cases per million)

The highest daily case count was 1,051.71 cases per million (Jan 30, 2022), and the total cases per million by the end of the period reached 97,323.55.

## Customizing the Analysis
The wave detection algorithm can be customized using the following parameters:

- `threshold_multiplier`: Multiplier of the mean used as threshold for wave detection (default: 1.5)
- `min_duration`: Minimum duration in days to consider as a wave (default: 14)
- `separation_days`: Minimum days between waves (default: 30)

Example:
```python
# Detect waves with more stringent criteria
waves = analyzer.detect_waves(threshold_multiplier=2.0, min_duration=21, separation_days=45)
```

## Contributing
Contributions to improve the analysis or add new features are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- Data sources for COVID-19 case information
- Python data science community for libraries and tools
- Contributors to pandas, matplotlib, seaborn, and numpy

## Contact
For questions or feedback, please contact [khairul.islam@hws.edu]
