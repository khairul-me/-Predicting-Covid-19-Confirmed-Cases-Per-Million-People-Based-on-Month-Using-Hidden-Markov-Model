# COVID-19 Case Prediction Using Hidden Markov Model (HMM)

## Project Overview
This project implements a Hidden Markov Model (HMM) to predict and analyze COVID-19 confirmed cases per million people based on the month of the year. We use the temporal patterns in COVID-19 data to build a predictive model that can:

1. Filter the level of infection for any given month
2. Predict the next 3 months of infection levels
3. Trace the most likely sequence of infection levels using the Viterbi algorithm

## Data Description
The project uses the "covid_data.csv" dataset, which contains the following information:
- **Date**: In MM/DD/YYYY format
- **New cases per million**: Daily new COVID-19 cases per million people
- **Total cases per million**: Cumulative case count per million people
- **New cases 7-day average**: 7-day moving average of new cases

For our HMM model, we aggregate the daily data into monthly sums and categorize them into 10 discrete infection levels.

## Hidden Markov Model Design

### Hidden States
We defined 10 hidden states representing different levels of infection:
1. level_0_1k: < 1,000 cases per million
2. level_1k_2k: 1,000 - 2,000 cases per million
3. level_2k_3k: 2,000 - 3,000 cases per million
4. level_3k_4k: 3,000 - 4,000 cases per million
5. level_4k_5k: 4,000 - 5,000 cases per million
6. level_5k_6k: 5,000 - 6,000 cases per million
7. level_6k_7k: 6,000 - 7,000 cases per million
8. level_7k_8k: 7,000 - 8,000 cases per million
9. level_8k_9k: 8,000 - 9,000 cases per million
10. level_9k_plus: > 9,000 cases per million

### Observable Evidence
The observable evidence in our model is the month of the year (January through December), represented as integers 1-12.

### Model Components
The HMM consists of three key probability distributions:
1. **Initial Probabilities**: The probability of starting in each infection level
2. **Transition Probabilities**: The probability of moving from one infection level to another
3. **Emission Probabilities**: The probability of observing a specific month given an infection level

## Implementation Details

### Data Processing
1. Load the COVID-19 data from CSV
2. Convert daily data to monthly sums
3. Categorize monthly data into discrete infection levels
4. Extract patterns for training the HMM

### Training the HMM
The HMM is trained by:
1. Counting initial state occurrences
2. Calculating transition probabilities between states
3. Deriving emission probabilities from states to months

### Key Algorithms

#### 1. Filtering
The `filter_current_month()` function estimates the current infection level given a month:
```python
def filter_current_month(self, month):
    """
    Filter the level of infection for the current month.
    
    Args:
        month (int): Month number (1-12)
        
    Returns:
        list: Probability distribution over states for the given month
    """
```

#### 2. Prediction
The `predict_next_months()` function forecasts infection levels for the next 3 months:
```python
def predict_next_months(self, month, num_months=3):
    """
    Predict the next n months' level of infection given the current month.
    
    Args:
        month (int): Current month number (1-12)
        num_months (int): Number of months to predict (default: 3)
        
    Returns:
        list: List of probability distributions for each predicted month
    """
```

#### 3. Viterbi Algorithm
The `viterbi_algorithm()` function finds the most likely sequence of infection levels:
```python
def viterbi_algorithm(self, start_month, end_month):
    """
    Use the Viterbi algorithm to find the most likely sequence of infection levels.
    
    Args:
        start_month (int): Starting month number (1-12)
        end_month (int): Ending month number (1-12)
        
    Returns:
        list: The most likely sequence of infection levels
    """
```

### Visualizations
The project includes several visualizations:
1. **Transition Probability Matrix**: Shows the likelihood of transitioning between infection levels
2. **Emission Probability Matrix**: Shows the relationship between infection levels and months
3. **Initial State Probabilities**: Shows the starting distribution of infection levels
4. **Viterbi Path Visualization**: Displays the most likely sequence of infection levels over time

## How to Run the Project

### Prerequisites
- Python 3.7 or higher
- Required libraries: pandas, numpy, matplotlib, seaborn

### Installation
```bash
pip install pandas numpy matplotlib seaborn
```

### Running the Code
1. Ensure "covid_data.csv" is in the same directory as the script
2. Run the script:
```bash
python CovidHMM.py
```

### Using the Interactive Demo
The program includes an interactive demo with the following options:
1. Filter level of infection for a current month
2. Predict next 3 months level of infection
3. Use Viterbi algorithm to explain infection levels
4. Visualize HMM components
5. Exit

## Results and Findings

### Model Accuracy
Our HMM model effectively captures seasonal patterns in COVID-19 transmission. By analyzing the transition probabilities, we can identify how infection levels typically evolve from month to month.

### Seasonal Patterns
The emission probability matrix reveals seasonal patterns in COVID-19 cases. Certain months show stronger associations with particular infection levels, reflecting seasonal variations in transmission.

### Prediction Capabilities
The model can predict future infection levels with reasonable accuracy, though predictions naturally become less certain the further into the future they extend.

## Limitations and Future Improvements

### Limitations
1. The model assumes that month is the primary factor in determining infection levels
2. Limited data span (only a few years of COVID-19 data)
3. Discrete states simplify what is essentially a continuous phenomenon

### Potential Improvements
1. Incorporate additional observable variables (e.g., vaccination rates, public policy changes)
2. Implement a continuous-state HMM for more nuanced modeling
3. Use more advanced techniques like the Baum-Welch algorithm for parameter estimation

## Conclusion
This Hidden Markov Model provides a useful probabilistic framework for understanding and predicting COVID-19 infection levels based on temporal patterns. The implementation successfully demonstrates the core algorithms of filtering, prediction, and explanation through the Viterbi algorithm.

While the model has limitations, it effectively captures the relationship between months and infection levels, providing insights into COVID-19 transmission patterns and enabling probabilistic forecasting of future trends.

## Contributors
- [Your Name]
- [Team Member Name (if applicable)]

## Acknowledgments
- The project was completed as part of [Course Name/Number]
- COVID-19 data sourced from [Data Source]
