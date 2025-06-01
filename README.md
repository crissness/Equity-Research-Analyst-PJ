# Equity-Research-Analyst-PJ
# Beta Calculator for DCF Analysis

A comprehensive Python tool for calculating stock beta coefficients relative to market indices, designed specifically for DCF (Discounted Cash Flow) valuation models. The tool downloads real-time financial data from Yahoo Finance and performs statistical regression analysis to determine systematic risk.

## 🎯 Purpose

Beta (β) measures a stock's volatility relative to the overall market, making it essential for:
- **DCF Analysis**: Calculating the cost of equity using CAPM (Capital Asset Pricing Model)
- **Risk Assessment**: Understanding systematic risk exposure
- **Portfolio Management**: Evaluating volatility characteristics
- **Investment Analysis**: Comparing stocks' market sensitivity

## 🚀 Features

### Core Functionality
- **Automated Data Download**: Fetches historical price data from Yahoo Finance
- **Beta Calculation**: Performs linear regression to calculate beta coefficient
- **Statistical Analysis**: Provides comprehensive statistical metrics (R², correlation, alpha)
- **Visual Analysis**: Creates scatter plots with regression lines
- **Multi-Format Export**: Saves results to Excel with multiple sheets and optional text summaries

### Advanced Features
- **Flexible Time Periods**: Supports various periods (1d to max) and intervals
- **Custom Date Ranges**: Option to specify exact start/end dates
- **Robust Data Handling**: Handles MultiIndex DataFrames and missing data
- **International Markets**: Works with global stock exchanges
- **Quality Assessment**: Evaluates beta reliability for DCF usage

## 📋 Requirements

```python
# Required packages
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
scipy>=1.10.0
openpyxl>=3.1.0  # For Excel export
```

Install with:
```bash
pip install yfinance pandas numpy matplotlib scipy openpyxl
```

## 🔧 Usage

### Quick Start
```python
from beta_calculator import BetaCalculator

# Initialize calculator
calc = BetaCalculator()

# Download data (example: Apple vs S&P 500)
calc.download_data('AAPL', '^GSPC', period='2y')

# Calculate returns and beta
calc.calculate_returns()
calc.calculate_beta()

# Show regression plot
calc.plot_regression()

# Save to Excel
calc.save_to_excel(stock_symbol='AAPL', index_symbol='SP500')
```

### Interactive Mode
Run the script directly for guided input:
```bash
python beta_calculator.py
```

The interactive mode will prompt for:
- Stock ticker symbol
- Reference index ticker
- Time period and interval
- Custom date range (optional)
- Output preferences

## 📊 Supported Markets & Tickers

### Stock Examples
- **Italian Market**: ENI.MI, ENEL.MI, ISP.MI, UCG.MI, TIT.MI
- **US Market**: AAPL, MSFT, GOOGL, TSLA, NVDA
- **Other Markets**: Add appropriate suffixes (.L for London, .TO for Toronto, etc.)

### Index Examples
- **FTSE MIB (Italy)**: FTSEMIB.MI
- **S&P 500 (USA)**: ^GSPC
- **NASDAQ**: ^IXIC
- **DAX (Germany)**: ^GDAXI
- **FTSE 100 (UK)**: ^FTSE

## 📈 Output & Results

### Console Output
```
BETA ANALYSIS RESULTS
==================================================
Beta: 1.2450
Alpha: 0.001234 (0.1234%)
R-squared: 0.6789
Correlation: 0.8234
P-value: 0.000001
Standard Error: 0.045623
Number of observations: 504

INTERPRETATION
==================================================
The stock is more volatile than the market (Beta > 1)
```

### Excel Export (Multiple Sheets)
1. **Beta Results**: Key metrics and analysis parameters
2. **Stock Data**: Historical price data for the stock
3. **Index Data**: Historical price data for the index
4. **Returns**: Daily/period returns for both securities
5. **Statistics**: Additional statistical measures
6. **Interpretation**: Qualitative analysis and DCF recommendations

### Key Metrics Explained
- **Beta (β)**: Market sensitivity coefficient
  - β > 1: More volatile than market
  - β < 1: Less volatile than market
  - β < 0: Moves opposite to market
- **Alpha (α)**: Excess return vs. market
- **R-squared**: Regression quality (how well market explains stock movements)
- **Correlation**: Linear relationship strength

## 🎛️ Configuration Options

### Time Periods
- Short-term: `1d`, `5d`, `1mo`, `3mo`
- Medium-term: `6mo`, `1y`, `2y`
- Long-term: `5y`, `10y`, `max`
- Current year: `ytd`

### Data Intervals
- **Daily**: `1d` (recommended for beta calculation)
- **Weekly**: `1wk`
- **Monthly**: `1mo`
- **Intraday**: `1m`, `5m`, `15m`, `30m`, `1h` (for short periods only)

## 🔍 Interpretation Guidelines

### Beta Interpretation for DCF
- **β > 1.2**: High systematic risk, very volatile
- **1.0 < β < 1.2**: Elevated risk, more volatile than market
- **0.8 < β < 1.0**: Moderate risk, similar to market
- **0.5 < β < 0.8**: Lower risk, less volatile than market
- **β < 0.5**: Low systematic risk or negative correlation

### R-squared Quality Assessment
- **R² > 0.7**: Excellent - Market explains stock movements well
- **R² > 0.5**: Good - Moderately strong relationship
- **R² > 0.3**: Average - Weak relationship
- **R² < 0.3**: Poor - Consider sector beta instead

### DCF Usage Recommendations
- **✅ Reliable**: R² > 0.5 with 100+ observations
- **⚠️ Caution**: R² > 0.3 with sufficient data
- **❌ Unreliable**: R² < 0.3, consider industry beta

## 🛠️ Advanced Usage

### Custom Analysis Period
```python
calc.download_data(
    stock_symbol='ENI.MI',
    index_symbol='FTSEMIB.MI',
    start_date='2020-01-01',
    end_date='2024-01-01'
)
```

### Batch Analysis
```python
stocks = ['AAPL', 'MSFT', 'GOOGL']
results = {}

for stock in stocks:
    calc = BetaCalculator()
    calc.download_data(stock, '^GSPC', period='2y')
    calc.calculate_returns()
    calc.calculate_beta()
    results[stock] = calc.beta
```

### Statistical Summary
```python
# Get comprehensive statistics
stats = calc.get_summary_stats()
print(f"Annualized Volatility: {stats['Stock_Volatility']:.2%}")
print(f"Annualized Return: {stats['Average_Stock_Return']:.2%}")
```

## ⚠️ Limitations & Considerations

1. **Data Dependency**: Relies on Yahoo Finance data availability
2. **Market Hours**: Some tickers may have limited historical data
3. **Currency Effects**: International stocks affected by FX movements
4. **Statistical Assumptions**: Assumes linear relationship between stock and market
5. **Look-back Period**: Beta changes over time; use appropriate historical period

## 🔧 Troubleshooting

### Common Issues
- **"Unable to download data"**: Check ticker symbols and internet connection
- **"No Close column found"**: Verify ticker format for your market
- **"Few data points"**: Increase time period or change interval
- **Low R-squared**: Consider sector-specific index or longer time period

### Data Quality Checks
- Minimum 30 observations recommended
- R-squared > 0.3 for reliable results
- Check for structural breaks or outliers in data

## 📝 License & Disclaimer

This tool is for educational and analytical purposes. Financial data is provided by Yahoo Finance. Users should verify results and consider multiple data sources for investment decisions. Beta calculations are historical and may not predict future performance.
