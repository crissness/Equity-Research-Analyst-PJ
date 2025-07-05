# Integrated WACC Calculator 📊

A comprehensive Python tool for calculating the Weighted Average Cost of Capital (WACC) through an integrated workflow that combines Beta calculation, Cost of Equity (CAPM), Cost of Debt, and final WACC computation.

## 🎯 Overview

This calculator provides a complete financial analysis workflow:
1. **Beta Calculation**: Regression analysis between stock and market index returns
2. **Cost of Equity**: Using the Capital Asset Pricing Model (CAPM)
3. **Cost of Debt**: Using synthetic credit ratings based on interest coverage ratios
4. **WACC**: Final weighted average cost of capital calculation

## 📋 Requirements

### Python Libraries
```bash
pip install yfinance pandas numpy matplotlib scipy openpyxl
```

### Excel Database Files
You'll need the following Excel files with market data:

1. **Bond.xlsx** - Contains 10-year government bond yields
   - Required columns: `Country`, `Yield 10y` (or similar)
   - Example: USA - 4.5%, Germany - 2.8%, etc.

2. **ERP 2025.xlsx** - Contains Equity Risk Premium by country
   - Required columns: `Country`, `Total Equity Risk Premium` (or similar)
   - Example: USA - 5.5%, Italy - 7.2%, etc.

3. **Synthetic Ratings.xlsx** - Contains credit rating mappings
   - Interest coverage ratio ranges for Large Cap, Small Cap, and Financial firms
   - Credit spreads for each rating category

## 🚀 Usage

### Basic Usage
```python
python integrated_wacc_calculator.py
```

### Step-by-Step Process

#### 1. Beta and Cost of Equity Calculation
```
Enter stock ticker: AAPL
Enter index ticker: ^GSPC
Bond Excel file path (default: 'Bond.xlsx'): 
ERP Excel file path (default: 'ERP 2025.xlsx'): 
Period (default: 10y): 
Interval (default: 1mo): 
```

The calculator will:
- Download historical price data from Yahoo Finance
- Calculate logarithmic returns
- Perform regression analysis to determine Beta
- Select risk-free rate from bond database
- Apply CAPM formula: `Cost of Equity = Rf + β × ERP`

#### 2. Cost of Debt Calculation
```
Enter current EBIT (in millions): 100000
Enter current Interest Expenses (in millions): 3500
Enter marginal tax rate (%): 21
```

The calculator will:
- Fetch company information from Yahoo Finance
- Calculate interest coverage ratio
- Assign synthetic credit rating
- Apply formula: `Cost of Debt = (Rf + Spread) × (1 - Tax Rate)`

#### 3. WACC Calculation
```
Use Market Values or Book Values? (market/book): market
Enter Total Debt from balance sheet (in millions): 120000
Enter Weighted Average Maturity of debt (years): 7.5
```

The calculator will:
- Calculate market value of debt using DCF approach
- Determine weights of equity and debt
- Apply formula: `WACC = (E/V × Re) + (D/V × Rd)`

## 📊 Output

### Console Output
- Real-time calculation results
- Step-by-step formula breakdowns
- Summary statistics and diagnostics

### Excel Output
A comprehensive Excel file containing:
- **Summary**: All key results at a glance
- **Beta_Statistics**: Regression analysis details
- **CAPM_Calculation**: Cost of equity breakdown
- **Cost_of_Debt_Calc**: Cost of debt computation
- **WACC_Calculation**: Final WACC breakdown
- **Historical Data**: Stock and index prices
- **Returns_Data**: Calculated returns

## 📈 Example

### Input Example
```
Stock: AAPL (Apple Inc.)
Index: ^GSPC (S&P 500)
EBIT: $100,000 million
Interest Expense: $3,500 million
Tax Rate: 21%
Debt: $120,000 million
Maturity: 7.5 years
```

### Output Example
```
Beta: 1.2456
Cost of Equity: 11.23%
Cost of Debt: 3.85%
WACC: 8.76%
```

## 🔧 Customization

### Supported Ticker Formats
- **US Stocks**: AAPL, MSFT, GOOGL
- **European Stocks**: ASML.AS, SAP.DE, ENI.MI
- **Market Indices**: ^GSPC, ^IXIC, ^FTSE, FTSEMIB.MI

### Period Options
- `1y`, `2y`, `5y`, `10y` (years)
- `1mo`, `3mo`, `6mo` (months)
- `ytd` (year to date)

### Interval Options
- `1d` (daily)
- `1wk` (weekly)
- `1mo` (monthly)

## ⚠️ Important Notes

1. **Internet Connection**: Required for downloading data from Yahoo Finance
2. **Data Quality**: Ensure your Excel databases are up-to-date
3. **Market Hours**: Some real-time data may be delayed
4. **Company Type**: The calculator automatically detects financial services firms
5. **Regression Quality**: Check R-squared values (>0.3 recommended)

## 🐛 Troubleshooting

### Common Issues

1. **"No data found for ticker"**
   - Verify the ticker symbol is correct
   - Check if the company is listed on Yahoo Finance

2. **"Bond/ERP data not found"**
   - Ensure Excel files are in the correct format
   - Check country names match between files

3. **"Invalid coverage ratio"**
   - Verify EBIT and Interest Expense values
   - Check for negative or zero values

## 📝 Formulas Reference

### CAPM (Cost of Equity)
```
Re = Rf + β × ERP
```

### Cost of Debt
```
Rd = (Rf + Credit Spread) × (1 - Tax Rate)
```

### WACC
```
WACC = (E/V × Re) + (D/V × Rd)
```

Where:
- E = Market value of equity
- D = Market value of debt
- V = E + D (Total value)
- Re = Cost of equity
- Rd = After-tax cost of debt

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Yahoo Finance for providing financial data
- Professor Aswath Damodaran for synthetic rating methodology
- Financial modeling best practices from CFA Institute

---

**Happy Financial Modeling! 📈**
