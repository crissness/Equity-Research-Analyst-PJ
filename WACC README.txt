# WACC Calculator

A comprehensive Python tool for calculating the Weighted Average Cost of Capital (WACC) for companies using either market values or book values.

## Overview

The Weighted Average Cost of Capital (WACC) is a critical financial metric that represents the average cost of capital from all sources (equity and debt) weighted by their respective proportions in the company's capital structure. This calculator provides an interactive command-line interface to compute WACC with professional-grade accuracy and detailed reporting.

**Formula:** `WACC = (Cost of Equity × Weight of Equity) + (Cost of Debt × Weight of Debt)`

## Features

- **Interactive Command-Line Interface**: Step-by-step guided input process
- **Dual Valuation Methods**: Choose between market values or book values
- **Real-Time Market Data**: Automatic fetching of market capitalization via Yahoo Finance
- **Market Value of Debt Calculation**: Precise calculation using present value formulas
- **Excel Export**: Comprehensive results saved to formatted Excel files
- **Input Validation**: Robust error handling and data validation
- **Professional Reporting**: Detailed calculation breakdowns and formatted outputs

## Installation

### Prerequisites

```bash
pip install yfinance pandas numpy openpyxl
```

### Required Libraries

- `yfinance`: For fetching real-time market data
- `pandas`: For data manipulation and Excel export
- `numpy`: For numerical calculations
- `openpyxl`: For Excel file generation

## Usage

### Basic Usage

```bash
python wacc_calculator.py
```

The calculator will guide you through the following steps:

1. **Cost of Capital Inputs**
   - Cost of Equity (%)
   - Cost of Debt (%)

2. **Valuation Method Selection**
   - Market Values (recommended)
   - Book Values

3. **Value Inputs** (based on selected method)
   - Market Values: Ticker symbol, debt details, maturity, interest expense
   - Book Values: Book value of equity and debt

4. **WACC Calculation**
   - Automatic calculation and detailed breakdown

5. **Excel Export** (optional)
   - Comprehensive results saved to Excel file

### Example Workflow

```
======================================================================
WACC CALCULATOR
======================================================================
This calculator computes the Weighted Average Cost of Capital using:
WACC = (Cost of Equity × Weight of Equity) + (Cost of Debt × Weight of Debt)
======================================================================

==================================================
COST OF CAPITAL INPUTS
==================================================
Enter Cost of Equity (%, e.g., 12.5): 10.5
✅ Cost of Equity: 10.50%
Enter Cost of Debt (%, e.g., 4.5): 3.8
✅ Cost of Debt: 3.80%

==================================================
VALUATION METHOD SELECTION
==================================================
Choose valuation approach for weights calculation:
• Market Values: Uses current market capitalization and market value of debt
• Book Values: Uses balance sheet values from financial statements

Use Market Values or Book Values? (market/book, default: market): market
✅ Selected: Market Values
```

## Calculation Methods

### Market Value Approach (Recommended)

**Market Value of Equity:**
- Fetched automatically using company ticker from Yahoo Finance
- Real-time market capitalization

**Market Value of Debt:**
- Formula: `(Interest Expense × (1 - (1 / (1 + Cost of Debt))) / Cost of Debt) + (Total Debt / ((1 + Cost of Debt)^Maturity))`
- Components:
  - Present Value of Interest Payments
  - Present Value of Principal Repayment

### Book Value Approach

Uses balance sheet values directly:
- Book Value of Equity from shareholders' equity
- Book Value of Debt from total debt

## Input Requirements

### For Market Value Calculation:
- **Company Ticker**: Valid stock symbol (e.g., AAPL, MSFT, TSLA)
- **Cost of Equity**: 0% - 50%
- **Cost of Debt**: 0% - 30%
- **Total Debt**: From balance sheet (in millions)
- **Weighted Average Maturity**: Average maturity of debt (years)
- **Interest Expense**: From income statement (in millions)

### For Book Value Calculation:
- **Cost of Equity**: 0% - 50%
- **Cost of Debt**: 0% - 30%
- **Book Value of Equity**: From balance sheet (in millions)
- **Book Value of Debt**: From balance sheet (in millions)

## Output

### Console Output
- Step-by-step calculation breakdown
- Formatted results with weights and components
- Final WACC percentage
- Professional summary

### Excel Export
The generated Excel file contains three sheets:

1. **WACC_Results**: Complete analysis summary
2. **WACC_Calculation**: Component-by-component breakdown
3. **Debt_Valuation**: Market value of debt calculation details (market method only)

## Example Output

```
============================================================
WACC CALCULATION
============================================================
FORMULA: WACC = (Cost of Equity × Weight of Equity) + (Cost of Debt × Weight of Debt)
============================================================
Valuation Method: Market Values
Value of Equity:         $2,500,000,000,000
Value of Debt:           $120,000,000,000
Total Value:             $2,620,000,000,000

Weight of Equity:        95.4%
Weight of Debt:          4.6%

Cost of Equity:          10.50%
Cost of Debt:            3.80%
============================================================
CALCULATION:
WACC = (10.50% × 95.4%) + (3.80% × 4.6%)
WACC = 10.02% + 0.17%
WACC = 10.19%
============================================================
```

## Use Cases

The calculated WACC can be used for:

- **DCF Valuation Models**: As the discount rate for free cash flows
- **Investment Project Evaluation**: Hurdle rate for capital budgeting
- **Corporate Financial Planning**: Cost of capital benchmarking
- **Performance Measurement**: Economic value added (EVA) calculations
- **Merger & Acquisition Analysis**: Valuation discount rates

## Technical Details

### Market Value of Debt Formula

The calculator uses the bond valuation approach to estimate market value of debt:

```
Market Value of Debt = PV(Interest Payments) + PV(Principal)

Where:
- PV(Interest Payments) = (Interest Expense × (1 - (1/(1 + r))) / r)
- PV(Principal) = Face Value / (1 + r)^n
- r = Cost of Debt
- n = Weighted Average Maturity
```

### Data Sources

- **Market Data**: Yahoo Finance API via `yfinance`
- **Financial Statements**: User input (manual entry required)

## Error Handling

The calculator includes comprehensive error handling for:

- Invalid ticker symbols
- Network connectivity issues
- Malformed input data
- Missing financial data
- File writing permissions

## Limitations

- Requires manual input of financial statement data
- Market value of debt calculation assumes simplified bond structure
- Yahoo Finance API dependency for market data
- Does not account for tax effects (after-tax cost of debt calculation not included)

## File Naming Convention

Excel output files follow the pattern:
```
wacc_analysis_{TICKER}_{TIMESTAMP}.xlsx
```

Example: `wacc_analysis_AAPL_20241215_143022.xlsx`

## Contributing

This calculator can be enhanced with:

- Tax adjustment options for cost of debt
- Multiple debt instrument handling
- Historical WACC analysis
- Sensitivity analysis features
- GUI interface development

## License

This tool is provided for educational and professional use. Ensure compliance with data source terms of service when using market data APIs.

---

**Note**: This calculator provides estimates based on input data and market conditions. Always verify results with additional analysis and consider consulting financial professionals for critical business decisions.