# Valuation Model Selection Tool

A Python-based interactive tool that helps you choose the appropriate valuation model for any company based on Aswath Damodaran’s model selection framework.

## Overview

This tool guides you through a series of questions about a company’s financial situation and recommends the most suitable valuation approach. It considers factors like earnings status, growth patterns, financial leverage, and cash flow characteristics to determine the optimal discounted cash flow model.

## Features

- **Interactive questionnaire** - Step-by-step input collection with clear prompts
- **Intelligent model selection** - Automatically recommends valuation approach based on company characteristics
- **FCFE calculation** - Computes Free Cash Flow to Equity when capital expenditure data is available
- **Growth pattern analysis** - Determines appropriate growth stages (stable, two-stage, three-stage, or n-stage)
- **Results export** - Option to save analysis results to a text file

## Installation

### Prerequisites

- Python 3.6 or higher

### Setup

1. Clone or download the script
1. Make it executable (Linux/Mac):
   
   ```bash
   chmod +x valuation_model_selector.py
   ```
1. Run the script:
   
   ```bash
   python3 valuation_model_selector.py
   ```

## Usage

Simply run the script and follow the interactive prompts:

```bash
python3 valuation_model_selector.py
```

### Input Categories

The tool will ask you about:

1. **Earnings Status** - Whether the company has positive or negative earnings
1. **Growth Expectations** - Expected growth rates and competitive advantages
1. **Financial Leverage** - Current debt ratio and expected changes
1. **Cash Flow Data** - Dividends, capital expenditures, and working capital
1. **Business Characteristics** - Cyclical nature, temporary issues, startup status

### Sample Session

```
============================================================
CHOOSING THE RIGHT VALUATION MODEL
This program helps you choose the right valuation model
============================================================

Are your earnings positive? (Yes/No): Yes
What is the expected inflation rate in the economy? [Default: 3.0%]: 2.5
What is the expected real growth rate in the economy? [Default: 2.0%]: 2.0
What is the expected growth rate in earnings (revenues) for this firm in the near future? [Default: 15.0%]: 12.0
Does this firm have a significant and sustainable advantage over competitors? (Yes/No): Yes
...
```

## Output

The tool provides recommendations for:

- **Model Type** - Discounted Cash Flow Model or Option Pricing Model
- **Earnings Level** - Current earnings or normalized earnings
- **Cash Flow Type** - FCFE, FCFF, or Dividends
- **Growth Pattern** - Stable, two-stage, three-stage, or n-stage growth
- **Growth Period Length** - Duration of high growth period

### Sample Output

```
============================================================
OUTPUT FROM THE MODEL
Based on your inputs, the recommended valuation approach is:
============================================================

Type of Model: Discounted CF Model

Level of Earnings to use: Current Earnings

Cashflows to discount: FCFE (Value equity)

Growth Pattern: Two-stage Growth

Length of Growth Period: 5 to 10 years

Calculated FCFE: $125.00
```

## Technical Details

### Key Calculations

**Free Cash Flow to Equity (FCFE):**

```
FCFE = Net Income - (Capital Spending - Depreciation) × (1 - Debt Ratio) - Working Capital Change × (1 - Debt Ratio)
```

**Growth Pattern Logic:**

- **Stable Growth**: Expected growth ≤ Economy growth + 1%
- **Two-stage Growth**: Economy growth + 1% < Expected growth ≤ Economy growth + 6%
- **Three-stage Growth**: Expected growth > Economy growth + 6%

### Decision Tree Logic

The tool uses a comprehensive decision tree that considers:

- Earnings positivity and sustainability
- Competitive advantages and market position
- Financial leverage and capital structure changes
- Cash flow generation and dividend policy
- Growth sustainability and time horizon

## File Structure

```
valuation_model_selector.py
├── ValuationModelSelector class
│   ├── Input collection methods
│   ├── Calculation logic
│   ├── Output generation
│   └── Helper functions
└── Main execution function
```

## Limitations

- Based on traditional DCF valuation principles
- Requires user judgment for qualitative inputs
- Does not account for market-specific factors
- Results are recommendations, not definitive valuations

## Based on Academic Framework

This tool implements the valuation model selection framework developed by Aswath Damodaran, Professor of Finance at NYU Stern School of Business. The methodology is widely used in corporate finance and investment analysis.

## Contributing

Feel free to submit issues and enhancement requests. The tool can be extended to include:

- Additional valuation models (relative valuation, real options)
- Industry-specific considerations
- Monte Carlo simulation capabilities
- Sensitivity analysis features

## License

This tool is provided as-is for educational and professional use. Please ensure compliance with your organization’s policies when using for commercial purposes.
