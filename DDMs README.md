# Dividend Discount Models (DDM) Suite

A comprehensive Python implementation of three essential dividend discount models for equity valuation in corporate finance. This suite provides professional-grade tools for valuing stocks based on expected dividend payments and growth patterns.

## 📋 Overview

This collection includes three sophisticated dividend discount models:

1. **Gordon Growth Model (DDM_ST.py)** - Single-stage stable growth valuation
2. **Two-Stage DDM (DDM_2ST.py)** - High growth followed by stable growth
3. **Three-Stage DDM (DDM_3ST.py)** - High growth, transition, and stable growth phases

Each model features interactive input collection, comprehensive validation, sensitivity analysis, and professional visualizations.

## 🚀 Features

### Core Functionality
- **Interactive User Interface**: Guided input collection with validation and defaults
- **Multiple Growth Rate Estimation Methods**: Historical, analyst forecasts, and fundamental analysis
- **Flexible Cost of Equity Calculation**: Direct input or CAPM-based calculation
- **Comprehensive Sensitivity Analysis**: Test various scenarios and assumptions
- **Professional Visualizations**: Enhanced charts and graphs for analysis
- **Results Export**: Save calculations and analysis to text files
- **Warning System**: Automatic detection of potentially problematic inputs

### Advanced Capabilities
- **Weighted Growth Rate Averaging**: Combine multiple growth estimation methods
- **Dynamic Parameter Adjustment**: Beta and payout ratio changes across phases
- **Gradient Transition Modeling**: Smooth transitions between growth phases
- **Present Value Calculations**: Detailed breakdown of value components
- **Value Decomposition**: Separate analysis of growth value vs. assets in place

## 📦 Requirements

```python
# Required packages
matplotlib >= 3.0.0
numpy >= 1.18.0
scipy >= 1.4.0
```

Install dependencies:
```bash
pip install matplotlib numpy scipy
```

## 🔧 Installation

1. Clone or download the repository
2. Ensure Python 3.6+ is installed
3. Install required dependencies
4. Run any of the model files directly

## 📖 Usage

### Gordon Growth Model (Single-Stage)
```bash
python DDM_ST.py
```

**Best for:**
- Mature, stable companies
- Consistent dividend payout history
- Predictable growth patterns
- Utility companies, REITs, consumer staples

**Key Inputs:**
- Current earnings per share
- Dividend payout ratio
- Cost of equity (direct or CAPM components)
- Expected perpetual growth rate

### Two-Stage DDM
```bash
python DDM_2ST.py
```

**Best for:**
- Companies transitioning from growth to maturity
- Clear inflection point in growth trajectory
- Mid-cap companies with defined growth phases

**Key Inputs:**
- High growth period length and rate
- Stable growth assumptions
- Changing payout ratios between phases
- Different beta for stable period (optional)

### Three-Stage DDM
```bash
python DDM_3ST.py
```

**Best for:**
- Young, high-growth companies
- Complex growth trajectories
- Technology and biotech companies
- Detailed long-term forecasting

**Key Inputs:**
- Three distinct growth phases
- Gradual transition parameters
- Dynamic cost of equity adjustments
- Evolving dividend policies

## 💡 Model Methodology

### Growth Rate Estimation
Each model supports multiple growth rate estimation methods:

1. **Historical Growth**: Based on EPS growth over past 5 years
2. **Analyst Estimates**: External growth predictions
3. **Fundamental Analysis**: ROE × Retention rate approach

Users assign weights to each method for a composite growth estimate.

### Cost of Equity Calculation
Two approaches available:

1. **Direct Input**: User provides cost of equity directly
2. **CAPM Method**: Risk-free rate + (Beta × Market risk premium)

### Valuation Formula

**Gordon Growth Model:**
```
Value = D₁ / (r - g)
```

**Two-Stage Model:**
```
Value = Σ(Dt/(1+r)^t) + [Dn+1/(r-g)]/(1+r)^n
```

**Three-Stage Model:**
```
Value = Σ(High Growth PV) + Σ(Transition PV) + Terminal Value PV
```

Where:
- D = Dividends
- r = Cost of equity
- g = Growth rate
- t = Time period

## 📊 Output Features

### Detailed Results Display
- Complete input summary
- Growth rate breakdown with weights
- Year-by-year dividend projections
- Present value calculations
- Value decomposition analysis

### Sensitivity Analysis
- Growth rate sensitivity (±10% range)
- Growth period sensitivity
- Cost of equity sensitivity
- Interactive parameter testing

### Professional Visualizations
- Sensitivity curves with gradient effects
- Tornado diagrams for impact analysis
- Percentage change comparisons
- Summary statistics boxes

### Warning System
Automatic alerts for:
- Unrealistic growth assumptions
- High beta values for stable firms
- Low payout ratios
- Negative inputs
- Mathematical inconsistencies

## 📁 File Structure

```
DDM_Suite/
├── DDM_ST.py          # Gordon Growth Model
├── DDM_2ST.py         # Two-Stage DDM
├── DDM_3ST.py         # Three-Stage DDM
├── README.md          # This file
└── results/           # Output directory (created automatically)
    ├── gordon_growth_results.txt
    ├── two_stage_ddm_results.txt
    └── three_stage_ddm_results.txt
```

## 🎯 Example Use Cases

### Technology Startup Valuation
Use Three-Stage DDM for a company expected to have:
- 10 years of high growth (25% annually)
- 5 years of declining growth (25% → 8%)
- Stable growth thereafter (8% annually)

### Utility Company Valuation
Use Gordon Growth Model for:
- Stable dividend-paying utility
- Consistent 4-6% growth
- Regulated business environment

### Mid-Cap Growth Company
Use Two-Stage DDM for:
- Current high growth phase (15% for 7 years)
- Transition to mature growth (8% thereafter)

## ⚠️ Important Considerations

### Model Limitations
- **Dividend Dependency**: Only suitable for dividend-paying companies
- **Growth Assumptions**: Sensitive to growth rate estimates
- **Terminal Value**: Large portion of value often in terminal period
- **Market Conditions**: Doesn't account for market sentiment or timing

### Best Practices
1. **Use Multiple Scenarios**: Run sensitivity analysis extensively
2. **Validate Assumptions**: Compare with historical performance
3. **Consider Industry Factors**: Adjust for sector-specific characteristics
4. **Regular Updates**: Refresh assumptions as new information becomes available

### Academic Foundation
These models are based on established financial theory:
- Gordon, M.J. (1959). Dividends, Earnings and Stock Prices
- Damodaran, A. Corporate Finance: Theory and Practice
- Brealey, R.A., Myers, S.C. Principles of Corporate Finance

## 🔍 Model Selection Guide

| Company Characteristics | Recommended Model |
|------------------------|-------------------|
| Mature, stable dividend policy | Gordon Growth Model |
| Clear growth-to-maturity transition | Two-Stage DDM |
| Young, complex growth trajectory | Three-Stage DDM |
| Utility, REIT, consumer staple | Gordon Growth Model |
| Mid-cap with defined phases | Two-Stage DDM |
| Technology, biotech, startup | Three-Stage DDM |

## 📈 Advanced Features

### Dynamic Parameter Modeling
- **Beta Adjustment**: Different betas for different growth phases
- **Payout Evolution**: Gradual changes in dividend policy
- **Cost of Equity Transitions**: Reflecting changing risk profiles

### Comprehensive Validation
- **Input Validation**: Real-time checking of user inputs
- **Mathematical Consistency**: Ensures model stability
- **Economic Reasonableness**: Flags unrealistic assumptions

### Professional Output
- **Formatted Results**: Clean, professional presentation
- **Export Functionality**: Save results for reporting
- **Visualization Suite**: Multiple chart types for analysis

## 🤝 Contributing

This is an educational and professional tool. Suggestions for improvements are welcome:

1. **Algorithm Enhancements**: More sophisticated growth modeling
2. **Additional Features**: Monte Carlo simulation, scenario analysis
3. **UI Improvements**: Enhanced user experience
4. **Documentation**: Additional examples and use cases

## 📄 License

This project is intended for educational and professional use. Please ensure compliance with your institution's or organization's policies regarding financial modeling tools.

## 📞 Support

For questions about implementation or financial theory:
1. Review the comprehensive inline documentation
2. Check the warning messages for guidance
3. Consult corporate finance textbooks for theoretical background
4. Consider professional financial advisory for investment decisions

---

**Disclaimer**: These models are for educational and analytical purposes only. They should not be used as the sole basis for investment decisions. Always consult with qualified financial professionals and consider multiple valuation approaches when making investment decisions.
