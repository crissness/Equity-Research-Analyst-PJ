# Cost of Equity Calculator with CAPM Analysis

A comprehensive Python tool for calculating the cost of equity using the Capital Asset Pricing Model (CAPM). This tool provides beta calculation with statistical analysis, automatic equity risk premium detection, and government bond yield integration for complete CAPM-based cost of equity analysis.

## 🚀 Features

- **Beta Calculation**: Regression-based beta using logarithmic returns with comprehensive statistics
- **Government Bond Integration**: Excel-based 10-year government bond yields database
- **Automatic ERP Detection**: Country-specific equity risk premiums from comprehensive database
- **Complete CAPM Analysis**: Full cost of equity calculation with step-by-step formula breakdown
- **Multiple Markets**: Support for stocks and indices from major global markets
- **Professional Excel Export**: Comprehensive results with formula documentation
- **Visual Analysis**: Beta regression plots with statistical indicators

## 📊 CAPM Formula Implementation

The tool implements the Capital Asset Pricing Model:

**Cost of Equity (Re) = Risk-free Rate (Rf) + Beta (β) × Equity Risk Premium (ERP)**

### Example Calculation Output:
```
COST OF EQUITY (CAPM):
==================================================
FORMULA: Re = Rf + β × ERP
==================================================
Risk-free rate (Rf): 4.50%
Beta (β): 1.2543
Equity Risk Premium (ERP): 4.33% (United States)
==================================================
CALCULATION:
Re = 4.50% + 1.2543 × 4.33%
Re = 4.50% + 5.43%
Re = 9.93%
==================================================
```

## 📊 Supported Markets

### Stock Exchanges
- **USA**: NASDAQ, NYSE (e.g., AAPL, MSFT, GOOGL)
- **Italy**: Borsa Italiana (e.g., ENI.MI, ENEL.MI, ISP.MI)
- **Germany**: XETRA (e.g., SAP.DE, VOW3.DE)
- **Netherlands**: Euronext Amsterdam (e.g., ASML.AS)
- **Switzerland**: SIX Swiss Exchange (e.g., NESN.SW)
- **And many more via Yahoo Finance**

### Market Indices
- **S&P 500**: ^GSPC
- **NASDAQ**: ^IXIC
- **FTSE MIB (Italy)**: FTSEMIB.MI
- **DAX (Germany)**: ^GDAXI
- **CAC 40 (France)**: ^FCHI
- **FTSE 100 (UK)**: ^FTSE
- **Nikkei (Japan)**: ^N225

## 💾 Required Excel Databases

### 1. Bond.xlsx
Contains 10-year government bond yields by country:
- **Column 1**: Country names
- **Column 3**: "Yield 10y" (10-year government bond yields)
- **Format**: Supports both decimal (0.0450) and percentage (4.50%) formats
- **Coverage**: Major developed and emerging markets

### 2. ERP 2025.xlsx  
Contains equity risk premiums by country:
- **Column 1**: Country names
- **Column 4**: "Total Equity Risk Premium"
- **Source**: Professional ERP database with country-specific risk assessments
- **Updates**: Annual updates recommended for accuracy

## 🛠️ Installation

### Prerequisites
```bash
pip install yfinance pandas numpy matplotlib scipy openpyxl
```

### Required Files Setup
```
your-project-folder/
├── CostofEquity.py          # Main Python script
├── Bond.xlsx                # Government bond yields database
└── ERP 2025.xlsx           # Equity risk premium database
```

## 🔧 Usage

### Basic Usage
```bash
python CostofEquity.py
```

### Interactive Workflow
The program guides you through a structured 4-step process:

#### **Step 1: Beta Calculation**
```
Enter stock ticker: AAPL
Enter index ticker: ^GSPC
Period: 10y (default)
Interval: 1mo (default)
```

#### **Step 2: Risk-Free Rate Selection**
```
Available countries in bond database:
AUSTRALIA       (4.25%) | AUSTRIA         (2.96%) | BELGIUM         (3.12%)
CANADA          (3.84%) | FRANCE          (3.15%) | GERMANY         (2.53%)
ITALY           (3.75%) | JAPAN           (1.41%) | USA             (4.50%)
...

Enter country name for 10Y bond yield: USA
✅ Risk-free rate set to: 4.50%
```

#### **Step 3: Equity Risk Premium**
```
✅ Country detected: United States
✅ Equity Risk Premium: 4.33%
Use automatic ERP (4.33%)? (y/n): y
```

#### **Step 4: Cost of Equity Calculation**
```
COST OF EQUITY (CAPM):
==================================================
FORMULA: Re = Rf + β × ERP
==================================================
Final Result: 9.93%
```

## 📈 Output Features

### Console Output
- **Beta Statistics**: Comprehensive regression analysis
- **Step-by-step CAPM**: Formula with actual values substituted
- **Quality Indicators**: R-squared, p-values, sample size
- **Professional Summary**: Ready-to-use cost of equity rate

### Excel Export
The tool generates a comprehensive Excel file with multiple sheets:

#### **1. CAPM_Results Sheet**
| Metric | Value | Percentage |
|--------|-------|------------|
| CAPM_Formula | Re = Rf + β × ERP | |
| Risk_Free_Rate_Rf | 0.0450 | 4.50% |
| Beta | 1.2543 | 1.2543 |
| Equity_Risk_Premium_ERP | 0.0433 | 4.33% |
| Cost_of_Equity_Re | 0.0993 | 9.93% |

#### **2. CAPM_Calculation Sheet**
| Step | Calculation | Description |
|------|-------------|-------------|
| Formula | Re = Rf + β × ERP | CAPM Cost of Equity Formula |
| Substitution | Re = 4.50% + 1.2543 × 4.33% | Substituting actual values |
| Risk Premium Component | Re = 4.50% + 5.43% | Beta × ERP calculation |
| Final Result | Re = 9.93% | Final Cost of Equity |

#### **3. Additional Sheets**
- **Stock_Data**: Historical stock price data
- **Index_Data**: Historical market index data  
- **Returns_Data**: Calculated logarithmic returns

## 📋 Methodology

### Beta Calculation
- **Returns**: Logarithmic returns for better statistical properties
- **Regression**: Ordinary Least Squares (OLS) regression
- **Period**: 10-year default for long-term perspective
- **Frequency**: Monthly data to reduce noise
- **Formula**: β = Cov(Rs, Rm) / Var(Rm)

### Statistical Output
- **Beta coefficient**: Systematic risk measure
- **R-squared**: Model explanatory power
- **T-statistic**: Statistical significance
- **P-value**: Probability testing
- **Standard error**: Estimation precision
- **Correlation**: Linear relationship strength

### Risk-Free Rate
- **Source**: 10-year government bond yields
- **Database**: Excel-based for reliability and control
- **Coverage**: Major developed and emerging markets
- **Currency**: Local currency yields for each country

### Equity Risk Premium  
- **Country-specific**: Tailored to local market conditions
- **Professional grade**: Based on comprehensive risk assessment
- **Automatic detection**: Index symbol → Country → ERP lookup
- **Manual override**: User can input custom ERP if needed

## 🎯 Use Cases

### DCF Valuation
- **Primary use**: Cost of equity for discount rate in DCF models
- **Integration ready**: Output directly usable in valuation spreadsheets
- **Documentation**: Complete audit trail for valuation reports
- **Professional standard**: Follows industry best practices

### Portfolio Management
- **Risk assessment**: Systematic risk analysis for portfolio construction
- **Benchmark comparison**: Beta vs. market indices
- **Performance attribution**: Risk-adjusted return analysis

### Academic Research
- **Empirical finance**: Beta estimation and CAPM testing
- **Market analysis**: Cross-country risk premium studies
- **Methodology validation**: Statistical robustness testing

### Investment Analysis
- **Equity valuation**: Cost of equity for investment decisions
- **Risk management**: Systematic risk measurement
- **Due diligence**: Professional-grade financial analysis

## 📚 CAPM Components Guide

### Risk-Free Rate Selection
Choose the government bond yield that matches your analysis:
- **Same country**: Use the country where the company operates
- **Same currency**: Match the currency of cash flows
- **10-year maturity**: Standard practice for long-term valuations
- **Current rates**: Use recent yields for accuracy

**Typical Current Ranges:**
- **USA**: 4.0% - 5.0%
- **Germany**: 2.0% - 3.0%
- **Italy**: 3.5% - 4.5%
- **Japan**: 0.5% - 1.5%
- **Emerging markets**: 5.0% - 15.0%

### Equity Risk Premium Guidance
ERP represents the additional return investors require for equity risk:

**By Region:**
- **USA/Western Europe**: 4.0% - 6.0%
- **Eastern Europe**: 6.0% - 9.0%
- **Emerging Asia**: 6.0% - 10.0%
- **Latin America**: 7.0% - 12.0%
- **Africa/Middle East**: 8.0% - 15.0%

**Research Sources:**
- **Academic**: Damodaran (NYU), Ibbotson Associates
- **Professional**: PwC, EY, KPMG valuation surveys
- **Market data**: Central bank reports, investment bank research

## ⚠️ Important Notes

### Data Quality
- **Excel databases**: Ensure Bond.xlsx and ERP 2025.xlsx are in the same directory
- **File format**: Use proper Excel format (.xlsx), not CSV
- **Data updates**: Update ERP database annually for accuracy
- **Bond yields**: Verify current market rates periodically

### Statistical Considerations
- **Sample size**: Minimum 30 observations recommended (2.5 years monthly data)
- **R-squared interpretation**: 
  - **> 0.50**: Reliable beta for DCF use
  - **0.30-0.50**: Use with caution, consider industry beta
  - **< 0.30**: Consider alternative beta sources
- **Statistical significance**: P-value < 0.05 indicates reliable beta

### CAPM Limitations
- **Assumptions**: Single-factor model, may not capture all risk
- **Market efficiency**: Assumes efficient markets
- **Beta stability**: Beta can change over time
- **Alternative models**: Consider Fama-French or other multi-factor models for comprehensive analysis

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **CAPM Theory**: [Corporate Finance Institute](https://corporatefinanceinstitute.com/resources/valuation/capm-capital-asset-pricing-model/)
- **Beta Analysis**: [Investopedia - Beta](https://www.investopedia.com/terms/b/beta.asp)
- **DCF Valuation**: [McKinsey Valuation Guide](https://www.mckinsey.com/business-functions/strategy-and-corporate-finance/our-insights/valuation)
- **Yahoo Finance API**: [yfinance documentation](https://pypi.org/project/yfinance/)

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to the methodology section for calculation details

---

**Disclaimer**: This tool is for educational and professional analysis purposes. Always verify calculations and consult financial professionals for investment decisions. CAPM results should be used as part of a comprehensive valuation analysis.
