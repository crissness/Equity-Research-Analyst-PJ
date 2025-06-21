# Cost of Debt Calculator

A comprehensive Python tool for calculating the after-tax cost of debt using synthetic credit ratings and market data. This calculator implements Professor Aswath Damodaran's synthetic rating methodology to determine corporate borrowing costs for financial analysis and valuation.

## 🏦 Overview

The Cost of Debt Calculator determines a company's borrowing cost by:
1. **Analyzing company fundamentals** from Yahoo Finance
2. **Calculating interest coverage ratios** from financial inputs  
3. **Assigning synthetic credit ratings** using Damodaran's methodology
4. **Determining credit spreads** based on company characteristics
5. **Computing after-tax cost of debt** with tax shield benefits

### Formula Implementation
**Cost of Debt = (Risk-free Rate + Credit Spread) × (1 - Marginal Tax Rate)**

## 🚀 Features

- **Automatic Company Analysis**: Market cap, sector classification, and financial services detection
- **Synthetic Credit Rating**: Implementation of Damodaran's rating methodology based on coverage ratios
- **Multi-Category Rating System**: Separate rating tables for large cap, small cap, and financial services firms
- **Government Bond Integration**: Risk-free rates from comprehensive bond yield database
- **Tax Shield Calculation**: Incorporates tax deductibility of interest payments
- **Professional Excel Export**: Complete analysis with step-by-step calculations
- **Real-time Market Data**: Yahoo Finance integration for current company information

## 📊 Synthetic Rating Methodology

This calculator implements **Professor Aswath Damodaran's synthetic rating approach**, a widely-used method in corporate finance for estimating credit ratings and default spreads. The methodology:

### **Theoretical Foundation**
- **Developer**: Professor Aswath Damodaran (NYU Stern School of Business)
- **Purpose**: Provides market-based credit ratings when official ratings are unavailable
- **Application**: Widely used in academic research and professional valuation practice
- **Data Source**: Based on historical analysis of rating agencies' methodologies

### **Rating Categories**
The system uses three distinct rating frameworks:

#### **1. Large Cap Companies (Market Cap > $5 Billion)**
- Higher coverage ratio thresholds for investment grade ratings
- Reflects rating agencies' preference for large, established companies
- Access to capital markets and diversified operations

#### **2. Small Cap Companies (Market Cap < $5 Billion)**  
- More conservative rating assignments
- Higher credit spreads reflecting increased default risk
- Limited access to capital markets and concentrated operations

#### **3. Financial Services Firms**
- Specialized rating criteria for banks, insurance, and financial institutions
- Different coverage ratio calculations reflecting financial leverage norms
- Regulatory oversight and systemic risk considerations

### **Coverage Ratio Analysis**
- **Interest Coverage Ratio**: EBIT ÷ Interest Expenses
- **Default Case**: Firms with no interest expenses assigned ratio of 20 (AAA equivalent)
- **Rating Assignment**: Coverage ratios mapped to credit ratings and default spreads
- **Market-Based Spreads**: Reflects actual market pricing of credit risk

## 💾 Required Data Files

### 1. Synthetic Ratings.xlsx
Contains Damodaran's synthetic rating tables with three sections:
- **Large Cap Ratings**: Coverage ratios and spreads for companies > $5B market cap
- **Small Cap Ratings**: Coverage ratios and spreads for companies < $5B market cap  
- **Financial Services**: Specialized ratings for financial institutions
- **Structure**: Coverage ratio ranges, credit ratings, and corresponding spreads

### 2. Bond.xlsx
Government bond yield database:
- **Column 1**: Country names
- **Column 3**: "Yield 10y" (10-year government bond yields)
- **Coverage**: Major developed and emerging market yields
- **Usage**: Provides risk-free rate baseline for cost of debt calculation

## 🛠️ Installation

### Prerequisites
```bash
pip install yfinance pandas numpy openpyxl
```

### File Setup
```
your-project-folder/
├── cost_of_debt_calculator.py
├── Synthetic Ratings.xlsx      # Damodaran's synthetic rating tables
└── Bond.xlsx                   # Government bond yields database
```

## 🔧 Usage

### Basic Usage
```bash
python cost_of_debt_calculator.py
```

### Interactive Workflow

#### **Step 1: Company Analysis**
```
Enter company ticker: AAPL
✅ Company: Apple Inc.
✅ Sector: Technology
✅ Market Cap: $3,500,000,000,000
✅ Company Type: large_cap
✅ Financial Services: No
```

#### **Step 2: Financial Inputs**
```
Enter current EBIT (in millions): 123,000
Enter current Interest Expenses (in millions): 3,000
✅ EBIT: $123,000,000,000
✅ Interest Expenses: $3,000,000,000
✅ Interest Coverage Ratio: 41.00
```

#### **Step 3: Synthetic Rating Assignment**
```
Company Type: Large Cap
Interest Coverage Ratio: 41.00
✅ Rating Assignment:
   • Coverage Ratio Range: 8.50 to 100000.00
   • Assigned Rating: Aaa/AAA
   • Credit Spread: 0.45%
```

#### **Step 4: Risk-Free Rate Selection**
```
Available countries in bond database:
AUSTRALIA       (4.25%) | AUSTRIA         (2.96%) | CANADA          (3.84%)
FRANCE          (3.15%) | GERMANY         (2.53%) | ITALY           (3.75%)
JAPAN           (1.41%) | NETHERLANDS     (2.76%) | USA             (4.50%)

Enter country for risk-free rate: USA
✅ Risk-free rate: 4.50% (USA)
```

#### **Step 5: Tax Considerations**
```
Enter marginal tax rate (%, e.g., 25): 21
✅ Marginal tax rate: 21.0%
```

#### **Step 6: Final Calculation**
```
COST OF DEBT CALCULATION
============================================================
FORMULA: Cost of Debt = (Rf + Spread) × (1 - Tax Rate)
============================================================
Risk-free Rate (Rf):     4.50%
Credit Spread:           0.45%
Pre-tax Cost of Debt:    4.95%
Marginal Tax Rate:       21.0%
============================================================
CALCULATION:
Cost of Debt = (4.50% + 0.45%) × (1 - 21.0%)
Cost of Debt = 4.95% × 79.0%
Cost of Debt = 3.91%
============================================================
```

## 📈 Output Features

### Console Output
- **Company Profile**: Automatic classification and financial services detection
- **Rating Analysis**: Coverage ratio calculation and synthetic rating assignment
- **Step-by-step Formula**: Complete CAPM calculation with actual values
- **Tax Shield Analysis**: Shows impact of tax deductibility

### Excel Export
The tool generates a comprehensive Excel file with two sheets:

#### **Cost_of_Debt_Results**
| Metric | Value | Formatted |
|--------|-------|-----------|
| Company_Ticker | AAPL | AAPL |
| Market_Cap | 3500000000000 | $3,500,000,000,000 |
| Interest_Coverage_Ratio | 41.00 | 41.00 |
| Assigned_Rating | Aaa/AAA | Aaa/AAA |
| Credit_Spread | 0.0045 | 0.45% |
| After_Tax_Cost_of_Debt | 0.0391 | 3.91% |

#### **Calculation_Steps**
| Step | Calculation | Description |
|------|-------------|-------------|
| Formula | Cost of Debt = (Rf + Spread) × (1 - Tax Rate) | After-tax Cost of Debt Formula |
| Risk-free Rate | Rf = 4.50% | Government bond yield |
| Credit Spread | Spread = 0.45% | Based on synthetic rating |
| Pre-tax Cost | Pre-tax = 4.95% | Cost before tax benefits |
| Final Result | Cost of Debt = 3.91% | After-tax cost of debt |

## 📋 Methodology

### Interest Coverage Ratio
- **Calculation**: EBIT ÷ Interest Expenses
- **Special Case**: Companies with no interest expenses assigned ratio of 20.0
- **Purpose**: Primary metric for synthetic rating assignment
- **Industry Standard**: Widely used by rating agencies for credit assessment

### Company Classification
- **Large Cap**: Market capitalization > $5 billion
- **Small Cap**: Market capitalization < $5 billion
- **Financial Services**: Auto-detected from Yahoo Finance sector/industry data
- **Rating Tables**: Each category uses different coverage ratio thresholds

### Credit Spread Determination
- **Rating-Based**: Spreads assigned based on synthetic credit rating
- **Market-Calibrated**: Reflects historical default spreads by rating category
- **Risk-Adjusted**: Higher spreads for lower ratings and smaller companies

### Tax Shield Calculation
- **Interest Deductibility**: Reflects tax benefits of debt financing
- **After-tax Cost**: Reduces borrowing cost by (Tax Rate × Pre-tax Cost)
- **Regional Variations**: User inputs appropriate marginal tax rate

## 🎯 Use Cases

### Corporate Finance
- **WACC Calculation**: Essential component for weighted average cost of capital
- **Capital Structure**: Debt vs. equity cost comparison
- **Financing Decisions**: Optimal capital structure analysis
- **Credit Analysis**: Understanding borrowing costs and credit risk

### Valuation Analysis
- **DCF Models**: Cost of debt for enterprise value calculations
- **LBO Analysis**: Debt pricing for leveraged buyout models
- **Credit Risk**: Default probability and recovery rate analysis
- **Comparative Analysis**: Peer group cost of debt benchmarking

### Academic Research
- **Capital Structure**: Empirical studies on financing decisions
- **Credit Risk**: Analysis of synthetic rating accuracy
- **Market Efficiency**: Cost of debt vs. market pricing studies
- **International Finance**: Cross-country borrowing cost analysis

## 📚 Theoretical Background

### Damodaran's Contribution
Professor Aswath Damodaran's synthetic rating methodology addresses key limitations in traditional credit analysis:

#### **Traditional Challenges**
- **Limited Coverage**: Not all companies have official credit ratings
- **Lagging Indicators**: Rating agencies may be slow to adjust ratings
- **Subjective Elements**: Rating process includes qualitative factors
- **Access Barriers**: Small companies often lack formal ratings

#### **Synthetic Rating Benefits**
- **Universal Application**: Can be applied to any company with financial data
- **Real-time Updates**: Based on current financial metrics
- **Objective Methodology**: Quantitative approach reduces subjectivity
- **Market-Based**: Reflects actual default risk pricing

#### **Academic Validation**
- **Empirical Testing**: Extensive validation against actual default rates
- **Industry Adoption**: Widely used in investment banking and consulting
- **Regulatory Acceptance**: Recognized methodology for regulatory capital
- **Continuous Refinement**: Updated based on market developments

## ⚠️ Important Considerations

### Data Requirements
- **Current Financials**: EBIT and interest expense data should be recent
- **Market Data**: Stock price and market cap reflect current valuation
- **Tax Rates**: Use appropriate marginal tax rate for jurisdiction
- **Government Bonds**: Risk-free rate should match currency and geography

### Methodology Limitations
- **Simplified Model**: Focuses primarily on coverage ratios
- **Industry Variations**: Some sectors may require specialized adjustments
- **Market Conditions**: Credit spreads vary with market cycles
- **Currency Risk**: Cross-currency borrowing may involve additional spreads

### Best Practices
- **Regular Updates**: Refresh analysis quarterly or after significant events
- **Peer Comparison**: Validate results against industry benchmarks
- **Sensitivity Analysis**: Test impact of different assumptions
- **Professional Review**: Consider consulting financial professionals for complex cases

## 📊 Example Companies by Category

### Large Cap Examples
- **Technology**: Apple (AAPL), Microsoft (MSFT), Google (GOOGL)
- **Healthcare**: Johnson & Johnson (JNJ), Pfizer (PFE)
- **Energy**: Exxon Mobil (XOM), Chevron (CVX)
- **Industrial**: General Electric (GE), Boeing (BA)

### Small Cap Examples
- **Regional Banks**: Community financial institutions
- **Specialty Retail**: Niche market retailers
- **Biotech**: Small pharmaceutical companies
- **Technology**: Emerging software companies

### Financial Services Examples
- **Banks**: JPMorgan Chase (JPM), Bank of America (BAC)
- **Insurance**: AIG, Prudential Financial
- **Investment Management**: BlackRock (BLK), Goldman Sachs (GS)
- **REITs**: Real Estate Investment Trusts

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request

### Areas for Enhancement
- **Additional Rating Factors**: Incorporate more financial metrics
- **Industry Adjustments**: Sector-specific rating modifications
- **International Markets**: Enhanced support for emerging markets
- **Real-time Data**: Live bond yield feeds

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

### Academic Sources
- **Damodaran, A.** "Investment Valuation: Tools and Techniques for Determining the Value of Any Asset"
- **Damodaran, A.** "Applied Corporate Finance" - Synthetic Rating Methodology
- **NYU Stern**: [Damodaran Online](http://pages.stern.nyu.edu/~adamodar/) - Updated datasets and methodology

### Professional Resources
- **Moody's**: Rating Methodology and Criteria
- **S&P Global**: Credit Rating Definitions and Process
- **Federal Reserve**: Corporate Bond Spreads and Risk Premiums
- **Bloomberg**: Credit Risk and Default Studies

### Data Sources
- **Yahoo Finance**: Market data and company information
- **Government Treasuries**: Official bond yield data
- **Central Banks**: Interest rate and monetary policy data

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to Damodaran's online resources for methodology questions
- Review academic literature for theoretical background

---

**Disclaimer**: This tool implements academic methodology for educational and professional analysis. The synthetic rating approach is based on Professor Damodaran's research and should be used as part of comprehensive financial analysis. Always verify results and consult financial professionals for investment decisions.

**Attribution**: Synthetic rating methodology developed by Professor Aswath Damodaran, NYU Stern School of Business. This implementation is for educational and analytical purposes.
