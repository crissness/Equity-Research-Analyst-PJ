# Beta Calculator with Risk-Free Rate

A comprehensive Python tool for calculating stock beta coefficients with confidence intervals and government bond yields for DCF valuation and CAPM cost of equity analysis.

## 🚀 Features

- **Beta Calculation**: Regression-based beta using logarithmic returns
- **Confidence Intervals**: Statistical confidence intervals for beta estimates (90%, 95%, 99%)
- **Government Bond Yields**: Automatic download of 10-year government bond yields as risk-free rate
- **CAPM Cost of Equity**: Complete cost of equity calculation using CAPM formula
- **Multiple Markets**: Support for stocks and indices from major global markets
- **Data Export**: Export results to Excel with comprehensive data sheets
- **Visualization**: Regression plots with confidence bands

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

### Government Bonds (10-Year)
- **USA**: US Treasury (^TNX)
- **Germany**: German Bund (^TNX-DE)
- **Italy**: Italian BTP (ITALY10Y-EUR)
- **France**: French OAT (FRANCE10Y-EUR)
- **Spain**: Spanish Bond (SPAIN10Y-EUR)
- **UK**: UK Gilt (^TNXUK)
- **Japan**: Japanese Bond (^TNX-JP)
- **Canada**: Canadian Bond (^TNX-CA)
- **Australia**: Australian Bond (^TNX-AU)

## 🛠️ Installation

### Prerequisites
```bash
pip install yfinance pandas numpy matplotlib scipy openpyxl
```

### Clone Repository
```bash
git clone https://github.com/yourusername/beta-calculator.git
cd beta-calculator
```

## 🔧 Usage

### Basic Usage
```bash
python beta_calculator.py
```

### Interactive Prompts
The program will guide you through:

1. **Stock Selection**: Enter ticker symbol (e.g., AAPL, ENI.MI)
2. **Market Index**: Choose benchmark index (e.g., ^GSPC, FTSEMIB.MI)
3. **Risk-Free Rate**: Select country for 10-year bond or enter manually
4. **Time Period**: Choose analysis period (recommended: 2y)
5. **Data Interval**: Select frequency (recommended: 1d)
6. **Confidence Intervals**: Optional statistical confidence bands
7. **Market Risk Premium**: Required for cost of equity calculation

### Example Session
```
Stock ticker: AAPL
Index ticker: ^GSPC
Country for 10Y bond: USA
Period: 2y
Interval: 1d
Calculate confidence intervals? y
Confidence level: 0.95
Market risk premium (%): 6.5
```

## 📈 Output

### Console Output
```
BETA RESULTS:
Beta: 1.2543
Alpha: 0.002147
R-squared: 0.7431
Correlation: 0.8620
Observations: 504

BETA WITH CONFIDENCE INTERVALS:
Beta: 1.2543
95% CI: [1.1205, 1.3881]
Standard Error: 0.067321
T-statistic: 18.634
P-value: 0.000000
R-squared: 0.7431

COST OF EQUITY (CAPM):
Risk-free rate: 4.25%
Beta: 1.2543
Market risk premium: 6.50%
Cost of Equity: 12.40%
95% CI: [11.53%, 13.27%]
```

### Excel Export
- **Results Sheet**: All calculated metrics and statistics
- **Stock Data**: Historical stock price data
- **Index Data**: Historical index data
- **Bond Data**: Government bond yield data (if available)
- **Returns**: Calculated logarithmic returns

## 📋 Methodology

### Beta Calculation
- Uses **logarithmic returns** for better statistical properties
- **Linear regression**: Stock returns vs. Market returns
- **Formula**: β = Cov(Rs, Rm) / Var(Rm)

### Confidence Intervals
- **T-distribution** based confidence intervals
- **Formula**: β ± t-critical × Standard Error
- Degrees of freedom: n - 2

### CAPM Cost of Equity
- **Formula**: Cost of Equity = Rf + β × (Rm - Rf)
- **Rf**: Risk-free rate (10-year government bond)
- **β**: Calculated beta coefficient
- **Market Risk Premium**: User-provided based on research

## 🎯 Use Cases

### DCF Valuation
- Calculate cost of equity for discount rate
- Assess systematic risk of investment
- Validate beta reliability with confidence intervals

### Portfolio Management
- Risk assessment and portfolio construction
- Benchmark comparison and tracking error analysis
- Performance attribution

### Academic Research
- Empirical finance studies
- Risk factor analysis
- Market efficiency testing

## 📚 Market Risk Premium Guidance

### Typical Ranges by Region
- **USA**: 5.0% - 7.0%
- **Europe (Developed)**: 5.0% - 8.0%
- **Emerging Markets**: 7.0% - 12.0%
- **Conservative Estimate**: 6.0%
- **Historical Long-term Average**: 6.5%

### Sources for Research
- **Academic Studies**: Damodaran, Ibbotson, Morningstar
- **Investment Banks**: Goldman Sachs, Morgan Stanley equity risk premium surveys
- **Central Banks**: Federal Reserve, ECB financial stability reports
- **Professional Services**: PwC, EY valuation surveys

## ⚠️ Important Notes

### Data Quality
- Data sourced from Yahoo Finance
- Some international bonds may have limited availability
- Manual input option available for all parameters

### Statistical Considerations
- Minimum 30 observations recommended for reliable beta
- R-squared indicates explanatory power of market relationship
- Confidence intervals help assess beta reliability

### DCF Usage Recommendations
- **R² > 0.5**: Reliable beta for DCF
- **R² 0.3-0.5**: Use with caution, consider industry beta
- **R² < 0.3**: Consider alternative beta sources

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Yahoo Finance API**: [yfinance documentation](https://pypi.org/project/yfinance/)
- **Beta in Finance**: [Investopedia - Beta](https://www.investopedia.com/terms/b/beta.asp)
- **CAPM Model**: [Corporate Finance Institute](https://corporatefinanceinstitute.com/resources/valuation/capm-capital-asset-pricing-model/)

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to the methodology section for calculation details

---

**Disclaimer**: This tool is for educational and research purposes. Always verify calculations and consult financial professionals for investment decisions.
