import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class BetaCalculator:
    """
    Streamlined class to calculate regression beta and download government bond yields.
    Focuses on calculations without interpretations.
    """

    def __init__(self):
        self.stock_data = None
        self.index_data = None
        self.stock_returns = None
        self.index_returns = None
        self.beta = None
        self.alpha = None
        self.r_squared = None
        self.correlation = None
        # Confidence interval attributes
        self.beta_std_err = None
        self.beta_confidence_interval = None
        self.confidence_level = None
        self.t_statistic = None
        self.p_value = None
        self.degrees_of_freedom = None
        # Government bond attributes
        self.bond_data = None
        self.current_risk_free_rate = None
        self.avg_risk_free_rate = None
        self.bond_symbol = None

    def download_data(self, stock_symbol, index_symbol, period="1y", interval="1d",
                      start_date=None, end_date=None):
        """Downloads historical data for stock and index."""
        print(f"Downloading data for {stock_symbol} and {index_symbol}...")

        try:
            if start_date and end_date:
                self.stock_data = yf.download(stock_symbol, start=start_date, end=end_date,
                                              interval=interval, progress=False)
                self.index_data = yf.download(index_symbol, start=start_date, end=end_date,
                                              interval=interval, progress=False)
            else:
                self.stock_data = yf.download(stock_symbol, period=period,
                                              interval=interval, progress=False)
                self.index_data = yf.download(index_symbol, period=period,
                                              interval=interval, progress=False)

            if self.stock_data.empty or self.index_data.empty:
                raise ValueError("Unable to download data. Check the entered symbols.")

            print(f"Data downloaded successfully!")
            print(
                f"Period: {self.stock_data.index[0].strftime('%Y-%m-%d')} - {self.stock_data.index[-1].strftime('%Y-%m-%d')}")
            print(f"Number of observations: {len(self.stock_data)}")
            return True

        except Exception as e:
            print(f"Error downloading data: {e}")
            return False

    def download_government_bond_yield(self, country="USA", period="1y", interval="1d"):
        """Downloads 10-year government bond yield data for risk-free rate."""

        bond_symbols = {
            "USA": "^TNX", "US": "^TNX", "UNITED STATES": "^TNX",
            "GERMANY": "^TNX-DE", "DE": "^TNX-DE", "DEUTSCHLAND": "^TNX-DE",
            "ITALY": "ITALY10Y-EUR", "IT": "ITALY10Y-EUR", "ITALIA": "ITALY10Y-EUR",
            "FRANCE": "FRANCE10Y-EUR", "FR": "FRANCE10Y-EUR",
            "SPAIN": "SPAIN10Y-EUR", "ES": "SPAIN10Y-EUR", "ESPAÑA": "SPAIN10Y-EUR",
            "UK": "^TNXUK", "UNITED KINGDOM": "^TNXUK", "BRITAIN": "^TNXUK",
            "JAPAN": "^TNX-JP", "JP": "^TNX-JP",
            "CANADA": "^TNX-CA", "CA": "^TNX-CA",
            "AUSTRALIA": "^TNX-AU", "AU": "^TNX-AU"
        }

        alternative_symbols = {
            "GERMANY": ["DE10Y-EUR", "^TNX-DE", "GERMANY10Y-EUR"],
            "ITALY": ["ITALY10Y-EUR", "IT10Y-EUR"],
            "FRANCE": ["FRANCE10Y-EUR", "FR10Y-EUR"],
            "SPAIN": ["SPAIN10Y-EUR", "ES10Y-EUR"],
            "UK": ["^TNXUK", "UK10Y-GBP", "^TNX-GB"],
            "JAPAN": ["^TNX-JP", "JP10Y-JPY"],
            "CANADA": ["^TNX-CA", "CA10Y-CAD"]
        }

        country_upper = country.upper()

        if country_upper not in bond_symbols:
            print(f"❌ Country '{country}' not supported.")
            return False

        primary_symbol = bond_symbols[country_upper]
        symbols_to_try = [primary_symbol]

        if country_upper in alternative_symbols:
            symbols_to_try.extend(alternative_symbols[country_upper])

        print(f"Downloading bond yield for {country}...")

        for symbol in symbols_to_try:
            try:
                print(f"Trying symbol: {symbol}...")

                if hasattr(self, 'stock_data') and self.stock_data is not None:
                    start_date = self.stock_data.index[0].strftime('%Y-%m-%d')
                    end_date = self.stock_data.index[-1].strftime('%Y-%m-%d')
                    self.bond_data = yf.download(symbol, start=start_date, end=end_date,
                                                 interval=interval, progress=False)
                else:
                    self.bond_data = yf.download(symbol, period=period,
                                                 interval=interval, progress=False)

                if not self.bond_data.empty:
                    self.bond_symbol = symbol
                    break

            except Exception as e:
                print(f"Failed with {symbol}: {e}")
                continue

        if self.bond_data is None or self.bond_data.empty:
            print(f"❌ Could not download bond data for {country}")
            return False

        try:
            # Extract yield data
            if hasattr(self.bond_data.columns, 'levels'):
                close_col = None
                for col in self.bond_data.columns:
                    if 'Close' in col[0]:
                        close_col = col
                        break
                if close_col is None:
                    close_col = self.bond_data.columns[0]
                bond_yields = self.bond_data[close_col]
            else:
                if 'Close' in self.bond_data.columns:
                    bond_yields = self.bond_data['Close']
                elif 'Adj Close' in self.bond_data.columns:
                    bond_yields = self.bond_data['Adj Close']
                else:
                    bond_yields = self.bond_data.iloc[:, 0]

            bond_yields = bond_yields.dropna()

            if len(bond_yields) == 0:
                raise ValueError("No valid yield data found")

            # Convert to decimal rates
            self.current_risk_free_rate = bond_yields.iloc[-1] / 100
            self.avg_risk_free_rate = bond_yields.mean() / 100

            print(f"✅ Bond yield data downloaded successfully!")
            print(f"Current 10Y yield: {self.current_risk_free_rate * 100:.2f}%")
            print(f"Average 10Y yield: {self.avg_risk_free_rate * 100:.2f}%")

            return True

        except Exception as e:
            print(f"❌ Error processing bond yield data: {e}")
            return False

    def get_manual_risk_free_rate(self):
        """Allows manual input of risk-free rate."""
        print("\n" + "=" * 50)
        print("MANUAL RISK-FREE RATE INPUT")
        print("=" * 50)
        print("Enter the current 10-year government bond yield.")
        print("You can find current rates at:")
        print("• USA: https://www.treasury.gov/")
        print("• Europe: https://www.ecb.europa.eu/")
        print("• Yahoo Finance, Bloomberg, MarketWatch")
        print("• Current typical ranges:")
        print("  - USA: 3.5% - 5.5%")
        print("  - Germany: 1.5% - 3.5%")
        print("  - Italy: 3.5% - 5.5%")
        print("  - Japan: 0.5% - 1.5%")

        while True:
            try:
                rate_input = input("\nEnter 10-year bond yield (in %, e.g., 4.25): ").strip()
                if not rate_input:
                    print("Please enter a valid yield percentage.")
                    continue

                rate_percent = float(rate_input)

                if not 0 <= rate_percent <= 50:
                    print("Invalid yield. Please enter a value between 0% and 50%.")
                    continue

                self.current_risk_free_rate = rate_percent / 100
                self.avg_risk_free_rate = rate_percent / 100
                self.bond_symbol = "MANUAL_INPUT"

                print(f"✅ Risk-free rate set to: {rate_percent:.2f}%")
                return True

            except ValueError:
                print("Invalid input. Please enter a numeric value (e.g., 4.25)")
                continue
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return False

    def calculate_returns(self):
        """Calculates logarithmic returns for stock and index."""
        if self.stock_data is None or self.index_data is None:
            print("Error: data not available. Run download_data() first")
            return False

        try:
            def get_close_price(data, symbol):
                if hasattr(data.columns, 'levels'):
                    level_0_values = data.columns.get_level_values(0)
                    if 'Adj Close' in level_0_values:
                        adj_close_cols = [(i, col) for i, col in enumerate(data.columns)
                                          if col[0] == 'Adj Close' and symbol.upper() in col[1].upper()]
                        if adj_close_cols:
                            return data[adj_close_cols[0][1]]
                    if 'Close' in level_0_values:
                        close_cols = [(i, col) for i, col in enumerate(data.columns)
                                      if col[0] == 'Close' and symbol.upper() in col[1].upper()]
                        if close_cols:
                            return data[close_cols[0][1]]
                elif 'Adj Close' in data.columns:
                    return data['Adj Close']
                elif 'Close' in data.columns:
                    return data['Close']
                raise KeyError(f"No 'Close' column found for {symbol}")

            stock_symbol = self.stock_data.columns[0][1] if hasattr(self.stock_data.columns, 'levels') else 'STOCK'
            index_symbol = self.index_data.columns[0][1] if hasattr(self.index_data.columns, 'levels') else 'INDEX'

            stock_close = get_close_price(self.stock_data, stock_symbol)
            index_close = get_close_price(self.index_data, index_symbol)

            if not isinstance(stock_close, pd.Series):
                stock_close = pd.Series(stock_close.values.flatten(), index=stock_close.index)
            if not isinstance(index_close, pd.Series):
                index_close = pd.Series(index_close.values.flatten(), index=index_close.index)

            # Calculate log returns
            self.stock_returns = np.log(stock_close / stock_close.shift(1)).dropna()
            self.index_returns = np.log(index_close / index_close.shift(1)).dropna()

            # Align dates
            common_dates = self.stock_returns.index.intersection(self.index_returns.index)
            self.stock_returns = self.stock_returns[common_dates]
            self.index_returns = self.index_returns[common_dates]

            print(f"Returns calculated for {len(self.stock_returns)} observations")
            return True

        except Exception as e:
            print(f"Error calculating returns: {e}")
            return False

    def calculate_beta(self):
        """Calculates basic beta using linear regression."""
        if self.stock_returns is None or self.index_returns is None:
            print("Error: returns not available. Run calculate_returns() first")
            return False

        combined_data = pd.DataFrame({
            'stock': self.stock_returns,
            'index': self.index_returns
        }).dropna()

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            combined_data['index'], combined_data['stock']
        )

        self.beta = slope
        self.alpha = intercept
        self.r_squared = r_value ** 2
        self.correlation = combined_data['stock'].corr(combined_data['index'])

        print(f"\nBETA RESULTS:")
        print(f"Beta: {self.beta:.4f}")
        print(f"Alpha: {self.alpha:.6f}")
        print(f"R-squared: {self.r_squared:.4f}")
        print(f"Correlation: {self.correlation:.4f}")
        print(f"Observations: {len(combined_data)}")

        return True

    def calculate_beta_with_confidence(self, confidence_level=0.95):
        """Calculates beta with confidence intervals."""
        if self.stock_returns is None or self.index_returns is None:
            return False

        combined_data = pd.DataFrame({
            'stock': self.stock_returns,
            'index': self.index_returns
        }).dropna()

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            combined_data['index'], combined_data['stock']
        )

        # Calculate confidence interval
        n = len(combined_data)
        degrees_of_freedom = n - 2
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)

        margin_of_error = t_critical * std_err
        beta_lower = slope - margin_of_error
        beta_upper = slope + margin_of_error

        # Store results
        self.beta = slope
        self.alpha = intercept
        self.r_squared = r_value ** 2
        self.correlation = combined_data['stock'].corr(combined_data['index'])
        self.beta_std_err = std_err
        self.beta_confidence_interval = (beta_lower, beta_upper)
        self.confidence_level = confidence_level
        self.t_statistic = slope / std_err
        self.p_value = p_value
        self.degrees_of_freedom = degrees_of_freedom

        print(f"\nBETA WITH CONFIDENCE INTERVALS:")
        print(f"Beta: {self.beta:.4f}")
        print(f"{confidence_level * 100:.0f}% CI: [{beta_lower:.4f}, {beta_upper:.4f}]")
        print(f"Standard Error: {std_err:.6f}")
        print(f"T-statistic: {slope / std_err:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"R-squared: {self.r_squared:.4f}")

        return True

    def calculate_cost_of_equity(self, market_risk_premium=None):
        """Calculates cost of equity using CAPM."""
        if self.beta is None or self.current_risk_free_rate is None:
            print("Error: Beta and risk-free rate required")
            return None

        if market_risk_premium is None:
            if hasattr(self, 'index_returns') and self.index_returns is not None:
                market_return = self.index_returns.mean() * 252
                market_risk_premium = market_return - self.avg_risk_free_rate
                print(f"Estimated market risk premium: {market_risk_premium * 100:.2f}%")
            else:
                print("Market risk premium not provided and cannot be estimated")
                return None

        cost_of_equity = self.current_risk_free_rate + self.beta * market_risk_premium

        print(f"\nCOST OF EQUITY (CAPM):")
        print(f"Risk-free rate: {self.current_risk_free_rate * 100:.2f}%")
        print(f"Beta: {self.beta:.4f}")
        print(f"Market risk premium: {market_risk_premium * 100:.2f}%")
        print(f"Cost of Equity: {cost_of_equity * 100:.2f}%")

        # Add confidence interval if available
        if self.beta_confidence_interval:
            beta_lower, beta_upper = self.beta_confidence_interval
            cost_equity_lower = self.current_risk_free_rate + beta_lower * market_risk_premium
            cost_equity_upper = self.current_risk_free_rate + beta_upper * market_risk_premium
            print(
                f"{self.confidence_level * 100:.0f}% CI: [{cost_equity_lower * 100:.2f}%, {cost_equity_upper * 100:.2f}%]")

        return cost_of_equity

    def plot_regression(self, figsize=(10, 6)):
        """Creates regression plot with confidence bands if available."""
        if self.beta is None:
            return

        combined_data = pd.DataFrame({
            'index_returns': self.index_returns,
            'stock_returns': self.stock_returns
        }).dropna()

        plt.figure(figsize=figsize)
        plt.scatter(combined_data['index_returns'], combined_data['stock_returns'],
                    alpha=0.6, s=20, color='blue')

        x_line = np.linspace(combined_data['index_returns'].min(),
                             combined_data['index_returns'].max(), 100)
        y_line = self.alpha + self.beta * x_line
        plt.plot(x_line, y_line, 'r-', linewidth=2,
                 label=f'β = {self.beta:.4f}')

        if self.beta_confidence_interval:
            beta_lower, beta_upper = self.beta_confidence_interval
            y_lower = self.alpha + beta_lower * x_line
            y_upper = self.alpha + beta_upper * x_line
            plt.plot(x_line, y_lower, 'r--', alpha=0.7, linewidth=1)
            plt.plot(x_line, y_upper, 'r--', alpha=0.7, linewidth=1)
            plt.fill_between(x_line, y_lower, y_upper, alpha=0.1, color='red')

        plt.xlabel('Index Log Returns')
        plt.ylabel('Stock Log Returns')
        plt.title(f'Beta Regression (R² = {self.r_squared:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x * 100:.1f}%'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x * 100:.1f}%'))
        plt.tight_layout()
        plt.show()

    def get_summary_stats(self):
        """Returns summary statistics."""
        if self.stock_returns is None or self.index_returns is None:
            return None

        summary = {
            'Beta': self.beta,
            'Alpha': self.alpha,
            'R_squared': self.r_squared,
            'Correlation': self.correlation,
            'Stock_Volatility_Annual': self.stock_returns.std() * np.sqrt(252),
            'Index_Volatility_Annual': self.index_returns.std() * np.sqrt(252),
            'Stock_Return_Annual': self.stock_returns.mean() * 252,
            'Index_Return_Annual': self.index_returns.mean() * 252,
        }

        if self.beta_confidence_interval:
            beta_lower, beta_upper = self.beta_confidence_interval
            summary.update({
                'Beta_Lower_CI': beta_lower,
                'Beta_Upper_CI': beta_upper,
                'Beta_CI_Width': beta_upper - beta_lower,
                'Beta_Std_Error': self.beta_std_err,
                'T_Statistic': self.t_statistic,
                'P_Value': self.p_value,
                'Confidence_Level': self.confidence_level
            })

        if self.current_risk_free_rate:
            summary.update({
                'Current_Risk_Free_Rate': self.current_risk_free_rate,
                'Average_Risk_Free_Rate': self.avg_risk_free_rate,
                'Bond_Symbol': self.bond_symbol
            })

        return summary

    def save_to_excel(self, filename=None, stock_symbol="STOCK", index_symbol="INDEX"):
        """Saves all data to Excel file."""
        if self.beta is None:
            return None

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"beta_analysis_{stock_symbol}_{timestamp}.xlsx"

        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Results sheet
                results_data = {
                    'Metric': ['Beta', 'Alpha', 'R_squared', 'Correlation', 'Observations'],
                    'Value': [self.beta, self.alpha, self.r_squared, self.correlation, len(self.stock_returns)]
                }

                if self.beta_confidence_interval:
                    beta_lower, beta_upper = self.beta_confidence_interval
                    results_data['Metric'].extend(
                        ['Beta_Lower_CI', 'Beta_Upper_CI', 'CI_Width', 'Std_Error', 'P_Value'])
                    results_data['Value'].extend(
                        [beta_lower, beta_upper, beta_upper - beta_lower, self.beta_std_err, self.p_value])

                if self.current_risk_free_rate:
                    results_data['Metric'].extend(['Risk_Free_Rate', 'Bond_Symbol'])
                    results_data['Value'].extend([self.current_risk_free_rate, self.bond_symbol])

                pd.DataFrame(results_data).to_excel(writer, sheet_name='Results', index=False)

                # Data sheets
                if self.stock_data is not None:
                    self.stock_data.to_excel(writer, sheet_name='Stock_Data')
                if self.index_data is not None:
                    self.index_data.to_excel(writer, sheet_name='Index_Data')
                if self.bond_data is not None:
                    self.bond_data.to_excel(writer, sheet_name='Bond_Data')

                # Returns
                returns_df = pd.DataFrame({
                    'Date': self.stock_returns.index,
                    'Stock_Returns': self.stock_returns.values,
                    'Index_Returns': self.index_returns.values
                })
                returns_df.to_excel(writer, sheet_name='Returns', index=False)

            print(f"✅ Data saved to: {filename}")
            return filename

        except Exception as e:
            print(f"❌ Error saving: {e}")
            return None


def get_user_input():
    """Collects user input for analysis."""
    print("=" * 70)
    print("BETA CALCULATOR WITH RISK-FREE RATE")
    print("=" * 70)

    # Stock ticker input with examples
    print("\nStock ticker examples:")
    print("• Italian stocks: ENI.MI, ENEL.MI, ISP.MI, UCG.MI, TIT.MI")
    print("• US stocks: AAPL, MSFT, GOOGL, TSLA, NVDA")
    print("• European stocks: ASML.AS, SAP.DE, NESN.SW")
    stock_symbol = input("\nEnter stock ticker: ").upper().strip()

    # Index ticker input with examples
    print("\nIndex ticker examples:")
    print("• FTSE MIB (Italy): FTSEMIB.MI")
    print("• S&P 500 (USA): ^GSPC")
    print("• NASDAQ: ^IXIC")
    print("• DAX (Germany): ^GDAXI")
    print("• CAC 40 (France): ^FCHI")
    print("• FTSE 100 (UK): ^FTSE")
    print("• Nikkei (Japan): ^N225")
    index_symbol = input("\nEnter index ticker: ").upper().strip()

    # Government bond selection with examples
    print("\n" + "=" * 50)
    print("RISK-FREE RATE (10-YEAR GOVERNMENT BOND)")
    print("=" * 50)
    print("Available countries:")
    print("• USA: US Treasury (^TNX)")
    print("• Germany: German Bund (^TNX-DE)")
    print("• Italy: Italian BTP (ITALY10Y-EUR)")
    print("• France: French OAT (FRANCE10Y-EUR)")
    print("• Spain: Spanish Bond (SPAIN10Y-EUR)")
    print("• UK: UK Gilt (^TNXUK)")
    print("• Japan: Japanese Bond (^TNX-JP)")
    print("• Canada: Canadian Bond (^TNX-CA)")
    print("• Australia: Australian Bond (^TNX-AU)")
    print("• Manual: Enter rate manually")
    bond_country = input("\nEnter country (e.g., USA, Germany, Italy) or 'manual': ").strip()

    # Period selection with examples
    print("\nAvailable periods:")
    print("• Short term: 1mo, 3mo, 6mo")
    print("• Medium term: 1y, 2y (recommended)")
    print("• Long term: 5y, 10y, max")
    print("• Recent: 1d, 5d, 1mo")
    period = input("\nEnter period (default: 2y): ").strip() or "2y"

    # Interval selection with examples
    print("\nAvailable intervals:")
    print("• Daily: 1d (recommended)")
    print("• Weekly: 1wk")
    print("• Monthly: 1mo")
    print("• Intraday: 1h, 30m, 15m, 5m, 1m")
    interval = input("\nEnter interval (default: 1d): ").strip() or "1d"

    # Custom dates option
    print("\nCustom date range:")
    use_custom_dates = input("Use specific start/end dates? (y/n, default: n): ").strip().lower()
    start_date, end_date = None, None

    if use_custom_dates in ['y', 'yes']:
        print("Date format: YYYY-MM-DD")
        print("Examples: 2022-01-01, 2023-12-31")
        start_date = input("Start date: ").strip()
        end_date = input("End date: ").strip()
        if start_date and end_date:
            period = None  # Don't use period if using custom dates

    return stock_symbol, index_symbol, bond_country, period, interval, start_date, end_date


if __name__ == "__main__":
    try:
        stock_symbol, index_symbol, bond_country, period, interval, start_date, end_date = get_user_input()
        calc = BetaCalculator()

        print(f"\n{'=' * 60}")
        print("DOWNLOADING DATA...")
        print(f"{'=' * 60}")

        # Download stock and index data
        success = calc.download_data(
            stock_symbol=stock_symbol,
            index_symbol=index_symbol,
            period=period,
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )

        if success:
            # Download bond data
            print(f"\n{'=' * 50}")
            print("DOWNLOADING RISK-FREE RATE...")
            print(f"{'=' * 50}")

            if bond_country.lower() == 'manual':
                calc.get_manual_risk_free_rate()
            else:
                bond_success = calc.download_government_bond_yield(
                    country=bond_country,
                    period=period,
                    interval=interval
                )
                if not bond_success:
                    print("\nAutomatic download failed.")
                    retry = input("Try manual input? (y/n, default: y): ").strip().lower()
                    if retry not in ['n', 'no']:
                        calc.get_manual_risk_free_rate()

            # Calculate returns and beta
            if calc.calculate_returns():
                calc.calculate_beta()

                # Ask for confidence intervals
                print(f"\n{'=' * 40}")
                print("CONFIDENCE INTERVAL OPTIONS")
                print(f"{'=' * 40}")
                print("Available confidence levels:")
                print("• 90% (0.90)")
                print("• 95% (0.95) - Standard")
                print("• 99% (0.99)")

                ci_input = input("\nCalculate confidence intervals? (y/n, default: y): ").strip().lower()
                if ci_input not in ['n', 'no']:
                    conf_input = input("Confidence level (default: 0.95): ").strip()
                    try:
                        conf_level = float(conf_input) if conf_input else 0.95
                        if not 0.5 <= conf_level <= 0.999:
                            conf_level = 0.95
                    except ValueError:
                        conf_level = 0.95

                    calc.calculate_beta_with_confidence(confidence_level=conf_level)

                # Calculate cost of equity
                if calc.current_risk_free_rate:
                    print(f"\n{'=' * 40}")
                    print("COST OF EQUITY CALCULATION")
                    print(f"{'=' * 40}")
                    print("Market risk premium is required for CAPM calculation.")
                    print("Typical market risk premiums by region:")
                    print("• USA: 5.0% - 7.0%")
                    print("• Europe (developed): 5.0% - 8.0%")
                    print("• Emerging markets: 7.0% - 12.0%")
                    print("• Conservative estimate: 6.0%")
                    print("• Historical long-term average: 6.5%")

                    while True:
                        try:
                            mrp_str = input("\nEnter market risk premium (%, e.g., 6.5): ").strip()
                            if mrp_str:
                                market_risk_premium = float(mrp_str) / 100
                                if 0 <= market_risk_premium <= 0.25:  # 0% to 25% sanity check
                                    cost_of_equity = calc.calculate_cost_of_equity(market_risk_premium)
                                    break
                                else:
                                    print("Please enter a value between 0% and 25%")
                            else:
                                print("Market risk premium is required for cost of equity calculation")
                        except ValueError:
                            print("Invalid input. Please enter a numeric value (e.g., 6.5)")
                        except KeyboardInterrupt:
                            print("Skipping cost of equity calculation")
                            break

                # Show plot
                print(f"\n{'=' * 30}")
                print("VISUALIZATION")
                print(f"{'=' * 30}")
                plot_input = input("Show regression plot? (y/n, default: y): ").strip().lower()
                if plot_input not in ['n', 'no']:
                    calc.plot_regression()

                # Save results
                print(f"\n{'=' * 30}")
                print("SAVE OPTIONS")
                print(f"{'=' * 30}")
                save_input = input("Save results to Excel? (y/n, default: y): ").strip().lower()
                if save_input not in ['n', 'no']:
                    calc.save_to_excel(stock_symbol=stock_symbol, index_symbol=index_symbol)

                print(f"\n{'=' * 60}")
                print("ANALYSIS COMPLETED SUCCESSFULLY!")
                print(f"{'=' * 60}")

                # Final summary
                if calc.beta_confidence_interval:
                    beta_lower, beta_upper = calc.beta_confidence_interval
                    print(f"✅ Beta: {calc.beta:.4f} [CI: {beta_lower:.4f}, {beta_upper:.4f}]")
                else:
                    print(f"✅ Beta: {calc.beta:.4f}")

                if calc.current_risk_free_rate:
                    print(f"✅ Risk-free rate: {calc.current_risk_free_rate * 100:.2f}%")

                print(f"📊 R²: {calc.r_squared:.4f} | Observations: {len(calc.stock_returns)}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Please check your inputs and try again.")