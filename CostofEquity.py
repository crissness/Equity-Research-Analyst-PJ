import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import openpyxl

warnings.filterwarnings('ignore')


class BetaCalculator:
    """
    Streamlined class to calculate regression beta and load bond yields from Excel.
    Includes automatic Equity Risk Premium lookup from country data.
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
        self.p_value = None
        self.std_error = None
        self.t_statistic = None
        # Bond yield attributes
        self.bond_data = None
        self.current_risk_free_rate = None
        self.avg_risk_free_rate = None
        self.bond_symbol = None
        self.bond_yield_data = None
        # ERP attributes
        self.erp_data = None
        self.detected_country = None
        self.equity_risk_premium = None

    def download_data(self, stock_symbol, index_symbol, period="10y", interval="1mo",
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

    def load_bond_yield_data(self, bond_file_path="Bond.xlsx"):
        """Loads 10-year government bond yield data from Excel file."""
        try:
            import os
            if not os.path.exists(bond_file_path):
                print(f"❌ Bond file '{bond_file_path}' not found in current directory")
                print(f"Current directory: {os.getcwd()}")
                return False

            print(f"📁 Loading bond yield data from: {bond_file_path}")

            df = pd.read_excel(bond_file_path, sheet_name=0)

            print(f"📊 Excel file loaded. Shape: {df.shape}")

            if 'Country' not in df.columns:
                print("❌ Excel file must have 'Country' column")
                return False

            # Look for yield column
            yield_column = None
            possible_yield_columns = ['Yield 10y', '10Y Yield', 'Yield', '10 Year Yield']

            for col in possible_yield_columns:
                if col in df.columns:
                    yield_column = col
                    break

            if yield_column is None:
                print("❌ Could not find yield column")
                print("Available columns:", list(df.columns))
                return False

            print(f"✅ Using yield column: '{yield_column}'")

            self.bond_yield_data = {}
            valid_entries = 0

            for _, row in df.iterrows():
                if pd.notna(row['Country']) and pd.notna(row[yield_column]):
                    try:
                        country = str(row['Country']).strip()
                        yield_value = float(row[yield_column])

                        # Handle both decimal and percentage formats
                        if yield_value > 1:
                            yield_value = yield_value / 100

                        self.bond_yield_data[country.upper()] = yield_value
                        valid_entries += 1
                    except (ValueError, TypeError):
                        continue

            print(f"✅ Bond yield data loaded for {valid_entries} countries")
            return True

        except Exception as e:
            print(f"❌ Error loading bond yield data: {e}")
            return False

    def load_erp_data(self, erp_file_path="ERP 2025.xlsx"):
        """Loads Equity Risk Premium data from Excel file."""
        try:
            import os
            if not os.path.exists(erp_file_path):
                print(f"❌ ERP file '{erp_file_path}' not found in current directory")
                return False

            print(f"📁 Loading ERP data from: {erp_file_path}")

            df = pd.read_excel(erp_file_path, sheet_name=0)

            if 'Country' not in df.columns:
                print("❌ Excel file must have 'Country' column")
                return False

            # Look for ERP column
            erp_column = None
            possible_erp_columns = ['Total Equity Risk Premium', 'Equity Risk Premium', 'ERP']

            for col in possible_erp_columns:
                if col in df.columns:
                    erp_column = col
                    break

            if erp_column is None:
                print("❌ Could not find ERP column")
                return False

            self.erp_data = {}
            valid_entries = 0

            for _, row in df.iterrows():
                if pd.notna(row['Country']) and pd.notna(row[erp_column]):
                    try:
                        country = str(row['Country']).strip()
                        erp = float(row[erp_column])
                        self.erp_data[country.upper()] = erp
                        valid_entries += 1
                    except (ValueError, TypeError):
                        continue

            print(f"✅ ERP data loaded for {valid_entries} countries")
            return True

        except Exception as e:
            print(f"❌ Error loading ERP data: {e}")
            return False

    def get_bond_yield_from_excel(self, country):
        """Gets 10-year government bond yield from Excel data."""

        if self.bond_yield_data is None:
            return None

        country_upper = country.upper()

        # Try exact match first
        if country_upper in self.bond_yield_data:
            yield_value = self.bond_yield_data[country_upper]
            print(f"✅ Found 10Y bond yield for {country}: {yield_value * 100:.2f}%")
            return yield_value

        # Try alternative country names
        country_alternatives = {
            "USA": ["UNITED STATES", "US", "AMERICA"],
            "UNITED STATES": ["USA", "US", "AMERICA"],
            "UK": ["UNITED KINGDOM", "BRITAIN", "GREAT BRITAIN"],
            "UNITED KINGDOM": ["UK", "BRITAIN", "GREAT BRITAIN"],
        }

        if country_upper in country_alternatives:
            for alt in country_alternatives[country_upper]:
                if alt in self.bond_yield_data:
                    yield_value = self.bond_yield_data[alt]
                    print(f"✅ Found 10Y bond yield for {country} (as {alt}): {yield_value * 100:.2f}%")
                    return yield_value

        print(f"❌ No bond yield data found for '{country}'")
        return None

    def select_bond_country_from_excel(self):
        """Allows user to select a country from the bond Excel data."""

        if self.bond_yield_data is None:
            print("❌ Bond yield data not loaded.")
            return None

        print(f"\n{'=' * 60}")
        print("SELECT 10-YEAR GOVERNMENT BOND")
        print(f"{'=' * 60}")
        print("Available countries in bond database:")

        countries_list = sorted(list(self.bond_yield_data.keys()))

        # Display countries in a nice format
        for i in range(0, len(countries_list), 3):
            row_countries = countries_list[i:i + 3]
            formatted_row = []
            for country in row_countries:
                yield_val = self.bond_yield_data[country]
                formatted_row.append(f"{country:<15} ({yield_val * 100:.2f}%)")
            print("  " + " | ".join(formatted_row))

        print(f"\n{'=' * 60}")

        while True:
            country_input = input("Enter country name for 10Y bond yield: ").strip()

            if not country_input:
                print("Please enter a country name.")
                continue

            yield_value = self.get_bond_yield_from_excel(country_input)

            if yield_value is not None:
                self.current_risk_free_rate = yield_value
                self.avg_risk_free_rate = yield_value
                self.bond_symbol = f"EXCEL_{country_input.upper()}_10Y"

                # Create simple bond_data for consistency
                today = datetime.now().strftime('%Y-%m-%d')
                self.bond_data = pd.DataFrame({
                    'Date': [today],
                    'Yield': [yield_value * 100]
                })
                self.bond_data['Date'] = pd.to_datetime(self.bond_data['Date'])
                self.bond_data.set_index('Date', inplace=True)

                print(f"✅ Risk-free rate set to: {yield_value * 100:.2f}%")
                return yield_value
            else:
                print("❌ Country not found in bond database.")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry in ['n', 'no']:
                    return None

    def detect_country_from_index(self, index_symbol):
        """Detects country from market index symbol."""

        index_country_map = {
            "^GSPC": "United States", "^DJI": "United States", "^IXIC": "United States",
            "^GDAXI": "Germany", "^FCHI": "France", "FTSEMIB.MI": "Italy",
            "^FTSE": "United Kingdom", "^AEX": "Netherlands", "^SSMI": "Switzerland",
            "^IBEX": "Spain", "^N225": "Japan", "^AORD": "Australia",
            "^GSPTSE": "Canada", "^BVSP": "Brazil", "^MXX": "Mexico"
        }

        alternative_patterns = {
            ".MI": "Italy", ".DE": "Germany", ".PA": "France", ".L": "United Kingdom",
            ".AS": "Netherlands", ".SW": "Switzerland", ".TO": "Canada", ".AX": "Australia"
        }

        index_upper = index_symbol.upper()

        # Direct mapping first
        if index_upper in index_country_map:
            return index_country_map[index_upper]

        # Pattern matching
        for pattern, country in alternative_patterns.items():
            if pattern.upper() in index_upper:
                return country

        return None

    def get_equity_risk_premium(self, index_symbol):
        """Gets equity risk premium for the country based on index symbol."""

        if self.erp_data is None:
            return None

        self.detected_country = self.detect_country_from_index(index_symbol)

        if self.detected_country is None:
            print(f"⚠️ Could not detect country from index '{index_symbol}'")
            return None

        country_upper = self.detected_country.upper()

        if country_upper in self.erp_data:
            self.equity_risk_premium = self.erp_data[country_upper]
            print(f"✅ Country detected: {self.detected_country}")
            print(f"✅ Equity Risk Premium: {self.equity_risk_premium * 100:.2f}%")
            return self.equity_risk_premium

        # Try alternative country names
        country_alternatives = {
            "UNITED STATES": ["USA", "US"],
            "UNITED KINGDOM": ["UK", "BRITAIN"],
        }

        for standard_name, alternatives in country_alternatives.items():
            if country_upper == standard_name:
                for alt in alternatives:
                    if alt in self.erp_data:
                        self.equity_risk_premium = self.erp_data[alt]
                        print(f"✅ Country detected: {self.detected_country}")
                        print(f"✅ Equity Risk Premium: {self.equity_risk_premium * 100:.2f}%")
                        return self.equity_risk_premium

        print(f"❌ No ERP data found for '{self.detected_country}'")
        return None

    def get_manual_equity_risk_premium(self):
        """Allows manual input of equity risk premium."""
        print("\n" + "=" * 50)
        print("MANUAL EQUITY RISK PREMIUM INPUT")
        print("=" * 50)

        if self.detected_country:
            print(f"Detected country: {self.detected_country}")

        print("Typical ranges:")
        print("• USA/Western Europe: 4.5% - 6.0%")
        print("• Eastern Europe: 6.0% - 9.0%")
        print("• Emerging markets: 7.0% - 12.0%")

        while True:
            try:
                erp_input = input("\nEnter equity risk premium (%, e.g., 6.5): ").strip()
                if not erp_input:
                    continue

                erp_percent = float(erp_input)

                if not 0 <= erp_percent <= 25:
                    print("Invalid ERP. Please enter a value between 0% and 25%.")
                    continue

                self.equity_risk_premium = erp_percent / 100
                print(f"✅ Equity Risk Premium set to: {erp_percent:.2f}%")
                return self.equity_risk_premium

            except ValueError:
                print("Invalid input. Please enter a numeric value")
                continue
            except KeyboardInterrupt:
                return None

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
                                          if col[0] == 'Adj Close']
                        if adj_close_cols:
                            return data[adj_close_cols[0][1]]
                    if 'Close' in level_0_values:
                        close_cols = [(i, col) for i, col in enumerate(data.columns)
                                      if col[0] == 'Close']
                        if close_cols:
                            return data[close_cols[0][1]]
                elif 'Adj Close' in data.columns:
                    return data['Adj Close']
                elif 'Close' in data.columns:
                    return data['Close']
                raise KeyError(f"No 'Close' column found")

            stock_close = get_close_price(self.stock_data, 'stock')
            index_close = get_close_price(self.index_data, 'index')

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
        """Calculates beta using linear regression with comprehensive statistics."""
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
        self.p_value = p_value
        self.std_error = std_err
        self.t_statistic = slope / std_err

        print(f"\nBETA REGRESSION RESULTS:")
        print(f"Beta: {self.beta:.4f}")
        print(f"Alpha: {self.alpha:.6f}")
        print(f"R-squared: {self.r_squared:.4f}")
        print(f"Correlation: {self.correlation:.4f}")
        print(f"Standard Error: {self.std_error:.6f}")
        print(f"T-statistic: {self.t_statistic:.4f}")
        print(f"P-value: {self.p_value:.6f}")
        print(f"Observations: {len(combined_data)}")

        return True

    def calculate_cost_of_equity(self, market_risk_premium=None):
        """Calculates cost of equity using CAPM."""
        if self.beta is None or self.current_risk_free_rate is None:
            print("Error: Beta and risk-free rate required")
            return None

        if market_risk_premium is None:
            market_risk_premium = self.equity_risk_premium

        if market_risk_premium is None:
            print("Error: Market risk premium/Equity risk premium must be provided")
            return None

        cost_of_equity = self.current_risk_free_rate + self.beta * market_risk_premium

        print(f"\nCOST OF EQUITY (CAPM):")
        print(f"{'=' * 50}")
        print(f"FORMULA: Re = Rf + β × ERP")
        print(f"{'=' * 50}")
        print(f"Risk-free rate (Rf): {self.current_risk_free_rate * 100:.2f}%")
        print(f"Beta (β): {self.beta:.4f}")

        if self.detected_country and self.equity_risk_premium == market_risk_premium:
            print(f"Equity Risk Premium (ERP): {market_risk_premium * 100:.2f}% ({self.detected_country})")
        else:
            print(f"Equity Risk Premium (ERP): {market_risk_premium * 100:.2f}%")

        print(f"{'=' * 50}")
        print(f"CALCULATION:")
        print(f"Re = {self.current_risk_free_rate * 100:.2f}% + {self.beta:.4f} × {market_risk_premium * 100:.2f}%")
        print(f"Re = {self.current_risk_free_rate * 100:.2f}% + {(self.beta * market_risk_premium) * 100:.2f}%")
        print(f"Re = {cost_of_equity * 100:.2f}%")
        print(f"{'=' * 50}")

        return cost_of_equity

    def plot_regression(self, figsize=(10, 6)):
        """Creates regression plot."""
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

        plt.xlabel('Index Log Returns')
        plt.ylabel('Stock Log Returns')
        plt.title(f'Beta Regression (R² = {self.r_squared:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x * 100:.1f}%'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x * 100:.1f}%'))
        plt.tight_layout()
        plt.show()

    def save_to_excel(self, filename=None, stock_symbol="STOCK", index_symbol="INDEX"):
        """Saves all data to Excel file."""
        if self.beta is None:
            return None

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"beta_analysis_{stock_symbol}_{timestamp}.xlsx"

        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Results sheet with CAPM formula
                results_data = {
                    'Metric': ['CAPM_Formula', 'Risk_Free_Rate_Rf', 'Beta', 'Equity_Risk_Premium_ERP',
                               'Cost_of_Equity_Re', '', 'Beta_Statistics', 'Alpha', 'R_squared',
                               'Correlation', 'Standard_Error', 'T_Statistic', 'P_Value', 'Observations'],
                    'Value': ['Re = Rf + β × ERP', self.current_risk_free_rate, self.beta,
                              self.equity_risk_premium,
                              self.current_risk_free_rate + self.beta * self.equity_risk_premium,
                              '', 'Beta Analysis Results', self.alpha, self.r_squared, self.correlation,
                              self.std_error, self.t_statistic, self.p_value, len(self.stock_returns)],
                    'Percentage': ['', f"{self.current_risk_free_rate * 100:.2f}%", f"{self.beta:.4f}",
                                   f"{self.equity_risk_premium * 100:.2f}%",
                                   f"{(self.current_risk_free_rate + self.beta * self.equity_risk_premium) * 100:.2f}%",
                                   '', '', f"{self.alpha * 100:.4f}%", f"{self.r_squared * 100:.2f}%",
                                   f"{self.correlation * 100:.2f}%", f"{self.std_error:.6f}",
                                   f"{self.t_statistic:.4f}", f"{self.p_value:.6f}", len(self.stock_returns)]
                }

                # Add detailed calculation breakdown
                beta_risk_premium = self.beta * self.equity_risk_premium
                calculation_data = {
                    'Step': ['Formula', 'Substitution', 'Risk Premium Component', 'Final Result'],
                    'Calculation': [
                        'Re = Rf + β × ERP',
                        f'Re = {self.current_risk_free_rate * 100:.2f}% + {self.beta:.4f} × {self.equity_risk_premium * 100:.2f}%',
                        f'Re = {self.current_risk_free_rate * 100:.2f}% + {beta_risk_premium * 100:.2f}%',
                        f'Re = {(self.current_risk_free_rate + beta_risk_premium) * 100:.2f}%'
                    ],
                    'Description': [
                        'CAPM Cost of Equity Formula',
                        'Substituting actual values',
                        'Beta × ERP calculation',
                        'Final Cost of Equity'
                    ]
                }

                if self.current_risk_free_rate:
                    results_data['Metric'].extend(['', 'Risk_Free_Rate_Source', 'Bond_Symbol'])
                    results_data['Value'].extend(['', 'Bond Database', self.bond_symbol])
                    results_data['Percentage'].extend(['', 'Excel Database', self.bond_symbol])

                if self.equity_risk_premium:
                    results_data['Metric'].extend(['', 'ERP_Source', 'Detected_Country'])
                    results_data['Value'].extend(['', 'ERP Database', self.detected_country])
                    results_data['Percentage'].extend(['', 'Excel Database', self.detected_country])

                # Save to Excel
                pd.DataFrame(results_data).to_excel(writer, sheet_name='CAPM_Results', index=False)
                pd.DataFrame(calculation_data).to_excel(writer, sheet_name='CAPM_Calculation', index=False)

                # Data sheets
                if self.stock_data is not None:
                    self.stock_data.to_excel(writer, sheet_name='Stock_Data')
                if self.index_data is not None:
                    self.index_data.to_excel(writer, sheet_name='Index_Data')

                # Returns
                returns_df = pd.DataFrame({
                    'Date': self.stock_returns.index,
                    'Stock_Returns': self.stock_returns.values,
                    'Index_Returns': self.index_returns.values,
                    'Stock_Returns_Percent': self.stock_returns.values * 100,
                    'Index_Returns_Percent': self.index_returns.values * 100
                })
                returns_df.to_excel(writer, sheet_name='Returns_Data', index=False)

            print(f"✅ Data saved to: {filename}")
            print(f"📊 Excel file includes:")
            print(f"   • CAPM_Results: Main results with formula")
            print(f"   • CAPM_Calculation: Step-by-step calculation")
            print(f"   • Stock_Data: Historical stock prices")
            print(f"   • Index_Data: Historical index prices")
            print(f"   • Returns_Data: Calculated returns")
            return filename

        except Exception as e:
            print(f"❌ Error saving: {e}")
            return None


def get_user_input():
    """Collects user input for analysis."""
    print("=" * 70)
    print("BETA CALCULATOR WITH AUTOMATIC EQUITY RISK PREMIUM")
    print("=" * 70)

    # Stock ticker input
    print("\nStock ticker examples:")
    print("• Italian stocks: ENI.MI, ENEL.MI, ISP.MI, UCG.MI, TIT.MI")
    print("• US stocks: AAPL, MSFT, GOOGL, TSLA, NVDA")
    print("• European stocks: ASML.AS, SAP.DE, NESN.SW")
    stock_symbol = input("\nEnter stock ticker: ").upper().strip()

    # Index ticker input
    print("\nIndex ticker examples:")
    print("• FTSE MIB (Italy): FTSEMIB.MI")
    print("• S&P 500 (USA): ^GSPC")
    print("• NASDAQ: ^IXIC")
    print("• DAX (Germany): ^GDAXI")
    print("• CAC 40 (France): ^FCHI")
    print("• FTSE 100 (UK): ^FTSE")
    print("• Nikkei (Japan): ^N225")
    index_symbol = input("\nEnter index ticker: ").upper().strip()

    # Bond file
    print("\n" + "=" * 50)
    print("BOND YIELD DATABASE")
    print("=" * 50)
    bond_file = input("Bond Excel file path (default: 'Bond.xlsx'): ").strip()
    if not bond_file:
        bond_file = "Bond.xlsx"

    # ERP file
    print("\n" + "=" * 50)
    print("EQUITY RISK PREMIUM DATABASE")
    print("=" * 50)
    erp_file = input("ERP Excel file path (default: 'ERP 2025.xlsx'): ").strip()
    if not erp_file:
        erp_file = "ERP 2025.xlsx"

    # Period and interval
    print("\nPeriod (default: 10y): ", end="")
    period = input().strip() or "10y"

    print("Interval (default: 1mo): ", end="")
    interval = input().strip() or "1mo"

    return stock_symbol, index_symbol, bond_file, erp_file, period, interval


if __name__ == "__main__":
    try:
        stock_symbol, index_symbol, bond_file, erp_file, period, interval = get_user_input()
        calc = BetaCalculator()

        print(f"\n{'=' * 70}")
        print("COST OF EQUITY CALCULATION - COMPREHENSIVE ANALYSIS")
        print(f"{'=' * 70}")

        # Load databases
        print("Loading databases...")
        calc.load_erp_data(erp_file)

        if not calc.load_bond_yield_data(bond_file):
            print("❌ Failed to load bond database. Exiting.")
            exit(1)

        # Download market data
        print(f"\nDownloading market data...")
        if not calc.download_data(stock_symbol, index_symbol, period, interval):
            print("❌ Failed to download market data. Exiting.")
            exit(1)

        # Calculate beta
        print("\nCalculating beta...")
        if not calc.calculate_returns() or not calc.calculate_beta():
            print("❌ Failed to calculate beta. Exiting.")
            exit(1)

        # Show plot
        plot_input = input("\nShow regression plot? (y/n): ").strip().lower()
        if plot_input == 'y':
            calc.plot_regression()

        # Get risk-free rate
        print(f"\n{'=' * 50}")
        print("STEP 2: RISK-FREE RATE")
        print(f"{'=' * 50}")

        if calc.select_bond_country_from_excel() is None:
            print("❌ Risk-free rate required. Exiting.")
            exit(1)

        # Get ERP
        print(f"\n{'=' * 50}")
        print("STEP 3: EQUITY RISK PREMIUM")
        print(f"{'=' * 50}")

        auto_erp = calc.get_equity_risk_premium(index_symbol)

        if auto_erp:
            use_auto = input(f"\nUse automatic ERP ({auto_erp * 100:.2f}%)? (y/n): ").strip().lower()
            if use_auto == 'n':
                if calc.get_manual_equity_risk_premium() is None:
                    print("❌ ERP required. Exiting.")
                    exit(1)
        else:
            if calc.get_manual_equity_risk_premium() is None:
                print("❌ ERP required. Exiting.")
                exit(1)

        # Calculate cost of equity
        print(f"\n{'=' * 50}")
        print("STEP 4: COST OF EQUITY")
        print(f"{'=' * 50}")

        cost_of_equity = calc.calculate_cost_of_equity()

        if cost_of_equity:
            print(f"\n{'=' * 60}")
            print(f"🎯 FINAL COST OF EQUITY RESULT")
            print(f"{'=' * 60}")
            print(f"CAPM Formula: Re = Rf + β × ERP")
            print(f"Final Result: {cost_of_equity * 100:.2f}%")
            print(f"{'=' * 60}")
            print(f"This rate can be used as the discount rate")
            print(f"for equity cash flows in DCF valuation.")
            print(f"{'=' * 60}")

        # Save results
        save_input = input("\nSave comprehensive results to Excel? (y/n): ").strip().lower()
        if save_input == 'y':
            calc.save_to_excel(stock_symbol=stock_symbol, index_symbol=index_symbol)

        print(f"\n{'=' * 70}")
        print("ANALYSIS COMPLETED!")
        print(f"{'=' * 70}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Please check your inputs and try again.")