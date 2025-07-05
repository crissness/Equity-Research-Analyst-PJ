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
        # Tax rate attributes
        self.tax_rate_data = None

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

    def load_tax_rate_data(self, erp_file_path="ERP 2025.xlsx"):
        """Loads marginal tax rate data from ERP Excel file."""
        try:
            import os
            if not os.path.exists(erp_file_path):
                print(f"❌ ERP file '{erp_file_path}' not found for tax rates")
                return False

            print(f"📁 Loading tax rate data from: {erp_file_path}")

            df = pd.read_excel(erp_file_path, sheet_name=0)

            if 'Country' not in df.columns:
                print("❌ Excel file must have 'Country' column")
                return False

            # Look for tax rate column
            tax_column = None
            possible_tax_columns = ['Corporate Tax Rate', 'Marginal Tax Rate', 'Tax Rate', 'Tax']

            for col in possible_tax_columns:
                if col in df.columns:
                    tax_column = col
                    break

            if tax_column is None:
                print("⚠️ No tax rate column found in ERP file")
                print("Available columns:", list(df.columns))
                return False

            print(f"✅ Using tax rate column: '{tax_column}'")

            self.tax_rate_data = {}
            valid_entries = 0

            for _, row in df.iterrows():
                if pd.notna(row['Country']) and pd.notna(row[tax_column]):
                    try:
                        country = str(row['Country']).strip()
                        tax_rate = float(row[tax_column])

                        # Handle both decimal and percentage formats
                        if tax_rate > 1:
                            tax_rate = tax_rate / 100

                        self.tax_rate_data[country.upper()] = tax_rate
                        valid_entries += 1
                    except (ValueError, TypeError):
                        continue

            print(f"✅ Tax rate data loaded for {valid_entries} countries")
            return True

        except Exception as e:
            print(f"❌ Error loading tax rate data: {e}")
            return False

    def get_tax_rate_from_country(self, country):
        """Gets marginal tax rate for a specific country."""
        if self.tax_rate_data is None:
            return None

        country_upper = country.upper()

        # Try exact match first
        if country_upper in self.tax_rate_data:
            tax_rate = self.tax_rate_data[country_upper]
            print(f"✅ Found tax rate for {country}: {tax_rate * 100:.1f}%")
            return tax_rate

        # Try alternative country names
        country_alternatives = {
            "USA": ["UNITED STATES", "US", "AMERICA"],
            "UNITED STATES": ["USA", "US", "AMERICA"],
            "UK": ["UNITED KINGDOM", "BRITAIN", "GREAT BRITAIN"],
            "UNITED KINGDOM": ["UK", "BRITAIN", "GREAT BRITAIN"],
        }

        if country_upper in country_alternatives:
            for alt in country_alternatives[country_upper]:
                if alt in self.tax_rate_data:
                    tax_rate = self.tax_rate_data[alt]
                    print(f"✅ Found tax rate for {country} (as {alt}): {tax_rate * 100:.1f}%")
                    return tax_rate

        print(f"⚠️ No tax rate data found for '{country}'")
        return None

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
                pd.DataFrame(calculation_data).to_excel(writer, sheet_name='CAMP_Calculation', index=False)

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


class CostOfDebtCalculator:
    """
    Comprehensive cost of debt calculator using synthetic ratings approach.
    Calculates: Cost of Debt = (Risk-free Rate + Spread) × (1 - Tax Rate)
    """

    def __init__(self):
        self.ticker = None
        self.company_info = None
        self.market_cap = None
        self.is_financial = None
        self.ebit = None
        self.interest_expense = None
        self.interest_coverage_ratio = None
        self.company_type = None  # 'large_cap', 'small_cap', 'financial'
        self.rating = None
        self.spread = None
        self.risk_free_rate = None
        self.tax_rate = None
        self.cost_of_debt = None
        # Database attributes
        self.synthetic_ratings_data = None
        self.bond_yield_data = None
        # Tax rate attributes
        self.tax_rate_data = None
        self.detected_tax_country = None

    def load_synthetic_ratings_data(self, ratings_file_path="Synthetic Ratings.xlsx"):
        """Loads synthetic ratings data from Excel file."""
        try:
            import os
            if not os.path.exists(ratings_file_path):
                print(f"❌ Ratings file '{ratings_file_path}' not found")
                return False

            print(f"📁 Loading synthetic ratings from: {ratings_file_path}")

            # Read the Excel file
            df = pd.read_excel(ratings_file_path, sheet_name=0, header=None)

            # Parse the complex structure
            self.synthetic_ratings_data = {
                'large_cap': [],
                'small_cap': [],
                'financial': []
            }

            # Extract data starting from row 3 (0-indexed row 2)
            # Skip the header rows and process actual data
            for i in range(3, len(df)):
                row = df.iloc[i]

                try:
                    # Large cap data (columns 0-3)
                    if pd.notna(row[0]) and pd.notna(row[1]) and pd.notna(row[2]) and pd.notna(row[3]):
                        # Skip rows with ">" symbols - these are the actual numeric ranges
                        if isinstance(row[0], (int, float)) and isinstance(row[1], (int, float)):
                            self.synthetic_ratings_data['large_cap'].append({
                                'min_ratio': float(row[0]),
                                'max_ratio': float(row[1]),
                                'rating': str(row[2]),
                                'spread': float(row[3])
                            })
                except (ValueError, TypeError):
                    pass  # Skip problematic rows

                try:
                    # Small cap data (columns 5-8)
                    if pd.notna(row[5]) and pd.notna(row[6]) and pd.notna(row[7]) and pd.notna(row[8]):
                        if isinstance(row[5], (int, float)) and isinstance(row[6], (int, float)):
                            self.synthetic_ratings_data['small_cap'].append({
                                'min_ratio': float(row[5]),
                                'max_ratio': float(row[6]),
                                'rating': str(row[7]),
                                'spread': float(row[8])
                            })
                except (ValueError, TypeError):
                    pass  # Skip problematic rows

                try:
                    # Financial services data (columns 10-13)
                    if pd.notna(row[10]) and pd.notna(row[11]) and pd.notna(row[12]) and pd.notna(row[13]):
                        if isinstance(row[10], (int, float)) and isinstance(row[11], (int, float)):
                            self.synthetic_ratings_data['financial'].append({
                                'min_ratio': float(row[10]),
                                'max_ratio': float(row[11]),
                                'rating': str(row[12]),
                                'spread': float(row[13])
                            })
                except (ValueError, TypeError):
                    pass  # Skip problematic rows

            # Verify we have data
            if not any(self.synthetic_ratings_data.values()):
                print("❌ No valid rating data found in file")
                return False

            print(f"✅ Loaded ratings data:")
            print(f"   • Large cap ratings: {len(self.synthetic_ratings_data['large_cap'])}")
            print(f"   • Small cap ratings: {len(self.synthetic_ratings_data['small_cap'])}")
            print(f"   • Financial services ratings: {len(self.synthetic_ratings_data['financial'])}")

            # Show sample data for verification
            if self.synthetic_ratings_data['large_cap']:
                sample = self.synthetic_ratings_data['large_cap'][0]
                print(
                    f"   • Sample large cap: {sample['min_ratio']}-{sample['max_ratio']} → {sample['rating']} ({sample['spread'] * 100:.2f}%)")

            return True

        except Exception as e:
            print(f"❌ Error loading synthetic ratings: {e}")
            print("Debug: Let's examine the file structure...")

            try:
                # Debug information
                df = pd.read_excel(ratings_file_path, sheet_name=0, header=None)
                print(f"File shape: {df.shape}")
                print("First few rows:")
                for i in range(min(8, len(df))):
                    print(f"Row {i}: {list(df.iloc[i])}")
            except Exception as debug_e:
                print(f"Debug error: {debug_e}")

            return False

    def load_bond_yield_data(self, bond_file_path="Bond.xlsx"):
        """Loads bond yield data from Excel file."""
        try:
            import os
            if not os.path.exists(bond_file_path):
                print(f"❌ Bond file '{bond_file_path}' not found")
                return False

            print(f"📁 Loading bond yield data from: {bond_file_path}")

            df = pd.read_excel(bond_file_path, sheet_name=0)

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
                return False

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

    def load_tax_rate_data(self, tax_rate_data):
        """Loads tax rate data from the beta calculator."""
        if tax_rate_data is None:
            print("⚠️ No tax rate data provided")
            return False

        self.tax_rate_data = tax_rate_data
        print(f"✅ Tax rate data loaded for {len(tax_rate_data)} countries")
        return True

    def get_tax_rate_from_country(self, country):
        """Gets marginal tax rate for a specific country."""
        if self.tax_rate_data is None:
            return None

        country_upper = country.upper()

        # Try exact match first
        if country_upper in self.tax_rate_data:
            tax_rate = self.tax_rate_data[country_upper]
            print(f"✅ Found tax rate for {country}: {tax_rate * 100:.1f}%")
            return tax_rate

        # Try alternative country names
        country_alternatives = {
            "USA": ["UNITED STATES", "US", "AMERICA"],
            "UNITED STATES": ["USA", "US", "AMERICA"],
            "UK": ["UNITED KINGDOM", "BRITAIN", "GREAT BRITAIN"],
            "UNITED KINGDOM": ["UK", "BRITAIN", "GREAT BRITAIN"],
        }

        if country_upper in country_alternatives:
            for alt in country_alternatives[country_upper]:
                if alt in self.tax_rate_data:
                    tax_rate = self.tax_rate_data[alt]
                    print(f"✅ Found tax rate for {country} (as {alt}): {tax_rate * 100:.1f}%")
                    return tax_rate

        print(f"⚠️ No tax rate data found for '{country}'")
        return None

    def get_company_info(self, ticker):
        """Downloads company information from Yahoo Finance."""
        print(f"📊 Fetching company information for {ticker}...")

        try:
            self.ticker = ticker.upper()
            stock = yf.Ticker(self.ticker)
            self.company_info = stock.info

            if not self.company_info:
                print(f"❌ No data found for ticker {self.ticker}")
                return False

            # Get market cap
            self.market_cap = self.company_info.get('marketCap', None)
            if self.market_cap is None:
                print(f"⚠️ Market cap not available for {self.ticker}")
                return False

            # Automatically detect if it's a financial services firm
            sector = self.company_info.get('sector', '').lower()
            industry = self.company_info.get('industry', '').lower()

            financial_keywords = ['financial', 'bank', 'insurance', 'credit', 'mortgage',
                                  'investment', 'securities', 'asset management']

            self.is_financial = any(keyword in sector for keyword in financial_keywords) or \
                                any(keyword in industry for keyword in financial_keywords)

            # Determine company type
            if self.is_financial:
                self.company_type = 'financial'
            elif self.market_cap > 5_000_000_000:  # > 5 billion
                self.company_type = 'large_cap'
            else:
                self.company_type = 'small_cap'

            print(f"✅ Company information retrieved:")
            print(f"   • Company: {self.company_info.get('longName', self.ticker)}")
            print(f"   • Sector: {self.company_info.get('sector', 'N/A')}")
            print(f"   • Industry: {self.company_info.get('industry', 'N/A')}")
            print(f"   • Market Cap: ${self.market_cap:,.0f}")
            print(f"   • Company Type: {self.company_type}")
            print(f"   • Financial Services: {'Yes' if self.is_financial else 'No'}")

            return True

        except Exception as e:
            print(f"❌ Error fetching company info: {e}")
            return False

    def get_financial_inputs(self):
        """Gets EBIT and interest expense from user."""
        print(f"\n{'=' * 50}")
        print("FINANCIAL INPUTS")
        print(f"{'=' * 50}")
        print(f"Please provide the following financial data for {self.ticker}:")

        # Get EBIT
        while True:
            try:
                ebit_input = input("\nEnter current EBIT (in millions, e.g., 1500): ").strip()
                if not ebit_input:
                    print("Please enter a valid EBIT value.")
                    continue

                self.ebit = float(ebit_input) * 1_000_000  # Convert to actual value
                print(f"✅ EBIT set to: ${self.ebit:,.0f}")
                break

            except ValueError:
                print("Invalid input. Please enter a numeric value.")
                continue

        # Get Interest Expense
        while True:
            try:
                interest_input = input("\nEnter current Interest Expenses (in millions, e.g., 75): ").strip()

                if not interest_input:
                    print("Please enter a value, or 0 if no interest expenses.")
                    continue

                interest_value = float(interest_input) * 1_000_000  # Convert to actual value

                if interest_value <= 0:
                    print("⚠️ No interest expenses detected. Setting coverage ratio to 20.")
                    self.interest_expense = 0
                    self.interest_coverage_ratio = 20.0
                else:
                    self.interest_expense = interest_value
                    self.interest_coverage_ratio = self.ebit / self.interest_expense

                print(f"✅ Interest Expenses: ${self.interest_expense:,.0f}")
                print(f"✅ Interest Coverage Ratio: {self.interest_coverage_ratio:.2f}")
                break

            except ValueError:
                print("Invalid input. Please enter a numeric value.")
                continue

        return True

    def assign_synthetic_rating(self):
        """Assigns synthetic rating based on company type and coverage ratio."""
        if self.synthetic_ratings_data is None:
            print("❌ Synthetic ratings data not loaded")
            return False

        print(f"\n{'=' * 50}")
        print("SYNTHETIC RATING ASSIGNMENT")
        print(f"{'=' * 50}")

        # Select appropriate rating table
        rating_table = self.synthetic_ratings_data[self.company_type]

        print(f"Company Type: {self.company_type.replace('_', ' ').title()}")
        print(f"Interest Coverage Ratio: {self.interest_coverage_ratio:.2f}")

        # Find appropriate rating
        for rating_entry in rating_table:
            min_ratio = rating_entry['min_ratio']
            max_ratio = rating_entry['max_ratio']

            if min_ratio <= self.interest_coverage_ratio <= max_ratio:
                self.rating = rating_entry['rating']
                self.spread = rating_entry['spread']

                print(f"✅ Rating Assignment:")
                print(f"   • Coverage Ratio Range: {min_ratio:.2f} to {max_ratio:.2f}")
                print(f"   • Assigned Rating: {self.rating}")
                print(f"   • Credit Spread: {self.spread * 100:.2f}%")

                return True

        # If no rating found, assign worst rating
        worst_rating = rating_table[0]  # First entry is usually worst
        self.rating = worst_rating['rating']
        self.spread = worst_rating['spread']

        print(f"⚠️ Coverage ratio outside normal range. Assigned worst rating:")
        print(f"   • Assigned Rating: {self.rating}")
        print(f"   • Credit Spread: {self.spread * 100:.2f}%")

        return True

    def select_risk_free_rate(self):
        """Allows user to select risk-free rate from bond database."""
        if self.bond_yield_data is None:
            print("❌ Bond yield data not loaded")
            return False

        print(f"\n{'=' * 50}")
        print("RISK-FREE RATE SELECTION")
        print(f"{'=' * 50}")
        print("Available countries in bond database:")

        countries_list = sorted(list(self.bond_yield_data.keys()))

        # Display countries
        for i in range(0, len(countries_list), 3):
            row_countries = countries_list[i:i + 3]
            formatted_row = []
            for country in row_countries:
                yield_val = self.bond_yield_data[country]
                formatted_row.append(f"{country:<15} ({yield_val * 100:.2f}%)")
            print("  " + " | ".join(formatted_row))

        while True:
            country_input = input(f"\nEnter country for risk-free rate: ").strip()

            if not country_input:
                print("Please enter a country name.")
                continue

            country_upper = country_input.upper()

            # Try exact match first
            if country_upper in self.bond_yield_data:
                self.risk_free_rate = self.bond_yield_data[country_upper]
                self.detected_tax_country = country_input  # Store for tax rate detection
                print(f"✅ Risk-free rate: {self.risk_free_rate * 100:.2f}% ({country_input})")
                return True

            # Try alternative names
            country_alternatives = {
                "USA": ["UNITED STATES", "US"],
                "UNITED STATES": ["USA", "US"],
                "UK": ["UNITED KINGDOM", "BRITAIN"],
                "UNITED KINGDOM": ["UK", "BRITAIN"],
            }

            found = False
            if country_upper in country_alternatives:
                for alt in country_alternatives[country_upper]:
                    if alt in self.bond_yield_data:
                        self.risk_free_rate = self.bond_yield_data[alt]
                        self.detected_tax_country = alt  # Store for tax rate detection
                        print(f"✅ Risk-free rate: {self.risk_free_rate * 100:.2f}% ({alt})")
                        found = True
                        break

            if found:
                return True

            print(f"❌ Country '{country_input}' not found in database.")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry in ['n', 'no']:
                return False

    def get_tax_rate(self):
        """Gets marginal tax rate from user or automatically from country data."""
        print(f"\n{'=' * 50}")
        print("MARGINAL TAX RATE")
        print(f"{'=' * 50}")

        # Try to detect country from risk-free rate selection
        auto_tax_rate = None
        if self.tax_rate_data and hasattr(self, 'detected_tax_country') and self.detected_tax_country:
            auto_tax_rate = self.get_tax_rate_from_country(self.detected_tax_country)

        if auto_tax_rate:
            print(f"\nAutomatic tax rate detected: {auto_tax_rate * 100:.1f}% ({self.detected_tax_country})")
            use_auto = input("Use this tax rate? (y/n, default: y): ").strip().lower()

            if use_auto != 'n':
                self.tax_rate = auto_tax_rate
                print(f"✅ Marginal tax rate set to: {auto_tax_rate * 100:.1f}%")
                return True
            else:
                print("\nManual tax rate input selected.")

        print("\nEnter the marginal tax rate for the company.")
        print("Typical ranges:")
        print("• USA Corporate Tax Rate: 21%")
        print("• European rates: 19% - 32%")
        print("• Emerging markets: 15% - 35%")

        while True:
            try:
                tax_input = input(f"\nEnter marginal tax rate (%, e.g., 25): ").strip()

                if not tax_input:
                    print("Please enter a tax rate.")
                    continue

                tax_percent = float(tax_input)

                if not 0 <= tax_percent <= 60:
                    print("Invalid tax rate. Please enter a value between 0% and 60%.")
                    continue

                self.tax_rate = tax_percent / 100
                print(f"✅ Marginal tax rate: {tax_percent:.1f}%")
                return True

            except ValueError:
                print("Invalid input. Please enter a numeric value.")
                continue

    def calculate_cost_of_debt(self):
        """Calculates the after-tax cost of debt."""
        if any(x is None for x in [self.risk_free_rate, self.spread, self.tax_rate]):
            print("❌ Missing required data for cost of debt calculation")
            return None

        # Cost of Debt = (Risk-free Rate + Spread) × (1 - Tax Rate)
        pre_tax_cost = self.risk_free_rate + self.spread
        self.cost_of_debt = pre_tax_cost * (1 - self.tax_rate)

        print(f"\n{'=' * 60}")
        print("COST OF DEBT CALCULATION")
        print(f"{'=' * 60}")
        print(f"FORMULA: Cost of Debt = (Rf + Spread) × (1 - Tax Rate)")
        print(f"{'=' * 60}")
        print(f"Risk-free Rate (Rf):     {self.risk_free_rate * 100:.2f}%")
        print(f"Credit Spread:           {self.spread * 100:.2f}%")
        print(f"Pre-tax Cost of Debt:    {pre_tax_cost * 100:.2f}%")
        print(f"Marginal Tax Rate:       {self.tax_rate * 100:.1f}%")
        print(f"{'=' * 60}")
        print(f"CALCULATION:")
        print(
            f"Cost of Debt = ({self.risk_free_rate * 100:.2f}% + {self.spread * 100:.2f}%) × (1 - {self.tax_rate * 100:.1f}%)")
        print(f"Cost of Debt = {pre_tax_cost * 100:.2f}% × {(1 - self.tax_rate) * 100:.1f}%")
        print(f"Cost of Debt = {self.cost_of_debt * 100:.2f}%")
        print(f"{'=' * 60}")

        return self.cost_of_debt

    def save_to_excel(self, filename=None):
        """Saves all results to Excel file."""
        if self.cost_of_debt is None:
            print("❌ No results to save")
            return None

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"cost_of_debt_{self.ticker}_{timestamp}.xlsx"

        try:
            # Prepare results data
            pre_tax_cost = self.risk_free_rate + self.spread

            results_data = {
                'Metric': [
                    'Company_Ticker',
                    'Company_Name',
                    'Market_Cap',
                    'Company_Type',
                    'Is_Financial_Services',
                    '',
                    'EBIT',
                    'Interest_Expenses',
                    'Interest_Coverage_Ratio',
                    '',
                    'Assigned_Rating',
                    'Credit_Spread',
                    'Risk_Free_Rate',
                    'Pre_Tax_Cost_of_Debt',
                    'Marginal_Tax_Rate',
                    '',
                    'Cost_of_Debt_Formula',
                    'After_Tax_Cost_of_Debt'
                ],
                'Value': [
                    self.ticker,
                    self.company_info.get('longName', self.ticker),
                    self.market_cap,
                    self.company_type,
                    'Yes' if self.is_financial else 'No',
                    '',
                    self.ebit,
                    self.interest_expense,
                    self.interest_coverage_ratio,
                    '',
                    self.rating,
                    self.spread,
                    self.risk_free_rate,
                    pre_tax_cost,
                    self.tax_rate,
                    '',
                    '(Rf + Spread) × (1 - Tax Rate)',
                    self.cost_of_debt
                ],
                'Formatted': [
                    self.ticker,
                    self.company_info.get('longName', self.ticker),
                    f"${self.market_cap:,.0f}",
                    self.company_type.replace('_', ' ').title(),
                    'Yes' if self.is_financial else 'No',
                    '',
                    f"${self.ebit:,.0f}",
                    f"${self.interest_expense:,.0f}",
                    f"{self.interest_coverage_ratio:.2f}",
                    '',
                    self.rating,
                    f"{self.spread * 100:.2f}%",
                    f"{self.risk_free_rate * 100:.2f}%",
                    f"{pre_tax_cost * 100:.2f}%",
                    f"{self.tax_rate * 100:.1f}%",
                    '',
                    '(Rf + Spread) × (1 - Tax Rate)',
                    f"{self.cost_of_debt * 100:.2f}%"
                ]
            }

            # Calculation breakdown
            calculation_data = {
                'Step': [
                    'Formula',
                    'Risk-free Rate',
                    'Credit Spread',
                    'Pre-tax Cost',
                    'Tax Shield',
                    'Final Result'
                ],
                'Calculation': [
                    'Cost of Debt = (Rf + Spread) × (1 - Tax Rate)',
                    f'Rf = {self.risk_free_rate * 100:.2f}%',
                    f'Spread = {self.spread * 100:.2f}%',
                    f'Pre-tax = {self.risk_free_rate * 100:.2f}% + {self.spread * 100:.2f}% = {pre_tax_cost * 100:.2f}%',
                    f'Tax Shield = 1 - {self.tax_rate * 100:.1f}% = {(1 - self.tax_rate) * 100:.1f}%',
                    f'Cost of Debt = {pre_tax_cost * 100:.2f}% × {(1 - self.tax_rate) * 100:.1f}% = {self.cost_of_debt * 100:.2f}%'
                ],
                'Description': [
                    'After-tax Cost of Debt Formula',
                    'Government bond yield (risk-free rate)',
                    'Credit spread based on synthetic rating',
                    'Cost of debt before tax benefits',
                    'Tax deductibility of interest payments',
                    'Final after-tax cost of debt'
                ]
            }

            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Save main results
                pd.DataFrame(results_data).to_excel(writer, sheet_name='Cost_of_Debt_Results', index=False)

                # Save calculation breakdown
                pd.DataFrame(calculation_data).to_excel(writer, sheet_name='Calculation_Steps', index=False)

            print(f"✅ Results saved to: {filename}")
            print(f"📊 Excel file includes:")
            print(f"   • Cost_of_Debt_Results: Complete analysis results")
            print(f"   • Calculation_Steps: Step-by-step calculation")

            return filename

        except Exception as e:
            print(f"❌ Error saving to Excel: {e}")
            return None


class WACCCalculator:
    """
    Weighted Average Cost of Capital (WACC) Calculator.
    Calculates: WACC = (Cost of Equity × Weight of Equity) + (Cost of Debt × Weight of Debt)
    """

    def __init__(self):
        self.cost_of_equity = None
        self.cost_of_debt = None
        self.valuation_method = None  # 'market' or 'book'
        self.ticker = None
        self.company_info = None

        # Market values
        self.market_value_equity = None
        self.market_value_debt = None

        # Book values
        self.book_value_equity = None
        self.book_value_debt = None

        # Final values used in calculation
        self.equity_value = None
        self.debt_value = None
        self.total_value = None
        self.weight_equity = None
        self.weight_debt = None
        self.wacc = None

        # For market value of debt calculation
        self.debt_maturity = None
        self.interest_expense = None

    def get_cost_inputs(self):
        """Gets cost of equity and cost of debt from user."""
        print("=" * 70)
        print("WACC CALCULATOR")
        print("=" * 70)
        print("This calculator computes the Weighted Average Cost of Capital using:")
        print("WACC = (Cost of Equity × Weight of Equity) + (Cost of Debt × Weight of Debt)")
        print("=" * 70)

        print(f"\n{'=' * 50}")
        print("COST OF CAPITAL INPUTS")
        print(f"{'=' * 50}")

        # Get Cost of Equity
        while True:
            try:
                equity_input = input("Enter Cost of Equity (%, e.g., 12.5): ").strip()
                if not equity_input:
                    print("Please enter the cost of equity.")
                    continue

                equity_percent = float(equity_input)

                if not 0 <= equity_percent <= 50:
                    print("Invalid cost of equity. Please enter a value between 0% and 50%.")
                    continue

                self.cost_of_equity = equity_percent / 100
                print(f"✅ Cost of Equity: {equity_percent:.2f}%")
                break

            except ValueError:
                print("Invalid input. Please enter a numeric value.")
                continue

        # Get Cost of Debt
        while True:
            try:
                debt_input = input("Enter Cost of Debt (%, e.g., 4.5): ").strip()
                if not debt_input:
                    print("Please enter the cost of debt.")
                    continue

                debt_percent = float(debt_input)

                if not 0 <= debt_percent <= 30:
                    print("Invalid cost of debt. Please enter a value between 0% and 30%.")
                    continue

                self.cost_of_debt = debt_percent / 100
                print(f"✅ Cost of Debt: {debt_percent:.2f}%")
                break

            except ValueError:
                print("Invalid input. Please enter a numeric value.")
                continue

        return True

    def select_valuation_method(self):
        """Allows user to choose between market and book values."""
        print(f"\n{'=' * 50}")
        print("VALUATION METHOD SELECTION")
        print(f"{'=' * 50}")
        print("Choose valuation approach for weights calculation:")
        print("• Market Values: Uses current market capitalization and market value of debt")
        print("• Book Values: Uses balance sheet values from financial statements")
        print("\nRecommendation: Market values are preferred for WACC calculation")
        print("as they reflect current investor expectations and market conditions.")

        while True:
            method_input = input("\nUse Market Values or Book Values? (market/book, default: market): ").strip().lower()

            if not method_input or method_input in ['market', 'm']:
                self.valuation_method = 'market'
                print("✅ Selected: Market Values")
                return True
            elif method_input in ['book', 'b']:
                self.valuation_method = 'book'
                print("✅ Selected: Book Values")
                return True
            else:
                print("Please enter 'market' or 'book'")
                continue

    def get_market_values(self):
        """Calculates market values of equity and debt."""
        print(f"\n{'=' * 50}")
        print("MARKET VALUE CALCULATION")
        print(f"{'=' * 50}")

        # Get ticker for market value of equity
        while True:
            ticker_input = input("Enter company ticker for market value of equity: ").strip()
            if not ticker_input:
                print("Please enter a valid ticker symbol.")
                continue

            self.ticker = ticker_input.upper()

            # Get market cap from Yahoo Finance
            try:
                stock = yf.Ticker(self.ticker)
                self.company_info = stock.info

                if not self.company_info:
                    print(f"❌ No data found for ticker {self.ticker}")
                    continue

                self.market_value_equity = self.company_info.get('marketCap', None)
                if self.market_value_equity is None:
                    print(f"❌ Market cap not available for {self.ticker}")
                    continue

                print(f"✅ Company: {self.company_info.get('longName', self.ticker)}")
                print(f"✅ Market Value of Equity: ${self.market_value_equity:,.0f}")
                break

            except Exception as e:
                print(f"❌ Error fetching data for {self.ticker}: {e}")
                continue

        # Get inputs for market value of debt calculation
        print(f"\nFor market value of debt calculation, please provide:")

        # Get book value of debt
        while True:
            try:
                debt_input = input("Enter Total Debt from balance sheet (in millions, e.g., 15000): ").strip()
                if not debt_input:
                    print("Please enter the total debt value.")
                    continue

                book_debt = float(debt_input) * 1_000_000  # Convert to actual value
                print(f"✅ Book Value of Debt: ${book_debt:,.0f}")
                break

            except ValueError:
                print("Invalid input. Please enter a numeric value.")
                continue

        # Get weighted average maturity
        while True:
            try:
                maturity_input = input("Enter Weighted Average Maturity of debt (years, e.g., 8.5): ").strip()
                if not maturity_input:
                    print("Please enter the average maturity.")
                    continue

                self.debt_maturity = float(maturity_input)
                if self.debt_maturity <= 0:
                    print("Maturity must be greater than 0.")
                    continue

                print(f"✅ Weighted Average Maturity: {self.debt_maturity:.1f} years")
                break

            except ValueError:
                print("Invalid input. Please enter a numeric value.")
                continue

        # Get interest expense
        while True:
            try:
                interest_input = input(
                    "Enter Interest Expense from income statement (in millions, e.g., 500): ").strip()
                if not interest_input:
                    print("Please enter the interest expense.")
                    continue

                self.interest_expense = float(interest_input) * 1_000_000  # Convert to actual value
                print(f"✅ Interest Expense: ${self.interest_expense:,.0f}")
                break

            except ValueError:
                print("Invalid input. Please enter a numeric value.")
                continue

        # Calculate market value of debt using the corrected formula
        print(f"\nCalculating market value of debt...")
        print(
            f"Formula: Market Value of Debt = (Interest Expense × (1 - (1 / (1 + Cost of Debt))) / Cost of Debt) + (Total Debt / ((1 + Cost of Debt)^Maturity))")

        # Present value of interest payments (corrected formula)
        pv_interest = (self.interest_expense * (1 - (1 / (1 + self.cost_of_debt)))) / self.cost_of_debt

        # Present value of principal repayment
        pv_principal = book_debt / ((1 + self.cost_of_debt) ** self.debt_maturity)

        self.market_value_debt = pv_interest + pv_principal

        print(f"\n📊 Market Value of Debt Calculation:")
        print(f"   • PV of Interest Payments: ${pv_interest:,.0f}")
        print(f"   • PV of Principal Repayment: ${pv_principal:,.0f}")
        print(f"   • Total Market Value of Debt: ${self.market_value_debt:,.0f}")

        # Set final values
        self.equity_value = self.market_value_equity
        self.debt_value = self.market_value_debt

        return True

    def get_book_values(self):
        """Gets book values of equity and debt from user."""
        print(f"\n{'=' * 50}")
        print("BOOK VALUE INPUTS")
        print(f"{'=' * 50}")
        print("Please provide balance sheet values:")

        # Get book value of equity
        while True:
            try:
                equity_input = input("Enter Book Value of Equity (in millions, e.g., 25000): ").strip()
                if not equity_input:
                    print("Please enter the book value of equity.")
                    continue

                self.book_value_equity = float(equity_input) * 1_000_000  # Convert to actual value
                print(f"✅ Book Value of Equity: ${self.book_value_equity:,.0f}")
                break

            except ValueError:
                print("Invalid input. Please enter a numeric value.")
                continue

        # Get book value of debt
        while True:
            try:
                debt_input = input("Enter Book Value of Debt (in millions, e.g., 15000): ").strip()
                if not debt_input:
                    print("Please enter the book value of debt.")
                    continue

                self.book_value_debt = float(debt_input) * 1_000_000  # Convert to actual value
                print(f"✅ Book Value of Debt: ${self.book_value_debt:,.0f}")
                break

            except ValueError:
                print("Invalid input. Please enter a numeric value.")
                continue

        # Set final values
        self.equity_value = self.book_value_equity
        self.debt_value = self.book_value_debt

        return True

    def calculate_wacc(self):
        """Calculates the Weighted Average Cost of Capital."""
        if any(x is None for x in [self.cost_of_equity, self.cost_of_debt, self.equity_value, self.debt_value]):
            print("❌ Missing required data for WACC calculation")
            return None

        # Calculate total value and weights
        self.total_value = self.equity_value + self.debt_value
        self.weight_equity = self.equity_value / self.total_value
        self.weight_debt = self.debt_value / self.total_value

        # Calculate WACC
        self.wacc = (self.cost_of_equity * self.weight_equity) + (self.cost_of_debt * self.weight_debt)

        print(f"\n{'=' * 60}")
        print("WACC CALCULATION")
        print(f"{'=' * 60}")
        print(f"FORMULA: WACC = (Cost of Equity × Weight of Equity) + (Cost of Debt × Weight of Debt)")
        print(f"{'=' * 60}")

        # Show values used
        valuation_type = "Market" if self.valuation_method == 'market' else "Book"
        print(f"Valuation Method: {valuation_type} Values")
        print(f"Value of Equity:         ${self.equity_value:,.0f}")
        print(f"Value of Debt:           ${self.debt_value:,.0f}")
        print(f"Total Value:             ${self.total_value:,.0f}")
        print(f"")
        print(f"Weight of Equity:        {self.weight_equity:.1%}")
        print(f"Weight of Debt:          {self.weight_debt:.1%}")
        print(f"")
        print(f"Cost of Equity:          {self.cost_of_equity:.2%}")
        print(f"Cost of Debt:            {self.cost_of_debt:.2%}")
        print(f"{'=' * 60}")
        print(f"CALCULATION:")
        print(
            f"WACC = ({self.cost_of_equity:.2%} × {self.weight_equity:.1%}) + ({self.cost_of_debt:.2%} × {self.weight_debt:.1%})")
        print(f"WACC = {self.cost_of_equity * self.weight_equity:.2%} + {self.cost_of_debt * self.weight_debt:.2%}")
        print(f"WACC = {self.wacc:.2%}")
        print(f"{'=' * 60}")

        return self.wacc

    def save_to_excel(self, filename=None):
        """Saves WACC analysis results to Excel file."""
        if self.wacc is None:
            print("❌ No WACC results to save")
            return None

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            company_name = self.ticker if self.ticker else "Company"
            filename = f"wacc_analysis_{company_name}_{timestamp}.xlsx"

        try:
            # Prepare results data
            results_data = {
                'Metric': [
                    'Company_Ticker',
                    'Company_Name',
                    'Valuation_Method',
                    '',
                    'Cost_of_Equity',
                    'Cost_of_Debt',
                    '',
                    'Value_of_Equity',
                    'Value_of_Debt',
                    'Total_Value',
                    '',
                    'Weight_of_Equity',
                    'Weight_of_Debt',
                    '',
                    'WACC_Formula',
                    'Weighted_Average_Cost_of_Capital'
                ],
                'Value': [
                    self.ticker or 'N/A',
                    self.company_info.get('longName', 'N/A') if self.company_info else 'N/A',
                    'Market Values' if self.valuation_method == 'market' else 'Book Values',
                    '',
                    self.cost_of_equity,
                    self.cost_of_debt,
                    '',
                    self.equity_value,
                    self.debt_value,
                    self.total_value,
                    '',
                    self.weight_equity,
                    self.weight_debt,
                    '',
                    '(Cost of Equity × Weight of Equity) + (Cost of Debt × Weight of Debt)',
                    self.wacc
                ],
                'Formatted': [
                    self.ticker or 'N/A',
                    self.company_info.get('longName', 'N/A') if self.company_info else 'N/A',
                    'Market Values' if self.valuation_method == 'market' else 'Book Values',
                    '',
                    f"{self.cost_of_equity:.2%}",
                    f"{self.cost_of_debt:.2%}",
                    '',
                    f"${self.equity_value:,.0f}",
                    f"${self.debt_value:,.0f}",
                    f"${self.total_value:,.0f}",
                    '',
                    f"{self.weight_equity:.1%}",
                    f"{self.weight_debt:.1%}",
                    '',
                    '(Cost of Equity × Weight of Equity) + (Cost of Debt × Weight of Debt)',
                    f"{self.wacc:.2%}"
                ]
            }

            # WACC calculation breakdown
            calculation_data = {
                'Component': [
                    'Equity Component',
                    'Debt Component',
                    'Total WACC'
                ],
                'Calculation': [
                    f'{self.cost_of_equity:.2%} × {self.weight_equity:.1%} = {self.cost_of_equity * self.weight_equity:.2%}',
                    f'{self.cost_of_debt:.2%} × {self.weight_debt:.1%} = {self.cost_of_debt * self.weight_debt:.2%}',
                    f'{self.cost_of_equity * self.weight_equity:.2%} + {self.cost_of_debt * self.weight_debt:.2%} = {self.wacc:.2%}'
                ],
                'Description': [
                    'Equity cost weighted by equity proportion',
                    'Debt cost weighted by debt proportion',
                    'Sum of weighted costs'
                ]
            }

            # Market value of debt details (if applicable)
            if self.valuation_method == 'market' and hasattr(self, 'debt_maturity'):
                pv_interest = (self.interest_expense * (1 - (1 / (1 + self.cost_of_debt)))) / self.cost_of_debt
                pv_principal = (self.debt_value - pv_interest)  # Approximate back-calculation

                debt_valuation_data = {
                    'Component': [
                        'Interest_Expense_Annual',
                        'Debt_Maturity_Years',
                        'Cost_of_Debt_Rate',
                        'PV_of_Interest_Payments',
                        'PV_of_Principal_Repayment',
                        'Market_Value_of_Debt'
                    ],
                    'Value': [
                        self.interest_expense,
                        self.debt_maturity,
                        self.cost_of_debt,
                        pv_interest,
                        self.debt_value - pv_interest,
                        self.debt_value
                    ],
                    'Formatted': [
                        f"${self.interest_expense:,.0f}",
                        f"{self.debt_maturity:.1f} years",
                        f"{self.cost_of_debt:.2%}",
                        f"${pv_interest:,.0f}",
                        f"${self.debt_value - pv_interest:,.0f}",
                        f"${self.debt_value:,.0f}"
                    ]
                }

            # Save to Excel
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Main results
                pd.DataFrame(results_data).to_excel(writer, sheet_name='WACC_Results', index=False)

                # Calculation breakdown
                pd.DataFrame(calculation_data).to_excel(writer, sheet_name='WACC_Calculation', index=False)

                # Market value of debt details (if applicable)
                if self.valuation_method == 'market' and hasattr(self, 'debt_maturity'):
                    pd.DataFrame(debt_valuation_data).to_excel(writer, sheet_name='Debt_Valuation', index=False)

            print(f"✅ WACC analysis saved to: {filename}")
            print(f"📊 Excel file includes:")
            print(f"   • WACC_Results: Complete analysis results")
            print(f"   • WACC_Calculation: Component breakdown")
            if self.valuation_method == 'market':
                print(f"   • Debt_Valuation: Market value of debt calculation")

            return filename

        except Exception as e:
            print(f"❌ Error saving to Excel: {e}")
            return None


# Main integrated function
def integrated_wacc_analysis():
    """
    Integrated WACC Analysis Workflow:
    1. Calculate Beta and Cost of Equity
    2. Calculate Cost of Debt
    3. Calculate WACC
    """

    print("=" * 70)
    print("INTEGRATED WACC ANALYSIS")
    print("=" * 70)
    print("This workflow will guide you through:")
    print("1. Beta calculation and Cost of Equity (CAPM)")
    print("2. Cost of Debt calculation")
    print("3. WACC calculation")
    print("=" * 70)

    try:
        # Store results
        cost_of_equity_result = None
        cost_of_debt_result = None
        company_ticker = None

        # Part 1: Beta and Cost of Equity
        print(f"\n{'=' * 70}")
        print("PART 1: BETA CALCULATION AND COST OF EQUITY")
        print(f"{'=' * 70}")

        # Get user inputs for beta calculation
        print("\nStock ticker examples:")
        print("• Italian stocks: ENI.MI, ENEL.MI, ISP.MI, UCG.MI, TIT.MI")
        print("• US stocks: AAPL, MSFT, GOOGL, TSLA, NVDA")
        print("• European stocks: ASML.AS, SAP.DE, NESN.SW")
        stock_symbol = input("\nEnter stock ticker: ").upper().strip()
        company_ticker = stock_symbol  # Store for later use

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

        # Create beta calculator instance
        beta_calc = BetaCalculator()

        # Load databases
        print("\nLoading databases...")
        beta_calc.load_erp_data(erp_file)
        beta_calc.load_tax_rate_data(erp_file)  # Load tax rate data

        if not beta_calc.load_bond_yield_data(bond_file):
            print("❌ Failed to load bond database. Exiting.")
            return

        # Download market data
        print(f"\nDownloading market data...")
        if not beta_calc.download_data(stock_symbol, index_symbol, period, interval):
            print("❌ Failed to download market data. Exiting.")
            return

        # Calculate beta
        print("\nCalculating beta...")
        if not beta_calc.calculate_returns() or not beta_calc.calculate_beta():
            print("❌ Failed to calculate beta. Exiting.")
            return

        # Show plot
        plot_input = input("\nShow regression plot? (y/n): ").strip().lower()
        if plot_input == 'y':
            beta_calc.plot_regression()

        # Get risk-free rate
        print(f"\n{'=' * 50}")
        print("RISK-FREE RATE SELECTION")
        print(f"{'=' * 50}")

        if beta_calc.select_bond_country_from_excel() is None:
            print("❌ Risk-free rate required. Exiting.")
            return

        # Get ERP
        print(f"\n{'=' * 50}")
        print("EQUITY RISK PREMIUM")
        print(f"{'=' * 50}")

        auto_erp = beta_calc.get_equity_risk_premium(index_symbol)

        if auto_erp:
            use_auto = input(f"\nUse automatic ERP ({auto_erp * 100:.2f}%)? (y/n): ").strip().lower()
            if use_auto == 'n':
                if beta_calc.get_manual_equity_risk_premium() is None:
                    print("❌ ERP required. Exiting.")
                    return
        else:
            if beta_calc.get_manual_equity_risk_premium() is None:
                print("❌ ERP required. Exiting.")
                return

        # Calculate cost of equity
        cost_of_equity_result = beta_calc.calculate_cost_of_equity()

        if cost_of_equity_result:
            print(f"\n{'=' * 60}")
            print(f"🎯 COST OF EQUITY RESULT")
            print(f"{'=' * 60}")
            print(f"Final Cost of Equity: {cost_of_equity_result * 100:.2f}%")
            print(f"{'=' * 60}")

        # Part 2: Cost of Debt
        print(f"\n{'=' * 70}")
        print("PART 2: COST OF DEBT CALCULATION")
        print(f"{'=' * 70}")

        # Create cost of debt calculator instance
        debt_calc = CostOfDebtCalculator()

        # Ratings file
        ratings_file = input("\nSynthetic Ratings file (default: 'Synthetic Ratings.xlsx'): ").strip()
        if not ratings_file:
            ratings_file = "Synthetic Ratings.xlsx"

        # Load databases
        print("\nLoading databases...")
        if not debt_calc.load_synthetic_ratings_data(ratings_file):
            print("❌ Failed to load synthetic ratings. Exiting.")
            return

        # Use the same bond file
        if not debt_calc.load_bond_yield_data(bond_file):
            print("❌ Failed to load bond data. Exiting.")
            return

        # Share tax rate data from beta calculator
        debt_calc.load_tax_rate_data(beta_calc.tax_rate_data)

        # Get company information
        print(f"\nAnalyzing company...")
        if not debt_calc.get_company_info(company_ticker):
            print("❌ Failed to get company information. Exiting.")
            return

        # Get financial inputs
        print(f"\nFinancial analysis...")
        debt_calc.get_financial_inputs()

        # Assign synthetic rating
        print(f"\nCredit rating assignment...")
        debt_calc.assign_synthetic_rating()

        # Select risk-free rate
        print(f"\nRisk-free rate selection...")
        if not debt_calc.select_risk_free_rate():
            print("❌ Risk-free rate required. Exiting.")
            return

        # Get tax rate
        print(f"\nTax considerations...")
        debt_calc.get_tax_rate()

        # Calculate cost of debt
        cost_of_debt_result = debt_calc.calculate_cost_of_debt()

        if cost_of_debt_result:
            print(f"\n🎯 COST OF DEBT RESULT:")
            print(f"After-tax Cost of Debt: {cost_of_debt_result * 100:.2f}%")

        # Part 3: WACC Calculation
        print(f"\n{'=' * 70}")
        print("PART 3: WACC CALCULATION")
        print(f"{'=' * 70}")

        if cost_of_equity_result is None or cost_of_debt_result is None:
            print("❌ Missing cost of equity or cost of debt. Cannot calculate WACC.")
            return

        # Create WACC calculator instance
        wacc_calc = WACCCalculator()

        # Set the calculated costs
        wacc_calc.cost_of_equity = cost_of_equity_result
        wacc_calc.cost_of_debt = cost_of_debt_result

        print(f"\nUsing calculated values:")
        print(f"• Cost of Equity: {cost_of_equity_result * 100:.2f}%")
        print(f"• Cost of Debt: {cost_of_debt_result * 100:.2f}%")

        # Select valuation method
        wacc_calc.select_valuation_method()

        # Get values based on selected method
        if wacc_calc.valuation_method == 'market':
            # Set ticker from earlier
            wacc_calc.ticker = company_ticker
            wacc_calc.company_info = debt_calc.company_info
            wacc_calc.market_value_equity = debt_calc.market_cap

            # Get debt values
            print(f"\nFor market value of debt calculation, please provide:")

            # Get book value of debt
            while True:
                try:
                    debt_input = input("Enter Total Debt from balance sheet (in millions, e.g., 15000): ").strip()
                    if not debt_input:
                        print("Please enter the total debt value.")
                        continue

                    book_debt = float(debt_input) * 1_000_000
                    print(f"✅ Book Value of Debt: ${book_debt:,.0f}")
                    break

                except ValueError:
                    print("Invalid input. Please enter a numeric value.")
                    continue

            # Get weighted average maturity
            while True:
                try:
                    maturity_input = input("Enter Weighted Average Maturity of debt (years, e.g., 8.5): ").strip()
                    if not maturity_input:
                        print("Please enter the average maturity.")
                        continue

                    wacc_calc.debt_maturity = float(maturity_input)
                    if wacc_calc.debt_maturity <= 0:
                        print("Maturity must be greater than 0.")
                        continue

                    print(f"✅ Weighted Average Maturity: {wacc_calc.debt_maturity:.1f} years")
                    break

                except ValueError:
                    print("Invalid input. Please enter a numeric value.")
                    continue

            # Use interest expense from cost of debt calculation
            wacc_calc.interest_expense = debt_calc.interest_expense
            print(f"✅ Using Interest Expense from earlier: ${wacc_calc.interest_expense:,.0f}")

            # Calculate market value of debt
            print(f"\nCalculating market value of debt...")
            pv_interest = (wacc_calc.interest_expense * (
                    1 - (1 / (1 + wacc_calc.cost_of_debt)))) / wacc_calc.cost_of_debt
            pv_principal = book_debt / ((1 + wacc_calc.cost_of_debt) ** wacc_calc.debt_maturity)
            wacc_calc.market_value_debt = pv_interest + pv_principal

            print(f"\n📊 Market Value of Debt Calculation:")
            print(f"   • PV of Interest Payments: ${pv_interest:,.0f}")
            print(f"   • PV of Principal Repayment: ${pv_principal:,.0f}")
            print(f"   • Total Market Value of Debt: ${wacc_calc.market_value_debt:,.0f}")

            # Set final values
            wacc_calc.equity_value = wacc_calc.market_value_equity
            wacc_calc.debt_value = wacc_calc.market_value_debt
        else:
            if not wacc_calc.get_book_values():
                print("❌ Failed to get book values. Exiting.")
                return

        # Calculate WACC
        wacc_result = wacc_calc.calculate_wacc()

        if wacc_result:
            print(f"\n{'=' * 60}")
            print("🎯 FINAL INTEGRATED RESULTS")
            print(f"{'=' * 60}")
            print(f"Cost of Equity:  {cost_of_equity_result * 100:.2f}%")
            print(f"Cost of Debt:    {cost_of_debt_result * 100:.2f}%")
            print(f"WACC:            {wacc_result * 100:.2f}%")
            print(f"{'=' * 60}")
            print(f"This WACC can be used as the discount rate for:")
            print(f"• DCF valuation models")
            print(f"• Investment project evaluation")
            print(f"• Corporate financial planning")
            print(f"• Performance measurement")
            print(f"{'=' * 60}")

        # Save all results to a single Excel file
        save_input = input(f"\nSave all analysis results to Excel? (y/n): ").strip().lower()
        if save_input == 'y':
            print("\nSaving comprehensive analysis to Excel...")

            # Create filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"complete_wacc_analysis_{company_ticker}_{timestamp}.xlsx"

            try:
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    # 1. Summary Sheet with all key results
                    summary_data = {
                        'Analysis Component': [
                            'FINAL RESULTS',
                            'Cost of Equity (CAPM)',
                            'Cost of Debt (After-tax)',
                            'WACC',
                            '',
                            'BETA ANALYSIS',
                            'Stock Ticker',
                            'Index Ticker',
                            'Beta',
                            'R-squared',
                            'Risk-free Rate',
                            'Equity Risk Premium',
                            'Country',
                            '',
                            'COST OF DEBT ANALYSIS',
                            'Company Name',
                            'Market Cap',
                            'EBIT',
                            'Interest Expense',
                            'Interest Coverage Ratio',
                            'Synthetic Rating',
                            'Credit Spread',
                            'Tax Rate',
                            '',
                            'WACC COMPONENTS',
                            'Valuation Method',
                            'Value of Equity',
                            'Value of Debt',
                            'Weight of Equity',
                            'Weight of Debt'
                        ],
                        'Value': [
                            '',
                            cost_of_equity_result,
                            cost_of_debt_result,
                            wacc_result,
                            '',
                            '',
                            stock_symbol,
                            index_symbol,
                            beta_calc.beta,
                            beta_calc.r_squared,
                            beta_calc.current_risk_free_rate,
                            beta_calc.equity_risk_premium,
                            beta_calc.detected_country or 'N/A',
                            '',
                            '',
                            debt_calc.company_info.get('longName', company_ticker),
                            debt_calc.market_cap,
                            debt_calc.ebit,
                            debt_calc.interest_expense,
                            debt_calc.interest_coverage_ratio,
                            debt_calc.rating,
                            debt_calc.spread,
                            debt_calc.tax_rate,
                            '',
                            '',
                            wacc_calc.valuation_method.capitalize(),
                            wacc_calc.equity_value,
                            wacc_calc.debt_value,
                            wacc_calc.weight_equity,
                            wacc_calc.weight_debt
                        ],
                        'Formatted': [
                            '',
                            f"{cost_of_equity_result * 100:.2f}%",
                            f"{cost_of_debt_result * 100:.2f}%",
                            f"{wacc_result * 100:.2f}%",
                            '',
                            '',
                            stock_symbol,
                            index_symbol,
                            f"{beta_calc.beta:.4f}",
                            f"{beta_calc.r_squared:.4f}",
                            f"{beta_calc.current_risk_free_rate * 100:.2f}%",
                            f"{beta_calc.equity_risk_premium * 100:.2f}%",
                            beta_calc.detected_country or 'N/A',
                            '',
                            '',
                            debt_calc.company_info.get('longName', company_ticker),
                            f"${debt_calc.market_cap:,.0f}",
                            f"${debt_calc.ebit:,.0f}",
                            f"${debt_calc.interest_expense:,.0f}",
                            f"{debt_calc.interest_coverage_ratio:.2f}",
                            debt_calc.rating,
                            f"{debt_calc.spread * 100:.2f}%",
                            f"{debt_calc.tax_rate * 100:.1f}%",
                            '',
                            '',
                            wacc_calc.valuation_method.capitalize(),
                            f"${wacc_calc.equity_value:,.0f}",
                            f"${wacc_calc.debt_value:,.0f}",
                            f"{wacc_calc.weight_equity:.1%}",
                            f"{wacc_calc.weight_debt:.1%}"
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

                    # 2. Beta Analysis Results
                    beta_results = {
                        'Metric': ['Beta', 'Alpha', 'R_squared', 'Correlation', 'Standard_Error',
                                   'T_Statistic', 'P_Value', 'Observations'],
                        'Value': [beta_calc.beta, beta_calc.alpha, beta_calc.r_squared,
                                  beta_calc.correlation, beta_calc.std_error, beta_calc.t_statistic,
                                  beta_calc.p_value, len(beta_calc.stock_returns)],
                        'Formatted': [f"{beta_calc.beta:.4f}", f"{beta_calc.alpha:.6f}",
                                      f"{beta_calc.r_squared:.4f}", f"{beta_calc.correlation:.4f}",
                                      f"{beta_calc.std_error:.6f}", f"{beta_calc.t_statistic:.4f}",
                                      f"{beta_calc.p_value:.6f}", len(beta_calc.stock_returns)]
                    }
                    pd.DataFrame(beta_results).to_excel(writer, sheet_name='Beta_Statistics', index=False)

                    # 3. CAPM Calculation
                    capm_calc = {
                        'Step': ['Formula', 'Risk-free Rate (Rf)', 'Beta (β)', 'Equity Risk Premium (ERP)',
                                 'Calculation', 'Result'],
                        'Detail': ['Re = Rf + β × ERP',
                                   f'{beta_calc.current_risk_free_rate * 100:.2f}%',
                                   f'{beta_calc.beta:.4f}',
                                   f'{beta_calc.equity_risk_premium * 100:.2f}%',
                                   f'{beta_calc.current_risk_free_rate * 100:.2f}% + {beta_calc.beta:.4f} × {beta_calc.equity_risk_premium * 100:.2f}%',
                                   f'{cost_of_equity_result * 100:.2f}%']
                    }
                    pd.DataFrame(capm_calc).to_excel(writer, sheet_name='CAPM_Calculation', index=False)

                    # 4. Cost of Debt Calculation
                    cod_calc = {
                        'Step': ['Formula', 'Risk-free Rate', 'Credit Spread', 'Pre-tax Cost',
                                 'Tax Rate', 'Tax Shield', 'Result'],
                        'Detail': ['Cost of Debt = (Rf + Spread) × (1 - Tax Rate)',
                                   f'{debt_calc.risk_free_rate * 100:.2f}%',
                                   f'{debt_calc.spread * 100:.2f}%',
                                   f'{(debt_calc.risk_free_rate + debt_calc.spread) * 100:.2f}%',
                                   f'{debt_calc.tax_rate * 100:.1f}%',
                                   f'{(1 - debt_calc.tax_rate) * 100:.1f}%',
                                   f'{cost_of_debt_result * 100:.2f}%']
                    }
                    pd.DataFrame(cod_calc).to_excel(writer, sheet_name='Cost_of_Debt_Calc', index=False)

                    # 5. WACC Calculation
                    wacc_calc_data = {
                        'Component': ['Formula', 'Equity Component', 'Debt Component', 'Total WACC'],
                        'Calculation': [
                            'WACC = (Cost of Equity × Weight of Equity) + (Cost of Debt × Weight of Debt)',
                            f'{wacc_calc.cost_of_equity * 100:.2f}% × {wacc_calc.weight_equity:.1%} = {(wacc_calc.cost_of_equity * wacc_calc.weight_equity) * 100:.2f}%',
                            f'{wacc_calc.cost_of_debt * 100:.2f}% × {wacc_calc.weight_debt:.1%} = {(wacc_calc.cost_of_debt * wacc_calc.weight_debt) * 100:.2f}%',
                            f'{(wacc_calc.cost_of_equity * wacc_calc.weight_equity) * 100:.2f}% + {(wacc_calc.cost_of_debt * wacc_calc.weight_debt) * 100:.2f}% = {wacc_result * 100:.2f}%'
                        ]
                    }
                    pd.DataFrame(wacc_calc_data).to_excel(writer, sheet_name='WACC_Calculation', index=False)

                    # 6. Historical Data - Stock Prices
                    if beta_calc.stock_data is not None:
                        beta_calc.stock_data.to_excel(writer, sheet_name='Stock_Prices')

                    # 7. Historical Data - Index Prices
                    if beta_calc.index_data is not None:
                        beta_calc.index_data.to_excel(writer, sheet_name='Index_Prices')

                    # 8. Returns Data
                    if beta_calc.stock_returns is not None and beta_calc.index_returns is not None:
                        returns_df = pd.DataFrame({
                            'Date': beta_calc.stock_returns.index,
                            'Stock_Returns': beta_calc.stock_returns.values,
                            'Index_Returns': beta_calc.index_returns.values,
                            'Stock_Returns_%': beta_calc.stock_returns.values * 100,
                            'Index_Returns_%': beta_calc.index_returns.values * 100
                        })
                        returns_df.to_excel(writer, sheet_name='Returns_Data', index=False)

                print(f"\n✅ Comprehensive analysis saved to: {filename}")
                print(f"\n📊 Excel file contains:")
                print(f"   • Summary: All key results in one view")
                print(f"   • Beta_Statistics: Detailed regression statistics")
                print(f"   • CAMP_Calculation: Cost of equity breakdown")
                print(f"   • Cost_of_Debt_Calc: Cost of debt breakdown")
                print(f"   • WACC_Calculation: Final WACC computation")
                print(f"   • Stock_Prices: Historical stock data")
                print(f"   • Index_Prices: Historical index data")
                print(f"   • Returns_Data: Calculated returns")

            except Exception as e:
                print(f"❌ Error saving to Excel: {e}")

        print(f"\n{'=' * 70}")
        print("INTEGRATED WACC ANALYSIS COMPLETED!")
        print(f"{'=' * 70}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Please check your inputs and try again.")


if __name__ == "__main__":
    integrated_wacc_analysis()
