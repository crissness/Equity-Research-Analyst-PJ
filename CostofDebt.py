import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


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
        """Gets marginal tax rate from user."""
        print(f"\n{'=' * 50}")
        print("MARGINAL TAX RATE")
        print(f"{'=' * 50}")
        print("Enter the marginal tax rate for the company.")
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


def get_user_input():
    """Collects user input for cost of debt analysis."""
    print("=" * 70)
    print("COST OF DEBT CALCULATOR")
    print("=" * 70)
    print("This calculator determines the after-tax cost of debt using:")
    print("• Company financials and synthetic credit rating")
    print("• Risk-free rate from government bonds")
    print("• Tax shield from interest deductibility")
    print("=" * 70)

    # Company ticker
    print("\nCompany ticker examples:")
    print("• US companies: AAPL, MSFT, JPM, BAC")
    print("• European companies: ASML.AS, SAP.DE, ENI.MI")
    ticker = input("\nEnter company ticker: ").strip()

    # File paths
    ratings_file = input("\nSynthetic Ratings file (default: 'Synthetic Ratings.xlsx'): ").strip()
    if not ratings_file:
        ratings_file = "Synthetic Ratings.xlsx"

    bond_file = input("Bond yields file (default: 'Bond.xlsx'): ").strip()
    if not bond_file:
        bond_file = "Bond.xlsx"

    return ticker, ratings_file, bond_file


if __name__ == "__main__":
    try:
        ticker, ratings_file, bond_file = get_user_input()
        calc = CostOfDebtCalculator()

        print(f"\n{'=' * 70}")
        print("COST OF DEBT ANALYSIS")
        print(f"{'=' * 70}")

        # Load databases
        print("Step 1: Loading databases...")
        if not calc.load_synthetic_ratings_data(ratings_file):
            print("❌ Failed to load synthetic ratings. Exiting.")
            exit(1)

        if not calc.load_bond_yield_data(bond_file):
            print("❌ Failed to load bond data. Exiting.")
            exit(1)

        # Get company information
        print(f"\nStep 2: Analyzing company...")
        if not calc.get_company_info(ticker):
            print("❌ Failed to get company information. Exiting.")
            exit(1)

        # Get financial inputs
        print(f"\nStep 3: Financial analysis...")
        calc.get_financial_inputs()

        # Assign synthetic rating
        print(f"\nStep 4: Credit rating assignment...")
        calc.assign_synthetic_rating()

        # Select risk-free rate
        print(f"\nStep 5: Risk-free rate selection...")
        if not calc.select_risk_free_rate():
            print("❌ Risk-free rate required. Exiting.")
            exit(1)

        # Get tax rate
        print(f"\nStep 6: Tax considerations...")
        calc.get_tax_rate()

        # Calculate cost of debt
        print(f"\nStep 7: Final calculation...")
        cost_of_debt = calc.calculate_cost_of_debt()

        if cost_of_debt:
            print(f"\n🎯 FINAL COST OF DEBT RESULT:")
            print(f"After-tax Cost of Debt: {cost_of_debt * 100:.2f}%")
            print(f"This rate represents the company's cost of borrowing")
            print(f"and can be used in WACC calculations.")

        # Save results
        save_input = input(f"\nSave results to Excel? (y/n): ").strip().lower()
        if save_input == 'y':
            calc.save_to_excel()

        print(f"\n{'=' * 70}")
        print("COST OF DEBT ANALYSIS COMPLETED!")
        print(f"{'=' * 70}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Please check your inputs and try again.")