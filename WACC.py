import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


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


def main():
    """Main function to run WACC calculation."""
    try:
        calc = WACCCalculator()

        # Step 1: Get cost inputs
        calc.get_cost_inputs()

        # Step 2: Select valuation method
        calc.select_valuation_method()

        # Step 3: Get values based on selected method
        if calc.valuation_method == 'market':
            if not calc.get_market_values():
                print("❌ Failed to get market values. Exiting.")
                return
        else:
            if not calc.get_book_values():
                print("❌ Failed to get book values. Exiting.")
                return

        # Step 4: Calculate WACC
        wacc_result = calc.calculate_wacc()

        if wacc_result:
            print(f"\n{'=' * 60}")
            print("🎯 FINAL WACC RESULT")
            print(f"{'=' * 60}")
            print(f"Weighted Average Cost of Capital: {wacc_result:.2%}")
            print(f"{'=' * 60}")
            print(f"This WACC can be used as the discount rate for:")
            print(f"• DCF valuation models")
            print(f"• Investment project evaluation")
            print(f"• Corporate financial planning")
            print(f"• Performance measurement")
            print(f"{'=' * 60}")

        # Step 5: Save results
        save_input = input(f"\nSave WACC analysis to Excel? (y/n): ").strip().lower()
        if save_input == 'y':
            calc.save_to_excel()

        print(f"\n{'=' * 60}")
        print("WACC ANALYSIS COMPLETED!")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Please check your inputs and try again.")


if __name__ == "__main__":
    main()