#!/usr/bin/env python3
"""
Valuation Model Selection Tool
Based on Aswath Damodaran's model selection framework
"""


class ValuationModelSelector:
    def __init__(self):
        self.inputs = {}
        self.outputs = {}

    def get_user_inputs(self):
        """Collect all necessary inputs from the user"""
        print("=" * 60)
        print("CHOOSING THE RIGHT VALUATION MODEL")
        print("This program helps you choose the right valuation model")
        print("=" * 60)
        print()

        # Basic question about earnings
        self.inputs['earnings_positive'] = self.get_yes_no_input(
            "Are your earnings positive?"
        )

        if self.inputs['earnings_positive']:
            self._get_positive_earnings_inputs()
        else:
            self._get_negative_earnings_inputs()

        # Financial leverage questions (asked for all companies)
        self._get_leverage_inputs()

        # Dividend and cash flow questions
        self._get_cash_flow_inputs()

    def _get_positive_earnings_inputs(self):
        """Get inputs for companies with positive earnings"""
        print("\nFor positive earnings companies:")

        self.inputs['inflation_rate'] = self.get_percentage_input(
            "What is the expected inflation rate in the economy? (It can be estimated from economists' forecasts.)",
            default=3.0
        )

        self.inputs['real_growth_rate'] = self.get_percentage_input(
            "What is the expected real growth rate in the economy?",
            default=2.0
        )

        self.inputs['expected_growth_rate'] = self.get_percentage_input(
            "What is the expected growth rate in earnings (revenues) for this firm in the near future?",
            default=15.0
        )

        self.inputs['competitive_advantage'] = self.get_yes_no_input(
            "Does this firm have a significant and sustainable advantage over competitors?"
        )

    def _get_negative_earnings_inputs(self):
        """Get inputs for companies with negative earnings"""
        print("\nFor negative earnings companies:")

        self.inputs['inflation_rate'] = self.get_percentage_input(
            "What is the expected inflation rate in the economy? (It can be estimated from economists' forecasts.)",
            default=3.0
        )

        self.inputs['real_growth_rate'] = self.get_percentage_input(
            "What is the expected real growth rate in the economy?",
            default=2.0
        )

        self.inputs['expected_growth_rate'] = self.get_percentage_input(
            "What is the expected growth rate in earnings (revenues) for this firm in the near future?",
            default=15.0
        )

        self.inputs['competitive_advantage'] = self.get_yes_no_input(
            "Does this firm have a significant and sustainable advantage over competitors?"
        )

        self.inputs['cyclical_business'] = self.get_yes_no_input(
            "Are the earnings negative because the firm is in a cyclical business?"
        )

        self.inputs['temporary_occurrence'] = self.get_yes_no_input(
            "Are the earnings negative because of a one-time or temporary occurrence?"
        )

        self.inputs['too_much_debt'] = self.get_yes_no_input(
            "Are the earnings negative because the firm has too much debt?"
        )

        if self.inputs['too_much_debt']:
            self.inputs['bankruptcy_likely'] = self.get_yes_no_input(
                "   If yes, is there a strong likelihood of bankruptcy?"
            )
        else:
            self.inputs['bankruptcy_likely'] = False

        self.inputs['startup'] = self.get_yes_no_input(
            "Are the earnings negative because the firm is just starting up?"
        )

    def _get_leverage_inputs(self):
        """Get financial leverage inputs"""
        print("\nFinancial Leverage:")

        self.inputs['debt_ratio'] = self.get_percentage_input(
            "What is the current debt ratio (in market value terms)?",
            default=4.0
        )

        self.inputs['debt_ratio_change'] = self.get_yes_no_input(
            "Is this debt ratio expected to change significantly?"
        )

    def _get_cash_flow_inputs(self):
        """Get dividend and cash flow inputs"""
        print("\nDividend Policy:")

        self.inputs['dividends'] = self.get_currency_input(
            "What did the firm pay out as dividends in the current year?",
            default=100.0
        )

        self.inputs['can_estimate_capex'] = self.get_yes_no_input(
            "Can you estimate capital expenditures and working capital requirements?"
        )

        if self.inputs['can_estimate_capex']:
            print("\nEnter the following inputs for computing FCFE:")

            self.inputs['net_income'] = self.get_currency_input(
                "Net Income (NI):",
                default=200.0
            )

            self.inputs['depreciation'] = self.get_currency_input(
                "Depreciation and Amortization:",
                default=50.0
            )

            self.inputs['capital_spending'] = self.get_currency_input(
                "Capital Spending (Including acquisitions):",
                default=100.0
            )

            self.inputs['working_capital_change'] = self.get_currency_input(
                "Change in Non-cash Working Capital:",
                default=25.0
            )

    def calculate_outputs(self):
        """Calculate the recommended valuation approach based on inputs"""

        # Calculate FCFE if applicable
        if self.inputs.get('can_estimate_capex'):
            debt_ratio = self.inputs['debt_ratio'] / 100.0
            self.outputs['fcfe'] = (
                    self.inputs['net_income'] -
                    (self.inputs['capital_spending'] - self.inputs['depreciation']) * (1 - debt_ratio) -
                    self.inputs['working_capital_change'] * (1 - debt_ratio)
            )
        else:
            self.outputs['fcfe'] = self.inputs['dividends']

        # Determine model type
        if self.inputs.get('bankruptcy_likely', False):
            self.outputs['model_type'] = "Option Pricing Model"
        else:
            self.outputs['model_type'] = "Discounted CF Model"

        # Determine earnings level
        if not self.inputs['earnings_positive']:
            if self.inputs.get('cyclical_business') or self.inputs.get('temporary_occurrence'):
                self.outputs['earnings_level'] = "Normalized Earnings"
            else:
                self.outputs['earnings_level'] = "Current Earnings"
        else:
            self.outputs['earnings_level'] = "Current Earnings"

        # Determine cash flow type
        self.outputs['cash_flow_type'] = self._determine_cash_flow_type()

        # Determine growth pattern
        self.outputs['growth_pattern'] = self._determine_growth_pattern()

        # Determine growth period length
        self.outputs['growth_period'] = self._determine_growth_period()

    def _determine_cash_flow_type(self):
        """Determine which cash flow measure to use"""
        if self.inputs['debt_ratio_change']:
            if self.inputs['can_estimate_capex']:
                return "FCFF (Value firm)"
            else:
                return "Dividends (Value equity)"
        else:
            if not self.inputs['earnings_positive']:
                return "FCFF (Value firm)"
            else:
                # Compare dividends to FCFE
                if self.inputs['dividends'] > 1.25 * self.outputs['fcfe']:
                    return "FCFE (Value equity)"
                elif self.inputs['dividends'] < 0.9 * self.outputs['fcfe']:
                    return "FCFE (Value equity)"
                else:
                    return "Dividends (Value equity)"

    def _determine_growth_pattern(self):
        """Determine the appropriate growth pattern"""
        growth_rate = self.inputs['expected_growth_rate'] / 100.0
        economy_growth = (self.inputs['inflation_rate'] + self.inputs['real_growth_rate']) / 100.0
        if not self.inputs['earnings_positive']:
            if self.outputs['earnings_level'] == "Normalized Earnings":
                if growth_rate < economy_growth + 0.0101:
                    return "Stable Growth"
                elif growth_rate < economy_growth + 0.06:
                    return "Two-stage Growth"
                else:
                    return "Three-stage Growth"
        else:
            if growth_rate < economy_growth + 0.0101:
                return "Stable Growth"
            elif growth_rate < economy_growth + 0.06:
                return "Two-stage Growth"
            else:
                return "Three-stage Growth"

    def _determine_growth_period(self):
        """Determine the length of the growth period"""
        growth_rate = self.inputs['expected_growth_rate'] / 100.0
        economy_growth = (self.inputs['inflation_rate'] + self.inputs['real_growth_rate']) / 100.0
        if self.outputs['growth_pattern'] == "Stable Growth":
            return "No high growth period"
        else:
            if self.inputs['earnings_positive']:
                if self.inputs['competitive_advantage']:
                   if growth_rate > economy_growth + 0.06:
                      return "10 or more years"
                   else:
                      return "5 to 10 years"
            else:
                if growth_rate > economy_growth + 0.06:
                    return "5 to 10 years"
                else:
                    return "5 years or less"

    def display_results(self):
        """Display the recommended valuation approach"""
        print("\n" + "=" * 60)
        print("OUTPUT FROM THE MODEL")
        print("Based on your inputs, the recommended valuation approach is:")
        print("=" * 60)

        print(f"\nType of Model: {self.outputs['model_type']}")
        if self.outputs['model_type'] == "Option Pricing Model":
            print("  ! Note: First do a DCF valuation before applying option pricing")

        print(f"\nLevel of Earnings to use: {self.outputs['earnings_level']}")

        print(f"\nCashflows to discount: {self.outputs['cash_flow_type']}")

        print(f"\nGrowth Pattern: {self.outputs['growth_pattern']}")
        if self.outputs['growth_pattern'] == "n-stage model":
            print("  ! In an n-stage model, estimate target operating margins (if valuing firm)")
            print("    or net margins (if valuing equity) and revenue growth each year")

        print(f"\nLength of Growth Period: {self.outputs['growth_period']}")

        if self.inputs.get('can_estimate_capex'):
            print(f"\nCalculated FCFE: ${self.outputs['fcfe']:.2f}")

        print("\n" + "=" * 60)

    # Helper methods for input collection
    def get_yes_no_input(self, prompt):
        """Get a yes/no input from user"""
        while True:
            response = input(f"{prompt} (Yes/No): ").strip().lower()
            if response in ['yes', 'y']:
                return True
            elif response in ['no', 'n']:
                return False
            else:
                print("Please enter Yes or No")

    def get_percentage_input(self, prompt, default=None):
        """Get a percentage input from user"""
        if default is not None:
            prompt += f" [Default: {default}%]"

        while True:
            response = input(f"{prompt}: ").strip()
            if response == "" and default is not None:
                return default
            try:
                value = float(response)
                if 0 <= value <= 100:
                    return value
                else:
                    print("Please enter a value between 0 and 100")
            except ValueError:
                print("Please enter a valid number")

    def get_currency_input(self, prompt, default=None):
        """Get a currency amount input from user"""
        if default is not None:
            prompt += f" [Default: ${default}]"

        while True:
            response = input(f"{prompt}: ").strip()
            if response == "" and default is not None:
                return default
            try:
                # Remove $ sign if present
                response = response.replace('$', '').replace(',', '')
                return float(response)
            except ValueError:
                print("Please enter a valid number")


def main():
    """Main function to run the valuation model selector"""
    selector = ValuationModelSelector()

    # Get inputs from user
    selector.get_user_inputs()

    # Calculate recommended approach
    selector.calculate_outputs()

    # Display results
    selector.display_results()

    # Option to save results
    save_results = input("\nWould you like to save these results to a file? (Yes/No): ").strip().lower()
    if save_results in ['yes', 'y']:
        filename = input("Enter filename (without extension): ").strip()
        if not filename:
            filename = "valuation_model_results"

        with open(f"{filename}.txt", 'w') as f:
            f.write("VALUATION MODEL SELECTION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write("INPUTS:\n")
            for key, value in selector.inputs.items():
                f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
            f.write("\nOUTPUTS:\n")
            for key, value in selector.outputs.items():
                f.write(f"  {key.replace('_', ' ').title()}: {value}\n")

        print(f"\nResults saved to {filename}.txt")


if __name__ == "__main__":
    main()