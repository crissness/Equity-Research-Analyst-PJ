#!/usr/bin/env python3
"""
Two-Stage Dividend Discount Model
Based on the dividend discount model for valuing equity in firms with two stages of growth
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


class TwoStageDDM:
    def __init__(self):
        self.inputs = {}
        self.outputs = {}
        self.warnings = []

    def display_intro(self):
        """Display the introduction and assumptions of the model"""
        print("=" * 60)
        print("TWO-STAGE DIVIDEND DISCOUNT MODEL")
        print("=" * 60)
        print("\nThis model is designed to value the equity in a firm, with two stages")
        print("of growth, an initial period of higher growth and a subsequent period")
        print("of stable growth.")
        print("\nAssumptions:")
        print("1. The firm is expected to grow at a higher growth rate in the first period.")
        print("2. The growth rate will drop at the end of the first period to the stable growth rate.")
        print("3. The dividend payout ratio is consistent with the expected growth rate.")
        print("\n" + "=" * 60)

    def get_user_inputs(self):
        """Collect all necessary inputs from the user"""
        print("\nThe user has to define the following inputs:")
        print("1. Length of high growth period")
        print("2. Expected growth rate in earnings during the high growth period.")
        print("3. Dividend payout ratio during the high growth period.")
        print("4. Expected growth rate in earnings during the stable growth period.")
        print("5. Expected payout ratio during the stable growth period.")
        print("6. Current Earnings per share")
        print("7. Inputs for the Cost of Equity")

        print("\nInputs to the model:")

        # Basic inputs
        self.inputs['current_eps'] = self.get_currency_input(
            "Current Earnings per share",
            default=5.49
        )

        # Get current dividends directly from user
        self.inputs['current_dividends'] = self.get_currency_input(
            "Current Dividends per share",
            default=2.31
        )

        # Growth period length
        self.inputs['growth_period'] = int(self.get_float_input(
            "Enter length of extraordinary growth period (years)",
            default=5
        ))

        # Cost of equity
        self._get_cost_of_equity_inputs()

        # Growth rate estimation
        self._get_growth_rate_inputs()

        # Stable period inputs
        self.inputs['stable_growth_rate'] = self.get_percentage_input(
            "\nEnter growth rate in stable growth period",
            default=8.0
        ) / 100.0

        # Calculate stable payout ratio from fundamentals
        self.outputs['stable_payout_fundamental'] = 1 - self.inputs['stable_growth_rate'] / self.stable_roe

        print(f"Stable payout ratio from fundamentals is = {self.outputs['stable_payout_fundamental']:.2%}")

        change_payout = self.get_yes_no_input(
            "Do you want to change this payout ratio?"
        )

        if change_payout:
            self.inputs['stable_payout'] = self.get_percentage_input(
                "Enter the stable payout ratio",
                default=70.0
            ) / 100.0
        else:
            self.inputs['stable_payout'] = self.outputs['stable_payout_fundamental']

        # Check if user wants different beta for stable period
        self.inputs['stable_beta_different'] = self.get_yes_no_input(
            "\nDo you want to use a different beta for the stable period?"
        )

        if self.inputs['stable_beta_different']:
            self.inputs['stable_beta'] = self.get_float_input(
                "Enter beta for stable period",
                default=1.0
            )
        else:
            self.inputs['stable_beta'] = self.inputs.get('beta', 1.0)

    def _get_cost_of_equity_inputs(self):
        """Get cost of equity inputs"""
        use_direct = self.get_yes_no_input(
            "Do you want to enter cost of equity directly?"
        )

        if use_direct:
            self.inputs['cost_of_equity'] = self.get_percentage_input(
                "Enter the cost of equity",
                default=12.225
            ) / 100.0
            self.inputs['use_capm'] = False

            # Still need beta for potential stable period calculation
            print("\nNote: Beta is still needed for potential stable period calculations.")
            self.inputs['beta'] = self.get_float_input(
                "Beta of the stock",
                default=0.95
            )

        else:
            print("\nEnter the inputs to the cost of equity:")
            self.inputs['beta'] = self.get_float_input(
                "Beta of the stock",
                default=0.95
            )
            self.inputs['risk_free_rate'] = self.get_percentage_input(
                "Riskfree rate",
                default=7.0
            ) / 100.0
            self.inputs['risk_premium'] = self.get_percentage_input(
                "Risk Premium",
                default=5.5
            ) / 100.0
            self.inputs['use_capm'] = True

    def _get_growth_rate_inputs(self):
        """Get inputs for growth rate estimation"""
        print("\n" + "-" * 40)
        print("Growth Rate Estimation")
        print("-" * 40)

        # Historical growth
        use_historical = self.get_yes_no_input(
            "Do you want to use the historical growth rate?"
        )

        if use_historical:
            self.inputs['eps_5_years_ago'] = self.get_currency_input(
                "Enter EPS from five years ago",
                default=1.62
            )
            self.inputs['use_historical'] = True
        else:
            self.inputs['use_historical'] = False
            self.inputs['eps_5_years_ago'] = 0

        # Outside estimate
        use_outside = self.get_yes_no_input(
            "\nDo you have an outside estimate of growth?"
        )

        if use_outside:
            self.inputs['outside_growth'] = self.get_percentage_input(
                "Enter the estimated growth",
                default=17.0
            ) / 100.0
            self.inputs['use_outside'] = True
        else:
            self.inputs['use_outside'] = False
            self.inputs['outside_growth'] = 0

        # ALWAYS get fundamental growth inputs (no longer asking if user wants it)
        print("\nFundamental Growth Rate Calculation:"
              "\nIf you don't want to use it assign it a 0% weight.")
        print("Enter the following inputs:")

        self.inputs['net_income'] = self.get_currency_input(
            "Net Income Currently",
            default=2122
        )
        self.inputs['book_value_current'] = self.get_currency_input(
            "Book Value of Equity (Current)",
            default=6237
        )
        self.inputs['book_value_last'] = self.get_currency_input(
            "Book Value of Equity (Last year)",
            default=6155
        )
        self.inputs['tax_rate'] = self.get_percentage_input(
            "Tax Rate on Income",
            default=40.0
        ) / 100.0

        # Calculate ROE and Retention
        self.outputs['roe'] = self.inputs['net_income'] / self.inputs['book_value_last']
        self.outputs['retention'] = 1 - self.inputs['current_dividends'] / self.inputs['current_eps']

        print(f"\nThe following will be the inputs to the fundamental growth formulation:")
        print(f"ROE = {self.outputs['roe']:.2%}")
        print(f"Retention = {self.outputs['retention']:.2%}")

        # Ask if user wants to change these for high growth period
        change_high = self.get_yes_no_input(
            "\nDo you want to change any of these inputs for the high growth period?"
        )

        if change_high:
            print("Please enter all variables:")
            self.inputs['high_roe'] = self.get_percentage_input(
                "ROE",
                default=self.outputs['roe'] * 100
            ) / 100.0
            self.inputs['high_retention'] = self.get_percentage_input(
                "Retention",
                default=self.outputs['retention'] * 100
            ) / 100.0
        else:
            self.inputs['high_roe'] = self.outputs['roe']
            self.inputs['high_retention'] = self.outputs['retention']

        # Ask if user wants to change these for stable period
        change_stable = self.get_yes_no_input(
            "\nDo you want to change any of these inputs for the stable growth period?"
        )

        if change_stable:
            print("Please enter all variables:")
            self.stable_roe = self.get_percentage_input(
                "ROE",
                default=self.outputs['roe'] * 100
            ) / 100.0
        else:
            self.stable_roe = self.inputs['high_roe']

        # Fundamental growth is always available now
        self.inputs['use_fundamental'] = True

        # Get weights for growth rates
        print("\nSpecify weights to be assigned to each of these growth rates:")
        print("Note: All weights must add up to exactly 100%")

        weights_valid = False
        while not weights_valid:
            # Always ask for all three weights
            print("\nEnter the weights for each growth rate method:")

            # Historical weight
            if self.inputs['use_historical']:
                self.inputs['weight_historical'] = self.get_percentage_input(
                    "Historical Growth Rate",
                    default=33.3
                ) / 100.0
            else:
                print("Historical Growth Rate: 0.0% (not selected)")
                self.inputs['weight_historical'] = 0.0

            # Outside estimate weight
            if self.inputs['use_outside']:
                self.inputs['weight_outside'] = self.get_percentage_input(
                    "Outside Prediction of Growth",
                    default=33.3
                ) / 100.0
            else:
                print("Outside Prediction of Growth: 0.0% (not selected)")
                self.inputs['weight_outside'] = 0.0

            # Fundamental weight (always available now)
            self.inputs['weight_fundamental'] = self.get_percentage_input(
                "Fundamental Estimate of Growth",
                default=33.4
            ) / 100.0

            # Check if weights add up to 100%
            total_weight = (self.inputs['weight_historical'] +
                            self.inputs['weight_outside'] +
                            self.inputs['weight_fundamental'])

            print(f"\nWeight Summary:")
            print(f"Historical Growth Rate: {self.inputs['weight_historical']:.1%}")
            print(f"Outside Prediction of Growth: {self.inputs['weight_outside']:.1%}")
            print(f"Fundamental Estimate of Growth: {self.inputs['weight_fundamental']:.1%}")
            print(f"Total Weight: {total_weight:.1%}")

            if abs(total_weight - 1.0) <= 0.001:  # Allow for small rounding errors
                weights_valid = True
                print("✓ Weights are valid (add up to 100%)")
            else:
                print(f"❌ ERROR: Weights do not add up to 100% (current total: {total_weight:.1%})")
                print("Please re-enter all weights. They must sum to exactly 100%.")

                retry = self.get_yes_no_input("Do you want to try again?")
                if not retry:
                    print("Setting equal weights as default...")
                    # Count available methods
                    available_methods = sum([self.inputs['use_historical'],
                                             self.inputs['use_outside'],
                                             True])  # Fundamental is always available

                    if available_methods == 1:
                        # Only fundamental is available
                        self.inputs['weight_historical'] = 0.0
                        self.inputs['weight_outside'] = 0.0
                        self.inputs['weight_fundamental'] = 1.0
                    elif available_methods == 2:
                        # Two methods available
                        if self.inputs['use_historical'] and self.inputs['use_outside']:
                            self.inputs['weight_historical'] = 0.5
                            self.inputs['weight_outside'] = 0.5
                            self.inputs['weight_fundamental'] = 0.0
                        elif self.inputs['use_historical']:
                            self.inputs['weight_historical'] = 0.5
                            self.inputs['weight_outside'] = 0.0
                            self.inputs['weight_fundamental'] = 0.5
                        else:  # use_outside
                            self.inputs['weight_historical'] = 0.0
                            self.inputs['weight_outside'] = 0.5
                            self.inputs['weight_fundamental'] = 0.5
                    else:
                        # All three methods available
                        self.inputs['weight_historical'] = 1 / 3
                        self.inputs['weight_outside'] = 1 / 3
                        self.inputs['weight_fundamental'] = 1 / 3

                    weights_valid = True
                    print("Default equal weights have been set.")

    def calculate_outputs(self):
        """Calculate all model outputs"""
        # Calculate cost of equity
        if not self.inputs['use_capm']:
            self.outputs['cost_of_equity'] = self.inputs['cost_of_equity']
        else:
            self.outputs['cost_of_equity'] = (
                    self.inputs['risk_free_rate'] +
                    self.inputs['beta'] * self.inputs['risk_premium']
            )

        # Calculate growth rates
        self.outputs['growth_rates'] = {}

        # Historical growth
        if self.inputs['use_historical'] and self.inputs['eps_5_years_ago'] > 0:
            self.outputs['growth_rates']['historical'] = (
                    (self.inputs['current_eps'] / self.inputs['eps_5_years_ago']) ** 0.2 - 1
            )
        else:
            self.outputs['growth_rates']['historical'] = 0

        # Outside growth
        if self.inputs['use_outside']:
            self.outputs['growth_rates']['outside'] = self.inputs['outside_growth']
        else:
            self.outputs['growth_rates']['outside'] = 0

        # Fundamental growth (always calculated now)
        self.outputs['growth_rates']['fundamental'] = (
                self.inputs['high_retention'] * self.inputs['high_roe']
        )

        # Weighted average growth rate
        self.outputs['high_growth_rate'] = (
                self.outputs['growth_rates']['historical'] * self.inputs['weight_historical'] +
                self.outputs['growth_rates']['outside'] * self.inputs['weight_outside'] +
                self.outputs['growth_rates']['fundamental'] * self.inputs['weight_fundamental']
        )

        # Payout ratio for high growth phase
        self.outputs['high_payout'] = 1 - self.inputs['high_retention']

        # Calculate dividends for high growth phase
        self.outputs['dividends'] = []
        for year in range(1, min(self.inputs['growth_period'] + 1, 11)):
            if year == 1:
                dividend = self.inputs['current_eps'] * (1 + self.outputs['high_growth_rate']) * self.outputs[
                    'high_payout']
            else:
                dividend = self.outputs['dividends'][-1] * (1 + self.outputs['high_growth_rate'])
            self.outputs['dividends'].append(dividend)

        # Calculate stable period cost of equity
        if self.inputs['stable_beta_different']:
            if not self.inputs['use_capm']:
                self.outputs['stable_cost_of_equity'] = self.outputs['cost_of_equity'] * (
                        self.inputs['stable_beta'] / self.inputs['beta'])
            else:
                self.outputs['stable_cost_of_equity'] = (
                        self.inputs['risk_free_rate'] +
                        self.inputs['stable_beta'] * self.inputs['risk_premium']
                )
        else:
            self.outputs['stable_cost_of_equity'] = self.outputs['cost_of_equity']

        # Terminal price
        terminal_eps = self.inputs['current_eps'] * (1 + self.outputs['high_growth_rate']) ** self.inputs[
            'growth_period']
        terminal_dividend = terminal_eps * (1 + self.inputs['stable_growth_rate']) * self.inputs['stable_payout']
        self.outputs['terminal_price'] = terminal_dividend / (
                self.outputs['stable_cost_of_equity'] - self.inputs['stable_growth_rate'])

        # Present values
        # PV of dividends in high growth phase
        if self.outputs['cost_of_equity'] != self.outputs['high_growth_rate']:
            self.outputs['pv_dividends'] = (
                    self.inputs['current_eps'] * self.outputs['high_payout'] * (1 + self.outputs['high_growth_rate']) *
                    (1 - (1 + self.outputs['high_growth_rate']) ** self.inputs['growth_period'] /
                     (1 + self.outputs['cost_of_equity']) ** self.inputs['growth_period']) /
                    (self.outputs['cost_of_equity'] - self.outputs['high_growth_rate'])
            )
        else:
            # Special case when growth rate equals cost of equity
            self.outputs['pv_dividends'] = (
                    self.inputs['current_eps'] * self.outputs['high_payout'] *
                    self.inputs['growth_period'] / (1 + self.outputs['cost_of_equity'])
            )

        # PV of terminal price
        self.outputs['pv_terminal'] = self.outputs['terminal_price'] / (1 + self.outputs['cost_of_equity']) ** \
                                      self.inputs['growth_period']

        # Total value
        self.outputs['stock_value'] = self.outputs['pv_dividends'] + self.outputs['pv_terminal']

        # Value decomposition
        self.outputs['value_assets'] = self.inputs['current_dividends'] / self.outputs['stable_cost_of_equity']
        self.outputs['value_stable_growth'] = (
                self.inputs['current_dividends'] * (1 + self.inputs['stable_growth_rate']) /
                (self.outputs['stable_cost_of_equity'] - self.inputs['stable_growth_rate']) -
                self.outputs['value_assets']
        )
        self.outputs['value_extraordinary_growth'] = (
                self.outputs['stock_value'] - self.outputs['value_stable_growth'] - self.outputs['value_assets']
        )

        # Generate warnings
        self._generate_warnings()

    def _generate_warnings(self):
        """Generate warnings based on inputs"""
        self.warnings = []

        if self.inputs['current_eps'] < 0:
            self.warnings.append("You have entered a negative current EPS. This model will not work")

        if self.inputs.get('eps_5_years_ago', 1) < 0:
            self.warnings.append("Historical Growth Rate cannot be calculated with negative EPS. Weight it at zero")

        if hasattr(self, 'stable_roe') and self.inputs.get('high_roe', 1) < self.inputs['stable_growth_rate']:
            self.warnings.append(
                "The ROE for the high growth period is very low. You will get a very low fundamental growth rate")

        if hasattr(self, 'stable_roe') and self.stable_roe <= self.inputs['stable_growth_rate']:
            self.warnings.append(
                "The ROE for the stable period is less (or =) than the stable growth rate. You cannot afford any dividends.")

        weights_sum = self.inputs['weight_historical'] + self.inputs['weight_outside'] + self.inputs[
            'weight_fundamental']
        if abs(weights_sum - 1.0) > 0.001:
            self.warnings.append("Your weights on the growth rates do not add up to one")

        if self.inputs['stable_growth_rate'] > 0.10:
            self.warnings.append("This is a high growth rate for a stable period")

    def display_results(self):
        """Display the model results"""
        print("\n" + "=" * 60)
        print("OUTPUT FROM THE PROGRAM")
        print("=" * 60)

        # Display warnings first
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"⚠️  {warning}")

        print(f"\nCost of Equity = {self.outputs['cost_of_equity']:.4%}")

        # Calculate and display current payout ratio
        current_payout_ratio = self.inputs['current_dividends'] / self.inputs['current_eps']
        print(f"Current Payout Ratio = {current_payout_ratio:.2%}")

        print(f"\nCurrent Earnings per share = ${self.inputs['current_eps']:.2f}")
        print(f"Current Dividends per share = ${self.inputs['current_dividends']:.4f}")

        print("\nGrowth Rate in Earnings per share:")
        print(f"{'':30} {'Growth Rate':>12} {'Weight':>10}")
        print("-" * 55)

        # Always show all three growth rates now
        print(
            f"{'Historical Growth':30} {self.outputs['growth_rates']['historical']:>11.2%} {self.inputs['weight_historical']:>10.2%}")
        print(
            f"{'Outside Estimates':30} {self.outputs['growth_rates']['outside']:>11.2%} {self.inputs['weight_outside']:>10.2%}")
        print(
            f"{'Fundamental Growth':30} {self.outputs['growth_rates']['fundamental']:>11.2%} {self.inputs['weight_fundamental']:>10.2%}")
        print(f"{'Weighted Average':30} {self.outputs['high_growth_rate']:>11.2%}")

        print(f"\nPayout Ratio for high growth phase = {self.outputs['high_payout']:.2%}")

        print("\nThe dividends for the high growth phase are shown below (up to 10 years):")
        print(f"{'Year':>6} {'Dividend':>12}")
        print("-" * 20)
        for i, div in enumerate(self.outputs['dividends']):
            print(f"{i + 1:>6} ${div:>11.4f}")

        print(f"\nGrowth Rate in Stable Phase = {self.inputs['stable_growth_rate']:.2%}")
        print(f"Payout Ratio in Stable Phase = {self.inputs['stable_payout']:.2%}")
        print(f"Cost of Equity in Stable Phase = {self.outputs['stable_cost_of_equity']:.4%}")
        print(f"Price at the end of growth phase = ${self.outputs['terminal_price']:.2f}")

        print("\n" + "-" * 50)
        print(f"Present Value of dividends in high growth phase = ${self.outputs['pv_dividends']:.2f}")
        print(f"Present Value of Terminal Price = ${self.outputs['pv_terminal']:.2f}")
        print(f"Value of the stock = ${self.outputs['stock_value']:.2f}")
        print("-" * 50)

        print("\nEstimating the value of growth:")
        print(f"Value of assets in place = ${self.outputs['value_assets']:.2f}")
        print(f"Value of stable growth = ${self.outputs['value_stable_growth']:.2f}")
        print(f"Value of extraordinary growth = ${self.outputs['value_extraordinary_growth']:.2f}")
        print(
            f"Value of the stock = ${self.outputs['value_assets'] + self.outputs['value_stable_growth'] + self.outputs['value_extraordinary_growth']:.2f}")

    def perform_sensitivity_analysis(self):
        """Perform sensitivity analysis on growth rate and growth period"""
        # Growth rate sensitivity
        growth_rates = []
        values_by_growth = []
        base_growth = self.outputs['high_growth_rate']

        for delta in range(-10, 11):
            test_growth = base_growth + (delta / 100.0)
            if test_growth >= 0:
                growth_rates.append(test_growth)
                value = self._calculate_value_with_growth(test_growth, self.inputs['growth_period'])
                values_by_growth.append(value)

        self.outputs['sensitivity_growth'] = {
            'growth_rates': growth_rates,
            'values': values_by_growth
        }

        # Growth period sensitivity
        periods = list(range(0, 11))
        values_by_period = []

        for period in periods:
            value = self._calculate_value_with_growth(self.outputs['high_growth_rate'], period)
            values_by_period.append(value)

        self.outputs['sensitivity_period'] = {
            'periods': periods,
            'values': values_by_period
        }

    def _calculate_value_with_growth(self, growth_rate, period):
        """Helper function to calculate value with different parameters"""
        # Value of extraordinary growth component only
        if period == 0:
            return 0

        # PV of dividends
        if self.outputs['cost_of_equity'] != growth_rate:
            pv_dividends = (
                    self.inputs['current_eps'] * self.outputs['high_payout'] * (1 + growth_rate) *
                    (1 - (1 + growth_rate) ** period / (1 + self.outputs['cost_of_equity']) ** period) /
                    (self.outputs['cost_of_equity'] - growth_rate)
            )
        else:
            pv_dividends = (
                    self.inputs['current_eps'] * self.outputs['high_payout'] *
                    period / (1 + self.outputs['cost_of_equity'])
            )

        # Terminal value
        terminal_eps = self.inputs['current_eps'] * (1 + growth_rate) ** period
        terminal_dividend = terminal_eps * (1 + self.inputs['stable_growth_rate']) * self.inputs['stable_payout']
        terminal_price = terminal_dividend / (self.outputs['stable_cost_of_equity'] - self.inputs['stable_growth_rate'])
        pv_terminal = terminal_price / (1 + self.outputs['cost_of_equity']) ** period

        # Return extraordinary growth value only
        total_value = pv_dividends + pv_terminal
        return total_value - self.outputs['value_stable_growth'] - self.outputs['value_assets']

    def plot_sensitivity(self):
        """Create visualizations of the sensitivity analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Enhanced color scheme
        primary_color = '#2c3e50'
        secondary_color = '#e74c3c'
        accent_color = '#3498db'
        positive_color = '#27ae60'
        background_color = '#ecf0f1'

        fig.patch.set_facecolor(background_color)

        # 1. Growth Rate Sensitivity (left)
        ax1.set_facecolor('white')

        growth_rates = self.outputs['sensitivity_growth']['growth_rates']
        values = self.outputs['sensitivity_growth']['values']

        # Add gradient background
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        ax1.imshow(gradient, extent=[min(growth_rates), max(growth_rates),
                                     min(values) * 0.9, max(values) * 1.1],
                   aspect='auto', alpha=0.03, cmap='coolwarm')

        # Create smooth curve
        if len(growth_rates) > 3:
            growth_smooth = np.linspace(min(growth_rates), max(growth_rates), 300)
            spl = make_interp_spline(growth_rates, values, k=3)
            values_smooth = spl(growth_smooth)
            ax1.plot(growth_smooth, values_smooth, color=primary_color, linewidth=3, alpha=0.8)

        # Plot points
        ax1.scatter(growth_rates, values, color=primary_color, s=50, zorder=5, edgecolors='white', linewidth=1)

        # Highlight current point
        current_idx = min(range(len(growth_rates)),
                          key=lambda i: abs(growth_rates[i] - self.outputs['high_growth_rate']))
        ax1.scatter([growth_rates[current_idx]], [values[current_idx]],
                    color=secondary_color, s=200, marker='*', zorder=10,
                    edgecolors='white', linewidth=2)

        # Add annotation
        ax1.annotate(f'Current\n{growth_rates[current_idx]:.1%}',
                     xy=(growth_rates[current_idx], values[current_idx]),
                     xytext=(growth_rates[current_idx] + 0.02, values[current_idx] + max(values) * 0.1),
                     fontsize=10, fontweight='bold', color=secondary_color,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=secondary_color),
                     arrowprops=dict(arrowstyle='->', color=secondary_color, lw=1.5))

        ax1.set_xlabel('High Growth Rate', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Value of Extraordinary Growth ($)', fontsize=12, fontweight='bold')
        ax1.set_title('Sensitivity to Growth Rate\n(Fixed Period: {} years)'.format(self.inputs['growth_period']),
                      fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        # Remove top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # 2. Growth Period Sensitivity (right)
        ax2.set_facecolor('white')

        periods = self.outputs['sensitivity_period']['periods']
        values = self.outputs['sensitivity_period']['values']

        # Create bars with gradient
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(periods)))
        bars = ax2.bar(periods, values, color=colors, edgecolor='white', linewidth=2, alpha=0.8)

        # Highlight current period
        current_period_idx = self.inputs['growth_period']
        bars[current_period_idx].set_edgecolor(secondary_color)
        bars[current_period_idx].set_linewidth(3)

        # Add value labels on bars
        for i, (period, value) in enumerate(zip(periods, values)):
            if i == current_period_idx:
                ax2.text(period, value + max(values) * 0.02, f'${value:.0f}',
                         ha='center', va='bottom', fontweight='bold', fontsize=10, color=secondary_color)
            elif i % 2 == 0:  # Show every other label
                ax2.text(period, value + max(values) * 0.02, f'${value:.0f}',
                         ha='center', va='bottom', fontsize=9, color='gray')

        ax2.set_xlabel('Growth Period (years)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Value of Extraordinary Growth ($)', fontsize=12, fontweight='bold')
        ax2.set_title(
            'Sensitivity to Growth Period\n(Fixed Growth Rate: {:.1%})'.format(self.outputs['high_growth_rate']),
            fontsize=14, fontweight='bold', pad=15)
        ax2.set_xticks(periods)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Remove top and right spines
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Add summary box
        summary_text = (f"Base Case Summary:\n"
                        f"• Stock Value: ${self.outputs['stock_value']:.2f}\n"
                        f"• High Growth: {self.outputs['high_growth_rate']:.1%}\n"
                        f"• Growth Period: {self.inputs['growth_period']} years\n"
                        f"• Stable Growth: {self.inputs['stable_growth_rate']:.1%}")

        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=primary_color))

        plt.suptitle('Two-Stage DDM - Sensitivity Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.show()

    def display_sensitivity_table(self):
        """Display sensitivity analysis in tabular format"""
        print("\n" + "=" * 60)
        print("SENSITIVITY ANALYSIS")
        print("=" * 60)

        print("\nSensitivity to Growth Rate (with {} year growth period):".format(self.inputs['growth_period']))
        print(f"{'Growth Rate':>12} | {'Extraordinary Growth Value':>25}")
        print("-" * 40)

        for rate, value in zip(self.outputs['sensitivity_growth']['growth_rates'],
                               self.outputs['sensitivity_growth']['values']):
            marker = " <--" if abs(rate - self.outputs['high_growth_rate']) < 0.001 else ""
            print(f"{rate:>11.1%} | ${value:>24.2f}{marker}")

        print("\nSensitivity to Growth Period (with {:.1%} growth rate):".format(self.outputs['high_growth_rate']))
        print(f"{'Period':>8} | {'Extraordinary Growth Value':>25}")
        print("-" * 36)

        for period, value in zip(self.outputs['sensitivity_period']['periods'],
                                 self.outputs['sensitivity_period']['values']):
            marker = " <--" if period == self.inputs['growth_period'] else ""
            print(f"{period:>7} | ${value:>24.2f}{marker}")

    def save_results(self, filename=None):
        """Save results to a file"""
        if filename is None:
            filename = "two_stage_ddm_results.txt"

        with open(filename, 'w') as f:
            f.write("TWO-STAGE DIVIDEND DISCOUNT MODEL RESULTS\n")
            f.write("=" * 60 + "\n\n")

            f.write("INPUTS:\n")
            f.write(f"  Current EPS: ${self.inputs['current_eps']:.2f}\n")
            f.write(f"  Current Dividends: ${self.inputs['current_dividends']:.2f}\n")
            f.write(f"  Growth Period: {self.inputs['growth_period']} years\n")
            f.write(f"  High Growth Rate: {self.outputs['high_growth_rate']:.2%}\n")
            f.write(f"  Stable Growth Rate: {self.inputs['stable_growth_rate']:.2%}\n")
            f.write(f"  Cost of Equity: {self.outputs['cost_of_equity']:.4%}\n")

            f.write("\nOUTPUTS:\n")
            f.write(f"  Stock Value: ${self.outputs['stock_value']:.2f}\n")
            f.write(f"  - Value of Assets in Place: ${self.outputs['value_assets']:.2f}\n")
            f.write(f"  - Value of Stable Growth: ${self.outputs['value_stable_growth']:.2f}\n")
            f.write(f"  - Value of Extraordinary Growth: ${self.outputs['value_extraordinary_growth']:.2f}\n")

            if self.warnings:
                f.write("\nWARNINGS:\n")
                for warning in self.warnings:
                    f.write(f"  - {warning}\n")

        print(f"\nResults saved to {filename}")

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
            response = input(f"{prompt} (in percent): ").strip()
            if response == "" and default is not None:
                return default
            try:
                value = float(response)
                return value
            except ValueError:
                print("Please enter a valid number")

    def get_currency_input(self, prompt, default=None):
        """Get a currency amount input from user"""
        if default is not None:
            prompt += f" [Default: ${default}]"

        while True:
            response = input(f"{prompt} (in currency): ").strip()
            if response == "" and default is not None:
                return default
            try:
                # Remove common currency formatting characters
                response = response.replace(',', '').replace('$', '').replace(' ', '')
                return float(response)
            except ValueError:
                print("Please enter a valid number")

    def get_float_input(self, prompt, default=None):
        """Get a float input from user"""
        if default is not None:
            prompt += f" [Default: {default}]"

        while True:
            response = input(f"{prompt}: ").strip()
            if response == "" and default is not None:
                return default
            try:
                return float(response)
            except ValueError:
                print("Please enter a valid number")


def main():
    """Main function to run the Two-Stage Dividend Discount Model"""
    model = TwoStageDDM()

    # Display introduction
    model.display_intro()

    # Get inputs
    model.get_user_inputs()

    # Calculate outputs
    model.calculate_outputs()

    # Perform sensitivity analysis
    model.perform_sensitivity_analysis()

    # Display results
    model.display_results()

    # Display sensitivity table
    model.display_sensitivity_table()

    # Offer to plot sensitivity
    if input("\nWould you like to see sensitivity plots? (Yes/No): ").strip().lower() in ['yes', 'y']:
        model.plot_sensitivity()

    # Offer to save results
    if input("\nWould you like to save these results? (Yes/No): ").strip().lower() in ['yes', 'y']:
        filename = input("Enter filename (without extension) [Default: two_stage_ddm_results]: ").strip()
        if filename:
            model.save_results(f"{filename}.txt")
        else:
            model.save_results()


if __name__ == "__main__":
    main()