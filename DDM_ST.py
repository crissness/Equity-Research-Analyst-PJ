#!/usr/bin/env python3
"""
Gordon Growth Model (Dividend Discount Model - Stable Growth)
Based on the dividend discount model for valuing equity in stable firms
"""

import matplotlib.pyplot as plt
import numpy as np


class GordonGrowthModel:
    def __init__(self):
        self.inputs = {}
        self.outputs = {}

    def display_intro(self):
        """Display the introduction and assumptions of the model"""
        print("=" * 60)
        print("GORDON GROWTH MODEL")
        print("=" * 60)
        print("\nThis model is designed to value the equity in a stable firm paying")
        print("dividends, which are roughly equal to Free Cashflows to Equity.")
        print("\nAssumptions in the model:")
        print("1. The firm is in steady state and will grow at a stable rate forever.")
        print("2. The firm pays out what it can afford to in dividends, i.e., Dividends = FCFE.")
        print("\n" + "=" * 60)

    def get_user_inputs(self):
        """Collect all necessary inputs from the user"""
        print("\nUser defined inputs")
        print("The user has to define the following inputs to the model:")
        print("1. Current Earnings per share and Payout ratio (Dividends/Earnings)")
        print("2. Cost of Equity or Inputs to the CAPM (Beta, Riskfree rate, Risk Premium)")
        print("3. Expected Growth Rate in Earnings and dividends forever.")
        print("\nPlease enter inputs to the model:")

        # Get earnings and payout ratio
        self.inputs['earnings_per_share'] = self.get_currency_input(
            "Current Earnings per share",
            default=4.33
        )

        self.inputs['payout_ratio'] = self.get_percentage_input(
            "Current Payout Ratio",
            default=63.0
        ) / 100.0  # Convert to decimal

        # Get cost of equity
        use_direct_coe = self.get_yes_no_input(
            "Are you directly entering the cost of equity?"
        )

        if use_direct_coe:
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

        # Get growth rate
        print("\nThe expected growth rate for a stable firm cannot be significantly")
        print("higher than the nominal growth rate in the economy in which the firm operates.")
        self.inputs['growth_rate'] = self.get_percentage_input(
            "Expected Growth Rate",
            default=6.0
        ) / 100.0

    def calculate_outputs(self):
        """Calculate the model outputs based on inputs"""
        # Calculate current dividends per share
        self.outputs['dividends_per_share'] = (
                self.inputs['earnings_per_share'] * self.inputs['payout_ratio']
        )

        # Calculate cost of equity if using CAPM
        if self.inputs['use_capm']:
            self.outputs['cost_of_equity'] = (
                    self.inputs['risk_free_rate'] +
                    self.inputs['beta'] * self.inputs['risk_premium']
            )
        else:
            self.outputs['cost_of_equity'] = self.inputs['cost_of_equity']

        # Store growth rate in outputs
        self.outputs['growth_rate'] = self.inputs['growth_rate']

        # Calculate Gordon Growth Model value
        if self.outputs['cost_of_equity'] <= self.outputs['growth_rate']:
            self.outputs['value_per_share'] = float('inf')
            self.outputs['valuation_error'] = True
            self.outputs['error_message'] = "Cost of equity must be greater than growth rate!"
        else:
            self.outputs['value_per_share'] = (
                    self.outputs['dividends_per_share'] * (1 + self.outputs['growth_rate']) /
                    (self.outputs['cost_of_equity'] - self.outputs['growth_rate'])
            )
            self.outputs['valuation_error'] = False

        # Generate warnings
        self.outputs['warnings'] = []
        if self.inputs['growth_rate'] > 0.10:
            self.outputs['warnings'].append("This is high for an infinite growth rate. Check it")
        if self.inputs['use_capm'] and self.inputs.get('beta', 0) > 1.5:
            self.outputs['warnings'].append("This Beta is high for a stable firm")
        if self.inputs['payout_ratio'] < 0.2:
            self.outputs['warnings'].append("Payout ratio is low for a stable firm")

    def perform_sensitivity_analysis(self):
        """Perform sensitivity analysis on growth rate"""
        base_growth = self.outputs['growth_rate']
        growth_rates = []
        values = []

        # Test growth rates from -6% to +2% of base rate
        for delta in range(-6, 3):
            test_growth = base_growth + (delta / 100.0)
            growth_rates.append(test_growth)

            if self.outputs['cost_of_equity'] > test_growth and test_growth >= 0:
                value = (
                        self.outputs['dividends_per_share'] * (1 + test_growth) /
                        (self.outputs['cost_of_equity'] - test_growth)
                )
                values.append(value)
            else:
                values.append(None)

        self.outputs['sensitivity_analysis'] = {
            'growth_rates': growth_rates,
            'values': values
        }

    def display_results(self):
        """Display the model results"""
        print("\n" + "=" * 60)
        print("This is the output from the Gordon Growth Model")
        print("=" * 60)

        print("\nFirm Details: from inputs on prior page")
        print(f"Current Dividends per share = ${self.outputs['dividends_per_share']:.4f}")
        print(f"\nCost of Equity = {self.outputs['cost_of_equity']:.4%}")
        print(f"Expected Growth rate = {self.outputs['growth_rate']:.2%}")

        # Display warnings
        for warning in self.outputs['warnings']:
            print(f"\n⚠️  {warning}")

        # Display valuation
        print("\n" + "-" * 40)
        if self.outputs['valuation_error']:
            print(f"Gordon Growth Model Value = ERROR: {self.outputs['error_message']}")
        else:
            print(f"Gordon Growth Model Value = ${self.outputs['value_per_share']:.2f}")
        print("-" * 40)

        # Display sensitivity analysis
        print("\nSensitivity Analysis - Value vs Growth Rate:")
        print(f"{'Growth Rate':>12} | {'Value':>12}")
        print("-" * 27)

        for i, (rate, value) in enumerate(zip(
                self.outputs['sensitivity_analysis']['growth_rates'],
                self.outputs['sensitivity_analysis']['values']
        )):
            if value is not None:
                marker = " <--" if abs(rate - self.outputs['growth_rate']) < 0.001 else ""
                print(f"{rate:>11.1%} | ${value:>10.2f}{marker}")
            else:
                print(f"{rate:>11.1%} | {'N/A':>11}")

    def plot_sensitivity(self):
        """Create enhanced visualizations of the sensitivity analysis"""
        growth_rates = [r for r, v in zip(
            self.outputs['sensitivity_analysis']['growth_rates'],
            self.outputs['sensitivity_analysis']['values']
        ) if v is not None]

        values = [v for v in self.outputs['sensitivity_analysis']['values'] if v is not None]

        if len(growth_rates) > 0:
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))

            # Define color scheme
            primary_color = '#2E86AB'
            secondary_color = '#A23B72'
            accent_color = '#F18F01'
            background_color = '#F5F5F5'

            # Set figure background
            fig.patch.set_facecolor(background_color)

            # 1. Main sensitivity curve (top left)
            ax1 = plt.subplot(2, 2, 1)
            ax1.set_facecolor('white')

            # Create gradient effect
            for i in range(len(growth_rates) - 1):
                ax1.fill_between(growth_rates[i:i + 2], 0, values[i:i + 2],
                                 alpha=0.1, color=primary_color)

            # Main line
            ax1.plot(growth_rates, values, color=primary_color, linewidth=3,
                     marker='o', markersize=8, markerfacecolor='white',
                     markeredgecolor=primary_color, markeredgewidth=2)

            # Highlight current point
            ax1.scatter([self.outputs['growth_rate']], [self.outputs['value_per_share']],
                        color=secondary_color, s=200, zorder=5, marker='*',
                        edgecolors='white', linewidth=2)

            # Add value labels
            for i, (rate, value) in enumerate(zip(growth_rates, values)):
                if i % 2 == 0:  # Show every other label to avoid crowding
                    ax1.annotate(f'${value:.0f}', (rate, value),
                                 textcoords="offset points", xytext=(0, 10),
                                 ha='center', fontsize=9, color='gray')

            # Styling
            ax1.set_xlabel('Growth Rate', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Value per Share ($)', fontsize=12, fontweight='bold')
            ax1.set_title('Valuation Sensitivity to Growth Rate', fontsize=14, fontweight='bold', pad=20)
            ax1.grid(True, alpha=0.2, linestyle='--')
            ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

            # Add reference lines
            ax1.axhline(y=self.outputs['value_per_share'], color=secondary_color,
                        linestyle='--', alpha=0.5, linewidth=1)
            ax1.axvline(x=self.outputs['growth_rate'], color=secondary_color,
                        linestyle='--', alpha=0.5, linewidth=1)

            # 2. Percentage change chart (top right)
            ax2 = plt.subplot(2, 2, 2)
            ax2.set_facecolor('white')

            base_value = self.outputs['value_per_share']
            pct_changes = [(v - base_value) / base_value * 100 for v in values]
            colors = [accent_color if pc >= 0 else secondary_color for pc in pct_changes]

            bars = ax2.bar(range(len(growth_rates)), pct_changes, color=colors, alpha=0.7, edgecolor='white',
                           linewidth=2)

            # Highlight current position
            current_idx = growth_rates.index(min(growth_rates, key=lambda x: abs(x - self.outputs['growth_rate'])))
            bars[current_idx].set_edgecolor('black')
            bars[current_idx].set_linewidth(3)

            ax2.set_xlabel('Growth Rate Scenarios', fontsize=12, fontweight='bold')
            ax2.set_ylabel('% Change from Base Value', fontsize=12, fontweight='bold')
            ax2.set_title('Relative Value Changes', fontsize=14, fontweight='bold', pad=20)
            ax2.set_xticks(range(len(growth_rates)))
            ax2.set_xticklabels([f'{r:.1%}' for r in growth_rates], rotation=45)
            ax2.grid(True, alpha=0.2, axis='y', linestyle='--')
            ax2.axhline(y=0, color='black', linewidth=1)

            # Add percentage labels on bars
            for i, (bar, pc) in enumerate(zip(bars, pct_changes)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + (2 if height > 0 else -5),
                         f'{pc:.0f}%', ha='center', va='bottom' if height > 0 else 'top',
                         fontsize=9, fontweight='bold')

            # 3. Tornado diagram (bottom left)
            ax3 = plt.subplot(2, 2, 3)
            ax3.set_facecolor('white')

            # Calculate impacts
            growth_impacts = []
            labels = []
            for i, (rate, value) in enumerate(zip(growth_rates, values)):
                if rate != self.outputs['growth_rate']:
                    impact = value - base_value
                    growth_impacts.append(impact)
                    labels.append(f'{rate:.1%}')

            # Sort by absolute impact
            sorted_data = sorted(zip(labels, growth_impacts), key=lambda x: abs(x[1]), reverse=True)
            labels, impacts = zip(*sorted_data[:6])  # Show top 6 impacts

            colors = [accent_color if impact >= 0 else secondary_color for impact in impacts]
            y_pos = np.arange(len(labels))

            ax3.barh(y_pos, impacts, color=colors, alpha=0.7, edgecolor='white', linewidth=2)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(labels)
            ax3.set_xlabel('Impact on Value ($)', fontsize=12, fontweight='bold')
            ax3.set_title('Impact of Growth Rate Changes', fontsize=14, fontweight='bold', pad=20)
            ax3.axvline(x=0, color='black', linewidth=1)
            ax3.grid(True, alpha=0.2, axis='x', linestyle='--')

            # Add value labels
            for i, impact in enumerate(impacts):
                ax3.text(impact + (2 if impact > 0 else -2), i, f'${impact:,.0f}',
                         va='center', ha='left' if impact > 0 else 'right', fontsize=10)

            # 4. Summary statistics (bottom right)
            ax4 = plt.subplot(2, 2, 4)
            ax4.axis('off')

            # Create summary box
            summary_text = f"""
            SENSITIVITY ANALYSIS SUMMARY

            Base Case:
            • Growth Rate: {self.outputs['growth_rate']:.1%}
            • Value: ${base_value:,.2f}

            Sensitivity Range:
            • Min Value: ${min(values):,.2f} (at {growth_rates[values.index(min(values))]:.1%})
            • Max Value: ${max(values):,.2f} (at {growth_rates[values.index(max(values))]:.1%})
            • Range: ${max(values) - min(values):,.2f}

            Key Insights:
            • 1% increase in growth → ${(values[current_idx + 1] - base_value):+,.2f} ({((values[current_idx + 1] - base_value) / base_value * 100):+.1f}%)
            • 1% decrease in growth → ${(values[current_idx - 1] - base_value):+,.2f} ({((values[current_idx - 1] - base_value) / base_value * 100):+.1f}%)

            Risk Assessment:
            • Upside potential: {((max(values) - base_value) / base_value * 100):+.1f}%
            • Downside risk: {((min(values) - base_value) / base_value * 100):+.1f}%
            """

            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=1', facecolor='white',
                               edgecolor=primary_color, linewidth=2))

            # Overall title
            fig.suptitle(f'Gordon Growth Model - Comprehensive Sensitivity Analysis\nCurrent Value: ${base_value:,.2f}',
                         fontsize=16, fontweight='bold', y=0.98)

            plt.tight_layout()
            plt.show()

    def save_results(self, filename=None):
        """Save results to a file"""
        if filename is None:
            filename = "gordon_growth_results.txt"

        with open(filename, 'w') as f:
            f.write("GORDON GROWTH MODEL RESULTS\n")
            f.write("=" * 60 + "\n\n")

            f.write("INPUTS:\n")
            f.write(f"  Current Earnings per share: ${self.inputs['earnings_per_share']:.2f}\n")
            f.write(f"  Current Payout Ratio: {self.inputs['payout_ratio']:.2%}\n")

            if self.inputs['use_capm']:
                f.write(f"  Beta: {self.inputs['beta']:.2f}\n")
                f.write(f"  Risk-free Rate: {self.inputs['risk_free_rate']:.2%}\n")
                f.write(f"  Risk Premium: {self.inputs['risk_premium']:.2%}\n")
            else:
                f.write(f"  Cost of Equity (direct): {self.inputs['cost_of_equity']:.2%}\n")

            f.write(f"  Expected Growth Rate: {self.inputs['growth_rate']:.2%}\n")

            f.write("\nOUTPUTS:\n")
            f.write(f"  Current Dividends per share: ${self.outputs['dividends_per_share']:.4f}\n")
            f.write(f"  Cost of Equity: {self.outputs['cost_of_equity']:.4%}\n")
            f.write(f"  Gordon Growth Model Value: ${self.outputs['value_per_share']:.2f}\n")

            if self.outputs['warnings']:
                f.write("\nWARNINGS:\n")
                for warning in self.outputs['warnings']:
                    f.write(f"  - {warning}\n")

            f.write("\nSENSITIVITY ANALYSIS:\n")
            f.write(f"{'Growth Rate':>12} | {'Value':>12}\n")
            f.write("-" * 27 + "\n")

            for rate, value in zip(
                    self.outputs['sensitivity_analysis']['growth_rates'],
                    self.outputs['sensitivity_analysis']['values']
            ):
                if value is not None:
                    f.write(f"{rate:>11.1%} | ${value:>10.2f}\n")
                else:
                    f.write(f"{rate:>11.1%} | {'N/A':>11}\n")

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
                if value < 0 or value > 999:
                    print("Please enter a reasonable percentage value")
                else:
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
                # Remove $ sign if present
                response = response.replace('$', '').replace(',', '')
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
    """Main function to run the Gordon Growth Model"""
    model = GordonGrowthModel()

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

    # Offer to plot sensitivity
    if input("\nWould you like to see a sensitivity plot? (Yes/No): ").strip().lower() in ['yes', 'y']:
        model.plot_sensitivity()

    # Offer to save results
    if input("\nWould you like to save these results? (Yes/No): ").strip().lower() in ['yes', 'y']:
        filename = input("Enter filename (without extension) [Default: gordon_growth_results]: ").strip()
        if filename:
            model.save_results(f"{filename}.txt")
        else:
            model.save_results()


if __name__ == "__main__":
    main()