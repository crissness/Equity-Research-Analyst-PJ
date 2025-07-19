#!/usr/bin/env python3
"""
Three-Stage Dividend Discount Model
Based on the dividend discount model for valuing equity in firms with three stages of growth:
1. Initial period of high growth
2. Transition period of declining growth
3. Final period of stable growth
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


class ThreeStageDDM:
    def __init__(self):
        self.inputs = {}
        self.outputs = {}
        self.warnings = []

    def display_intro(self):
        """Display the introduction and assumptions of the model"""
        print("=" * 70)
        print("THREE-STAGE DIVIDEND DISCOUNT MODEL")
        print("=" * 70)
        print("\nThis model is designed to value the equity in a firm with three stages")
        print("of growth - an initial period of high growth, a transition period of")
        print("declining growth and a final period of stable growth.")
        print("\nAssumptions:")
        print("1. The firm is assumed to be in an extraordinary growth phase currently.")
        print("2. This extraordinary growth is expected to last for an initial period that has to be specified.")
        print("3. The growth rate declines linearly over the transition period to a stable growth rate.")
        print("4. The firm's dividend payout ratio changes consistently with the growth rate.")
        print("\n" + "=" * 70)

    def get_user_inputs(self):
        """Collect all necessary inputs from the user"""
        print("\nThe user should enter the following inputs:")
        print("1. Length of each growth phase")
        print("2. Growth rate in each growth phase")
        print("3. Dividend payout ratios in each growth phase.")
        print("4. Costs of Equity in each growth phase")

        print("\nInputs to the model:")

        # Basic inputs
        self.inputs['current_eps'] = self.get_currency_input(
            "Current Earnings per share",
            default=1.43
        )

        self.inputs['current_dividends'] = self.get_currency_input(
            "Current Dividends per share",
            default=0.56
        )

        # Cost of equity
        self._get_cost_of_equity_inputs()

        # Growth rate estimation for high growth phase
        self._get_high_growth_inputs()

        # Transition period inputs
        self._get_transition_period_inputs()

        # Stable period inputs
        self._get_stable_period_inputs()

    def _get_cost_of_equity_inputs(self):
        """Get cost of equity inputs"""
        use_direct = self.get_yes_no_input(
            "\nDo you want to enter cost of equity directly?"
        )

        if use_direct:
            self.inputs['cost_of_equity'] = self.get_percentage_input(
                "Enter the cost of equity",
                default=13.05
            ) / 100.0
            self.inputs['use_capm'] = False

            # Still need beta for transition calculations even if using direct cost of equity
            print("\nWe still need beta for transition period calculations:")
            self.inputs['beta'] = self.get_float_input(
                "Beta of the stock",
                default=1.1
            )
        else:
            print("\nEnter the inputs to the cost of equity:")
            self.inputs['beta'] = self.get_float_input(
                "Beta of the stock",
                default=1.1
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

    def _get_high_growth_inputs(self):
        """Get inputs for high growth phase"""
        print("\n" + "-" * 50)
        print("GROWTH RATE DURING THE INITIAL HIGH GROWTH PHASE")
        print("-" * 50)

        # Growth period length
        self.inputs['high_growth_period'] = int(self.get_float_input(
            "Enter length of extraordinary growth period (years)",
            default=10
        ))

        # Historical growth
        use_historical = self.get_yes_no_input(
            "\nDo you want to use the historical growth rate?"
        )

        if use_historical:
            self.inputs['eps_5_years_ago'] = self.get_currency_input(
                "Enter EPS from five years ago",
                default=0.61
            )
            self.inputs['use_historical'] = True
        else:
            self.inputs['use_historical'] = False

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

        # ALWAYS get fundamental growth inputs (no longer asking if user wants it)
        print("\nFundamental Growth Rate Calculation:"
              "\nIf you don't want to use it assign it a 0% weight.")
        print("Enter the following inputs:")

        self.inputs['net_income_current'] = self.get_currency_input(
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
        self.outputs['roe'] = self.inputs['net_income_current'] / self.inputs['book_value_last']
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

    def _get_transition_period_inputs(self):
        """Get inputs for transition period"""
        print("\n" + "-" * 50)
        print("GROWTH RATE DURING THE TRANSITION PERIOD")
        print("-" * 50)

        self.inputs['transition_period'] = int(self.get_float_input(
            "Enter length of the transition period (years)",
            default=8
        ))

        # Payout ratio adjustment
        self.inputs['payout_adjust_gradually'] = self.get_yes_no_input(
            "\nDo you want the payout ratio to adjust gradually to stable payout?"
        )

        if not self.inputs['payout_adjust_gradually']:
            self.inputs['transition_payout'] = self.get_percentage_input(
                "Enter the payout ratio for the transition period",
                default=50.0
            ) / 100.0

        # Beta adjustment
        self.inputs['beta_adjust_gradually'] = self.get_yes_no_input(
            "\nDo you want the beta to adjust gradually to stable beta?"
        )

        if not self.inputs['beta_adjust_gradually']:
            self.inputs['transition_beta'] = self.get_float_input(
                "Enter the beta for the transition period",
                default=1.0
            )

    def _get_stable_period_inputs(self):
        """Get inputs for stable period"""
        print("\n" + "-" * 50)
        print("GROWTH RATE DURING THE STABLE PHASE")
        print("-" * 50)

        self.inputs['stable_growth_rate'] = self.get_percentage_input(
            "Enter growth rate in stable growth period",
            default=8.0
        ) / 100.0

        # Calculate stable payout ratio from fundamentals
        if hasattr(self.inputs, 'stable_roe') and self.inputs['stable_roe'] > 0:
            stable_payout_fundamental = 1 - self.inputs['stable_growth_rate'] / self.inputs['stable_roe']
        else:
            stable_payout_fundamental = 0.60  # Default

        print(f"Stable payout ratio from fundamentals is = {stable_payout_fundamental:.2%}")

        change_payout = self.get_yes_no_input(
            "Do you want to change this payout ratio?"
        )

        if change_payout:
            self.inputs['stable_payout'] = self.get_percentage_input(
                "Enter the stable payout ratio",
                default=60.0
            ) / 100.0
        else:
            self.inputs['stable_payout'] = stable_payout_fundamental

        # Check if beta changes in stable period
        self.inputs['stable_beta_different'] = self.get_yes_no_input(
            "\nWill the beta change in the stable period?"
        )

        if self.inputs['stable_beta_different']:
            self.inputs['stable_beta'] = self.get_float_input(
                "Enter the beta for stable period",
                default=1.0
            )
        else:
            self.inputs['stable_beta'] = self.inputs.get('beta', 1.0)

    def calculate_outputs(self):
        """Calculate all model outputs"""
        # Calculate cost of equity
        if self.inputs['use_capm']:
            self.outputs['cost_of_equity'] = (
                    self.inputs['risk_free_rate'] +
                    self.inputs['beta'] * self.inputs['risk_premium']
            )
        else:
            self.outputs['cost_of_equity'] = self.inputs['cost_of_equity']

        # Calculate growth rates for high growth phase
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

        # Fundamental growth
        if self.inputs['use_fundamental'] and self.inputs.get('high_roe', 0) > 0:
            self.outputs['growth_rates']['fundamental'] = (
                    self.inputs['high_retention'] * self.inputs['high_roe']
            )
        else:
            self.outputs['growth_rates']['fundamental'] = 0

        # Weighted average growth rate for high growth phase
        self.outputs['high_growth_rate'] = (
                self.outputs['growth_rates']['historical'] * self.inputs['weight_historical'] +
                self.outputs['growth_rates']['outside'] * self.inputs['weight_outside'] +
                self.outputs['growth_rates']['fundamental'] * self.inputs['weight_fundamental']
        )

        # Payout ratio for high growth phase
        self.outputs['high_payout'] = 1 - self.inputs['high_retention']

        # Calculate high growth phase dividends and present values
        self._calculate_high_growth_phase()

        # Calculate transition phase
        self._calculate_transition_phase()

        # Calculate stable phase and terminal value
        self._calculate_stable_phase()

        # Calculate total stock value
        self.outputs['stock_value'] = (
                self.outputs['pv_high_growth_dividends'] +
                self.outputs['pv_transition_dividends'] +
                self.outputs['pv_terminal_price']
        )

        # Generate warnings
        self._generate_warnings()

    def _calculate_high_growth_phase(self):
        """Calculate high growth phase dividends and present values"""
        self.outputs['high_growth_years'] = []
        self.outputs['high_growth_earnings'] = []
        self.outputs['high_growth_dividends'] = []
        self.outputs['high_growth_pv'] = []

        # Calculate for up to 10 years or the specified period, whichever is smaller
        years_to_calculate = min(self.inputs['high_growth_period'], 10)

        for year in range(1, years_to_calculate + 1):
            self.outputs['high_growth_years'].append(year)

            # Earnings
            earnings = self.inputs['current_eps'] * (1 + self.outputs['high_growth_rate']) ** year
            self.outputs['high_growth_earnings'].append(earnings)

            # Dividends
            if year == 1:
                dividend = self.inputs['current_eps'] * (1 + self.outputs['high_growth_rate']) * self.outputs[
                    'high_payout']
            else:
                dividend = self.outputs['high_growth_dividends'][-1] * (1 + self.outputs['high_growth_rate'])
            self.outputs['high_growth_dividends'].append(dividend)

            # Present value
            pv = dividend / (1 + self.outputs['cost_of_equity']) ** year
            self.outputs['high_growth_pv'].append(pv)

        self.outputs['pv_high_growth_dividends'] = sum(self.outputs['high_growth_pv'])

    def _calculate_transition_phase(self):
        """Calculate transition phase with linearly declining growth rates"""
        self.outputs['transition_years'] = []
        self.outputs['transition_growth_rates'] = []
        self.outputs['transition_payout_ratios'] = []
        self.outputs['transition_earnings'] = []
        self.outputs['transition_dividends'] = []
        self.outputs['transition_betas'] = []
        self.outputs['transition_cost_of_equity'] = []
        self.outputs['transition_pv'] = []

        # Calculate for up to 10 years or the specified period, whichever is smaller
        years_to_calculate = min(self.inputs['transition_period'], 10)

        # Calculate stable cost of equity
        if self.inputs['stable_beta_different']:
            if self.inputs['use_capm']:
                stable_cost_of_equity = (
                        self.inputs['risk_free_rate'] +
                        self.inputs['stable_beta'] * self.inputs['risk_premium']
                )
            else:
                # If cost of equity was entered directly, use that for stable period too
                stable_cost_of_equity = self.inputs['cost_of_equity']
        else:
            stable_cost_of_equity = self.outputs['cost_of_equity']

        # Get final earnings from high growth phase
        final_high_growth_earnings = (
                self.inputs['current_eps'] *
                (1 + self.outputs['high_growth_rate']) ** self.inputs['high_growth_period']
        )

        for year in range(1, years_to_calculate + 1):
            transition_year = self.inputs['high_growth_period'] + year
            self.outputs['transition_years'].append(transition_year)

            # Growth rate declines linearly
            if self.inputs['transition_period'] > 0:
                growth_rate = (
                        self.outputs['high_growth_rate'] -
                        (self.outputs['high_growth_rate'] - self.inputs['stable_growth_rate']) *
                        year / self.inputs['transition_period']
                )
            else:
                growth_rate = self.inputs['stable_growth_rate']
            self.outputs['transition_growth_rates'].append(growth_rate)

            # Payout ratio adjusts gradually if specified
            if self.inputs['payout_adjust_gradually']:
                if self.inputs['transition_period'] > 0:
                    payout_ratio = (
                            self.outputs['high_payout'] +
                            (self.inputs['stable_payout'] - self.outputs['high_payout']) *
                            year / self.inputs['transition_period']
                    )
                else:
                    payout_ratio = self.inputs['stable_payout']
            else:
                payout_ratio = self.inputs.get('transition_payout', self.outputs['high_payout'])
            self.outputs['transition_payout_ratios'].append(payout_ratio)

            # Beta adjusts gradually if specified
            if self.inputs['beta_adjust_gradually']:
                if self.inputs['transition_period'] > 0:
                    beta = (
                            self.inputs['beta'] -
                            (self.inputs['beta'] - self.inputs['stable_beta']) *
                            year / self.inputs['transition_period']
                    )
                else:
                    beta = self.inputs['stable_beta']
            else:
                beta = self.inputs.get('transition_beta', self.inputs['beta'])
            self.outputs['transition_betas'].append(beta)

            # Cost of equity based on beta
            if self.inputs['use_capm']:
                cost_of_equity = self.inputs['risk_free_rate'] + self.inputs['risk_premium'] * beta
            else:
                # If cost of equity was entered directly, use that value for transition period too
                cost_of_equity = self.inputs['cost_of_equity']
            self.outputs['transition_cost_of_equity'].append(cost_of_equity)

            # Earnings
            if year == 1:
                earnings = final_high_growth_earnings * (1 + growth_rate)
            else:
                earnings = self.outputs['transition_earnings'][-1] * (1 + growth_rate)
            self.outputs['transition_earnings'].append(earnings)

            # Dividends
            dividend = earnings * payout_ratio
            self.outputs['transition_dividends'].append(dividend)

            # Present value (compound discount factor)
            discount_factor = (1 + self.outputs['cost_of_equity']) ** self.inputs['high_growth_period']
            for i in range(year):
                discount_factor *= (1 + self.outputs['transition_cost_of_equity'][i])
            pv = dividend / discount_factor
            self.outputs['transition_pv'].append(pv)

        self.outputs['pv_transition_dividends'] = sum(self.outputs['transition_pv'])

    def _calculate_stable_phase(self):
        """Calculate stable phase terminal value"""
        # Calculate stable cost of equity
        if self.inputs['stable_beta_different']:
            if self.inputs['use_capm']:
                self.outputs['stable_cost_of_equity'] = (
                        self.inputs['risk_free_rate'] +
                        self.inputs['stable_beta'] * self.inputs['risk_premium']
                )
            else:
                # If cost of equity was entered directly, use that for stable period too
                self.outputs['stable_cost_of_equity'] = self.inputs['cost_of_equity']
        else:
            self.outputs['stable_cost_of_equity'] = self.outputs['cost_of_equity']

        # Calculate terminal earnings (at the end of transition period)
        terminal_earnings = self.inputs['current_eps']

        # Compound through high growth phase
        terminal_earnings *= (1 + self.outputs['high_growth_rate']) ** self.inputs['high_growth_period']

        # Compound through transition phase
        for growth_rate in self.outputs['transition_growth_rates']:
            terminal_earnings *= (1 + growth_rate)

        # Terminal dividend (first year of stable phase)
        terminal_dividend = terminal_earnings * (1 + self.inputs['stable_growth_rate']) * self.inputs['stable_payout']

        # Terminal price using Gordon growth model
        self.outputs['terminal_price'] = terminal_dividend / (
                    self.outputs['stable_cost_of_equity'] - self.inputs['stable_growth_rate'])

        # Present value of terminal price
        discount_factor = (1 + self.outputs['cost_of_equity']) ** self.inputs['high_growth_period']
        for cost_of_equity in self.outputs['transition_cost_of_equity']:
            discount_factor *= (1 + cost_of_equity)
        self.outputs['pv_terminal_price'] = self.outputs['terminal_price'] / discount_factor

    def _generate_warnings(self):
        """Generate warnings based on inputs"""
        self.warnings = []

        if self.inputs['current_eps'] < 0:
            self.warnings.append("You have entered a negative current EPS. This model will not work")

        if self.inputs.get('eps_5_years_ago', 1) < 0:
            self.warnings.append("Historical Growth Rate cannot be calculated with negative EPS. Weight it at zero")

        if self.inputs.get('high_roe', 1) < self.inputs['stable_growth_rate']:
            self.warnings.append(
                "The ROE for the high growth period is very low. You will get a very low fundamental growth rate")

        if self.inputs.get('stable_roe', 1) <= self.inputs['stable_growth_rate']:
            self.warnings.append(
                "The ROE for the stable period is less (or =) than the stable growth rate. You cannot afford any dividends.")

        weights_sum = self.inputs['weight_historical'] + self.inputs['weight_outside'] + self.inputs[
            'weight_fundamental']
        if abs(weights_sum - 1.0) > 0.001:
            self.warnings.append("Your weights on the growth rates do not add up to one")

        if self.inputs['stable_growth_rate'] > 0.10:
            self.warnings.append("This is a high growth rate for a stable period")

        if self.inputs['high_growth_period'] > 20:
            self.warnings.append("High growth period is very long - consider if this is realistic")

        if self.inputs['transition_period'] > 15:
            self.warnings.append("Transition period is very long - consider if this is realistic")

    def display_results(self):
        """Display the model results"""
        print("\n" + "=" * 70)
        print("OUTPUT FROM THE PROGRAM")
        print("=" * 70)

        # Display warnings first
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"⚠️  {warning}")

        print("\n" + "=" * 50)
        print("INITIAL HIGH GROWTH PHASE")
        print("=" * 50)

        print(f"Cost of Equity = {self.outputs['cost_of_equity']:.4%}")

        # Calculate and display current payout ratio
        current_payout_ratio = self.inputs['current_dividends'] / self.inputs['current_eps']
        print(f"Current Payout Ratio = {current_payout_ratio:.2%}")

        print(f"Current Earnings per share = ${self.inputs['current_eps']:.2f}")
        print(f"Current Dividends per share = ${self.inputs['current_dividends']:.4f}")

        print("\nGrowth Rate in Earnings per share - Initial High Growth phase:")
        print(f"{'':30} {'Growth Rate':>12} {'Weight':>10}")
        print("-" * 55)

        if self.inputs['use_historical']:
            print(
                f"{'Historical Growth':30} {self.outputs['growth_rates']['historical']:>11.2%} {self.inputs['weight_historical']:>10.2%}")
        if self.inputs['use_outside']:
            print(
                f"{'Outside Estimates':30} {self.outputs['growth_rates']['outside']:>11.2%} {self.inputs['weight_outside']:>10.2%}")
        if self.inputs['use_fundamental']:
            print(
                f"{'Fundamental Growth':30} {self.outputs['growth_rates']['fundamental']:>11.2%} {self.inputs['weight_fundamental']:>10.2%}")
        print(f"{'Weighted Average':30} {self.outputs['high_growth_rate']:>11.2%}")

        print(f"\nPayout Ratio for high growth phase = {self.outputs['high_payout']:.2%}")

        print("\nThe dividends for the high growth phase are shown below (up to 10 years):")
        print(f"{'Year':>6} {'Earnings':>12} {'Dividends':>12} {'Present Value':>15}")
        print("-" * 50)
        for i, (year, earnings, dividend, pv) in enumerate(zip(
                self.outputs['high_growth_years'],
                self.outputs['high_growth_earnings'],
                self.outputs['high_growth_dividends'],
                self.outputs['high_growth_pv']
        )):
            print(f"{year:>6} ${earnings:>11.4f} ${dividend:>11.4f} ${pv:>14.4f}")

        print("\n" + "=" * 50)
        print("TRANSITION PERIOD (up to ten years)")
        print("=" * 50)

        if self.outputs['transition_years']:
            print(
                f"{'Year':>6} {'Growth Rate':>12} {'Payout Ratio':>13} {'Earnings':>12} {'Dividends':>12} {'Beta':>8} {'Cost of Eq.':>11} {'Present Value':>15}")
            print("-" * 100)
            for i, (year, growth, payout, earnings, dividend, beta, cost_eq, pv) in enumerate(zip(
                    self.outputs['transition_years'],
                    self.outputs['transition_growth_rates'],
                    self.outputs['transition_payout_ratios'],
                    self.outputs['transition_earnings'],
                    self.outputs['transition_dividends'],
                    self.outputs['transition_betas'],
                    self.outputs['transition_cost_of_equity'],
                    self.outputs['transition_pv']
            )):
                print(
                    f"{year:>6} {growth:>11.2%} {payout:>12.2%} ${earnings:>11.4f} ${dividend:>11.4f} {beta:>7.2f} {cost_eq:>10.2%} ${pv:>14.4f}")
        else:
            print("No transition period specified.")

        print("\n" + "=" * 50)
        print("STABLE GROWTH PHASE")
        print("=" * 50)

        print(f"Growth Rate in Stable Phase = {self.inputs['stable_growth_rate']:.2%}")
        print(f"Payout Ratio in Stable Phase = {self.inputs['stable_payout']:.2%}")
        print(f"Cost of Equity in Stable Phase = {self.outputs['stable_cost_of_equity']:.4%}")
        print(f"Price at the end of growth phase = ${self.outputs['terminal_price']:.2f}")

        print("\n" + "=" * 50)
        print("VALUATION SUMMARY")
        print("=" * 50)

        print(f"Present Value of dividends in high growth phase = ${self.outputs['pv_high_growth_dividends']:.2f}")
        print(f"Present Value of dividends in transition phase = ${self.outputs['pv_transition_dividends']:.2f}")
        print(f"Present Value of Terminal Price = ${self.outputs['pv_terminal_price']:.2f}")
        print("-" * 60)
        print(f"Value of the stock = ${self.outputs['stock_value']:.2f}")
        print("=" * 60)

    def perform_sensitivity_analysis(self):
        """Perform sensitivity analysis on key parameters"""
        # High growth rate sensitivity
        self._sensitivity_high_growth_rate()

        # High growth period sensitivity
        self._sensitivity_high_growth_period()

        # Transition period sensitivity
        self._sensitivity_transition_period()

        # Stable growth rate sensitivity
        self._sensitivity_stable_growth_rate()

    def _sensitivity_high_growth_rate(self):
        """Sensitivity analysis for high growth rate"""
        base_rate = self.outputs['high_growth_rate']
        rates = []
        values = []

        for delta in range(-10, 11, 2):
            test_rate = base_rate + (delta / 100.0)
            if test_rate >= 0:
                rates.append(test_rate)
                value = self._calculate_value_with_different_high_growth(test_rate)
                values.append(value)

        self.outputs['sensitivity_high_growth'] = {
            'rates': rates,
            'values': values
        }

    def _sensitivity_high_growth_period(self):
        """Sensitivity analysis for high growth period"""
        base_period = self.inputs['high_growth_period']
        periods = list(range(max(1, base_period - 5), base_period + 6))
        values = []

        for period in periods:
            value = self._calculate_value_with_different_high_period(period)
            values.append(value)

        self.outputs['sensitivity_high_period'] = {
            'periods': periods,
            'values': values
        }

    def _sensitivity_transition_period(self):
        """Sensitivity analysis for transition period"""
        base_period = self.inputs['transition_period']
        periods = list(range(max(0, base_period - 3), base_period + 4))
        values = []

        for period in periods:
            value = self._calculate_value_with_different_transition_period(period)
            values.append(value)

        self.outputs['sensitivity_transition_period'] = {
            'periods': periods,
            'values': values
        }

    def _sensitivity_stable_growth_rate(self):
        """Sensitivity analysis for stable growth rate"""
        base_rate = self.inputs['stable_growth_rate']
        rates = []
        values = []

        for delta in range(-3, 4):
            test_rate = base_rate + (delta / 100.0)
            if test_rate >= 0 and test_rate < self.outputs['stable_cost_of_equity']:
                rates.append(test_rate)
                value = self._calculate_value_with_different_stable_growth(test_rate)
                values.append(value)

        self.outputs['sensitivity_stable_growth'] = {
            'rates': rates,
            'values': values
        }

    def _calculate_value_with_different_high_growth(self, new_high_growth_rate):
        """Calculate stock value with different high growth rate"""
        # This is a simplified calculation for sensitivity analysis
        # We'll recalculate just the key components

        # High growth dividends
        pv_high = 0
        for year in range(1, min(self.inputs['high_growth_period'] + 1, 11)):
            dividend = (self.inputs['current_eps'] *
                        (1 + new_high_growth_rate) ** year *
                        self.outputs['high_payout'])
            pv = dividend / (1 + self.outputs['cost_of_equity']) ** year
            pv_high += pv

        # Keep transition the same for simplicity
        pv_transition = self.outputs['pv_transition_dividends']

        # Recalculate terminal value
        terminal_earnings = (self.inputs['current_eps'] *
                             (1 + new_high_growth_rate) ** self.inputs['high_growth_period'])

        # Apply transition growth
        for growth_rate in self.outputs['transition_growth_rates']:
            terminal_earnings *= (1 + growth_rate)

        terminal_dividend = terminal_earnings * (1 + self.inputs['stable_growth_rate']) * self.inputs['stable_payout']
        terminal_price = terminal_dividend / (self.outputs['stable_cost_of_equity'] - self.inputs['stable_growth_rate'])

        # Discount terminal price
        discount_factor = (1 + self.outputs['cost_of_equity']) ** self.inputs['high_growth_period']
        for cost_of_equity in self.outputs['transition_cost_of_equity']:
            discount_factor *= (1 + cost_of_equity)
        pv_terminal = terminal_price / discount_factor

        return pv_high + pv_transition + pv_terminal

    def _calculate_value_with_different_high_period(self, new_period):
        """Calculate stock value with different high growth period"""
        # Simplified calculation
        pv_high = 0
        for year in range(1, min(new_period + 1, 11)):
            dividend = (self.inputs['current_eps'] *
                        (1 + self.outputs['high_growth_rate']) ** year *
                        self.outputs['high_payout'])
            pv = dividend / (1 + self.outputs['cost_of_equity']) ** year
            pv_high += pv

        # Estimate the impact on terminal value
        # This is a simplified approach
        adjustment_factor = (self.inputs['high_growth_period'] - new_period) * 0.1
        adjusted_terminal = self.outputs['pv_terminal_price'] * (1 + adjustment_factor)

        return pv_high + self.outputs['pv_transition_dividends'] + adjusted_terminal

    def _calculate_value_with_different_transition_period(self, new_period):
        """Calculate stock value with different transition period"""
        # Simplified calculation - adjust transition value proportionally
        if self.inputs['transition_period'] > 0:
            adjustment_factor = new_period / self.inputs['transition_period']
        else:
            adjustment_factor = 1

        adjusted_transition = self.outputs['pv_transition_dividends'] * adjustment_factor

        return (self.outputs['pv_high_growth_dividends'] +
                adjusted_transition +
                self.outputs['pv_terminal_price'])

    def _calculate_value_with_different_stable_growth(self, new_stable_growth):
        """Calculate stock value with different stable growth rate"""
        # Recalculate terminal value with new stable growth
        terminal_earnings = self.inputs['current_eps']

        # Compound through all phases
        terminal_earnings *= (1 + self.outputs['high_growth_rate']) ** self.inputs['high_growth_period']
        for growth_rate in self.outputs['transition_growth_rates']:
            terminal_earnings *= (1 + growth_rate)

        terminal_dividend = terminal_earnings * (1 + new_stable_growth) * self.inputs['stable_payout']

        if self.outputs['stable_cost_of_equity'] > new_stable_growth:
            terminal_price = terminal_dividend / (self.outputs['stable_cost_of_equity'] - new_stable_growth)
        else:
            terminal_price = 0  # Invalid case

        # Discount terminal price
        discount_factor = (1 + self.outputs['cost_of_equity']) ** self.inputs['high_growth_period']
        for cost_of_equity in self.outputs['transition_cost_of_equity']:
            discount_factor *= (1 + cost_of_equity)
        pv_terminal = terminal_price / discount_factor

        return (self.outputs['pv_high_growth_dividends'] +
                self.outputs['pv_transition_dividends'] +
                pv_terminal)

    def display_sensitivity_analysis(self):
        """Display sensitivity analysis results"""
        print("\n" + "=" * 70)
        print("SENSITIVITY ANALYSIS")
        print("=" * 70)

        # High growth rate sensitivity
        if 'sensitivity_high_growth' in self.outputs:
            print(f"\nSensitivity to High Growth Rate (Base: {self.outputs['high_growth_rate']:.2%}):")
            print(f"{'Growth Rate':>15} | {'Stock Value':>12}")
            print("-" * 30)
            for rate, value in zip(self.outputs['sensitivity_high_growth']['rates'],
                                   self.outputs['sensitivity_high_growth']['values']):
                marker = " <--" if abs(rate - self.outputs['high_growth_rate']) < 0.001 else ""
                print(f"{rate:>14.2%} | ${value:>11.2f}{marker}")

        # High growth period sensitivity
        if 'sensitivity_high_period' in self.outputs:
            print(f"\nSensitivity to High Growth Period (Base: {self.inputs['high_growth_period']} years):")
            print(f"{'Period (years)':>15} | {'Stock Value':>12}")
            print("-" * 30)
            for period, value in zip(self.outputs['sensitivity_high_period']['periods'],
                                     self.outputs['sensitivity_high_period']['values']):
                marker = " <--" if period == self.inputs['high_growth_period'] else ""
                print(f"{period:>14} | ${value:>11.2f}{marker}")

        # Stable growth rate sensitivity
        if 'sensitivity_stable_growth' in self.outputs:
            print(f"\nSensitivity to Stable Growth Rate (Base: {self.inputs['stable_growth_rate']:.2%}):")
            print(f"{'Growth Rate':>15} | {'Stock Value':>12}")
            print("-" * 30)
            for rate, value in zip(self.outputs['sensitivity_stable_growth']['rates'],
                                   self.outputs['sensitivity_stable_growth']['values']):
                marker = " <--" if abs(rate - self.inputs['stable_growth_rate']) < 0.001 else ""
                print(f"{rate:>14.2%} | ${value:>11.2f}{marker}")

    def plot_sensitivity(self):
        """Create visualizations of the sensitivity analysis"""
        # Count how many sensitivity analyses we have
        plots_to_show = []
        if 'sensitivity_high_growth' in self.outputs:
            plots_to_show.append('high_growth')
        if 'sensitivity_high_period' in self.outputs:
            plots_to_show.append('high_period')
        if 'sensitivity_stable_growth' in self.outputs:
            plots_to_show.append('stable_growth')

        if not plots_to_show:
            print("No sensitivity analysis data available for plotting.")
            return

        num_plots = len(plots_to_show)
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

        if num_plots == 1:
            axes = [axes]

        # Enhanced color scheme
        primary_color = '#2c3e50'
        secondary_color = '#e74c3c'
        background_color = '#ecf0f1'

        fig.patch.set_facecolor(background_color)

        plot_idx = 0

        # High growth rate sensitivity
        if 'high_growth' in plots_to_show:
            ax = axes[plot_idx]
            ax.set_facecolor('white')

            rates = self.outputs['sensitivity_high_growth']['rates']
            values = self.outputs['sensitivity_high_growth']['values']

            ax.plot(rates, values, color=primary_color, linewidth=3, marker='o', markersize=6)

            # Highlight current point
            current_rate = self.outputs['high_growth_rate']
            current_idx = min(range(len(rates)), key=lambda i: abs(rates[i] - current_rate))
            ax.scatter([rates[current_idx]], [values[current_idx]],
                       color=secondary_color, s=200, marker='*', zorder=10)

            ax.set_xlabel('High Growth Rate', fontweight='bold')
            ax.set_ylabel('Stock Value ($)', fontweight='bold')
            ax.set_title('Sensitivity to High Growth Rate', fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

            plot_idx += 1

        # High growth period sensitivity
        if 'high_period' in plots_to_show:
            ax = axes[plot_idx]
            ax.set_facecolor('white')

            periods = self.outputs['sensitivity_high_period']['periods']
            values = self.outputs['sensitivity_high_period']['values']

            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(periods)))
            bars = ax.bar(periods, values, color=colors, alpha=0.8)

            # Highlight current period
            current_period = self.inputs['high_growth_period']
            if current_period in periods:
                current_idx = periods.index(current_period)
                bars[current_idx].set_edgecolor(secondary_color)
                bars[current_idx].set_linewidth(3)

            ax.set_xlabel('High Growth Period (years)', fontweight='bold')
            ax.set_ylabel('Stock Value ($)', fontweight='bold')
            ax.set_title('Sensitivity to High Growth Period', fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, axis='y')

            plot_idx += 1

        # Stable growth rate sensitivity
        if 'stable_growth' in plots_to_show:
            ax = axes[plot_idx]
            ax.set_facecolor('white')

            rates = self.outputs['sensitivity_stable_growth']['rates']
            values = self.outputs['sensitivity_stable_growth']['values']

            ax.plot(rates, values, color=primary_color, linewidth=3, marker='s', markersize=6)

            # Highlight current point
            current_rate = self.inputs['stable_growth_rate']
            current_idx = min(range(len(rates)), key=lambda i: abs(rates[i] - current_rate))
            ax.scatter([rates[current_idx]], [values[current_idx]],
                       color=secondary_color, s=200, marker='*', zorder=10)

            ax.set_xlabel('Stable Growth Rate', fontweight='bold')
            ax.set_ylabel('Stock Value ($)', fontweight='bold')
            ax.set_title('Sensitivity to Stable Growth Rate', fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        # Remove spines for all plots
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.suptitle('Three-Stage DDM - Sensitivity Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()

    def save_results(self, filename=None):
        """Save results to a file"""
        if filename is None:
            filename = "three_stage_ddm_results.txt"

        with open(filename, 'w') as f:
            f.write("THREE-STAGE DIVIDEND DISCOUNT MODEL RESULTS\n")
            f.write("=" * 70 + "\n\n")

            f.write("INPUTS:\n")
            f.write(f"  Current EPS: ${self.inputs['current_eps']:.2f}\n")
            f.write(f"  Current Dividends: ${self.inputs['current_dividends']:.2f}\n")
            f.write(f"  High Growth Period: {self.inputs['high_growth_period']} years\n")
            f.write(f"  High Growth Rate: {self.outputs['high_growth_rate']:.2%}\n")
            f.write(f"  Transition Period: {self.inputs['transition_period']} years\n")
            f.write(f"  Stable Growth Rate: {self.inputs['stable_growth_rate']:.2%}\n")
            f.write(f"  Cost of Equity: {self.outputs['cost_of_equity']:.4%}\n")

            f.write("\nOUTPUTS:\n")
            f.write(f"  Stock Value: ${self.outputs['stock_value']:.2f}\n")
            f.write(f"  - PV of High Growth Dividends: ${self.outputs['pv_high_growth_dividends']:.2f}\n")
            f.write(f"  - PV of Transition Dividends: ${self.outputs['pv_transition_dividends']:.2f}\n")
            f.write(f"  - PV of Terminal Price: ${self.outputs['pv_terminal_price']:.2f}\n")

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
    """Main function to run the Three-Stage Dividend Discount Model"""
    model = ThreeStageDDM()

    # Display introduction
    model.display_intro()

    # Get inputs
    model.get_user_inputs()

    # Calculate outputs
    model.calculate_outputs()

    # Display results
    model.display_results()

    # Perform sensitivity analysis
    model.perform_sensitivity_analysis()

    # Display sensitivity analysis
    model.display_sensitivity_analysis()

    # Offer to plot sensitivity
    if input("\nWould you like to see sensitivity plots? (Yes/No): ").strip().lower() in ['yes', 'y']:
        model.plot_sensitivity()

    # Offer to save results
    if input("\nWould you like to save these results? (Yes/No): ").strip().lower() in ['yes', 'y']:
        filename = input("Enter filename (without extension) [Default: three_stage_ddm_results]: ").strip()
        if filename:
            model.save_results(f"{filename}.txt")
        else:
            model.save_results()


if __name__ == "__main__":
    main()