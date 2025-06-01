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
    Class to calculate the regression beta of a stock relative to a market index.
    Uses Yahoo Finance data to download historical prices.
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

    def download_data(self, stock_symbol, index_symbol, period="1y", interval="1d",
                      start_date=None, end_date=None):
        """
        Downloads historical data for the stock and reference index.

        Parameters:
        -----------
        stock_symbol : str
            Stock symbol (e.g. 'AAPL', 'ENI.MI')
        index_symbol : str
            Reference index symbol (e.g. '^GSPC' for S&P500, 'FTSEMIB.MI' for FTSE MIB)
        period : str
            Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval : str
            Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        """

        print(f"Downloading data for {stock_symbol} and {index_symbol}...")

        try:
            # Download stock data
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

            # Verify that data was downloaded correctly
            if self.stock_data.empty or self.index_data.empty:
                raise ValueError("Unable to download data. Check the entered symbols.")

            # Debug: show information about downloaded data
            print(f"Stock data shape: {self.stock_data.shape}")
            print(f"Index data shape: {self.index_data.shape}")

            print(f"Data downloaded successfully!")
            print(
                f"Period: {self.stock_data.index[0].strftime('%Y-%m-%d')} - {self.stock_data.index[-1].strftime('%Y-%m-%d')}")
            print(f"Number of observations: {len(self.stock_data)}")

        except Exception as e:
            print(f"Error downloading data: {e}")
            return False

        return True

    def calculate_returns(self):
        """
        Calculates percentage returns for the stock and index.
        """
        if self.stock_data is None or self.index_data is None:
            print("Error: data not available. Run download_data() first")
            return False

        try:
            # Debug: show data structure
            print("Debug - stock_data columns:", list(self.stock_data.columns))
            print("Debug - column type:", type(self.stock_data.columns))

            # Robust method to extract closing prices
            def get_close_price(data, symbol):
                if hasattr(data.columns, 'levels'):
                    # MultiIndex - look for 'Adj Close' first, then 'Close'
                    level_0_values = data.columns.get_level_values(0)
                    level_1_values = data.columns.get_level_values(1)

                    if 'Adj Close' in level_0_values:
                        # Find column with 'Adj Close' and correct symbol
                        adj_close_cols = [(i, col) for i, col in enumerate(data.columns)
                                          if col[0] == 'Adj Close' and symbol.upper() in col[1].upper()]
                        if adj_close_cols:
                            col_name = adj_close_cols[0][1]
                            return data[col_name]

                    # If 'Adj Close' not found, use 'Close'
                    if 'Close' in level_0_values:
                        print(f"Warning: using 'Close' instead of 'Adj Close' for {symbol}")
                        close_cols = [(i, col) for i, col in enumerate(data.columns)
                                      if col[0] == 'Close' and symbol.upper() in col[1].upper()]
                        if close_cols:
                            col_name = close_cols[0][1]
                            return data[col_name]

                # Fallback for simple columns
                elif 'Adj Close' in data.columns:
                    return data['Adj Close']
                elif 'Close' in data.columns:
                    print(f"Warning: using 'Close' instead of 'Adj Close' for {symbol}")
                    return data['Close']

                raise KeyError(f"No 'Close' column found for {symbol}")

            # Extract symbols from parameters (assume they were saved)
            # For now, take the first available symbol from columns
            stock_symbol = self.stock_data.columns[0][1] if hasattr(self.stock_data.columns, 'levels') else 'STOCK'
            index_symbol = self.index_data.columns[0][1] if hasattr(self.index_data.columns, 'levels') else 'INDEX'

            # Extract closing prices
            stock_close = get_close_price(self.stock_data, stock_symbol)
            index_close = get_close_price(self.index_data, index_symbol)

            # Ensure they are pandas Series
            if not isinstance(stock_close, pd.Series):
                stock_close = pd.Series(stock_close.values.flatten(), index=stock_close.index)
            if not isinstance(index_close, pd.Series):
                index_close = pd.Series(index_close.values.flatten(), index=index_close.index)

            # Calculate returns
            self.stock_returns = stock_close.pct_change().dropna()
            self.index_returns = index_close.pct_change().dropna()

            # Align dates (take only common dates)
            common_dates = self.stock_returns.index.intersection(self.index_returns.index)
            self.stock_returns = self.stock_returns[common_dates]
            self.index_returns = self.index_returns[common_dates]

            print(f"Returns calculated for {len(self.stock_returns)} observations")
            return True

        except Exception as e:
            print(f"Error calculating returns: {e}")
            print("Stock data structure:")
            print(self.stock_data.head())
            print("\nIndex data structure:")
            print(self.index_data.head())
            return False

        # Align dates (take only common dates)
        common_dates = self.stock_returns.index.intersection(self.index_returns.index)
        self.stock_returns = self.stock_returns[common_dates]
        self.index_returns = self.index_returns[common_dates]

        print(f"Returns calculated for {len(self.stock_returns)} observations")
        return True

    def calculate_beta(self):
        """
        Calculates regression beta using linear regression.
        Beta = Cov(R_stock, R_market) / Var(R_market)
        """
        if self.stock_returns is None or self.index_returns is None:
            print("Error: returns not available. Run calculate_returns() first")
            return False

        # Remove any NaN values
        combined_data = pd.DataFrame({
            'stock': self.stock_returns,
            'index': self.index_returns
        }).dropna()

        if len(combined_data) < 30:
            print("Warning: few data points available for calculation (less than 30 observations)")

        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            combined_data['index'], combined_data['stock']
        )

        self.beta = slope
        self.alpha = intercept
        self.r_squared = r_value ** 2
        self.correlation = combined_data['stock'].corr(combined_data['index'])

        print(f"\n{'=' * 50}")
        print(f"BETA ANALYSIS RESULTS")
        print(f"{'=' * 50}")
        print(f"Beta: {self.beta:.4f}")
        print(f"Alpha: {self.alpha:.6f} ({self.alpha * 100:.4f}%)")
        print(f"R-squared: {self.r_squared:.4f}")
        print(f"Correlation: {self.correlation:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Standard Error: {std_err:.6f}")
        print(f"Number of observations: {len(combined_data)}")

        # Beta interpretation
        print(f"\n{'=' * 50}")
        print(f"INTERPRETATION")
        print(f"{'=' * 50}")
        if self.beta > 1:
            print(f"The stock is more volatile than the market (Beta > 1)")
        elif self.beta < 1 and self.beta > 0:
            print(f"The stock is less volatile than the market (0 < Beta < 1)")
        elif self.beta < 0:
            print(f"The stock moves inversely to the market (Beta < 0)")
        else:
            print(f"The stock is not correlated to the market (Beta ≈ 0)")

        return True

    def plot_regression(self, figsize=(10, 6)):
        """
        Creates a scatter plot with regression line.
        """
        if self.beta is None:
            print("Error: beta not calculated. Run calculate_beta() first")
            return

        # Prepare data for plot
        combined_data = pd.DataFrame({
            'index_returns': self.index_returns,
            'stock_returns': self.stock_returns
        }).dropna()

        # Create plot
        plt.figure(figsize=figsize)
        plt.scatter(combined_data['index_returns'], combined_data['stock_returns'],
                    alpha=0.6, s=20, color='blue', label='Observations')

        # Regression line
        x_line = np.linspace(combined_data['index_returns'].min(),
                             combined_data['index_returns'].max(), 100)
        y_line = self.alpha + self.beta * x_line
        plt.plot(x_line, y_line, 'r-', linewidth=2,
                 label=f'Regression (β = {self.beta:.4f})')

        plt.xlabel('Index Returns (%)')
        plt.ylabel('Stock Returns (%)')
        plt.title(f'Beta Analysis - Linear Regression\nR² = {self.r_squared:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Format axes as percentages
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x * 100:.1f}%'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x * 100:.1f}%'))

        plt.tight_layout()
        plt.show()

    def get_summary_stats(self):
        """
        Returns a summary of descriptive statistics.
        """
        if self.stock_returns is None or self.index_returns is None:
            print("Error: returns not available.")
            return None

        summary = {
            'Beta': self.beta,
            'Alpha': self.alpha,
            'R_squared': self.r_squared,
            'Correlation': self.correlation,
            'Stock_Volatility': self.stock_returns.std() * np.sqrt(252),  # Annualized
            'Index_Volatility': self.index_returns.std() * np.sqrt(252),  # Annualized
            'Average_Stock_Return': self.stock_returns.mean() * 252,  # Annualized
            'Average_Index_Return': self.index_returns.mean() * 252,  # Annualized
        }

        return summary

    def save_to_excel(self, filename=None, stock_symbol="STOCK", index_symbol="INDEX"):
        """
        Saves all data and results to an Excel file with separate sheets.

        Parameters:
        -----------
        filename : str, optional
            Excel file name. If None, generates automatically
        stock_symbol : str
            Stock symbol name for sheet names
        index_symbol : str
            Index symbol name for sheet names
        """
        if self.beta is None:
            print("Error: analysis not completed. Run calculate_beta() first")
            return None

        # Generate filename if not specified
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"beta_analysis_{stock_symbol}_{timestamp}.xlsx"

        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:

                # 1. BETA RESULTS SHEET
                results_data = {
                    'Metric': [
                        'Beta',
                        'Alpha',
                        'Alpha (%)',
                        'R-squared',
                        'Correlation',
                        'Stock Volatility (annual)',
                        'Index Volatility (annual)',
                        'Average Stock Return (annual)',
                        'Average Index Return (annual)',
                        'Number of Observations',
                        'Start Date',
                        'End Date',
                        'Stock Symbol',
                        'Index Symbol'
                    ],
                    'Value': [
                        self.beta,
                        self.alpha,
                        self.alpha * 100,
                        self.r_squared,
                        self.correlation,
                        self.stock_returns.std() * np.sqrt(252) * 100,  # Annual %
                        self.index_returns.std() * np.sqrt(252) * 100,  # Annual %
                        self.stock_returns.mean() * 252 * 100,  # Annual %
                        self.index_returns.mean() * 252 * 100,  # Annual %
                        len(self.stock_returns),
                        self.stock_data.index[0].strftime('%Y-%m-%d'),
                        self.stock_data.index[-1].strftime('%Y-%m-%d'),
                        stock_symbol,
                        index_symbol
                    ]
                }

                results_df = pd.DataFrame(results_data)
                results_df.to_excel(writer, sheet_name='Beta Results', index=False)

                # 2. STOCK DATA SHEET
                stock_clean = self.clean_dataframe_for_excel(self.stock_data, stock_symbol)
                stock_clean.to_excel(writer, sheet_name=f'{stock_symbol} Data')

                # 3. INDEX DATA SHEET
                index_clean = self.clean_dataframe_for_excel(self.index_data, index_symbol)
                index_clean.to_excel(writer, sheet_name=f'{index_symbol} Data')

                # 4. RETURNS SHEET
                returns_data = pd.DataFrame({
                    'Date': self.stock_returns.index,
                    f'{stock_symbol} Returns': self.stock_returns.values,
                    f'{index_symbol} Returns': self.index_returns.values,
                    f'{stock_symbol} Returns (%)': self.stock_returns.values * 100,
                    f'{index_symbol} Returns (%)': self.index_returns.values * 100
                })
                returns_data.to_excel(writer, sheet_name='Returns', index=False)

                # 5. DETAILED STATISTICS SHEET
                stats = self.get_summary_stats()
                if stats:
                    stats_data = {
                        'Statistic': list(stats.keys()),
                        'Value': [f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                                  for v in stats.values()],
                        'Value %': [f"{v * 100:.2f}%" if isinstance(v, (int, float)) and
                                                         ('Volatility' in k or 'Return' in k) else 'N/A'
                                    for k, v in stats.items()]
                    }
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name='Statistics', index=False)

                # 6. INTERPRETATION SHEET
                interpretation_data = {
                    'Aspect': [
                        'Systematic Risk',
                        'Market Sensitivity',
                        'Regression Quality',
                        'Market Correlation',
                        'Significant Alpha?',
                        'DCF Recommendation'
                    ],
                    'Interpretation': [
                        self.interpret_beta(),
                        self.interpret_sensitivity(),
                        self.interpret_r_squared(),
                        self.interpret_correlation(),
                        self.interpret_alpha(),
                        self.dcf_recommendation()
                    ]
                }
                interpretation_df = pd.DataFrame(interpretation_data)
                interpretation_df.to_excel(writer, sheet_name='Interpretation', index=False)

            print(f"\n✅ Data saved successfully to: {filename}")
            print(f"📊 File contains {len(writer.sheets)} sheets:")
            print("   • Beta Results")
            print(f"   • {stock_symbol} Data")
            print(f"   • {index_symbol} Data")
            print("   • Returns")
            print("   • Statistics")
            print("   • Interpretation")

            return filename

        except Exception as e:
            print(f"❌ Error saving: {e}")
            return None

    def clean_dataframe_for_excel(self, df, symbol):
        """
        Cleans a DataFrame with MultiIndex for Excel saving.
        """
        clean_df = df.copy()

        # If it has MultiIndex, flatten columns
        if hasattr(clean_df.columns, 'levels'):
            # Create simple column names
            new_columns = []
            for col in clean_df.columns:
                if len(col) == 2:
                    new_columns.append(f"{col[0]}")
                else:
                    new_columns.append(str(col))
            clean_df.columns = new_columns

        # Ensure index is a normal column
        clean_df.reset_index(inplace=True)

        return clean_df

    def interpret_beta(self):
        """Interprets beta value"""
        if self.beta > 1.2:
            return f"High systematic risk (β={self.beta:.3f}). Very volatile stock"
        elif self.beta > 1.0:
            return f"Elevated systematic risk (β={self.beta:.3f}). More volatile than market"
        elif self.beta > 0.8:
            return f"Moderate systematic risk (β={self.beta:.3f}). Similar to market"
        elif self.beta > 0.5:
            return f"Contained systematic risk (β={self.beta:.3f}). Less volatile than market"
        elif self.beta > 0:
            return f"Low systematic risk (β={self.beta:.3f}). Weakly correlated to market"
        else:
            return f"Negative correlation (β={self.beta:.3f}). Moves against market"

    def interpret_sensitivity(self):
        """Interprets market sensitivity"""
        sensitivity = abs(self.beta) * 100
        if abs(self.beta) > 1:
            return f"For every 1% market movement, stock moves {sensitivity:.1f}%"
        else:
            return f"For every 1% market movement, stock moves {sensitivity:.1f}%"

    def interpret_r_squared(self):
        """Interprets R-squared"""
        r2_pct = self.r_squared * 100
        if self.r_squared > 0.7:
            return f"Excellent quality ({r2_pct:.1f}%). Market explains stock movements well"
        elif self.r_squared > 0.5:
            return f"Good quality ({r2_pct:.1f}%). Moderately strong relationship with market"
        elif self.r_squared > 0.3:
            return f"Average quality ({r2_pct:.1f}%). Weak relationship with market"
        else:
            return f"Low quality ({r2_pct:.1f}%). Market explains little of stock movements"

    def interpret_correlation(self):
        """Interprets correlation"""
        corr_pct = abs(self.correlation) * 100
        if abs(self.correlation) > 0.8:
            return f"Very strong correlation ({corr_pct:.1f}%)"
        elif abs(self.correlation) > 0.6:
            return f"Strong correlation ({corr_pct:.1f}%)"
        elif abs(self.correlation) > 0.4:
            return f"Moderate correlation ({corr_pct:.1f}%)"
        else:
            return f"Weak correlation ({corr_pct:.1f}%)"

    def interpret_alpha(self):
        """Interprets alpha"""
        alpha_pct = self.alpha * 100
        if abs(alpha_pct) > 1:
            return f"Significant alpha ({alpha_pct:.2f}% monthly). Abnormal return vs market"
        else:
            return f"Non-significant alpha ({alpha_pct:.2f}% monthly). In line with market"

    def dcf_recommendation(self):
        """Recommendation for DCF use"""
        if self.r_squared > 0.5 and len(self.stock_returns) > 100:
            return f"✅ Reliable beta for DCF (R²={self.r_squared:.3f}, n={len(self.stock_returns)})"
        elif self.r_squared > 0.3:
            return f"⚠️ Beta usable with caution for DCF (R²={self.r_squared:.3f})"
        else:
            return f"❌ Beta unreliable for DCF. Consider sector beta (R²={self.r_squared:.3f})"


def get_user_input():
    """
    Collects user input for beta analysis.
    """
    print("=" * 60)
    print("BETA CALCULATION - DCF ANALYSIS")
    print("=" * 60)

    # Stock ticker input
    print("\nTicker examples:")
    print("• Italian stocks: ENI.MI, ENEL.MI, ISP.MI, UCG.MI, TIT.MI")
    print("• US stocks: AAPL, MSFT, GOOGL, TSLA, NVDA")
    stock_symbol = input("\nEnter stock ticker: ").upper().strip()

    # Reference index input
    print("\nIndex examples:")
    print("• FTSE MIB (Italy): FTSEMIB.MI")
    print("• S&P 500 (USA): ^GSPC")
    print("• NASDAQ: ^IXIC")
    print("• DAX (Germany): ^GDAXI")
    index_symbol = input("\nEnter reference index ticker: ").upper().strip()

    # Period input
    print("\nAvailable periods:")
    print("• 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")
    period = input("\nEnter period (default: 2y): ").strip() or "2y"

    # Interval input
    print("\nAvailable intervals:")
    print("• 1d (daily), 1wk (weekly), 1mo (monthly)")
    print("• For intraday data: 1m, 5m, 15m, 30m, 1h")
    interval = input("\nEnter interval (default: 1d): ").strip() or "1d"

    # Custom dates option
    use_custom_dates = input("\nUse custom dates? (y/n, default: n): ").strip().lower()
    start_date, end_date = None, None

    if use_custom_dates in ['y', 'yes']:
        start_date = input("Start date (YYYY-MM-DD): ").strip()
        end_date = input("End date (YYYY-MM-DD): ").strip()
        period = None  # Don't use period if using custom dates

    return stock_symbol, index_symbol, period, interval, start_date, end_date


# Example usage with user input
if __name__ == "__main__":
    try:
        # Collect user input
        stock_symbol, index_symbol, period, interval, start_date, end_date = get_user_input()

        # Initialize calculator
        calc = BetaCalculator()

        print(f"\n{'=' * 60}")
        print("STARTING ANALYSIS...")
        print(f"{'=' * 60}")

        # Download data
        success = calc.download_data(
            stock_symbol=stock_symbol,
            index_symbol=index_symbol,
            period=period,
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )

        if success:
            # Calculate returns
            if calc.calculate_returns():
                # Calculate beta
                if calc.calculate_beta():
                    # Ask if showing regression plot
                    show_plot = input("\nShow regression plot? (y/n, default: y): ").strip().lower()
                    if show_plot not in ['n', 'no']:
                        calc.plot_regression()

                    # Show additional statistics
                    stats = calc.get_summary_stats()
                    if stats:
                        print(f"\n{'=' * 60}")
                        print(f"ADDITIONAL STATISTICS")
                        print(f"{'=' * 60}")
                        for key, value in stats.items():
                            if 'Return' in key or 'Volatility' in key:
                                print(f"{key}: {value:.2%}")
                            else:
                                print(f"{key}: {value:.4f}")

                    # Save results
                    save_excel = input("\nSave results to Excel? (y/n, default: y): ").strip().lower()
                    if save_excel not in ['n', 'no']:
                        excel_file = calc.save_to_excel(stock_symbol=stock_symbol, index_symbol=index_symbol)
                        if excel_file:
                            print(f"📁 Excel file created: {excel_file}")

                    # Option to save text summary (kept for compatibility)
                    save_text = input("Save text summary too? (y/n, default: n): ").strip().lower()
                    if save_text in ['y', 'yes']:
                        filename = f"beta_summary_{stock_symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(f"BETA ANALYSIS - {stock_symbol} vs {index_symbol}\n")
                            f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(
                                f"Period: {calc.stock_data.index[0].strftime('%Y-%m-%d')} - {calc.stock_data.index[-1].strftime('%Y-%m-%d')}\n")
                            f.write(f"Interval: {interval}\n\n")
                            f.write(f"Beta: {calc.beta:.4f}\n")
                            f.write(f"Alpha: {calc.alpha:.6f} ({calc.alpha * 100:.4f}%)\n")
                            f.write(f"R-squared: {calc.r_squared:.4f}\n")
                            f.write(f"Correlation: {calc.correlation:.4f}\n")
                            f.write(f"\nBeta interpretation: {calc.interpret_beta()}\n")
                            f.write(f"DCF recommendation: {calc.dcf_recommendation()}\n")
                            if stats:
                                f.write(f"\nAdditional statistics:\n")
                                for key, value in stats.items():
                                    if 'Return' in key or 'Volatility' in key:
                                        f.write(f"{key}: {value:.2%}\n")
                                    else:
                                        f.write(f"{key}: {value:.4f}\n")
                        print(f"📄 Text summary saved to: {filename}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Check entered tickers and try again.")