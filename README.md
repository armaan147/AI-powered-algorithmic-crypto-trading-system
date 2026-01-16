# AI-Powered Algorithmic Crypto Trading System

A sophisticated algorithmic trading system for cryptocurrency that leverages machine learning, technical analysis, and correlation modeling to identify optimal trading opportunities across multiple digital assets.

## Overview

This system analyzes Bitcoin (BTC) price trends using regression analysis and identifies the best-performing altcoins through correlation and sensitivity metrics. It employs Exponential Moving Averages (EMAs) to generate trade signals and provides actionable buy/sell recommendations with take-profit and stop-loss targets.

## Key Features

- **Trend Analysis**: Linear regression-based BTC trend identification (Positive/Negative/Flat)
- **EMA Optimization**: Automated search for the best-performing Exponential Moving Averages
- **Velocity Metrics**: Measures price reaction velocity at EMA touch points
- **Bounce Efficiency**: Calculates the efficiency of price bounces from support/resistance levels
- **Cross-Asset Correlation**: Analyzes correlation, sensitivity, and trend relationships between BTC and other cryptocurrencies
- **Trade Signals**: Generates long/short trade recommendations with dynamic take-profit and stop-loss levels
- **Interactive Dashboard**: Real-time visualization using Streamlit and Plotly

## System Architecture

### Data Pipeline
1. **Data Ingestion**: Real-time data fetching from Binance API
2. **Data Processing**: Cleaning, normalization, and feature engineering
3. **Analysis**: Regression, correlation, and EMA calculations
4. **Signal Generation**: Automated trade signal generation
5. **Backtesting**: Historical performance validation

### Analysis Components

- **Regression Analysis**: Determines price trend direction
- **EMA Strategy**: Tests EMA periods (15-75) to find optimal parameters
- **Velocity Analysis**: Measures price movement speed relative to EMA
- **Bounce Efficiency**: Quantifies support/resistance effectiveness
- **Correlation Matrix**: Evaluates cryptocurrency dependencies

## Project Structure

```
.
├── notebooks/              # Jupyter analysis notebooks
│   ├── 1loading data.ipynb
│   ├── 2Running Regression.ipynb
│   ├── 3Plotting Regression.ipynb
│   ├── 4calculating velocity.ipynb
│   ├── 5Calculating Bounce Rate Efficiency.ipynb
│   ├── 6Finding the best EMA.ipynb
│   ├── 7Calculating the Correlations between BTC and Other Coins.ipynb
│   ├── 8Plotting Suggesting Trades.ipynb
│   └── 9Backtesting.ipynb
├── src/                    # Production code
│   └── corr_v2.py         # Main Streamlit application
├── data/
│   └── processed/         # Cleaned and processed datasets
├── results/               # Analysis outputs and reports
└── requirements.txt       # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Binance API credentials (obtain from [Binance](https://www.binance.com/en/account/api-management))

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-crypto-trading-system
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API credentials:
   - Update `API_KEY` and `API_SECRET` in `src/corr_v2.py`
   - Or set environment variables: `BINANCE_API_KEY` and `BINANCE_API_SECRET`

## Usage

### Running the Dashboard

```bash
streamlit run src/corr_v2.py
```

Then open your browser to `http://localhost:8501`

### Using the Notebooks

Execute notebooks sequentially for detailed analysis:

1. Load and clean data
2. Perform regression analysis
3. Analyze price velocity
4. Calculate bounce efficiency
5. Optimize EMA parameters
6. Evaluate cryptocurrency correlations
7. Generate trade suggestions
8. Backtest strategies

## Configuration

### Analysis Parameters

Adjustable settings in the Streamlit interface:

- **Days of Data**: 1-30 days of historical data
- **Time Interval**: 1m, 5m, 15m, 30m, 1h, 4h, 1d
- **EMA Range**: Default 15-75 periods
- **Bounce Efficiency Threshold**: Default 0.005 (0.5%)
- **Velocity Window**: Default k=10 periods

### Scoring Metrics

- **Velocity Score**: 41% weight in EMA selection
- **Efficiency Score**: 59% weight in EMA selection
- **Correlation Score**: 31% weight in coin selection
- **Sensitivity Score**: 32% weight in coin selection
- **Trend Score**: 37% weight in coin selection

## Supported Cryptocurrencies

Primary focus: Bitcoin (BTCUSDT)

Analyzed altcoins:
- Ethereum (ETHUSDT)
- Dogecoin (DOGEUSDT)
- Solana (SOLUSDT)
- Ripple (XRPUSDT)
- Binance Coin (BNBUSDT)
- Cardano (ADAUSDT)
- TRON (TRXUSDT)
- Avalanche (AVAXUSDT)
- Chainlink (LINKUSDT)
- Uniswap (UNIUSDT)
- Litecoin (LTCUSDT)
- Arbitrum (ARBITUSDT)
- Shiba Inu (SHIBUSDT)
- dYdX (DYDXUSDT)

## Algorithm Details

### EMA Selection Logic

1. Tests EMA periods from 15 to 75 with step of 3
2. Calculates velocity (price movement speed) for each EMA
3. Calculates bounce efficiency (support/resistance effectiveness)
4. Scales both metrics (0-1 range)
5. Computes combined score: `0.41 × velocity + 0.59 × efficiency`
6. Selects EMA with highest score

### Trade Signal Generation

**Long Signal**: 
- BTC trend is positive
- Current price > Selected EMA

**Short Signal**:
- BTC trend is negative
- Current price < Selected EMA

### Position Sizing

- **Take Profit**: EMA ± (variance/6)
- **Stop Loss**: Take Profit / 4

## API Reference

### Main Functions

```python
fetch_crypto_data(symbol, interval, days)
  - Fetches OHLCV data from Binance
  
calculate_regression(df)
  - Returns trend direction and regression line
  
identify_best_ma_ema_and_plot(df)
  - Finds optimal EMA period
  
calculate_correlation_and_sensitivity(base_df, target_df)
  - Analyzes crypto pair relationships
  
suggest_trades(base_df, target_df, best_metric, trend)
  - Generates trade signals and targets
```

## Performance Metrics

The system evaluates performance through:

- **Historical Win Rate**: Percentage of profitable signals
- **Average Return per Trade**: Mean profit/loss per transaction
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Correlation Strength**: Measure of crypto pair synchronization

## Risk Disclaimer

**⚠️ DISCLAIMER**: This is an educational trading tool. Cryptocurrency trading carries substantial risk of loss. Past performance does not guarantee future results.

- Always use stop-loss orders
- Start with small position sizes
- Never invest more than you can afford to lose
- Validate signals before executing trades
- Consider market conditions and external factors

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Static data visualization
- **plotly**: Interactive visualizations
- **streamlit**: Web dashboard framework
- **python-binance**: Binance API client
- **Pillow**: Image processing

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

## Security

- Never commit API keys to version control
- Use environment variables for sensitive credentials
- Validate all user inputs
- Implement rate limiting for API calls
- Monitor account activity regularly

## Roadmap

- [ ] Machine learning model integration (LSTM/GRU)
- [ ] Multi-timeframe analysis
- [ ] Advanced risk management
- [ ] Portfolio optimization
- [ ] Mobile app integration
- [ ] Real-time alerts and notifications
- [ ] Paper trading mode

## Support

For issues and questions:
- Check existing GitHub issues
- Review the notebooks for detailed analysis
- Consult Binance API documentation

## License

This project is provided for educational purposes. Use at your own risk.

## Author

Armaan Yadav

---

**Last Updated**: January 2026

**Status**: Active Development

For questions or improvements, please open an issue or contact the maintainer.
