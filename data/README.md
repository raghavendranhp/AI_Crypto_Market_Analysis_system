# Crypto Trading Dataset

This dataset provides hourly historical trading data for three major cryptocurrencies: Bitcoin, Ethereum, and Solana.

## Data Schema

The `crypto_trading_dataset.csv` file contains the following columns:

- **`timestamp`**: The standard date and time of the recorded hourly window (e.g., `2025-01-01 00:00:00`).
- **`coin_name`**: The name of the cryptocurrency being tracked (`Bitcoin`, `Ethereum`, or `Solana`).
- **`open_price`**: The price of the asset at the beginning of the hourly window (in USD).
- **`close_price`**: The price of the asset at the end of the hourly window (in USD). This is the primary feature used for model prediction and trend analysis.
- **`high_price`**: The highest price the asset reached during the hourly window (in USD).
- **`low_price`**: The lowest price the asset reached during the hourly window (in USD).
- **`volume`**: The total trading volume of the asset during the hourly period.
- **`market_cap`**: The total market capitalization of the asset at the time of recording (in USD).

## Usage in Models
This dataset drives both Phase 1 model derivations:
1. **Target Feature (`target_trend`)**: Derived by checking whether the `close_price` increases, decreases, or stays flat in the **next** chronological period.
2. **Predictive Features**:
    - **`price_change_pct`**: The percentage difference between sequential `close_price` data points.
    - **`ma7` / `ma30`**: The 7-period and 30-period simple moving averages of the `close_price`.
    - **`volatility_index`**: The standard deviation of the `price_change_pct` extending over a rolling 7-period window.
    - **`volume_spike_ratio`**: A dynamic ratio isolating the current `volume` relative to its 7-period moving average.

*(Dataset spans early January 2025 across multiple intervals per featured asset.)*
