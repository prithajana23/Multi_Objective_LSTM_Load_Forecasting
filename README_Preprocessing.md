## Dataset & Preprocessing

### Dataset Source
This work uses the **UCI Individual Household Electric Power Consumption** dataset:
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The raw data file required for experiments is:
- `household_power_consumption.txt`

### Data Loading
- The dataset is loaded using `pandas.read_csv()' with semicolon (`;') delimiter.
- The `Date' and `Time' columns are merged into a single `DateTime` column and used as the time index.

### Missing Value Handling
- Missing values marked with ``?'' are replaced with `NaN'.
- All columns are converted to numeric using `errors='coerce''.
- Missing values are filled using:
  - forward-fill (`ffill`)
  - backward-fill (`bfill`)

### Target Variable
The forecasting target used in this implementation is:
- `Global_active_power' (univariate forecasting)

All other features are discarded.

### Temporal Resampling (Forecasting Horizons)
To evaluate forecasting performance across multiple time resolutions, the time series is resampled using **mean aggregation**:

- **Minutely**: `T'
- **Hourly**: `H'
- **Daily**: `D'
- **Weekly**: `W'

After resampling, missing values (if any) are again filled with `ffill' and `bfill'.

### Train / Validation / Test Split
The dataset is split **chronologically** to avoid data leakage:

- 60% training
- 20% validation
- 20% testing

### Normalization
Min–Max scaling is applied to map values into `[0, 1]':

- The scaler is **fit only on training data**
- The trained scaler is applied to validation and test data

This ensures no information leakage from validation/test sets into training.

### Sliding Window (Supervised Sequence Creation)
The normalized time series is converted into supervised learning samples using a sliding window:

- Input: past `time_step' values
- Output: next-step prediction

The lookback window (`time_step') is defined for each resolution as:

- Minutely: 60
- Hourly: 60
- Daily: 30
- Weekly: 8

Each input sequence is reshaped into:
`(samples, time_step, 1)' for LSTM training.

