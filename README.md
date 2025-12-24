# Stock Market Predictor

A machine learning-powered stock market prediction application built with Python, Keras, and Streamlit. This application analyzes historical stock data and uses LSTM neural networks to predict future stock prices.

## Features

- Real-time stock data fetching using Yahoo Finance API
- Interactive visualization of stock prices and moving averages (MA50, MA100, MA200)
- Historical price prediction using trained LSTM model
- Future price prediction for customizable time periods (1-365 days)
- Clean and intuitive web interface powered by Streamlit

## Requirements

- Python 3.8 or higher
- TensorFlow/Keras
- Streamlit
- yfinance
- pandas
- numpy
- matplotlib
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Crucio104/stock-market-predictor.git
cd stock-market-predictor
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the trained model file `Stock Prediction Model.keras` is in the project directory.

## Usage

### Windows

Simply run the batch file:
```bash
.\run.bat
```

The application will automatically start and open in your default browser.

### Linux/Mac or Manual Start

Run the following command:
```bash
streamlit run Stock-market.py
```

Then navigate to `http://localhost:8501` in your web browser.

## How It Works

1. **Data Collection**: Fetches historical stock data from Yahoo Finance starting from 2012-01-01
2. **Data Processing**: Splits data into training (80%) and testing (20%) sets
3. **Normalization**: Uses MinMaxScaler to normalize data between 0 and 1
4. **Prediction**: Uses a pre-trained LSTM model to predict stock prices
5. **Visualization**: Displays multiple charts showing moving averages and predictions

## Model Architecture

The application uses an LSTM (Long Short-Term Memory) neural network trained on historical stock data with a 100-day lookback period.

## Project Structure

```
stock-market-predictor/
│
├── Stock-market.py              # Main application file
├── Stock Prediction Model.keras # Pre-trained LSTM model
├── run.bat                      # Windows batch file for easy execution
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore file
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## Configuration

The Streamlit configuration is stored in `.streamlit/config.toml` and includes:
- Server settings
- Browser configuration
- Port settings (default: 8501)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Crucio104**

## Disclaimer

This application is for educational purposes only. Stock market predictions are not guaranteed to be accurate. Always do your own research and consult with financial advisors before making investment decisions.
