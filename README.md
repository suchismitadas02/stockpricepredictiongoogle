# Stock Price Prediction using LSTM with Hyperparameter Tuning

## Overview

This project demonstrates the use of Long Short-Term Memory (LSTM) neural networks to predict stock prices. The model is trained on historical stock data from Google (GOOGL) and leverages hyperparameter tuning using Keras Tuner to optimize the model's performance. The project includes data preprocessing, model building, training, evaluation, and visualization of the results.

## Features

- **Data Preprocessing**: The dataset is preprocessed using Min-Max scaling to normalize the stock prices.
- **LSTM Model**: A sequential LSTM model is built with multiple layers, including dropout layers to prevent overfitting.
- **Hyperparameter Tuning**: The Keras Tuner is used to find the optimal hyperparameters for the LSTM model, including the number of units, dropout rate, and learning rate.
- **Training and Evaluation**: The model is trained on 80% of the dataset and evaluated on the remaining 20%. Early stopping is used to prevent overfitting during training.
- **Visualization**: The results are visualized to compare the actual and predicted stock prices.

## Dataset

The dataset used in this project is the historical stock price data for Google (GOOGL), which includes the following columns:

- **Date**: The date of the recorded data.
- **High**: The highest price of the stock on the given day.
- **Low**: The lowest price of the stock on the given day.
- **Open**: The opening price of the stock on the given day.
- **Close**: The closing price of the stock on the given day.
- **Volume**: The total number of shares traded during the day.
- **Adj Close**: The adjusted closing price of the stock after accounting for corporate actions like splits or dividends.

## Requirements

To run this project, you need the following Python libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `keras`
- `keras-tuner`
- `matplotlib`
- `seaborn`

You can install these libraries using pip:

```bash
pip install numpy pandas scikit-learn keras keras-tuner matplotlib seaborn
```

## Usage

1. **Load the Dataset**: The dataset is loaded from a CSV file containing historical stock prices.

2. **Data Preprocessing**: The data is preprocessed using Min-Max scaling to normalize the stock prices.

3. **Create Training and Test Datasets**: The dataset is split into training and test sets, with 80% of the data used for training and 20% for testing.

4. **Build the LSTM Model**: The LSTM model is built using Keras, with hyperparameters tuned using Keras Tuner.

5. **Train the Model**: The model is trained on the training dataset, with early stopping to prevent overfitting.

6. **Evaluate the Model**: The model is evaluated on the test dataset, and the test loss is calculated.

7. **Predict Stock Prices**: The model is used to predict stock prices on the test dataset.

8. **Visualize the Results**: The actual and predicted stock prices are plotted to visualize the model's performance.

## Results

The model's performance is visualized by plotting the actual and predicted stock prices. The plot shows how well the model predicts the stock prices compared to the actual values.

![Stock Price Prediction](stock_price_prediction.png)

## Conclusion

This project demonstrates the effectiveness of LSTM neural networks in predicting stock prices. By leveraging hyperparameter tuning, the model is optimized to achieve better performance. The results show that the model can capture the trends in stock prices, making it a useful tool for stock market analysis.

## Future Work

- **Feature Engineering**: Incorporate additional features such as technical indicators (e.g., moving averages, RSI) to improve model performance.
- **Model Optimization**: Experiment with different architectures and hyperparameters to further optimize the model.
- **Real-time Prediction**: Implement the model for real-time stock price prediction using live data feeds.

## Acknowledgments

- [Keras](https://keras.io/) for providing the deep learning framework.
- [Keras Tuner](https://keras-team.github.io/keras-tuner/) for hyperparameter tuning.
- [Pandas](https://pandas.pydata.org/) for data manipulation and analysis.
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for data visualization.

