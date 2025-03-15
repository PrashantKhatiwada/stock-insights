# Stock Market Insights & Prediction App

Welcome to the Stock Market Insights & Prediction App, a web application built using Streamlit, Python, and various machine learning libraries. This app provides advanced stock analysis, sentiment analysis from news articles, and predictions for stock prices using historical data and an LSTM model.

## Features

-   **Stock Data Visualization:** Displaying the historical stock data for any company (6 months of data by default) including technical indicators like SMA, EMA, and VWAP.
-   **Stock Price Prediction:** Predict the stock price for the next 10-15 days using a trained LSTM (Long Short-Term Memory) model.
-   **News Sentiment Analysis:** Analyzes the sentiment of recent news articles related to the stock, providing an overview of positive, neutral, or negative sentiment.
-   **Email Alerts:** Users can enter their email address to receive alerts about the stock sentiment.
-   **Interactive Charts:** Interactive stock price charts powered by Plotly for both historical data and predicted prices.

## Technologies Used

### Python Libraries:

-   `yfinance`: To fetch stock data.
-   `pandas`, `numpy`: For data manipulation and numerical operations.
-   `streamlit`: For building the web interface.
-   `plotly`: For interactive data visualization.
-   `TextBlob`: For performing sentiment analysis on news articles.
-   `BeautifulSoup`: To scrape news articles.
-   `ta`: Technical analysis library for stock data indicators.
-   `tensorflow`: For training and using the LSTM model.
-   `sklearn`: For preprocessing and data scaling.
-   `smtplib`: For sending email alerts.
-   `aiohttp`: For asynchronous HTTP requests.

## Setup Instructions

1.  **Clone the Repository**

    ```bash
    git clone [https://github.com/yourusername/stock-insights.git](https://github.com/yourusername/stock-insights.git)
    cd stock-insights
    ```

2.  **Install Dependencies**

    Make sure you have `pip` installed. Create a virtual environment and install the required dependencies:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # For macOS/Linux
    venv\Scripts\activate     # For Windows
    pip install -r requirements.txt
    ```

3.  **Set Up Environment Variables**

    This app requires certain environment variables, such as your email credentials, to send email alerts. You can store these credentials in a `.env` file:

    ```bash
    EMAIL_SENDER=your-email@example.com
    EMAIL_PASSWORD=your-email-password
    ```

    Make sure to create the `.env` file in the root of the project directory with the above variables.

4.  **Run the Application**

    Once everything is set up, you can run the app locally using the following command:

    ```bash
    streamlit run main.py
    ```

    The app should open in your default web browser.

## Usage

-   **Enter Stock Ticker:** Input the stock ticker (e.g., AAPL, TSLA) in the provided text box.
-   **Email Alerts:** Optionally, input your email to receive sentiment updates for the stock.
-   **Analyze:** Click on the "Analyze" button to get the stock's historical data, predictions for the next 10-15 days, and news sentiment analysis.
-   **Visualize:** Interactive charts will display the stock price and its predictions.

## Deployment

You can deploy this app using Streamlit Cloud, Heroku, or any platform that supports Python apps.

-   **Streamlit Cloud**
-   **Heroku**

## Contributing

We welcome contributions to this project! Feel free to fork this repository, submit issues, and create pull requests with any improvements or bug fixes.

### Steps for Contributing

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -am 'Add feature'`).
5.  Push to the branch (`git push origin feature-name`).
6.  Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

-   `yfinance` - Yahoo Finance API wrapper.
-   `TensorFlow` - Deep learning library for training LSTM models.
-   `Streamlit` - Framework for building interactive web apps.
-   `Plotly` - For interactive data visualization.
-   `TextBlob` - Natural Language Processing (NLP) library for sentiment analysis.
-   `BeautifulSoup` - For web scraping news articles.