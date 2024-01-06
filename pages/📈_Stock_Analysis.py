import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
import base64
import plotly.express as px
from datetime import datetime


## ............................................... ##
# Set page configuration (Call this once and make changes as needed)
st.set_page_config(page_title='Regression Stocks Prediction',  layout='wide', page_icon=':rocket:')

## ............................................... ##
# Add a dictonary of stock tickers and their company names and make a drop down menu to select the stock to predict
stock_tickers = {
    "MAPI":"MAPI.JK","MAP Aktif": "MAPA.JK","MAP Boga": "MAPB.JK",
    "Tesla": "TSLA", "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL", 
    "Meta": "META", "Amazon": "AMZN", "Netflix": "NFLX", "Alphabet": "GOOG", 
    "Nvidia": "NVDA", "Paypal": "PYPL", "Adobe": "ADBE", "Intel": "INTC", 
    "Cisco": "CSCO", "Comcast": "CMCSA", "Pepsi": "PEP", "Costco": "COST", 
    "Starbucks": "SBUX", "Walmart": "WMT", "Disney": "DIS", "Visa": "V", 
    "Mastercard": "MA", "Boeing": "BA", "IBM": "IBM", "McDonalds": "MCD", 
    "Nike": "NKE", "Exxon": "XOM", "Chevron": "CVX", "Verizon": "VZ", 
    "AT&T": "T", "Home Depot": "HD", "Salesforce": "CRM", "Oracle": "ORCL", 
    "Qualcomm": "QCOM", "AMD": "AMD"
}

st.sidebar.title("Stock Option")
# Custom CSS to change the sidebar color
sidebar_css = """
<style>
    div[data-testid="stSidebar"] > div:first-child {
        width: 350px;  # Adjust the width as needed
        background-color: #FF6969;
    }
</style>
"""

# User Input
#default_index = stock_tickers.keys().index("MAPI.JK") if "MAPI.JK" in stock_tickers.keys() else 0
#st.markdown(sidebar_css, unsafe_allow_html=True)
#user_input = st.sidebar.selectbox("Select a Stock", list(stock_tickers.keys()), index=default_index , key="main_selectbox")
#stock_name = user_input
#user_input = stock_tickers[user_input]

user_input = st.sidebar.text_input("Select a Stock", "MAPI.JK")

# User input for start and end dates using calendar widget
start_date = st.sidebar.date_input("Select start date:", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Select end date:", datetime(2023, 12, 1))


st.sidebar.markdown("----")
st.sidebar.markdown("© 2023 Stocks Prediction App")
st.sidebar.markdown("----")
st.sidebar.markdown("Example Stock Tickers")
st.sidebar.markdown(stock_tickers)

## ............................................... ##
# Page Title and Description
st.title("Stock Price Analysis and Prediction")
st.write("Created by Bayhaqy")
st.write("Using Dataset MAPI to Train and Test the Model")

## ............................................... ##

# Try to fetch information from Yahoo Finance
try:
    stock_info = yf.Ticker(user_input).info

    # Check if the 'longName' key exists in stock_info
    if 'longName' in stock_info:
        stock_name = stock_info['longName']
    else:
        stock_name = user_input

    # Fetch the latest news using yfinance
    news_data = yf.Ticker(user_input).news

    # Create a Streamlit app
    title = f"<h1 style='color: red; font-size: 25px; text-align: center; '>{stock_name}'s Stock Fundamental Analysis</h1>"
    st.markdown(title, unsafe_allow_html=True)

    with st.expander("See Details"):
        ## ............................................... ##
        # Display the retrieved stock information
        if 'longName' in stock_info:
            company_name = stock_info['longName']
            st.write(f"Company Name: {company_name}")
        else:
            st.write("Company name information not available for this stock.")

        if 'industry' in stock_info:
            Industry = stock_info['industry']
            st.write(f"Industry: {Industry}")
        else:
            st.write("Industry information not available for this stock.")

        if 'sector' in stock_info:
            Sector = stock_info['sector']
            st.write(f"Sector: {Sector}")
        else:
            st.write("Sector information not available for this stock.")

        if 'website' in stock_info:
            Website = stock_info['website']
            st.write(f"Sector: {Website}")
        else:
            st.write("Website information not available for this stock.")

        if 'marketCap' in stock_info:
            MarketCap = stock_info['marketCap']
            st.write(f"Market Cap: {MarketCap}")
        else:
            st.write("Market Cap information not available for this stock.")

        if 'previousClose' in stock_info:
            PreviousClose = stock_info['previousClose']
            st.write(f"Previous Close: {PreviousClose}")
        else:
            st.write("Previous Close information not available for this stock.")

        if 'dividendYield' in stock_info:
            dividend_yield = stock_info['dividendYield'] * 100  # Convert to percentage
            st.write(f"Dividend Yield: {dividend_yield:.2f}%")
        else:
            st.write("Dividend Yield information not available for this stock.")

        ## ............................................... ##
        # Display financial metrics
        st.subheader('Financial Metrics')
        if 'trailingEps' in stock_info:
            trailing_eps = stock_info['trailingEps']
            st.write(f"Earnings Per Share (EPS): {trailing_eps:.2f}")
        else:
            st.write("Earnings Per Share (EPS) information not available for this stock.")

        if 'trailingPE' in stock_info:
            trailing_pe = stock_info['trailingPE']
            st.write(f"Price-to-Earnings (P/E) Ratio: {trailing_pe:.2f}")
        else:
            st.write("Price-to-Earnings (P/E) Ratio information not available for this stock.")

        if 'priceToSalesTrailing12Months' in stock_info:
            priceToSalesTrailing_12Months = stock_info['priceToSalesTrailing12Months']
            st.write(f"Price-to-Sales (P/S) Ratio: {priceToSalesTrailing_12Months:.2f}")
        else:
            st.write("Price-to-Sales (P/S) Ratio information not available for this stock.")

        if 'priceToBook' in stock_info:
            price_ToBook = stock_info['priceToBook']
            st.write(f"Price-to-Book (P/B) Ratio: {price_ToBook:.2f}")
        else:
            st.write("Price-to-Book (P/B) Ratio information not available for this stock.")

        ## ............................................... ##
        # Display more detailed information
        st.subheader('Company Summary')
        if 'longBusinessSummary' in stock_info:
            LongBusinessSummary = stock_info['longBusinessSummary']
            st.write(f"{LongBusinessSummary}")
        else:
            st.write("Company Summary information not available for this stock.")

        ## ............................................... ##
        # Display information about company officers
        st.subheader('Company Officers')
        if 'fullTimeEmployees' in stock_info:
            FullTimeEmployees = stock_info['fullTimeEmployees']
            st.write(f"Full Time Employees: {FullTimeEmployees}")
        else:
            st.write("Full Time Employees information not available for this stock.")

        if 'companyOfficers' in stock_info:
            officers = stock_info['companyOfficers']
            officer_data = []

            # Create a DataFrame
            df = pd.DataFrame(columns=["Name", "Title", "Age"])

            # Populate the DataFrame with officer information
            #for officer in officers:
            #    officer_data.append([officer['name'], officer['title'], officer['age']])

            for officer in officers:
                name = officer.get('name', 'N/A')
                title = officer.get('title', 'N/A')
                age = officer.get('age', 'N/A')
                officer_data.append([name, title, age])


            df = pd.concat([df, pd.DataFrame(officer_data, columns=["Name", "Title", "Age"])], ignore_index=True)

            # Display the DataFrame using Markdown without the index column
            st.markdown(df.to_markdown(index=False))

        else:
            st.write("Company Officers information not available for this stock.")

        ## ............................................... ##
        # Display the latest news
        st.subheader('Latest News')

        # Prepare the data for the DataFrame
        news_data_for_dataframe = []
        for news_item in news_data:
            news_title = news_item['title']
            news_publisher = news_item['publisher']
            news_provider_publish_time = pd.to_datetime(news_item['providerPublishTime'], unit='s')
            news_type = news_item['type']
            news_link = f"[Link]({news_item['link']})"
            news_data_for_dataframe.append([news_title, news_publisher, news_provider_publish_time, news_type, news_link])

        # Create a Pandas DataFrame
        df = pd.DataFrame(news_data_for_dataframe, columns=["Title", "Publisher", "Provider Publish Time", "Type", "Link"])

        # Display the DataFrame using Markdown without the index column
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)

    ## ............................................... ##
    # Enhanced title with larger font size and a different color
    title = f"<h1 style='color: red; font-size: 25px; text-align: center; '>{stock_name}'s Stock Technical Analysis</h1>"
    st.markdown(title, unsafe_allow_html=True)

    with st.expander("See Details"):
        # Describing the data
        st.subheader(f'Data from {start_date} - {end_date}')
        data = yf.download(user_input, start_date, end_date)

        if data.empty:
            # Display a message to the user when no data is available
            st.warning(f"No data available for stock symbol {stock_name} in the specified date range.")
        else:
            # Reset the index to add the date column
            data = data.reset_index()

            # Display data in a Plotly table
            fig = go.Figure(data=[go.Table(
                    header=dict(values=list(data.columns),
                                font=dict(size=12, color='white'),
                                fill_color='#264653',
                                line_color='rgba(255,255,255,0.2)',
                                align=['left', 'center'],
                                height=20),
                    cells=dict(values=[data[k].tolist() for k in data.columns],
                              font=dict(size=12),
                              align=['left', 'center'],
                              line_color='rgba(255,255,255,0.2)',
                              height=20))])

            fig.update_layout(title_text=f"Data for {stock_name}", title_font_color='#264653', title_x=0, margin=dict(l=0, r=10, b=10, t=30))

            st.plotly_chart(fig, use_container_width=True)


            st.markdown(f"<h2 style='text-align: center; color: #264653;'>Data Overview for {stock_name}</h2>", unsafe_allow_html=True)
            # Get the description of the data
            description = data.describe()

            # Dictionary of columns and rows to highlight
            highlight_dict = {
                "Open": ["mean", "min", "max", "std"],
                "High": ["mean", "min", "max", "std"],
                "Low": ["mean", "min", "max", "std"],
                "Close": ["mean", "min", "max", "std"],
                "Adj Close": ["mean", "min", "max", "std"]
            }

            # Colors for specific rows
            color_dict = {
                "mean": "lightgreen",
                "min": "salmon",
                "max": "lightblue",
                "std": "lightyellow"
            }

            # Function to highlight specific columns and rows based on the dictionaries
            def highlight_specific_cells(val, col_name, row_name):
                if col_name in highlight_dict and row_name in highlight_dict[col_name]:
                    return f'background-color: {color_dict[row_name]}'
                return ''

            styled_description = description.style.apply(lambda row: [highlight_specific_cells(val, col, row.name) for col, val in row.items()], axis=1)

            # Display the styled table in Streamlit
            st.table(styled_description)

            ### ............................................... ##
            # Stock Price Over Time
            g1, g2, g3 = st.columns((1.2,1.2,1))

            fig1 = px.line(data, x='Date', y='Close', template='seaborn')
            fig1.update_traces(line_color='#264653')
            fig1.update_layout(title_text="Stock Price Over Time", title_x=0, margin=dict(l=20, r=20, b=20, t=30), yaxis_title=None, xaxis_title=None, height=400, width=700)
            g1.plotly_chart(fig1, use_container_width=True)

            # Volume of Stocks Traded Over Time
            fig2 = px.bar(data, x='Date', y='Volume', template='seaborn')
            fig2.update_traces(marker_color='#7A9E9F')
            fig2.update_layout(title_text="Volume of Stocks Traded Over Time", title_x=0, margin=dict(l=20, r=20, b=20, t=30), yaxis_title=None, xaxis_title=None, height=400, width=700)
            g2.plotly_chart(fig2, use_container_width=True)

            # Moving Averages
            short_window = 40
            long_window = 100
            data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
            data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
            fig3 = px.line(data, x='Date', y='Close', template='seaborn')
            fig3.add_scatter(x=data['Date'], y=data['Short_MA'], mode='lines', line=dict(color="red"), name=f'Short {short_window}D MA')
            fig3.add_scatter(x=data['Date'], y=data['Long_MA'], mode='lines', line=dict(color="blue"), name=f'Long {long_window}D MA')
            fig3.update_layout(title_text="Stock Price with Moving Averages", title_x=0, margin=dict(l=20, r=20, b=20, t=30), yaxis_title=None, xaxis_title=None, legend=dict(orientation="h", yanchor="bottom", y=0.9, xanchor="right", x=0.99), height=400, width=700)
            g3.plotly_chart(fig3, use_container_width=True)

            ## ............................................... ##
            # Daily Returns
            g4, g5, g6 = st.columns((1,1,1))
            data['Daily_Returns'] = data['Close'].pct_change()
            fig4 = px.line(data, x='Date', y='Daily_Returns', template='seaborn')
            fig4.update_traces(line_color='#E76F51')
            fig4.update_layout(title_text="Daily Returns", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
            g4.plotly_chart(fig4, use_container_width=True)

            # Cumulative Returns
            data['Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod()
            fig5 = px.line(data, x='Date', y='Cumulative_Returns', template='seaborn')
            fig5.update_traces(line_color='#2A9D8F')
            fig5.update_layout(title_text="Cumulative Returns", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
            g5.plotly_chart(fig5, use_container_width=True)
            # Stock Price Distribution
            fig6 = px.histogram(data, x='Close', template='seaborn', nbins=50)
            fig6.update_traces(marker_color='#F4A261')
            fig6.update_layout(title_text="Stock Price Distribution", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
            g6.plotly_chart(fig6, use_container_width=True)

            ## ............................................... ##

            # Bollinger Bands
            g7, g8, g9 = st.columns((1,1,1))
            rolling_mean = data['Close'].rolling(window=20).mean()
            rolling_std = data['Close'].rolling(window=20).std()
            data['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
            data['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)
            fig7 = px.line(data, x='Date', y='Close', template='seaborn')
            fig7.add_scatter(x=data['Date'], y=data['Bollinger_Upper'], mode='lines', line=dict(color="green"), name='Upper Bollinger Band')
            fig7.add_scatter(x=data['Date'], y=data['Bollinger_Lower'], mode='lines', line=dict(color="red"), name='Lower Bollinger Band')
            fig7.update_layout(title_text="Bollinger Bands", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
            g7.plotly_chart(fig7, use_container_width=True)

            # Stock Price vs. Volume
            fig8 = px.line(data, x='Date', y='Close', template='seaborn')
            fig8.add_bar(x=data['Date'], y=data['Volume'], name='Volume')
            fig8.update_layout(title_text="Stock Price vs. Volume", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
            g8.plotly_chart(fig8, use_container_width=True)

            # MACD
            data['12D_EMA'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['26D_EMA'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = data['12D_EMA'] - data['26D_EMA']
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
            fig9 = px.line(data, x='Date', y='MACD', template='seaborn', title="MACD")
            fig9.add_scatter(x=data['Date'], y=data['Signal_Line'], mode='lines', line=dict(color="orange"), name='Signal Line')
            fig9.update_layout(title_text="MACD", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
            g9.plotly_chart(fig9, use_container_width=True)

            ### ............................................... ##

            # Relative Strength Index (RSI)
            g10, g11, g12 = st.columns((1,1,1))
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))
            fig10 = px.line(data, x='Date', y='RSI', template='seaborn')
            fig10.update_layout(title_text="Relative Strength Index (RSI)", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
            g10.plotly_chart(fig10, use_container_width=True)

            # Candlestick Chart
            fig11 = go.Figure(data=[go.Candlestick(x=data['Date'],
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'])])
            fig11.update_layout(title_text="Candlestick Chart", title_x=0, margin=dict(l=0, r=10, b=10, t=30))
            g11.plotly_chart(fig11, use_container_width=True)

            # Correlation Matrix
            corr_matrix = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
            fig12 = px.imshow(corr_matrix, template='seaborn')
            fig12.update_layout(title_text="Correlation Matrix", title_x=0, margin=dict(l=0, r=10, b=10, t=30))
            g12.plotly_chart(fig12, use_container_width=True)

            ### ............................................... ##
            # Price Rate of Change (ROC)
            g13, g14, g15 = st.columns((1,1,1))
            n = 12
            data['ROC'] = ((data['Close'] - data['Close'].shift(n)) / data['Close'].shift(n)) * 100
            fig13 = px.line(data, x='Date', y='ROC', template='seaborn')
            fig13.update_layout(title_text="Price Rate of Change (ROC)", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
            g13.plotly_chart(fig13, use_container_width=True)

            # Stochastic Oscillator
            low_min = data['Low'].rolling(window=14).min()
            high_max = data['High'].rolling(window=14).max()
            data['%K'] = (100 * (data['Close'] - low_min) / (high_max - low_min))
            data['%D'] = data['%K'].rolling(window=3).mean()
            fig14 = px.line(data, x='Date', y='%K', template='seaborn')
            fig14.add_scatter(x=data['Date'], y=data['%D'], mode='lines', line=dict(color="orange"), name='%D (3-day SMA of %K)')
            fig14.update_layout(title_text="Stochastic Oscillator", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
            g14.plotly_chart(fig14, use_container_width=True)

            # Historical Volatility
            data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Historical_Volatility'] = data['Log_Return'].rolling(window=252).std() * np.sqrt(252)
            fig15 = px.line(data, x='Date', y='Historical_Volatility', template='seaborn')
            fig15.update_layout(title_text="Historical Volatility (252-day)", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
            g15.plotly_chart(fig15, use_container_width=True)

            ### ............................................... ##

            # Visualizing the data and want to get the data when hovering over the graph
            st.subheader('Closing Price vs Time Chart')
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
            fig1.layout.update(hovermode='x')
            # Display the figure in Streamlit
            st.plotly_chart(fig1,use_container_width=True)

            st.subheader('Closing Price vs Time Chart with 100MA')
            ma100 = data['Close'].rolling(100).mean()
            fig2 = go.Figure()
            # Add traces for 100MA and Closing Price
            fig2.add_trace(go.Scatter(x=data.index, y=ma100, mode='lines', name='100MA'))
            fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price'))
            fig2.layout.update(hovermode='x')
            # Display the figure in Streamlit
            st.plotly_chart(fig2,use_container_width=True)

            st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
            ma100 = data['Close'].rolling(100).mean()
            ma200 = data['Close'].rolling(200).mean()
            fig3 = go.Figure()
            # Add traces for 100MA and Closing Price
            fig3.add_trace(go.Scatter(x=data.index, y=ma100, mode='lines', name='100MA'))
            fig3.add_trace(go.Scatter(x=data.index, y=ma200, mode='lines', name='200MA'))
            fig3.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price'))
            fig3.layout.update(hovermode='x')

            # Display the figure in Streamlit
            st.plotly_chart(fig3,use_container_width=True)

    ## ............................................... ##
    # Enhanced title with larger font size and a different color
    title = f"<h1 style='color: red; font-size: 25px; text-align: center; '>{stock_name}'s Stock Prediction</h1>"
    st.markdown(title, unsafe_allow_html=True)

    with st.expander("See Details"):
        # Define the minimum number of days required for predictions
        minimum_days_for_predictions = 100

        # Check if the available data is less than the minimum required days
        if len(data) < minimum_days_for_predictions:
            st.warning(f"The available data for {stock_name} is less than {minimum_days_for_predictions} days, which is insufficient for predictions.")
        else:
            # Splitting the data into training and testing data
            data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
            data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70): int(len(data))])

            # Scaling the data
            scaler = MinMaxScaler(feature_range=(0,1))
            data_training_array = scaler.fit_transform(data_training)

            # load the model
            model = load_model('best_model_MAPI.h5')

            # Testing the model
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days,data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df)

            x_test = []
            y_test = []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i,0])

            x_test, y_test = np.array(x_test), np.array(y_test)

            y_predicted = model.predict(x_test)

            scaler = scaler.scale_
            scale_factor = 1/scaler[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # Visualizing the results
            st.subheader('Predictions vs Actual')
            fig4 = go.Figure()
            # Add traces for Actual and Predicted Price
            fig4.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_test, mode='lines', name='Actual Price'))
            fig4.add_trace(go.Scatter(x=data.index[-len(y_predicted):], y=y_predicted[:,0], mode='lines', name='Predicted Price'))
            fig4.layout.update(hovermode='x')
            # Display the figure in Streamlit
            st.plotly_chart(fig4,use_container_width=True)

except Exception as e:
    st.error(f"An error fetching stock information: {str(e)}")
