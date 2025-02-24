import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def generate_all_figures(df):
    """Generate all figures based on the provided dataframe"""
    if df is None or df.empty:
        st.warning("No data available to generate figures.")
        return None

    try:
        # st.write("### 👀 Basic Analysis")
        # Convert 'ship_date' to datetime
        df['ship_date'] = pd.to_datetime(df['ship_date'])

        # Fill missing values using .loc to avoid SettingWithCopyWarning
        mode_license = df['license'].mode()[0] if not df['license'].empty else "Unknown"
        df.loc[:, 'license'] = df['license'].fillna(mode_license)

        mode_city = df['city'].mode()[0] if not df['city'].empty else "Unknown"
        df.loc[:, 'city'] = df['city'].fillna(mode_city)

        # Convert columns to numeric using .loc
        df.loc[:, 'extension'] = pd.to_numeric(df['extension'], errors='coerce')
        df.loc[:, 'ordered'] = pd.to_numeric(df['ordered'], errors='coerce')
        df.loc[:, 'shipped'] = pd.to_numeric(df['shipped'], errors='coerce')
        df.loc[:, 'price'] = pd.to_numeric(df['price'], errors='coerce')

        # Extract year from 'ship_date'
        df.loc[:, 'year'] = df['ship_date'].dt.year.astype(str)  # Convert year to string

        # ==================================================
        # Figure 1: Total Revenue by Product Category
        # ==================================================
        revenue_by_category = df.groupby(['category', 'year'])['extension'].sum().reset_index()
        fig1 = px.bar(revenue_by_category, x='category', y='extension', color='year',
                    title='Total Revenue by Product Category (2023 vs 2024)',
                    labels={'extension': 'Total Revenue', 'category': 'Product Category'},
                    barmode='group')

        # ==================================================
        # Figure 2: Total Revenue by Customer Class
        # ==================================================
        revenue_by_class = df.groupby(['customer_class', 'year'])['extension'].sum().reset_index()
        fig2 = px.bar(revenue_by_class, x='customer_class', y='extension', color='year',
                    title='Total Revenue by Customer Class (2023 vs 2024)',
                    labels={'extension': 'Total Revenue', 'customer_class': 'Customer Class'},
                    barmode='group')

        # ==================================================
        # Figure 3: Daily Revenue Trend
        # ==================================================
        daily_revenue = df.groupby('ship_date')['extension'].sum().reset_index()
        fig3 = px.line(daily_revenue, x='ship_date', y='extension', 
                    title='Daily Revenue Trend', 
                    labels={'extension': 'Total Revenue', 'ship_date': 'Date'},
                    markers=True)

        # ==================================================
        # Figure 4: Price Distribution
        # ==================================================
        fig4 = px.histogram(df, x='price', nbins=50, 
                        title='Price Distribution', 
                        labels={'price': 'Product Price'},
                        marginal='box')  # Adds a boxplot on top

        # ==================================================
        # Figure 5: Total Sales by Destination State
        # ==================================================
        df.loc[:, 'state'] = df['state'].str.upper()
        state_sales = df.groupby('state')['extension'].sum().reset_index()
        state_sales['extension'] = pd.to_numeric(state_sales['extension'], errors='coerce')
        fig5 = px.choropleth(state_sales, 
                     locations='state', 
                     locationmode='USA-states', 
                     color='extension',
                     scope='usa',
                     title="Total Sales by Destination State",
                     color_continuous_scale="blues",
                     labels={'extension': 'Total Revenue'})             
        

        # st.write("### 👀 Analysis of Return Patterns")
        # ==================================================
        # Figure 6: Daily Returns Trend
        # ==================================================
        returns = df[df['extension'] < 0].copy()  # Use .copy() to avoid SettingWithCopyWarning
        daily_returns = returns.groupby('ship_date')['extension'].sum().reset_index()
        fig6 = px.line(daily_returns, x='ship_date', y='extension', 
                    title='Daily Returns Trend',
                    labels={'extension': 'Total Returns', 'ship_date': 'Date'},
                    markers=True)

        # ==================================================
        # Figure 7: Returns by Product Category
        # ==================================================
        returns_by_category = returns.groupby('category')['extension'].sum().reset_index()
        returns_by_category.loc[:, 'extension'] = returns_by_category['extension'].abs()
        returns_by_category['extension'] = pd.to_numeric(returns_by_category['extension'], errors='coerce')
        fig7 = px.bar(returns_by_category, x='category', y='extension', 
                    title='Returns by Product Category',
                    labels={'extension': 'Total Returns', 'category': 'Product Category'},
                    color='extension')

        # ==================================================
        # Figure 8: Returns by Customer Class
        # ==================================================
        returns_by_class = returns.groupby('customer_class')['extension'].sum().reset_index()
        returns_by_class.loc[:, 'extension'] = returns_by_class['extension'].abs()
        returns_by_class['extension'] = pd.to_numeric(returns_by_class['extension'], errors='coerce')
        fig8 = px.bar(returns_by_class, x='customer_class', y='extension', 
                    title='Returns by Customer Class',
                    labels={'extension': 'Total Returns', 'customer_class': 'Customer Class'},
                    color='extension')

        # ==================================================
        # Figure 9: Returns by Destination State
        # ==================================================
        returns_by_state = returns.groupby('state')['extension'].sum().reset_index()
        returns_by_state.loc[:, 'extension'] = returns_by_state['extension'].abs()
        returns_by_state['extension'] = pd.to_numeric(returns_by_state['extension'], errors='coerce')
        fig9 = px.choropleth(returns_by_state, 
                            locations='state', 
                            locationmode='USA-states', 
                            color='extension',
                            scope='usa',
                            title="Returns by Destination State",
                            color_continuous_scale="reds",
                            labels={'extension': 'Total Returns'})
        
        # st.write("### 👀 Adavanced Analysis")
        # ==================================================
        # Figure 10: Return Rate by Product Category
        # ==================================================
        # st.write("1️⃣ Instead of just looking at absolute returns, let's calculate the return percentage for each category and customer class.")
        # st.write("📌 Insight: This shows which categories have the highest return rate, helping to identify problematic products.")
        
        returns_by_category = df[df['extension'] < 0].groupby('category')['extension'].sum().abs()
        total_sales_by_category = df.groupby('category')['extension'].sum()
        return_rate = (returns_by_category / total_sales_by_category).reset_index().rename(columns={'extension': 'return_rate'})
        fig10 = px.bar(return_rate, x='category', y='return_rate', 
                    title='Return Rate by Product Category',
                    labels={'return_rate': 'Return Rate (%)', 'category': 'Product Category'},
                    text_auto='.2%')

        # ==================================================
        # Figure 11: Predicted Returns for Next 30 Days (Prophet)
        # ==================================================
        # st.write("2️⃣ We can use Facebook Prophet to forecast future returns -- Time-Series Forecasting (Predicting Future Returns)")
        # st.write("📌 Insight: This helps predict future return spikes, allowing proactive adjustments.")
        
        returns = df[df['extension'] < 0].groupby('ship_date')['extension'].sum().reset_index()
        returns = returns.rename(columns={'ship_date': 'ds', 'extension': 'y'})
        returns.loc[:, 'y'] = abs(returns['y'])  # Convert returns to positive for modeling
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(returns)
        future = model.make_future_dataframe(periods=30, freq='D')  # Predict next 30 days
        forecast = model.predict(future)
        fig11 = px.line(forecast, x='ds', y='yhat', title='Predicted Returns for Next 30 Days')
        fig11.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot'))
        fig11.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot'))
               
        # ==================================================
        # Figure 12: Anomaly Detection in Returns
        # ==================================================
        # st.write("3️⃣ We can detect unusual return spikes using Z-score method. -- Anomaly Detection (Detect Unusual Return Spikes)")
        # st.write("📌 Insight: This highlights unusual return days that might indicate fraud, defective products, or seasonal spikes.")
        
        returns = df[df['extension'] < 0].groupby('ship_date')['extension'].sum().reset_index()
        returns.loc[:, 'z_score'] = (returns['extension'] - returns['extension'].mean()) / returns['extension'].std()
        returns.loc[:, 'anomaly'] = returns['z_score'].abs() > 2.5
        fig12 = go.Figure()
        fig12.add_trace(go.Scatter(x=returns['ship_date'], y=returns['extension'], mode='lines', name='Returns'))
        fig12.add_trace(go.Scatter(x=returns[returns['anomaly']]['ship_date'], 
                                y=returns[returns['anomaly']]['extension'], 
                                mode='markers', name='Anomalies', marker=dict(color='red', size=10)))
        fig12.update_layout(title='Anomaly Detection in Returns')

        # ==================================================
        # Figure 13: Factors Influencing Returns (Random Forest)
        # ==================================================
        # st.write("4️⃣ We can use Random Forest to find the top factors driving returns -- Return Factor Correlation (What Causes Returns?)")
        # st.write("📌 Insight: This identifies the biggest factors driving returns, helping optimize pricing, shipping, or product quality.")

        df.loc[:, 'is_return'] = df['extension'].apply(lambda x: 1 if x < 0 else 0)
        features = ['price', 'ordered', 'shipped', 'customer_class', 'category']
        df = df.dropna(subset=features)
        for col in ['customer_class', 'category']:
            df.loc[:, col] = LabelEncoder().fit_transform(df[col])
        X = df[features]
        y = df['is_return']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        fig13 = px.bar(importances, x=importances.index, y=importances.values, 
                    title="Factors Influencing Returns",
                    labels={'y': 'Importance Score', 'x': 'Feature'})

        # Return all figures
        return {
            'fig1': fig1, 'fig2': fig2, 'fig3': fig3, 'fig4': fig4, 'fig5': fig5,
            'fig6': fig6, 'fig7': fig7, 'fig8': fig8, 'fig9': fig9, 'fig10': fig10,
            'fig11': fig11, 'fig12': fig12, 'fig13': fig13
        }

    except Exception as e:
        st.error(f"Error generating figures: {str(e)}")
        return None

def show_prebuilt_analysis():
    """Display the prebuilt analysis in Streamlit"""

    # Check if there's data in the session state
    if 'df' in st.session_state and st.session_state.df is not None:
        st.write("### 👀 Prebuilt Analysis")

        # Generate all figures using the DataFrame from session state
        figures = generate_all_figures(st.session_state.df)
        
        # Display figures if they were generated successfully
        if figures:
            for fig_name, fig in figures.items():
                if fig is not None:
                    st.plotly_chart(fig)

    else:
        st.warning("Please upload a file first.")
