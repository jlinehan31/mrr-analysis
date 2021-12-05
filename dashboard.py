# Import packages. If package not recognized use `pip install` in your terminal
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly import tools
import plotly.offline as py
import streamlit as st

# Jolt Hex Color Codes: 
# Blue: #0077C8
# Green: #00B74F

st.set_page_config(layout='wide')
st.title('ARR Dashboard')

# Create drag-and-drop file import functionality
df = st.sidebar.file_uploader('Select a CSV file to import (Default provided)')
@st.cache()
def load_file(df):
    if df is not None:
        df = pd.read_csv(df, 
                        index_col=False,
                        parse_dates=['Date']
        )
    else:
        st.stop()
    return df
df = load_file(df)

# Copy dataframe to a working version and change `Date` type to date
arr_df = df.copy()
arr_df['Date'] = pd.to_datetime(arr_df['Date']).dt.date

# Create beginning and end date drop downs
start_date = st.sidebar.selectbox('Start Date', arr_df['Date'].unique())
end_date = st.sidebar.selectbox('End Date', arr_df['Date'].unique()[::-1])

# Brand Option
select_brand = st.sidebar.multiselect('Brand', arr_df['Brand'].unique(),
                                      help='Select one or many')
if len(select_brand) > 0:
    select_brand = select_brand
else:
    select_brand = arr_df['Brand'].unique()

# Company Name Option
select_account = st.sidebar.multiselect('Account Name', 
                                        arr_df['Company Name'].unique(),
                                        help='Select one or many')
if len(select_account) > 0:
    select_account = select_account
else:
    select_account = arr_df['Company Name'].unique()

# Create ARR data frame and make date selection dynamic
arr_df = (arr_df[(arr_df['Date'] >= start_date) &
                (arr_df['Date'] <= end_date) &
                (arr_df['Brand'].isin(select_brand)) &
                (arr_df['Company Name'].isin(select_account))]
                .reset_index(drop=True))

beg_arr = arr_df[arr_df['Date'] == start_date]['Beginning'].sum() * 12
end_arr = arr_df[arr_df['Date'] == end_date]['Ending'].sum() * 12

# Show the ARR value for the selected `end_date`
st.metric('ARR (% Change)', 
          '${:,.0f}'.format(end_arr),
          '{:.1%}'.format(end_arr / beg_arr - 1)
)

# Time series ARR visualization
fig1, ax1 = plt.subplots()
arr_viz = arr_df.groupby(by=['Date'])['Ending'].sum().to_frame().reset_index()

fig1 = px.bar(arr_viz,
              x='Date',
              y=arr_viz['Ending'] * 12,
              title='ARR by Month'
)
fig1.update_xaxes(title='', showgrid=False)
fig1.update_yaxes(title='', showgrid=False)

st.plotly_chart(fig1)

# Data frame of all data that fit within chosen parameters
st.header('MRR Data by Account')
st.dataframe(arr_df)

# Create a button for quick csv export
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

mrr_csv = convert_df(arr_df)
st.download_button(
    label='Export Data',
    data=mrr_csv,
    file_name=f'MRR Data ({start_date} - {end_date}).csv',
    mime='text/csv'
)

# Create view to show MRR churn
st.header('Churn')

churn_df = arr_df.copy()
churn_df['MRR Churn'] = churn_df[['Contraction', 'Lost']].sum(axis=1)
churn_df = (churn_df[churn_df['MRR Churn'] >= 1]
           .sort_values(by='MRR Churn', ascending=False)
           .reset_index(drop=True)
)
churn_df['Type'] = (churn_df['Contraction'].apply(
                    lambda x: 'Contraction' if x > 0 else 'Lost')
)

churn_value = churn_df['MRR Churn'].sum()
st.metric('MRR Churn', 
          '${:,.0f}'.format(churn_value),
          '{:.1%}'.format(churn_value / (beg_arr/12)),
          delta_color='inverse'
)

# Time series MRR churn visualization
fig2, ax2 = plt.subplots()
churn_viz = (churn_df.groupby(by=['Date', 'Type'])['MRR Churn'].sum()
            .to_frame().reset_index()
)

fig2 = px.bar(churn_viz,
              x='Date',
              y='MRR Churn',
              barmode='stack',
              color='Type',
              title='MRR Churn by Month'
)
fig2.update_xaxes(title='', showgrid=False)
fig2.update_yaxes(title='', showgrid=False)

# MRR churn distrbution (box plot)
fig3, ax3 = plt.subplots()

fig3 = px.scatter(churn_df,
                x='Date',
                y='MRR Churn',
                hover_name='Company Name',
                size='MRR Churn',
                color='Type',
                title='MRR Churn Distribution by Account'
)
fig3.update_xaxes(title='', showgrid=False)
fig3.update_yaxes(title='', showgrid=False)

# Arrange layout and plot churn charts
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig2)
with col2:
    st.plotly_chart(fig3)


# Drop `MRR Churn` column to keep df consistent as above
churn_df = churn_df.drop(['MRR Churn', 'Type'], axis=1)

# Data frame of all data that fit within chosen parameters
st.header('MRR Churn by Account')
st.dataframe(churn_df)

# Create a button for quick csv export
churn_csv = convert_df(churn_df)
st.download_button(
    label='Export Data',
    data=churn_csv,
    file_name=f'Churn Data ({start_date} - {end_date}).csv',
    mime='text/csv'
)

# Remove "Made with Streamlit"
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
