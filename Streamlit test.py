import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Hosp_Data_Sample.csv")

df = df.drop(columns=[col for col in df.columns if col.lower() == 'booking_id'], errors='ignore')

st.title("Booking Data Correlation Dashboard")

st.header("Correlation Matrix (Numerical Features)")
numeric_df = df.select_dtypes(include='number')

if not numeric_df.empty:
    corr = numeric_df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
else:
    st.write("No numerical features found.")

st.header("Categorical Feature Distributions")

exclude_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
exclude_cols += ['booking_datetime', 'booking_id']
non_numeric_df = df.select_dtypes(exclude='number').drop(columns=exclude_cols, errors='ignore')

if 'country_code' in non_numeric_df.columns:
    top_codes = df['country_code'].value_counts().head(10)
    fig, ax = plt.subplots()
    top_codes.plot(kind='bar', ax=ax)
    ax.set_title("Top 10 Country Codes")
    ax.set_ylabel("Count")
    ax.set_xlabel("Country Code (ISO 2-letter)")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    non_numeric_df = non_numeric_df.drop(columns=['country_code'])

for col in non_numeric_df.columns:
    st.subheader(f"{col}")
    value_counts = non_numeric_df[col].value_counts(dropna=False)
    fig, ax = plt.subplots()
    value_counts.plot(kind='bar', ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel(col)
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.header("Compare Categorical vs Numeric Feature (Box Plot)")

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
exclude_x = ['booking_id', 'booking_datetime', 'country_code']
categorical_cols = [
    col for col in categorical_cols
    if all(x not in col.lower() for x in ['date', 'time']) and col not in exclude_x
]

numeric_cols = df.select_dtypes(include='number').columns.tolist()

selected_cat = st.selectbox("Select Categorical Feature (X-Axis)", categorical_cols)
selected_num = st.selectbox("Select Numeric Feature (Y-Axis)", numeric_cols)

if selected_cat and selected_num:
    df_plot = df[[selected_cat, selected_num]].dropna().copy()

    if df_plot[selected_cat].nunique() > 20:
        top_categories = df_plot[selected_cat].value_counts().nlargest(20).index
        df_plot = df_plot[df_plot[selected_cat].isin(top_categories)]

    st.subheader(f"Box Plot of {selected_num} by {selected_cat}")
    fig, ax = plt.subplots()
    sns.boxplot(data=df_plot, x=selected_cat, y=selected_num, ax=ax)
    ax.set_xlabel(selected_cat)
    ax.set_ylabel(selected_num)
    plt.xticks(rotation=45)
    st.pyplot(fig)


st.header("📝 Insights & Recommendations")
st.markdown("""

The overall takeaway from all visuals and insights into this dataset is a lack of correlation between different factors, most notably with the overall price. The most actionable approach to making more profit based on the dataset will be to target how to attract more groups of adults that will, in turn, purchase more separate rooms, which directly relates to how much is being spent. After that, this company needs to think about taking better advantage of different packages and possibly different pricing during different times of the year.

""")
