######## Import libraries ########
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


######## Page, Config, style ########
st.set_page_config(page_title="Part A - Data Exploration")


######## Load data ########
## path + filename
data_path = "./raw_data/dataset.parquet"

## load data into DF
df = pd.read_parquet(data_path)


######## Main, Data Exploration ########

st.markdown("""

# Part A - Data Exploration

## *California Housing Dataset*

""")


######## Dataset, shape ########

col1, col2, col3, col4 = st.columns(4)
col1.metric("Samples", df.shape[0])
col2.metric("Features", df.shape[1]-1)
col3.metric("Target", "1")
col4.metric("Duplicates / NaN", f"{df.duplicated().sum()} / {df.isna().sum().sum()}")

st.caption('*The dataset is dense.')


######## Features / Description ########

st.write('## *Features / Description*')

@st.experimental_memo
def feat_discr():

    df_descr = df.describe().round(2)

    col1, col2, col3, col4 = st.columns(4)

    for i in range(9):
        col1.metric(f"**{df.columns[i]}** / {df_descr.index[1]}", df_descr.iloc[1,i])

    for i in range(9):
        col2.metric(f"{df_descr.index[2]}", df_descr.iloc[2,i])

    for i in range(9):
        col3.metric(f"{df_descr.index[3]}", df_descr.iloc[3,i])

    for i in range(9):
        col4.metric(f"{df_descr.index[7]}", df_descr.iloc[7,i])

feat_discr()

st.caption("""*MedInc: median income in block group / HouseAge: median house age in block group / AveRooms: average number of rooms per household /
AveBedrms: average number of bedrooms per household / Population: block group population / AveOccup: average number of household members / Latitude: block group latitude /
Longitude: block group longitude / MedHouseVal: median house value for California districts, expressed in hundreds of thousands of dollars ($100,000)""")


######## Outliers / Boxplot ########

st.write('## *Outliers / Boxplot*')

## color, line, style properties for the boxplot
flierprops = dict(marker='+', markerfacecolor='g', markersize=7,
                  linestyle='none', markeredgecolor='r', linewidth=0.1, alpha=.7)
color=dict(boxes='r', whiskers='black', medians='#fde725', caps='black')
boxprops = dict(facecolor= "#1E9B8AFF", linestyle='-', linewidth=1, color='black', alpha=.7)

@st.experimental_memo
def box_plot():
    a = 3  # number of rows
    b = 3  # number of columns
    c = 1  # initialize plot counter

    fig1 = plt.figure(figsize=(13,13))

    for i in df:
        plt.subplot(a, b, c)
        df.boxplot(i, flierprops=flierprops, color=color, boxprops=boxprops, patch_artist=True);
        c+=1
    st.pyplot(fig1)

box_plot()

st.caption("""*The feature MedInc has many outliers, but they seem reasonable (very high incomes). The features AveRooms, AveBedrms,
Population and AveOccup potentially have plenty of outliers due to the fact that there might be other building typologies than houses - such as hotels,
etc.. - in the dataset as well. The target MedHouseVal has many outliers at 500k since the range is limited to this amount and all values above
become culmulated.""")


######## Correlation / Matrix ########

st.write('## *Correlation / Matrix*')

@st.experimental_memo
def corr_mat():
    ## create correlation matrix
    cm = df.corr()

    ## change figsize
    fig2 = plt.figure(figsize=(13,13))

    ## plot heatmap
    sns.heatmap(cm,
                    cbar=False,
                    annot=True,
                    square=True,
                    fmt='.2f',
                    annot_kws={'size': 12},
                    yticklabels=df.columns,
                    xticklabels=df.columns,
                    cmap="viridis")
    st.pyplot(fig2)

corr_mat()

st.caption("""*The correlation coefficient for the features MedInc and the target MedHouseVal
indicates a positive linear relationship. Furthermore, there seems to be multicollinearity
between the features AveRooms and AveBedrms.""")


######## Distribution / Pairplot ########

sns.set_style(style='whitegrid')

st.write('## *Distribution / Pairplot*')

@st.experimental_memo
def pair_plt():
    ## Drop lon/lat columns
    drop = ["Longitude", "Latitude"]
    df_subset = df.iloc[:].drop(columns=drop)

    ## Discretize variable into equal-sized buckets
    df_subset["MedHouseVal"] = pd.qcut(df_subset["MedHouseVal"], 5, retbins=False)

    ## Plot pairwise relationships in a dataset
    fig3 = sns.pairplot(data=df_subset, hue="MedHouseVal", palette="viridis");

    ## Style plot
    sns.set_style(style='whitegrid')
    handles = fig3._legend_data.values()
    labels = fig3._legend_data.keys()
    fig3._legend.remove()
    fig3.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=5, frameon=False, fontsize='large', borderaxespad=-0.55, title='Median house value in $100K', title_fontsize='large')
    st.pyplot(fig3)
    return fig3

pair_plt()

st.caption("""*The hue of the pairplot indicates the Median house value of the pairwise relationship
of the features. The layered kernel density estimate shows the distribution of the data and affirms the
conclusions concerning the outliers from the boxplot. As with the correlation matrix, the pairplot renders the
relationship between MedInc and MedHouseVal visible. Hence non-linear-investigations might be necessary in
order to gain further insights.""")


######## Population / Map ########

st.write('## *Population / Map*')

@st.experimental_memo
def pop_loc():
    fig4 = plt.figure(figsize=(13,13))
    sns.set_style(style='whitegrid')
    sns.scatterplot(data=df, x='Latitude', y='Longitude', size='Population', sizes=(4, 200), palette='viridis', alpha=0.1);
    st.pyplot(fig4)

pop_loc()

st.caption("""*The scatter plot shows the density of the population. Metropolitan areas
as well as the coastline become visible.""")


######## Median house prices / Map ########

st.write('## *Median House Prices / Map*')

@st.experimental_memo
def price_loc():
    fig5 = plt.figure(figsize=(13,13))
    sns.set_style(style='whitegrid')
    sns.scatterplot(data=df, x='Latitude', y='Longitude', hue='MedHouseVal', palette='viridis', alpha=0.1);
    st.pyplot(fig5)

price_loc()

st.caption("""*The scatter plot indicates higher house prices closer to the coastline as well as in
metropolitan areas. """)


######## Median income / Map ########

st.write('## *Median Income / Map*')

@st.experimental_memo
def inc_loc():
    fig6 = plt.figure(figsize=(13,13))
    sns.set_style(style='whitegrid')
    sns.scatterplot(data=df, x='Latitude', y='Longitude', hue='MedInc', palette='viridis', alpha=0.1);
    st.pyplot(fig6)

inc_loc()

st.caption("""*The median income is higher in metropolitan areas and alongside the coastline.""")


######## Income - House value / Regplot ########

st.write('## *Median Income - House Value / Regplot*')

@st.experimental_memo
def reg_pl():
    fig7 = plt.figure(figsize=(13,13))
    sns.set_style(style='whitegrid')
    sns.regplot(data= df, x='MedInc', y='MedHouseVal',
                scatter_kws={"color": '#3b528b', 'alpha':0.05}, line_kws={"color": "red"});
    st.pyplot(fig7)

reg_pl()

st.caption("""*The regression plot between the features MedInc and MedHouseVal indicates a linear relationship
though an increasing variance with higher values. Further the graph shows that the range of the feature MedHouseVal
seems to be too narrow - values above 500k become cumulated at its limit.""")
