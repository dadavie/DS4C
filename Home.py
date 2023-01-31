import streamlit as st
import numpy as np
import pandas as pd

## style
st.set_page_config(page_title ="Home - Assignment", layout="wide")

st.markdown("""# Data Scientist 4 Construction

## Assignment / Data Exploration and Data Science Product

- Part A: In-depth analysis of the dataset; communication of gained insights in a proper fashion to stakeholders and customers

- Part B: Conceptualization of a data science product that can predict house prices

## Data / California Housing Dataset

The dataset was derived from the 1990 U.S. census, using one row per census
block group. A block group is the smallest geographical unit for which the U.S.
Census Bureau publishes sample data (a block group typically has a population
of 600 to 3,000 people).

The target variable is the median house value for California districts,
expressed in hundreds of thousands of dollars ($100,000).

The dataset was obtained from the StatLib repository.
https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

Source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html#sklearn.datasets.fetch_california_housing""")
