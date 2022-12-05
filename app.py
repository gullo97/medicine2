#%%
import streamlit as st
import pandas as pd
import umap
import plotly.express as px
from streamlit_plotly_events import plotly_events
import numpy as np
#%%
rev_norm = st.checkbox("Reverse Normalization", value=False)
# get mean and std from the data
data = pd.read_excel("data.xls")
#drop unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
#drop last 2 columns
data = data.iloc[:, :-2]
means = {col: data[col].mean() for col in data.columns}
stds = {col: data[col].std() for col in data.columns}

# means['Sesso','MGMTmet','IDH1'] = 0
# stds['Sesso','MGMTmet','IDH1'] = 1
# st.write(means)
#%%
# st.write("Hello World!")
#%% read csv
data_norm = pd.read_csv("data_norm.csv")
#%%
labels = data_norm.iloc[:, -2]
mapper = umap.UMAP(random_state=42).fit(data_norm.iloc[:, :-2])
#%%
fig = px.scatter(mapper.embedding_, x=0, y=1, color=labels, color_continuous_scale=px.colors.diverging.PiYG, color_continuous_midpoint=np.mean(labels))
selected_points = plotly_events(fig)
# get x and y coordinates of selected points
x = [point['x'] for point in selected_points]
y = [point['y'] for point in selected_points]
st.write('Point chosen: ',(x[0],y[0]))
click_coord = np.array([x[0],y[0]]).reshape(1, -1)
inverse = mapper.inverse_transform(click_coord)

if rev_norm:
    for i, col in enumerate(data.columns):
        inverse[0][i] = inverse[0][i] * stds[col] + means[col]
bar_fig = px.bar(x=data_norm.columns[:-2], y=inverse[0])
st.write(bar_fig)

