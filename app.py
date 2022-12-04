#%%
import streamlit as st
import pandas as pd
import umap
import plotly.express as px
from streamlit_plotly_events import plotly_events
import numpy as np

#%%
# st.write("Hello World!")
#%% read csv
data = pd.read_csv("data_norm.csv")
#%%
labels = data.iloc[:, -2]
mapper = umap.UMAP(random_state=42).fit(data.iloc[:, :-2])
#%%
fig = px.scatter(mapper.embedding_, x=0, y=1, color=labels)
selected_points = plotly_events(fig)
# get x and y coordinates of selected points
x = [point['x'] for point in selected_points]
y = [point['y'] for point in selected_points]
st.write('Point chosen: ',(x[0],y[0]))
click_coord = np.array([x[0],y[0]]).reshape(1, -1)
inverse = mapper.inverse_transform(click_coord)
# bar plot of inverse
# st.write(inverse)
bar_fig = px.bar(x=data.columns[:-2], y=inverse[0])
st.write(bar_fig)

