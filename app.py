#%%
import streamlit as st
import pandas as pd
import umap
import plotly.express as px
from streamlit_plotly_events import plotly_events
import numpy as np
#%%
st.title("Data Exploration")
st.write('Each point on the graph represents a single patient. The color of the point represents the patient\'s survival time. Click on a point to calculate the patiet\'s data.')

################################################################################
# get mean and std from the data to reverse the normalization if needed
data = pd.read_excel("data_eng.xlsx")
#drop unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
#drop last 2 columns
data = data.iloc[:, :-2]
means = {col: data[col].mean() for col in data.columns}
stds = {col: data[col].std() for col in data.columns}
################################################################################

#%%
# st.write("Hello World!")
#%% read csv
data_norm = pd.read_csv("data_norm.csv")
#%%
option  = st.selectbox('Select one :', ('OS (months)', 'PFS (months)'))
# labels = data_norm.iloc[:, -2]
labels = data_norm[option]
mapper = umap.UMAP(random_state=42).fit(data_norm.iloc[:, :-2])
#%%
fig = px.scatter(mapper.embedding_, x=0, y=1, color=labels, color_continuous_scale=px.colors.diverging.PiYG, color_continuous_midpoint=np.mean(labels), labels={'color': 'Months', '0': 'UMAP 1', '1': 'UMAP 2'})

selected_points = plotly_events(fig)
#overlay selected points on top of the original figure


# get x and y coordinates of selected points
x = [point['x'] for point in selected_points]
y = [point['y'] for point in selected_points]

try:
    st.write('Point chosen: ',(x[0],y[0])) 
    # col1, _ , col2 = st.columns([5,5,5]) 
    # col1.write(fig)  
    # col2.write(fig) 
    # fig.add_trace(px.scatter(selected_points, x=x[0], y=y[0], 
    #         marker=dict(
    #             color='LightSkyBlue',
    #             size=120,
    #             line=dict(
    #                 color='MediumPurple',
    #                 width=12
    #         ))))
    # st.write(fig)
    click_coord = np.array([x[0],y[0]]).reshape(1, -1)
    inverse = mapper.inverse_transform(click_coord)

    rev_norm = st.checkbox("Reverse Normalization", value=False)
    if rev_norm:
        for i, col in enumerate(data.columns):
            inverse[0][i] = inverse[0][i] * stds[col] + means[col]
    bar_fig = px.bar(x=data_norm.columns[:-2], y=inverse[0], labels={'x': '', 'y': ''})
    # drop axis labels
    st.write(bar_fig)
except: st.write('No point chosen')
