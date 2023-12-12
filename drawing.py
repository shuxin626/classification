from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import scipy.io
import numpy as np
import pickle
from sklearn.metrics import precision_recall_curve, average_precision_score

color_dict = {'m':'#FF0000', 'g': '#18AE18',
              '(b-t)': '#854836', 'b': '#3447FE',
              't': '#FF9900', 'CD4': '#0C99C1', 'CD8': '#F30CD4'}



def scatter3d_draw(df, cell_types):
    colors = [color_dict[cell_types[value]] for value in df['type']]
    trace = go.Scatter3d(
    x=df['x'],
    y=df['y'],
    z=df['z'],
    mode='markers',
    marker=dict(
        size=5,
        color=colors,  # Set individual colors for each point
    )
)
    # Set layout for the 3D plot
    layout = go.Layout(
        scene=dict(
            xaxis=dict(range=[-150, 150]),  # Set range for x-axis
            yaxis=dict(range=[-150, 150]),  # Set range for y-axis
            zaxis=dict(range=[-150, 150]),  # Set range for z-axis
        )
    )

    # Create the figure with the trace and layout
    fig = go.Figure(data=[trace], layout=layout)

    # Show the plot
    fig.write_html('scatter3d.html')
    fig.show()



def prc_draw(result_dict, cell_type):
    traces = []
    for i in range(result_dict['score'].shape[1]):
        y_true_binary = (result_dict['target'] == i).int()
        precision, recall, _ = precision_recall_curve(y_true_binary, result_dict['score'][:, i])
        trace = go.Scatter(x=recall, y=precision, name='', mode='lines', marker_color=color_dict[cell_type[i]])
        traces.append(trace)

    layout = go.Layout(
        title='Precision-Recall Curves per Class',
        xaxis=dict(title='Recall'),
        yaxis=dict(title='Precision'))

    fig = go.Figure(data=traces, layout=layout)
    fig.show()




