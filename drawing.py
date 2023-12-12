from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import scipy.io
import numpy as np
import pickle
from sklearn.metrics import precision_recall_curve, average_precision_score

color_dict = {'Monocyte':'#FF0000', 'Granulocyte': '#18AE18',
              'Lymphocyte': '#854836', 'B lymphocyte': '#3447FE',
              'T lymphocyte': '#FF9900', 'CD4': '#0C99C1', 'CD8': '#F30CD4'}



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


# def bar_draw(fig, file, color_sequence, type_sequence, row, col, show_legend=True, y_title='',
#              x_title=''):
#   acu_matrix = pd.read_excel(file)
#   acu_matrix = pd.melt(acu_matrix, id_vars=['Type'],
#                        value_vars=['Recall','Precision','F1-score'], var_name='Matrix')
#   sub_fig = px.bar(acu_matrix, x='Matrix', y='value', color='Type', barmode='group')
#   for i, d in enumerate(sub_fig.data):
#     fig.add_trace((go.Bar(x=d['x'], y=d['y'], name='',
#                           marker_color=color_sequence[i], legendgroup=type_sequence[i],
#                           showlegend=show_legend, legendgrouptitle_text=type_sequence[i])), row=row, col=col)
#   fig.update_yaxes(title_text=y_title, range=[0, 1.02], col=col, row=row)
#   fig.update_xaxes(title_text=x_title, col=col, row=row)

# def prc_draw(fig, file, cell_type, color_dict, row, col, show_legend=True,
#              y_title='', x_title=''):
#   with open(file, 'rb') as f:
#     result_dict = pickle.load(f)
#   for i in range(result_dict['score'].shape[1]):
#     y_true = utils.to_categorical(result_dict['target']).astype(int)
#     y_score = result_dict['score']
#     precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
#     fig.add_trace(
#       go.Scatter(x=recall, y=precision, name='', mode='lines', legendgroup=cell_type[i],
#                  legendgrouptitle_text=cell_type[i], marker_color=color_dict[cell_type[i]], showlegend=show_legend), row=row, col=col)
#     fig.update_xaxes(title_text=x_title, range=[0, 1.02], col=col, row=row)
#     fig.update_yaxes(title_text=y_title, range=[0, 1.02], col=col, row=row)




