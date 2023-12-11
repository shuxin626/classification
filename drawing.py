from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import scipy.io
import numpy as np
import pickle
from sklearn.metrics import precision_recall_curve, average_precision_score
from keras import utils
color_dict = {'Monocyte':'#FF0000', 'Granulocyte': '#18AE18',
              'Lymphocyte': '#854836', 'B lymphocyte': '#3447FE',
              'T lymphocyte': '#FF9900', 'CD4': '#0C99C1', 'CD8': '#F30CD4'}

def process_data(file, cell_type, suffix=''):
  df = scipy.io.loadmat(file)
  type_num = df['ts_target' + suffix]
  type_array = np.empty(shape=(np.shape(type_num)), dtype='object')
  for i in range(np.max(type_num)):
    type_array[type_num == i + 1] = cell_type[i]
  data_tag_name = 'ts_data' + suffix
  result_df = pd.DataFrame({'x': df[data_tag_name][:, 0], 'y': df[data_tag_name][:, 1],
                         'z': df[data_tag_name][:, 2], 'type': type_array[0]})
  return result_df

def scatter3d_draw(fig, file, cell_type, color_dict, row, col, suffix='',
                   show_legend=True):
  df = process_data(file, cell_type, suffix=suffix)
  sub_fig = px.scatter_3d(df, x='x', y='y', z='z',
                       color='type')
  for i, d in enumerate(sub_fig.data):
    fig.add_trace((go.Scatter3d(x=d['x'], y=d['y'], z=d['z'], mode='markers',
                                marker={'size': 1, 'color': color_dict[cell_type[i]]},
                                name='', showlegend=show_legend,
                                legendgroup=cell_type[i], legendgrouptitle_text=cell_type[i])),
                                row=row, col=col)

def bar_draw(fig, file, color_sequence, type_sequence, row, col, show_legend=True, y_title='',
             x_title=''):
  acu_matrix = pd.read_excel(file)
  acu_matrix = pd.melt(acu_matrix, id_vars=['Type'],
                       value_vars=['Recall','Precision','F1-score'], var_name='Matrix')
  sub_fig = px.bar(acu_matrix, x='Matrix', y='value', color='Type', barmode='group')
  for i, d in enumerate(sub_fig.data):
    fig.add_trace((go.Bar(x=d['x'], y=d['y'], name='',
                          marker_color=color_sequence[i], legendgroup=type_sequence[i],
                          showlegend=show_legend, legendgrouptitle_text=type_sequence[i])), row=row, col=col)
  fig.update_yaxes(title_text=y_title, range=[0, 1.02], col=col, row=row)
  fig.update_xaxes(title_text=x_title, col=col, row=row)

def prc_draw(fig, file, cell_type, color_dict, row, col, show_legend=True,
             y_title='', x_title=''):
  with open(file, 'rb') as f:
    result_dict = pickle.load(f)
  for i in range(result_dict['score'].shape[1]):
    y_true = utils.to_categorical(result_dict['target']).astype(int)
    y_score = result_dict['score']
    precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
    fig.add_trace(
      go.Scatter(x=recall, y=precision, name='', mode='lines', legendgroup=cell_type[i],
                 legendgrouptitle_text=cell_type[i], marker_color=color_dict[cell_type[i]], showlegend=show_legend), row=row, col=col)
    fig.update_xaxes(title_text=x_title, range=[0, 1.02], col=col, row=row)
    fig.update_yaxes(title_text=y_title, range=[0, 1.02], col=col, row=row)

def box_draw(fig, file, param, cell_type, color_dict, row, col, y_title='',
             x_title='', showlegend=True):
  df = pd.read_excel(file)
  for cell_subtype in cell_type:
    fig.add_trace(
      go.Box(x=df['Test donor'], y=df[cell_subtype], name=cell_subtype,
             boxpoints='outliers', marker={'size': 0.5}, marker_color=color_dict[cell_subtype],
             showlegend=showlegend, line_width=1),
             row=row, col=col)
  fig.update_yaxes(title_text=y_title, row=row, col=col)
  fig.update_xaxes(title_text=x_title, row=row, col=col)


#==================================Figure_1====================================
# # start drawing
# fig = make_subplots(rows=3, cols=3, specs=[[{'colspan':2}, None, {}],
#                                            [{}, {}, {}],[{"type": "scatter3d"},
#                                            {"type": "scatter3d"},
#                                            {"type": "scatter3d"}]],
#                     subplot_titles=('(a)', '(f)', '(b)', '(c)', '(g)',
#                                     '(d)', '(e)', '(h)'))
#
# # draw 3d plot
# scatter3d_draw(fig, 'all_3tsne_data.mat', ['Monocyte', 'Granulocyte', 'Lymphocyte'],
#                color_dict, 3, 1, '', True)
# scatter3d_draw(fig, 'all_3tsne_data_bt.mat', ['B lymphocyte', 'T lymphocyte'],
#                color_dict, 3, 2, '_bt', True)
# scatter3d_draw(fig, 'CD_tsne_data.mat', ['CD4', 'CD8'],
#                color_dict, 3, 3, '_cd', True)
#
# # draw bar
# bar_draw(fig, 'mgl-classification.xlsx', ['#FF0000','#18AE18','#3447FE','#FF9900'],
#          ['Monocyte', 'Granulocyte', 'B lymphocyte', 'T lymphocyte'], 1, 1, True)
# bar_draw(fig, 'cd-classification.xlsx', ['#0C99C1','#F30CD4'], ['CD4', 'CD8'], 1, 3,
#          True)
#
# # draw PRC
# prc_draw(fig, 'mgl.pkl', ['Monocyte', 'Granulocyte', 'Lymphocyte'], color_dict,
#          2, 1, True, 'Precision', 'Recall')
# prc_draw(fig, 'bt.pkl', ['B lymphocyte', 'T lymphocyte'], color_dict, 2, 2, True,
#          '', 'Recall')
# prc_draw(fig, 'cd.pkl', ['CD4', 'CD8'], color_dict, 2, 3, True, '', 'Recall')
# # figure show
# fig.update_layout(height=800, width=750)
# fig.update_layout(legend=dict(orientation='h', itemsizing='constant'))
# fig.update_layout(scene_aspectmode='cube')
# fig.update_layout(scene2_aspectmode='cube')
# fig.update_layout(scene3_aspectmode='cube')
# fig.update_layout(scene = dict(
#                           xaxis = dict(nticks=3, range=[-15,15], tickvals=[-10, 0, 10]),
#                           yaxis = dict(nticks=3, range=[-15,15], tickvals=[-10, 0 ,10]),
#                           zaxis = dict(nticks=3, range=[-15, 15], tickvals=[-10, 0 ,10])))
# fig.update_layout(scene2 = dict(
#                           xaxis = dict(nticks=3, range=[-15,15], tickvals=[-10, 0 ,10]),
#                           yaxis = dict(nticks=3, range=[-15,5], tickvals=[-10, -5 ,0]),
#                           zaxis = dict(nticks=4, range=[-15, 10], tickvals=[-10, -5, 0, 5])))
# fig.update_layout(scene3 = dict(
#                           xaxis = dict(nticks=3, range=[-150,150], tickvals=[-150, 0, 150]),
#                           yaxis = dict(nticks=3, range=[-100,100], tickvals=[-50, 0, 50]),
#                           zaxis = dict(nticks=4, range=[-150, 150], tickvals=[-100, 0, 100])))
# # fig.update_layout(font=dict(size=10))
# fig.show()
# fig.write_html('export.html')

#================================Figure2=======================================
#start drawing
fig = make_subplots(rows=3, cols=1, specs=[[{}],[{}],[{}]], subplot_titles=('(a)', '(b)', '(c)'))
type = ['Monocyte', 'Granulocyte', 'B lymphocyte', 'T lymphocyte']


# draw bar
acu_matrix = pd.read_excel('cross-donor-classification.xlsx')
acu_matrix = pd.melt(acu_matrix, id_vars=['Test donor'],
                     value_vars=['Monocyte', 'Granulocyte', 'B lymphocyte', 'T lymphocyte'],
                     var_name='Type')
sub_fig = px.bar(acu_matrix, x='Test donor', y='value', color='Type',
                 barmode='group')
for i, d in enumerate(sub_fig.data):
  fig.add_trace(go.Bar(x=d['x'], y=d['y'], name=d['name'],
                        marker_color=color_dict[type[i]], xperiodalignment='middle'), row=3, col=1)
fig.update_yaxes(title_text='F1 Score', row=3, col=1)

# box plot
box_draw(fig, 'individual-mass.xlsx', 'Mass', type, color_dict, 2, 1, y_title='Dry mass(pg)',
         showlegend=False)
box_draw(fig, 'individual-area.xlsx', 'Area', type, color_dict, 1, 1, y_title='Area(\u03BCm2)',
         showlegend=False)

# show figure
fig.update_layout(boxmode='group', boxgap=0, boxgroupgap=0.15)
fig.update_layout(height=650, width=750)
fig.update_layout(legend=dict(orientation='h', itemsizing='constant'))
fig.write_html('export2.html')
fig.show()
