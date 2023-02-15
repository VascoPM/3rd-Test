# This is a Dash dashboard script.
# It creates a dashboard to view the Clustering kMeans Solutions in an Auto-Encoders Latent Space.
# It is dependent on the file structure and naming policy used on the -Third Test- experiment.
# Vasco Mergulhao - 15/02/2023

from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import os
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from sklearn.preprocessing import StandardScaler

###########################################################
# Generic functions used in Dashboard
###########################################################
# Folder and File name Navigator Functions
def retrive_sol_files (dataset_name):
    extension = 'csv'
    os.chdir(f"../ModelResults/Clustering/{dataset_name}")
    solution_files = glob.glob('*.{}'.format(extension))
    os.chdir("../../../Time-Series-PhD")
    
    return solution_files

def retrive_AEmodel_options (sol_files):    
    # Given a Solution File list (.csv names), return Auto Encoder Model types
    # This is dependent of the file naming policy 
    
    AEmodels_options = []
    for i, sol in enumerate(sol_files):
        # Finds Model types in folder
        AEmodel = "_".join(sol.split('_')[:2]) # This is dependent on naming policy
        AEmodels_options.append(AEmodel)
        AEmodels_options = list(set(AEmodels_options)) # Only keeps unique values
        AEmodels_options.sort()
    
    return AEmodels_options

def retrive_LatentSpace_options (sol_files, AEmodel_type):
    # Finds Latent Space Sizes for a given Auto_Encoder type in the specified dataset-folder.
    # This is dependent of the file naming policy.
    
    LatentSpace_options = []
    for i, sol in enumerate(sol_files):
        # Skippes solutions of other AE model types
        if AEmodel_type in sol:
            # Finds Latent-Space sizes in folder
            Latent_Space = int(sol.split('_')[2].split('-')[-1])
            LatentSpace_options.append(Latent_Space)
            LatentSpace_options = list(set(LatentSpace_options))
            LatentSpace_options.sort()
    
    return LatentSpace_options

def retrive_SolNames_options (sol_files, AEmodel_type, LSpace_size):
    # Finds all solutions that respect all three filters: Dataset, AE-model, and Latent Space Size 
    # This is dependent of the file naming policy.
    
    SolNames_options = []
    for i, sol in enumerate(sol_files):
        # Skippes solutions of other AE model types
        if AEmodel_type in sol:
            # Finds Latent-Space size of sol
            # This way only checks is integer -LSpace_size- exists in the relevant section of name, and not the whole string.
            Latent_Space = int(sol.split('_')[2].split('-')[-1])
            if LSpace_size == Latent_Space:
                Sol_Name = sol.split('_')[-2]
                SolNames_options.append(Sol_Name)
                SolNames_options.sort()

    return SolNames_options

def zscore_custom(reference_ts, inverse_ts=None, mode='scale-down', all_const_replace = -1):
    #takes in a numpy arrays of single window profiles and returns them zscored or inverted based on reference, according to mode.
    
    # check and adjusting array shape:
    if reference_ts.shape == (len(reference_ts),):
        reference_ts = reference_ts.reshape(-1, 1)
    
    # check if input is constant (e.g., all -7)
    not_constant = True
    if len(set(np.ravel(reference_ts.reshape(1,-1))))<=1:
        not_constant = False

    if not_constant:
        #Creats Scaler and  fits it to the reference data    
        zscore_scaler = StandardScaler()
        zscore_scaler.fit(reference_ts)

        if mode == 'scale-down':
            # Zscores window
            return np.ravel(zscore_scaler.transform(reference_ts))

        elif mode == 'inverse':
            # Reverse Zscores winodw with reference to the original data (pre-encoding)
            return np.ravel(zscore_scaler.inverse_transform(inverse_ts))
    else:
        # For cases where input is all constant (e.g., -7)
        if mode == 'scale-down':
            # Return an array of constants with a designated value (-1)
            return np.full((len(reference_ts),), all_const_replace)
        elif mode == 'inverse':
            # Centers the encoded data arround the original constant value
            # Does it by removing the mean and adding the constant too all values in array
            return np.ravel(inverse_ts - np.mean(inverse_ts) + reference_ts[0])


###########################################################
# Navigating File Structure to find all available solutions
# First, see list of Datasets used, defined by the solution folders

Dataset_options = next(os.walk("../ModelResults/Clustering"))[1] # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory?page=1&tab=scoredesc#tab-top
if '.ipynb_checkpoints' in Dataset_options:
    Dataset_options.remove('.ipynb_checkpoints')
    
solution_files = retrive_sol_files (Dataset_options[0])
# The following are just to initiate the variables (default values)
AEmodels_options = retrive_AEmodel_options (solution_files)
LatentSpace_options = retrive_LatentSpace_options (solution_files, AEmodels_options[0])
SolNames_options = retrive_SolNames_options (solution_files, AEmodels_options[0], LatentSpace_options[0])


###########################################################
# Data Imports to defined default values and launch data.

# Original Dataset Import 
dataset_folder = "_".join(Dataset_options[0].split('_')[:-1])
Data_orig = pd.read_csv(f'../Data/{dataset_folder}/{Dataset_options[0]}.csv')
used_cols = len(Data_orig.columns)-3
window_col_names = Data_orig.columns[-used_cols:]

# Default Reconstruction Solution
Data_Sol = pd.read_csv(f'../ModelResults/Clustering/{Dataset_options[0]}/{solution_files[0]}')
used_cols = len(Data_Sol.columns)-5
clustering_cols = Data_Sol.columns[-used_cols:]
min_cluster = Data_Sol[clustering_cols[-1]].min()
max_cluster = Data_Sol[clustering_cols[-1]].max()
id_range =[Data_Sol['short_ID'].min(), Data_Sol['short_ID'].max()] 





###########################################################
# Dashboard Per se
###########################################################
# Creating App with Theme
app=Dash(external_stylesheets=[dbc.themes.DARKLY])

######################################################
# Layout Elements

# Header Bar
navbar = dbc.NavbarSimple(
    brand = 'Clustering Visualization',
    fluid = True, #fill horizontal space,
)

# Dropdown Solutions Available
Dropdown_Solutions = dcc.Dropdown(id = 'dropdown-solutions',
                                  options = SolNames_options,
                                  value = SolNames_options[0],
                                  style = {'color':'Black'}
                                 )

# Dropdown Clustering Options
Dropdown_ClusteringOptions = dcc.Dropdown(id = 'dropdown-clustering',
                                  options = clustering_cols,
                                  value = clustering_cols[-1],
                                          style = {'color':'Black'}
                                 )
# Scatter Graph Title using a Card for aesthetic reasons
Scatter_Graph_Title = dbc.Card(
    html.H5("2D UMAP Representation - Clusters"),
    body=True)

# 2D UMAP Plot
Scatter_Graph = dcc.Graph(id = 'scatter-graph')

# Dropdown Clustering Options
Dropdown_ActiveClusters = dcc.Dropdown(id = 'dropdown-ActiveClusters',
                                       options = np.sort(Data_Sol[clustering_cols[-1]].unique()),
                                       value = [],
                                       style = {'color':'Black'},
                                       multi = True
                                      )


# ID vs Cluster Plot
IDvsCluster_Graph = dcc.Graph(id = 'IDvsCluster-graph')

# Radio Items Selection Mode
Radio_IDMode = html.Div(
    [
        dbc.RadioItems(
            id = 'mode-ID-graph',
            options = [
                {"label": "Graph", "value": 'click_mode'},
                {"label": "Manual", "value": 'manual_mode'},
            ],
            value = 'click_mode',
            inline=True,
        )
    ]
)

# Manual ID input
Manual_input = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Input(
                id= 'id-input',
                placeholder="ID",
                type="number",
                min=Data_Sol['short_ID'].min(),
                max=Data_Sol['short_ID'].max(),
                step=1,
                style={"width": 80, "height": 25}
            ),
        ], width=2),
        dbc.Col([
             dbc.Input(
                id= 'window-input',
                placeholder="Window",
                type="number",
                min=Data_Sol['window_ID'].min(),
                max=Data_Sol['window_ID'].max(),
                step=1,
                style={"width": 80, "height": 25}
            ),
        ], width=2),
        dbc.Col([
            dbc.FormText(id = 'input-under-text', children = f"ID range: [{id_range[0]}, {id_range[1]}]")
        ])
    ])
])

# Cluster Plot
Cluster_Graph = dcc.Graph(id = 'Cluster-graph')

# Cluser Mean-Median Radio Items
MeanMedian_Radioitems = dbc.RadioItems(
    id="MeanMedian_Radioitems",
    options=[
        {"label": "Mean", "value": 'mean'},
        {"label": "Median", "value": 'median'},
    ],
    value='mean',
    inline=True,
)

# Line Plot
SelectedIDs_Graph = dcc.Graph(id = 'selectedIDs-graph')

######################################################
# Overall Layout
app.layout = html.Div([navbar, html.Br(),
                       dbc.Row([
                           dbc.Card([
                               dbc.CardBody([
                                   html.H5('Solution Under Analysis'),
                                   Dropdown_Solutions])                                   
                               ])
                           ]),
                       dbc.Row([
                          dbc.Col([
                              dbc.Card([
                                  dbc.CardBody([
                                      html.H5('Clustering Solution'),
                                      Dropdown_ClusteringOptions,
                                      Scatter_Graph_Title,
                                      Scatter_Graph,
                                      dbc.Card([
                                          dbc.CardBody([
                                              html.H6("Cluster Selection", style={'textAlign': 'left'}),
                                              Dropdown_ActiveClusters
                                          ])
                                      ])
                                  ])
                              ])
                          ],
                              width=6),

                          dbc.Col([
                              dbc.Card([
                                  dbc.CardBody([
                                      dbc.Row([
                                          dbc.Col([
                                              html.H5('Single ID')
                                          ], width=2),
                                          dbc.Col([
                                              Radio_IDMode
                                          ], width=3),
                                          dbc.Col([
                                              Manual_input
                                          ])
                                      ]),
                                      IDvsCluster_Graph,
                                      html.Br(),
                                      dbc.Row([
                                          dbc.Col([
                                              html.H5('Clusters')
                                          ], width=2),
                                          dbc.Col([
                                              MeanMedian_Radioitems
                                          ])
                                      ]),
                                      Cluster_Graph,
                                      html.Br(),
                                      html.H5('Aggregate Selected Points'),
                                      SelectedIDs_Graph,
                                  ])
                              ])
                          ],
                              width=6)
                       ])
                      ])


# ######################################################
# # Callbacks


# ######################################################
# # Update Active Cluster Dropdown
# @app.callback(
#     Output('dropdown-ActiveClusters', 'options'),    
#     Input('dropdown-solutions', 'value'),
#     Input('dropdown-clustering', 'value')
# )
# def update_RangeSlider(solution_selected, clustering_solution):
    
#     if (solution_selected is None) or (clustering_solution is None):
#         return {}    
#     else:
#         #load new solution
#         df_sol = pd.read_csv(f'../ModelResults/ClusteringUmap/{solution_selected}.csv')
#         #update dropdown options and values
#         options = np.sort(df_sol[clustering_solution].unique())

#         return options


# ######################################################
# # Update Scatter plot
# @app.callback(
#     Output('scatter-graph', 'figure'),
#     Input('dropdown-solutions', 'value'),
#     Input('dropdown-clustering', 'value'),
#     Input('dropdown-ActiveClusters', 'value'),
# )
# def update_scatter_graph(solution_selected, clustering_solution, active_clusters):
    
#     # Error Case:
#     if (solution_selected is None) or (clustering_solution is None) or (active_clusters is None):
#         return {}
    
#     # Any other time:
#     else:
#         if not active_clusters:
#             #########################
#             # Pre-Plotting Operations
#             # Load Solution
#             df_sol = pd.read_csv(f'../ModelResults/ClusteringUmap/{solution_selected}.csv') 

#             # Local df with relevant clustering solution
#             df_clusters = df_sol[['local_id', 'window_id', 'UMAP_V1', 'UMAP_V2',clustering_solution]]
#             cols = window_col_names.values.tolist()
#             cols.append('local_id')
#             cols.append('window_id')
#             df_clusters = df_clusters.merge(df_zscored[cols], how = 'left', on=['local_id', 'window_id'])

#             #########################
#             # Actual Plot
#             # Start Figure
#             fig = go.Figure()

#             # Clustered Points by Colour
#             # Show the selected clusters by their respective colours
#             fig.add_trace(go.Scattergl(
#                 x = df_clusters['UMAP_V1'],
#                 y = df_clusters['UMAP_V2'],
#                 mode='markers',
#                 customdata = np.stack((df_clusters[clustering_solution], df_clusters['local_id'], df_clusters['window_id']), axis=-1),
#                 hovertemplate ='<b>Cluster: %{customdata[0]}</b><br>ID: %{customdata[1]}<br>Window: %{customdata[2]}<extra></extra>',
#                 marker=dict(
#                     color= df_clusters[clustering_solution],
#                     cmax = df_sol[clustering_solution].max(),
#                     cmin = df_sol[clustering_solution].min(),                
#                     colorscale= 'portland',  #turbo, rainbow, jet one of plotly colorscales
#                     showscale= True    #set color equal to a variable
#                 )
#             )
#                          )

#             # Customising Appearance
#             fig.update_layout(
#                 margin=dict(l=20, r=20, t=20, b=20),
#                 template= 'plotly_dark',
#                 showlegend=False,
#                 annotations=[go.layout.Annotation(
#                                 font = {'size': 14},
#                                 align='left',
#                                 showarrow=False,
#                                 xref='paper',
#                                 yref='paper',
#                                 x=0.01,
#                                 y=0.95,
#                                 )]
#             )            

#             return fig

#         elif active_clusters:
#             #########################
#             # Pre-Plotting Operations

#             # Load Solution
#             df_sol = pd.read_csv(f'../ModelResults/ClusteringUmap/{solution_selected}.csv') 

#             # Local df with relevant clustering solution
#             df_clusters = df_sol[['local_id', 'window_id', 'UMAP_V1', 'UMAP_V2',clustering_solution]]
#             cols = window_col_names.values.tolist()
#             cols.append('local_id')
#             cols.append('window_id')
#             df_clusters = df_clusters.merge(df_zscored[cols], how = 'left', on=['local_id', 'window_id'])

#             # Filtering based of RangeSilder Cluster
#             df_filtered = df_clusters[df_clusters[clustering_solution].isin(active_clusters)]
#             df_NegativeFilter = df_clusters[~df_clusters[clustering_solution].isin(active_clusters)]

#             #Calculating number of points highlighted (HP)
#             filtered_points = len(df_filtered.index)
#             percentage_fp = (filtered_points / len(df_sol.index)) * 100
#             if percentage_fp >= 1:
#                 percentage_fp = np.round(percentage_fp, 0)
#             else:
#                 percentage_fp =  np.round(percentage_fp , -int(np.floor(np.log10(abs(percentage_fp))))) 

#             #########################
#             # Actual Plot
#             # Start Figure
#             fig = go.Figure()

#             # Grey Points Plot
#             # Show points out of range in grey colour, for reference
#             fig.add_trace(go.Scattergl(
#                 x = df_NegativeFilter['UMAP_V1'],
#                 y = df_NegativeFilter['UMAP_V2'],
#                 mode='markers',
#                 hoverinfo='skip',
#                 marker=dict(
#                         color= 'rgba(100,100,100, 0.7)',
#                     )
#                 )
#             )

#             # Clustered Points by Colour
#             # Show the selected clusters by their respective colours
#             fig.add_trace(go.Scattergl(
#                 x = df_filtered['UMAP_V1'],
#                 y = df_filtered['UMAP_V2'],
#                 mode='markers',
#                 customdata = np.stack((df_filtered[clustering_solution], df_filtered['local_id'], df_filtered['window_id']), axis=-1),
#                 hovertemplate ='<b>Cluster: %{customdata[0]}</b><br>ID: %{customdata[1]}<br>Window: %{customdata[2]}<extra></extra>',
#                 marker=dict(
#                     color= df_filtered[clustering_solution],
#                     cmax = df_sol[clustering_solution].max(),
#                     cmin = df_sol[clustering_solution].min(),                
#                     colorscale= 'portland',  #turbo, rainbow, jet one of plotly colorscales
#                     showscale= True    #set color equal to a variable
#                 )
#             )
#                          )

#             # Customising Appearance
#             fig.update_layout(
#                 margin=dict(l=20, r=20, t=20, b=20),
#                 template= 'plotly_dark',
#                 showlegend=False,
#                 annotations=[go.layout.Annotation(
#                                 text=f'[HP: {filtered_points} ({percentage_fp}%)]',
#                                 font = {'size': 14},
#                                 align='left',
#                                 showarrow=False,
#                                 xref='paper',
#                                 yref='paper',
#                                 x=0.01,
#                                 y=0.95,
#                                 )]
#             )            

#             return fig

    
# ##################################
# # Placeholder Plot  
# def placeholder_fig (message):
#     layout = go.Layout(
#         margin=dict(l=20, r=20, t=28, b=20),
#         template= 'plotly_dark',
#         height = 250,
#         annotations=[go.layout.Annotation(
#                         text= message,
#                         font = {'size': 14},
#                         align='left',
#                         showarrow=False,
#                         xref='paper',
#                         yref='paper',
#                         x=0.5,
#                         y=0.5,
#                         )]
#         )        
#     return go.Figure(layout=layout)     

# ######################################################    
# # Update Selected IDs Graph
# @app.callback(
#     Output('selectedIDs-graph', 'figure'),
#     Input('scatter-graph', 'selectedData'),
# )
# def update_AggIDs_graph(selectedData):
#     if (selectedData is None):
#         #Placeholder plot
#         return placeholder_fig('Make a Group Selection.')
    
#     else:
#         # Un-Nesting selected points 
#         selected_ids = []
#         selected_windows = []
#         for p in selectedData['points']:
#             # Ignores Grey Points:
#             if "customdata" in p:
#                 selected_ids.append(p['customdata'][1])
#                 selected_windows.append(p['customdata'][2])
#         # Maintaining id to window order
#         df_selected = pd.DataFrame(columns=['local_id', 'window_id'])        
#         df_selected['local_id'] =  selected_ids
#         df_selected['window_id'] =  selected_windows   
#         # Filtering for selected points        
#         cols = window_col_names.values.tolist()
#         cols.append('local_id')
#         cols.append('window_id')
#         df_selected = df_selected.merge(df_zscored[cols], how = 'left', on=['local_id', 'window_id'])
        
#         # Figure Per Se
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=window_col_names,
#                         y=df_selected[window_col_names].mean(),
#                         mode='lines',
#                         line_color = 'rgba(100,100,255, 0.8)',         
#                         name= f'Mean'
#                         ))
#         fig.add_trace(go.Scatter(x=window_col_names,
#                         y=df_selected[window_col_names].median(),
#                         mode='lines',
#                         line_color = 'rgba(200,200,200, 0.8)',         
#                         name= f'Median'
#                         ))
#         fig.update_layout(
#             margin=dict(l=20, r=28, t=20, b=20),
#             template= 'plotly_dark',
#             height = 250,
#             annotations=[go.layout.Annotation(
#                 text= f'# Points:<br>{len(selected_ids)}',
#                 font = {'size': 12},
#                 align='left',
#                 showarrow=False,
#                 xref='paper',
#                 yref='paper',
#                 x=1.025,
#                 y=0.7,
#                 xanchor="left",
#                 )]
#         )
#         return fig 
    
# ######################################################    
# # Update Clusters plot
# @app.callback(
#     Output('Cluster-graph', 'figure'),
#     Input('dropdown-solutions', 'value'),
#     Input('dropdown-clustering', 'value'),
#     Input('MeanMedian_Radioitems', 'value'),    
# )
# def update_clusters_graph(solution_selected, clustering_solution, radio_option):    

#     # Error Case:
#     if (solution_selected is None) or (clustering_solution is None) or (radio_option is None):
#         return placeholder_fig('Select Solution.')    
#     # Any other time:
#     else:
#         # Load Solution
#         df_sol = pd.read_csv(f'../ModelResults/ClusteringUmap/{solution_selected}.csv') 
        
#         # Local df with relevant clustering solution
#         df_clusters = df_sol[['local_id', 'window_id', clustering_solution]]
#         cols = window_col_names.values.tolist()
#         cols.append('local_id')
#         cols.append('window_id')
#         df_clusters = df_clusters.merge(df_zscored[cols], how = 'left', on=['local_id', 'window_id'])
        
#         if radio_option == 'mean': 
#             fig = go.Figure()
#             for c in df_clusters[clustering_solution].sort_values().unique():
#                 fig.add_trace(go.Scatter(x=window_col_names,
#                                 y= df_clusters[df_clusters[clustering_solution] == c][window_col_names].mean(),
#                                 mode='lines',
#                                 name= f'Cluster: {c}'
#                                 ))
#         elif radio_option == 'median':
#             fig = go.Figure()
#             for c in df_clusters[clustering_solution].sort_values().unique():
#                 fig.add_trace(go.Scatter(x=window_col_names,
#                                 y= df_clusters[df_clusters[clustering_solution] == c][window_col_names].median(),
#                                 mode='lines',
#                                 name= f'Cluster: {c}'
#                                 ))
#         else:
#             return {}

#         fig.update_layout(
#             margin=dict(l=20, r=20, t=20, b=20),
#             template= 'plotly_dark',
#             height = 250
#         )
    
#         return fig #placeholder_fig('test') 
    
# ######################################################    
# # Update ID vs Cluster Plot
# @app.callback(
#     Output('IDvsCluster-graph', 'figure'),
#     Input('dropdown-solutions', 'value'),
#     Input('id-input', 'value'),
#     Input('window-input', 'value'),
#     Input('scatter-graph', 'clickData'),
#     Input('dropdown-clustering', 'value')
# )
# def update_IDvsCluster_graph(solution_selected, input_id, input_win, clickData, clustering_solution):
    
#     if (clickData is None):
#         return placeholder_fig('Select Point on Graph.')
#     else:
#         if input_id is None:
#             return placeholder_fig('Type a valid ID.')
#         else:
#             if input_win is None:
#                 return placeholder_fig('Type a valid Window.')
#             else:        
#                 # Retriving Window Time Series
#                 selected_id = df_zscored[(df_zscored['local_id'] == input_id) & (df_zscored['window_id'] == input_win)]
#                 df_id = pd.DataFrame()
#                 df_id['days'] = window_col_names
#                 df_id['values'] = selected_id[window_col_names].values[0]                
        
#                 # Load Solution
#                 df_sol = pd.read_csv(f'../ModelResults/ClusteringUmap/{solution_selected}.csv') 
#                 input_cluster = df_sol[(df_sol['local_id'] == input_id) & (df_sol['window_id'] == input_win)][clustering_solution].values[0]
#                 # Local df with relevant clustering solution
#                 df_cluster = df_sol[['local_id', 'window_id', clustering_solution]]
#                 df_cluster = df_cluster[df_cluster[clustering_solution] == input_cluster]
#                 cols = window_col_names.values.tolist()
#                 cols.append('local_id')
#                 cols.append('window_id')
#                 df_cluster = df_cluster.merge(df_zscored[cols], how = 'left', on=['local_id', 'window_id'])
                
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatter(x=window_col_names,
#                                 y= df_id['values'],
#                                 mode='lines',
#                                 line_color = 'rgba(100,100,255, 0.8)',         
#                                 name= f'ID'
#                                 ))
#                 fig.add_trace(go.Scatter(x=window_col_names,
#                                 y=df_cluster[window_col_names].mean(),
#                                 mode='lines',
#                                 line_color = 'rgba(200,200,200, 0.8)',         
#                                 name= f'Cluster Mean'
#                                 ))
#                 fig.add_trace(go.Scatter(x=window_col_names,
#                                 y=df_cluster[window_col_names].median(),
#                                 mode='lines',
#                                 line_color = 'rgba(255,100,100, 0.8)',         
#                                 name= f'Cluster Median'
#                                 ))
#                 fig.update_layout(
#                     margin=dict(l=20, r=20, t=20, b=20),
#                     template= 'plotly_dark',
#                     height = 250,
#                     annotations=[go.layout.Annotation(
#                         text= f'Cluster: {input_cluster}<br>ID: {input_id}<br>Window: {input_win}',
#                         font = {'size': 12},
#                         align='left',
#                         showarrow=False,
#                         xref='paper',
#                         yref='paper',
#                         x=1.025,
#                         y=0.5,
#                         xanchor="left",
#                         )]

#                 )

#                 return fig #json.dumps(selectedData, indent=2)


# ######################################################    
# # Update Input Boxes
# @app.callback(
#     Output('input-under-text', 'children'),
#     Output('window-input', 'min'), 
#     Output('window-input', 'max'),
#     Output('window-input', 'disabled'),
#     Output('id-input', 'value'),
#     Output('window-input', 'value'),
#     Input('id-input', 'value'),
#     Input('window-input', 'value'),
#     Input('dropdown-solutions', 'value'),
#     Input('scatter-graph', 'clickData'),
#     Input('mode-ID-graph', 'value'),
# )
# def update_input(id_input, window_input, solution_selected, clickData, mode_option):
    
#     if mode_option == 'click_mode':
#         #Graph mode
#         message = 'Select point on graph'
#         if (clickData is None):
#             user_id = id_input
#             window = window_input
#         else:
#             user_id = clickData['points'][0]['customdata'][1]
#             window = clickData['points'][0]['customdata'][2]
#     else:
#         user_id = id_input
#         window = window_input
        
#     if id_input is None:
#         return f"ID range: {[id_range[0], id_range[1]]}", 0, 10, True, user_id, window
#     else:
#         df_reconstruct = pd.read_csv(f'../ModelResults/Reconstruction/{solution_selected}.csv')       
        
#         win_range = [
#             df_reconstruct[df_reconstruct['local_id']==user_id].window_id.min(),
#             df_reconstruct[df_reconstruct['local_id']==user_id].window_id.max()]
        
#         if mode_option == 'manual_mode':
#             message = f'Window range: {[win_range[0], win_range[1]]}'
               
#         return message, win_range[0], win_range[1], False, user_id, window
    
######################################################
# Running Dashboard
if __name__ == '__main__':
    app.run_server(debug=True, port=8052)
