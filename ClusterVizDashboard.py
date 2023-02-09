from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import os
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json


# Importing Data
df_zscored = pd.read_csv(f'../Data/W90_Zscored.csv') 
used_cols = len(df_zscored.columns)-3
window_col_names = df_zscored.columns[-used_cols:]

# Retriving all solution file names
extension = 'csv'
os.chdir("../ModelResults/ClusteringUmap")
solution_files = glob.glob('*.{}'.format(extension))
solution_list = [i[:-4] for i in solution_files ]
os.chdir("../../DashBoards")

# Setting defeault solution
df_sol = pd.read_csv(f'../ModelResults/ClusteringUmap/{solution_list[0]}.csv')  
used_cols = len(df_sol.columns)-5
clustering_cols = df_sol.columns[-used_cols:]
min_cluster = df_sol[clustering_cols[-1]].min()
max_cluster = df_sol[clustering_cols[-1]].max()
id_range =[df_sol['local_id'].min(), df_sol['local_id'].max()] 



######################################################
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
                                  options = solution_list,
                                  value = solution_list[0],
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
                                       options = np.sort(df_sol[clustering_cols[-1]].unique()),
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
                min=df_sol['local_id'].min(),
                max=df_sol['local_id'].max(),
                step=1,
                style={"width": 80, "height": 25}
            ),
        ], width=2),
        dbc.Col([
             dbc.Input(
                id= 'window-input',
                placeholder="Window",
                type="number",
                min=df_sol['window_id'].min(),
                max=df_sol['window_id'].max(),
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


######################################################
# Callbacks


######################################################
# Update Active Cluster Dropdown
@app.callback(
    Output('dropdown-ActiveClusters', 'options'),    
    Input('dropdown-solutions', 'value'),
    Input('dropdown-clustering', 'value')
)
def update_RangeSlider(solution_selected, clustering_solution):
    
    if (solution_selected is None) or (clustering_solution is None):
        return {}    
    else:
        #load new solution
        df_sol = pd.read_csv(f'../ModelResults/ClusteringUmap/{solution_selected}.csv')
        #update dropdown options and values
        options = np.sort(df_sol[clustering_solution].unique())

        return options


######################################################
# Update Scatter plot
@app.callback(
    Output('scatter-graph', 'figure'),
    Input('dropdown-solutions', 'value'),
    Input('dropdown-clustering', 'value'),
    Input('dropdown-ActiveClusters', 'value'),
)
def update_scatter_graph(solution_selected, clustering_solution, active_clusters):
    
    # Error Case:
    if (solution_selected is None) or (clustering_solution is None) or (active_clusters is None):
        return {}
    
    # Any other time:
    else:
        if not active_clusters:
            #########################
            # Pre-Plotting Operations
            # Load Solution
            df_sol = pd.read_csv(f'../ModelResults/ClusteringUmap/{solution_selected}.csv') 

            # Local df with relevant clustering solution
            df_clusters = df_sol[['local_id', 'window_id', 'UMAP_V1', 'UMAP_V2',clustering_solution]]
            cols = window_col_names.values.tolist()
            cols.append('local_id')
            cols.append('window_id')
            df_clusters = df_clusters.merge(df_zscored[cols], how = 'left', on=['local_id', 'window_id'])

            #########################
            # Actual Plot
            # Start Figure
            fig = go.Figure()

            # Clustered Points by Colour
            # Show the selected clusters by their respective colours
            fig.add_trace(go.Scattergl(
                x = df_clusters['UMAP_V1'],
                y = df_clusters['UMAP_V2'],
                mode='markers',
                customdata = np.stack((df_clusters[clustering_solution], df_clusters['local_id'], df_clusters['window_id']), axis=-1),
                hovertemplate ='<b>Cluster: %{customdata[0]}</b><br>ID: %{customdata[1]}<br>Window: %{customdata[2]}<extra></extra>',
                marker=dict(
                    color= df_clusters[clustering_solution],
                    cmax = df_sol[clustering_solution].max(),
                    cmin = df_sol[clustering_solution].min(),                
                    colorscale= 'portland',  #turbo, rainbow, jet one of plotly colorscales
                    showscale= True    #set color equal to a variable
                )
            )
                         )

            # Customising Appearance
            fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                template= 'plotly_dark',
                showlegend=False,
                annotations=[go.layout.Annotation(
                                font = {'size': 14},
                                align='left',
                                showarrow=False,
                                xref='paper',
                                yref='paper',
                                x=0.01,
                                y=0.95,
                                )]
            )            

            return fig

        elif active_clusters:
            #########################
            # Pre-Plotting Operations

            # Load Solution
            df_sol = pd.read_csv(f'../ModelResults/ClusteringUmap/{solution_selected}.csv') 

            # Local df with relevant clustering solution
            df_clusters = df_sol[['local_id', 'window_id', 'UMAP_V1', 'UMAP_V2',clustering_solution]]
            cols = window_col_names.values.tolist()
            cols.append('local_id')
            cols.append('window_id')
            df_clusters = df_clusters.merge(df_zscored[cols], how = 'left', on=['local_id', 'window_id'])

            # Filtering based of RangeSilder Cluster
            df_filtered = df_clusters[df_clusters[clustering_solution].isin(active_clusters)]
            df_NegativeFilter = df_clusters[~df_clusters[clustering_solution].isin(active_clusters)]

            #Calculating number of points highlighted (HP)
            filtered_points = len(df_filtered.index)
            percentage_fp = (filtered_points / len(df_sol.index)) * 100
            if percentage_fp >= 1:
                percentage_fp = np.round(percentage_fp, 0)
            else:
                percentage_fp =  np.round(percentage_fp , -int(np.floor(np.log10(abs(percentage_fp))))) 

            #########################
            # Actual Plot
            # Start Figure
            fig = go.Figure()

            # Grey Points Plot
            # Show points out of range in grey colour, for reference
            fig.add_trace(go.Scattergl(
                x = df_NegativeFilter['UMAP_V1'],
                y = df_NegativeFilter['UMAP_V2'],
                mode='markers',
                hoverinfo='skip',
                marker=dict(
                        color= 'rgba(100,100,100, 0.7)',
                    )
                )
            )

            # Clustered Points by Colour
            # Show the selected clusters by their respective colours
            fig.add_trace(go.Scattergl(
                x = df_filtered['UMAP_V1'],
                y = df_filtered['UMAP_V2'],
                mode='markers',
                customdata = np.stack((df_filtered[clustering_solution], df_filtered['local_id'], df_filtered['window_id']), axis=-1),
                hovertemplate ='<b>Cluster: %{customdata[0]}</b><br>ID: %{customdata[1]}<br>Window: %{customdata[2]}<extra></extra>',
                marker=dict(
                    color= df_filtered[clustering_solution],
                    cmax = df_sol[clustering_solution].max(),
                    cmin = df_sol[clustering_solution].min(),                
                    colorscale= 'portland',  #turbo, rainbow, jet one of plotly colorscales
                    showscale= True    #set color equal to a variable
                )
            )
                         )

            # Customising Appearance
            fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                template= 'plotly_dark',
                showlegend=False,
                annotations=[go.layout.Annotation(
                                text=f'[HP: {filtered_points} ({percentage_fp}%)]',
                                font = {'size': 14},
                                align='left',
                                showarrow=False,
                                xref='paper',
                                yref='paper',
                                x=0.01,
                                y=0.95,
                                )]
            )            

            return fig

    
##################################
# Placeholder Plot  
def placeholder_fig (message):
    layout = go.Layout(
        margin=dict(l=20, r=20, t=28, b=20),
        template= 'plotly_dark',
        height = 250,
        annotations=[go.layout.Annotation(
                        text= message,
                        font = {'size': 14},
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.5,
                        y=0.5,
                        )]
        )        
    return go.Figure(layout=layout)     

######################################################    
# Update Selected IDs Graph
@app.callback(
    Output('selectedIDs-graph', 'figure'),
    Input('scatter-graph', 'selectedData'),
)
def update_AggIDs_graph(selectedData):
    if (selectedData is None):
        #Placeholder plot
        return placeholder_fig('Make a Group Selection.')
    
    else:
        # Un-Nesting selected points 
        selected_ids = []
        selected_windows = []
        for p in selectedData['points']:
            # Ignores Grey Points:
            if "customdata" in p:
                selected_ids.append(p['customdata'][1])
                selected_windows.append(p['customdata'][2])
        # Maintaining id to window order
        df_selected = pd.DataFrame(columns=['local_id', 'window_id'])        
        df_selected['local_id'] =  selected_ids
        df_selected['window_id'] =  selected_windows   
        # Filtering for selected points        
        cols = window_col_names.values.tolist()
        cols.append('local_id')
        cols.append('window_id')
        df_selected = df_selected.merge(df_zscored[cols], how = 'left', on=['local_id', 'window_id'])
        
        # Figure Per Se
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=window_col_names,
                        y=df_selected[window_col_names].mean(),
                        mode='lines',
                        line_color = 'rgba(100,100,255, 0.8)',         
                        name= f'Mean'
                        ))
        fig.add_trace(go.Scatter(x=window_col_names,
                        y=df_selected[window_col_names].median(),
                        mode='lines',
                        line_color = 'rgba(200,200,200, 0.8)',         
                        name= f'Median'
                        ))
        fig.update_layout(
            margin=dict(l=20, r=28, t=20, b=20),
            template= 'plotly_dark',
            height = 250,
            annotations=[go.layout.Annotation(
                text= f'# Points:<br>{len(selected_ids)}',
                font = {'size': 12},
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=1.025,
                y=0.7,
                xanchor="left",
                )]
        )
        return fig 
    
######################################################    
# Update Clusters plot
@app.callback(
    Output('Cluster-graph', 'figure'),
    Input('dropdown-solutions', 'value'),
    Input('dropdown-clustering', 'value'),
    Input('MeanMedian_Radioitems', 'value'),    
)
def update_clusters_graph(solution_selected, clustering_solution, radio_option):    

    # Error Case:
    if (solution_selected is None) or (clustering_solution is None) or (radio_option is None):
        return placeholder_fig('Select Solution.')    
    # Any other time:
    else:
        # Load Solution
        df_sol = pd.read_csv(f'../ModelResults/ClusteringUmap/{solution_selected}.csv') 
        
        # Local df with relevant clustering solution
        df_clusters = df_sol[['local_id', 'window_id', clustering_solution]]
        cols = window_col_names.values.tolist()
        cols.append('local_id')
        cols.append('window_id')
        df_clusters = df_clusters.merge(df_zscored[cols], how = 'left', on=['local_id', 'window_id'])
        
        if radio_option == 'mean': 
            fig = go.Figure()
            for c in df_clusters[clustering_solution].sort_values().unique():
                fig.add_trace(go.Scatter(x=window_col_names,
                                y= df_clusters[df_clusters[clustering_solution] == c][window_col_names].mean(),
                                mode='lines',
                                name= f'Cluster: {c}'
                                ))
        elif radio_option == 'median':
            fig = go.Figure()
            for c in df_clusters[clustering_solution].sort_values().unique():
                fig.add_trace(go.Scatter(x=window_col_names,
                                y= df_clusters[df_clusters[clustering_solution] == c][window_col_names].median(),
                                mode='lines',
                                name= f'Cluster: {c}'
                                ))
        else:
            return {}

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            template= 'plotly_dark',
            height = 250
        )
    
        return fig #placeholder_fig('test') 
    
######################################################    
# Update ID vs Cluster Plot
@app.callback(
    Output('IDvsCluster-graph', 'figure'),
    Input('dropdown-solutions', 'value'),
    Input('id-input', 'value'),
    Input('window-input', 'value'),
    Input('scatter-graph', 'clickData'),
    Input('dropdown-clustering', 'value')
)
def update_IDvsCluster_graph(solution_selected, input_id, input_win, clickData, clustering_solution):
    
    if (clickData is None):
        return placeholder_fig('Select Point on Graph.')
    else:
        if input_id is None:
            return placeholder_fig('Type a valid ID.')
        else:
            if input_win is None:
                return placeholder_fig('Type a valid Window.')
            else:        
                # Retriving Window Time Series
                selected_id = df_zscored[(df_zscored['local_id'] == input_id) & (df_zscored['window_id'] == input_win)]
                df_id = pd.DataFrame()
                df_id['days'] = window_col_names
                df_id['values'] = selected_id[window_col_names].values[0]                
        
                # Load Solution
                df_sol = pd.read_csv(f'../ModelResults/ClusteringUmap/{solution_selected}.csv') 
                input_cluster = df_sol[(df_sol['local_id'] == input_id) & (df_sol['window_id'] == input_win)][clustering_solution].values[0]
                # Local df with relevant clustering solution
                df_cluster = df_sol[['local_id', 'window_id', clustering_solution]]
                df_cluster = df_cluster[df_cluster[clustering_solution] == input_cluster]
                cols = window_col_names.values.tolist()
                cols.append('local_id')
                cols.append('window_id')
                df_cluster = df_cluster.merge(df_zscored[cols], how = 'left', on=['local_id', 'window_id'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=window_col_names,
                                y= df_id['values'],
                                mode='lines',
                                line_color = 'rgba(100,100,255, 0.8)',         
                                name= f'ID'
                                ))
                fig.add_trace(go.Scatter(x=window_col_names,
                                y=df_cluster[window_col_names].mean(),
                                mode='lines',
                                line_color = 'rgba(200,200,200, 0.8)',         
                                name= f'Cluster Mean'
                                ))
                fig.add_trace(go.Scatter(x=window_col_names,
                                y=df_cluster[window_col_names].median(),
                                mode='lines',
                                line_color = 'rgba(255,100,100, 0.8)',         
                                name= f'Cluster Median'
                                ))
                fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    template= 'plotly_dark',
                    height = 250,
                    annotations=[go.layout.Annotation(
                        text= f'Cluster: {input_cluster}<br>ID: {input_id}<br>Window: {input_win}',
                        font = {'size': 12},
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=1.025,
                        y=0.5,
                        xanchor="left",
                        )]

                )

                return fig #json.dumps(selectedData, indent=2)


######################################################    
# Update Input Boxes
@app.callback(
    Output('input-under-text', 'children'),
    Output('window-input', 'min'), 
    Output('window-input', 'max'),
    Output('window-input', 'disabled'),
    Output('id-input', 'value'),
    Output('window-input', 'value'),
    Input('id-input', 'value'),
    Input('window-input', 'value'),
    Input('dropdown-solutions', 'value'),
    Input('scatter-graph', 'clickData'),
    Input('mode-ID-graph', 'value'),
)
def update_input(id_input, window_input, solution_selected, clickData, mode_option):
    
    if mode_option == 'click_mode':
        #Graph mode
        message = 'Select point on graph'
        if (clickData is None):
            user_id = id_input
            window = window_input
        else:
            user_id = clickData['points'][0]['customdata'][1]
            window = clickData['points'][0]['customdata'][2]
    else:
        user_id = id_input
        window = window_input
        
    if id_input is None:
        return f"ID range: {[id_range[0], id_range[1]]}", 0, 10, True, user_id, window
    else:
        df_reconstruct = pd.read_csv(f'../ModelResults/Reconstruction/{solution_selected}.csv')       
        
        win_range = [
            df_reconstruct[df_reconstruct['local_id']==user_id].window_id.min(),
            df_reconstruct[df_reconstruct['local_id']==user_id].window_id.max()]
        
        if mode_option == 'manual_mode':
            message = f'Window range: {[win_range[0], win_range[1]]}'
               
        return message, win_range[0], win_range[1], False, user_id, window
    
######################################################
# Running Dashboard
if __name__ == '__main__':
    app.run_server(debug=False)
