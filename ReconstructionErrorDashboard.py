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
id_range =[df_sol['local_id'].min(), df_sol['local_id'].max()] 

# Setting default Slider values
df_reconstruct = pd.read_csv(f'../ModelResults/Reconstruction/{solution_list[0]}.csv')
min_mse = np.round(df_reconstruct['MSE'].min(), 1)
max_mse = np.round(df_reconstruct['MSE'].max(), 1)


######################################################
# Creating App with Theme
app=Dash(external_stylesheets=[dbc.themes.DARKLY])

######################################################
# Layout Elements

# Header Bar
navbar = dbc.NavbarSimple(
    brand = 'Reconstruction Error',
    fluid = True, #fill horizontal space,
)

# Dropdown Solutions Available
Dropdown_Solutions = dcc.Dropdown(id = 'dropdown-solutions',
                                  options = solution_list,
                                  value = solution_list[0],
                                  style = {'color':'Black'}
                                 )

# MSE Slider
RangeSlider_MSE = dcc.RangeSlider(id='rangeSlider-mse',
                                 min = min_mse,
                                 max = max_mse,
                                 value = [min_mse, max_mse],
                                 tooltip={"placement": "bottom", "always_visible": True})


# Scatter Graph Title using a Card for aesthetic reasons
Scatter_Graph_Title = dbc.Card("2D UMAP Representation - MSE", body=True)

# 2D UMAP Plot
Scatter_Graph = dcc.Graph(id = 'scatter-graph')

# Line Plot
Line_Graph = dcc.Graph(id = 'line-graph')

# Radio Items Selection Mode
Radio_LineMode = html.Div(
    [
        dbc.RadioItems(
            id = 'mode-linegraph',
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
Manual_input = html.Div(
    [
        dbc.Input(
            id= 'id-input',
            placeholder="ID",
            type="number",
            min=df_sol['local_id'].min(),
            max=df_sol['local_id'].max(),
            step=1,
            style={"width": 150, "height": 25}
        ),
        dbc.Input(
            id= 'window-input',
            placeholder="Window",
            type="number",
            min=df_sol['window_id'].min(),
            max=df_sol['window_id'].max(),
            step=1,
            style={"width": 150, "height": 25}
        ),
        dbc.FormText(id = 'input-under-text', children = f"ID range: [{id_range[0]}, {id_range[1]}]")
    ]
)



######################################################
# Overall Layout
app.layout = html.Div([navbar, html.Br(),
                       dbc.Row([
                           dbc.Card([
                               dbc.CardBody([
                                   html.H6('Auto Encoder Model'),
                                   Dropdown_Solutions]),
                               ])
                           ]),                           
                       dbc.Row([
                          dbc.Col([Scatter_Graph_Title,
                                   Scatter_Graph,
                                   dbc.Card([
                                        dbc.CardBody([
                                            html.H6("MSE value range", style={'textAlign': 'center'}),
                                            RangeSlider_MSE
                                        ]),                                      
                                       ]),
                                   ],
                                  width=6),
                          dbc.Col([
                              dbc.Card([
                                  dbc.CardBody([
                                      html.H6('Time Credit: Original VS Recontructed'),
                                      Line_Graph,
                                      Radio_LineMode,
                                      Manual_input
                                  ])
                              ])                             
                            ],
                              width=6),
                           ]),
                        ])


######################################################
# Callbacks

######################################################
# Update RangeSlider
@app.callback(
    Output('rangeSlider-mse', 'min'),
    Output('rangeSlider-mse', 'max'),
    Output('rangeSlider-mse', 'value'),
    Input('dropdown-solutions', 'value'),
)
def update_RangeSlider(solution_selected):
    
    if (solution_selected is None):
        return {}    
    else:
        #load new solution
        df_reconstruct = pd.read_csv(f'../ModelResults/Reconstruction/{solution_selected}.csv')
        #update slider range and preselected values
        min_mse = np.round(df_reconstruct['MSE'].min(), 1)
        max_mse = np.round(df_reconstruct['MSE'].max(), 1)    
        val_mse = [min_mse, max_mse]

        return min_mse, max_mse, val_mse

######################################################
# Update Scatter plot
@app.callback(
    Output('scatter-graph', 'figure'),
    Input('dropdown-solutions', 'value'),
    Input('rangeSlider-mse', 'value'),
)
def update_scatter_graph(solution_selected, mse_range):
    
    if (solution_selected is None) or (mse_range is None):
        return {}
    else:
        #Load solution
        df_reconstruct = pd.read_csv(f'../ModelResults/Reconstruction/{solution_selected}.csv')
        #Filter Base on Range Slider
        df_filtered = df_reconstruct[df_reconstruct['MSE'].between(mse_range[0], mse_range[1])]
        df_NegativeFilter = df_reconstruct[~df_reconstruct['MSE'].between(mse_range[0], mse_range[1])]
        
        #Calculating number of points highlighted (HP)
        filtered_points = len(df_filtered.index)
        percentage_fp = (filtered_points / len(df_reconstruct.index)) * 100
        if percentage_fp >= 1:
            percentage_fp = np.round(percentage_fp, 0)
        else:
            percentage_fp =  np.round(percentage_fp , -int(np.floor(np.log10(abs(percentage_fp)))))
        
        fig = go.Figure()
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
        
        # Add points based on MSE
        fig.add_trace(go.Scattergl(
            x = df_filtered['UMAP_V1'],
            y = df_filtered['UMAP_V2'],
            mode='markers',
            customdata = np.stack((df_filtered['MSE'], df_filtered['local_id'], df_filtered['window_id']), axis=-1),
            hovertemplate ='<b>MSE: %{customdata[0]}</b><br>ID: %{customdata[1]}<br>Window: %{customdata[2]}<extra></extra>',
            marker=dict(
                    color= df_filtered['MSE'],
                    cmax = df_reconstruct['MSE'].max(),
                    cmin = df_reconstruct['MSE'].min(),
                    opacity= 0.7,
                    colorscale= 'portland',  #turbo, rainbow, jet one of plotly colorscales
                    showscale= True#set color equal to a variable
                )
            )
        )
        
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
        
######################################################    
# Line Plot function
def line_plot (df_orig, df_recons, selected_MSE, selected_id, selected_window):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window_col_names,
                    y= df_orig['values'],
                    mode='lines',
                    line_color = 'rgba(100,100,255, 0.8)',         
                    name= f'Input'
                    ))
    fig.add_trace(go.Scatter(x=window_col_names,
                    y= df_recons['values'],
                    mode='lines',
                    line_color = 'rgba(255,100,100, 0.8)',         
                    name= f'Reconstruction'
                    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=28, b=20),
        template= 'plotly_dark',
        height = 250,
         annotations=[go.layout.Annotation(
                        text= f'MSE: {np.round(selected_MSE,3)}<br>ID: {selected_id}<br>Window: {selected_window}',
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

# Update Line Plot
@app.callback(
    Output('line-graph', 'figure'),
    Input('dropdown-solutions', 'value'),
    Input('id-input', 'value'),
    Input('window-input', 'value'),
    Input('scatter-graph', 'clickData'),
)
def update_line_graph(solution_selected, input_id, input_win, clickData):
    
    if (clickData is None):
        return placeholder_fig('Select Point on Graph.')
    else:
        if input_id is None:
            return placeholder_fig('Type a valid ID.')
        else:
            if input_win is None:
                return placeholder_fig('Type a valid Window.')
            else:
                df_reconstruct = pd.read_csv(f'../ModelResults/Reconstruction/{solution_selected}.csv')        

                # Retriving Original Data (zscored)
                orig_ts = df_zscored[(df_zscored['local_id'] == input_id) & (df_zscored['window_id'] == input_win)]
                df_orig = pd.DataFrame()
                df_orig['days'] = window_col_names
                df_orig['values'] = orig_ts[window_col_names].values[0]

                # Selected window data
                recons_ts = df_reconstruct[(df_reconstruct['local_id'] == input_id) & (df_reconstruct['window_id'] == input_win)]
                df_recons = pd.DataFrame()
                df_recons['days'] = window_col_names
                df_recons['values'] = recons_ts[window_col_names].values[0] 

                return line_plot (df_orig, df_recons, recons_ts['MSE'].values[0], input_id, input_win)

    
    
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
    Input('mode-linegraph', 'value'),
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
    app.run_server(debug=True)
