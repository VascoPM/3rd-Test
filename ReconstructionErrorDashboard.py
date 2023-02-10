# This is a Dash dashboard script.
# It creates a dashboard to view the Reconstruction Error (MSE) of Auto-Encoders.
# It is dependent on the file structure and naming policy used on the -Third Test- experiment.
# Vasco Mergulhao - 10/02/2023


from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from dash.dash import no_update
import os
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json


# What needs to be done
# Fix Dropdown options.
    # For some reason it doesn't update the dateset option, always stays in Kenya
    # Make it so, when changed, dependent dropdown defaults to first option on list.
# Fix Line Graph.
    # Creat Radio Options for Original Scale and Zscore.
    # Zcore (or inverse it) inside the callback function.

###########################################################
# Generic functions used in Dashboard
###########################################################
# Folder and File name Navigator Functions
def retrive_sol_files (dataset_name):
    extension = 'csv'
    os.chdir(f"../ModelResults/AE_Reconstruction/{dataset_name}")
    solution_files = glob.glob('*.{}'.format(extension))
    os.chdir("../../../Time-Series-PhD")
    
    return solution_files

def retrive_AEmodel_options (sol_files):    
    # Given a Solution File list (.csv names), return Auto Encoder Model types
    # This is dependent of the file naming policy 
    
    AEmodels_options = []
    for i, sol in enumerate(solution_files):
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
    for i, sol in enumerate(solution_files):
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
    for i, sol in enumerate(solution_files):
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

###########################################################
# Navigating File Structure to find all available solutions
# First, see list of Datasets used, defined by the solution folders

Dataset_options = next(os.walk("../ModelResults/AE_Reconstruction"))[1] # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory?page=1&tab=scoredesc#tab-top
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
Data_reconstruct = pd.read_csv(f'../ModelResults/AE_Reconstruction/{Dataset_options[0]}/{solution_files[0]}')
min_mse = np.round(Data_reconstruct['MSE'].min(), 1)
max_mse = np.round(Data_reconstruct['MSE'].max(), 1)
id_range =[Data_reconstruct['short_ID'].min(), Data_reconstruct['short_ID'].max()] 




###########################################################
# Dashboard Per se
###########################################################
# Creating App with Theme
app=Dash(external_stylesheets=[dbc.themes.DARKLY])

######################################################
# Layout Elements

# Header Bar
navbar = dbc.NavbarSimple(
    brand = 'Reconstruction Error',
    fluid = True, #fill horizontal space,
)

# Dropdown W&B Solutions Options
Dropdown_SolNames = dcc.Dropdown(
    id = 'dropdown-SolNames',
    # options = SolNames_options,
    # value = SolNames_options[0],
    style = {'color':'Black'}
)

# Dropdown for Dataset Options
Dropdown_Dataset = dcc.Dropdown(
    id = 'dropdown-Dataset',
    options = Dataset_options,
    # value = Dataset_options[0],
    style = {'color':'Black'}
)

# Dropdown for Auto Encoder Models Options
Dropdown_AEmodels = dcc.Dropdown(
    id = 'dropdown-AEmodels',
    # options = AEmodels_options,
    # value = AEmodels_options[0],
    style = {'color':'Black'}
)

# Dropdown for Latent Space Options
Dropdown_LatentSpace = dcc.Dropdown(
    id = 'dropdown-LatentSpace',
    # options = LatentSpace_options,
    # value = LatentSpace_options[0],
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
            min=Data_reconstruct['short_ID'].min(),
            max=Data_reconstruct['short_ID'].max(),
            step=1,
            style={"width": 150, "height": 25}
        ),
        dbc.Input(
            id= 'window-input',
            placeholder="Window",
            type="number",
            min=Data_reconstruct['window_ID'].min(),
            max=Data_reconstruct['window_ID'].max(),
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
                                   dbc.Row([
                                       dbc.Col([
                                           html.H6('W&B Solution Name'),
                                           Dropdown_SolNames
                                       ], width=2),
                                       dbc.Col([
                                           html.H6('Dataset'),
                                           Dropdown_Dataset
                                       ], width=2),
                                       dbc.Col([
                                           html.H6('Auto Encoder Type'),
                                           Dropdown_AEmodels
                                       ], width=2),
                                       dbc.Col([
                                           html.H6('Latent Space Size'),
                                           Dropdown_LatentSpace
                                       ], width=2),
                                   ])
                               ])
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
# Update Dropdown_AEmodels
@app.callback(
    Output('dropdown-AEmodels', 'options'),
    Input('dropdown-Dataset', 'value')
)
def update_Dropdown_AEmodels(dataset_selected):
    
    if dataset_selected:
    
        sol_files = retrive_sol_files (dataset_selected)
        AEmodels_options = retrive_AEmodel_options (sol_files)

        return AEmodels_options
    
    else:
        return no_update

######################################################
# Update Dropdown_LatentSpace
@app.callback(
    Output('dropdown-LatentSpace', 'options'),
    Input('dropdown-Dataset', 'value'),
    Input('dropdown-AEmodels', 'value'),
)
def update_Dropdown_LatentSpace(dataset_selected, AEmodel_selected):
    
    if dataset_selected and AEmodel_selected:
    
        sol_files = retrive_sol_files (dataset_selected)
        LatentSpace_options = retrive_LatentSpace_options (sol_files, AEmodel_selected)

        return LatentSpace_options
    
    else:
        return no_update

######################################################
# Update Dropdown_SolNames
@app.callback(
    Output('dropdown-SolNames', 'options'),
    Input('dropdown-Dataset', 'value'),
    Input('dropdown-AEmodels', 'value'),
    Input('dropdown-LatentSpace', 'value'),
)
def update_Dropdown_SolNames(dataset_selected, AEmodel_selected, LatentSpace_selected):
    
    if dataset_selected and AEmodel_selected and LatentSpace_selected:
        sol_files = retrive_sol_files (dataset_selected)
        SolNames_options = retrive_SolNames_options (sol_files, AEmodel_selected,  LatentSpace_selected)

        return SolNames_options
    else:
        return no_update

######################################################
# Update RangeSlider
@app.callback(
    Output('rangeSlider-mse', 'min'),
    Output('rangeSlider-mse', 'max'),
    Output('rangeSlider-mse', 'value'),
    Input('dropdown-Dataset', 'value'),
    Input('dropdown-SolNames', 'value'),
)
def update_RangeSlider(dataset_selected, solution_selected):
    
    if (solution_selected is None) or (dataset_selected is None):
        return None, None, None  
    
    else:        
        #Load solution
        sol_files = retrive_sol_files (dataset_selected)
        solution_file = None
        for sol in sol_files:
            if solution_selected in sol:
                solution_file = sol
        if solution_file:
            Data_reconstruct = pd.read_csv(f'../ModelResults/AE_Reconstruction/{dataset_selected}/{solution_file}')

            #update slider range and preselected values
            min_mse = np.round(Data_reconstruct['MSE'].min(), 1)
            max_mse = np.round(Data_reconstruct['MSE'].max(), 1)    
            val_mse = [min_mse, max_mse]

            return min_mse, max_mse, val_mse
        
        else:
            return None, None, None

######################################################
# Update Scatter plot
@app.callback(
    Output('scatter-graph', 'figure'),
    Input('dropdown-Dataset', 'value'),
    Input('dropdown-SolNames', 'value'),
    Input('rangeSlider-mse', 'value'),
)
def update_scatter_graph(dataset_selected, solution_selected, mse_range):
    
    if (dataset_selected is None) or (solution_selected is None) or (mse_range is None):
        return {}
    else:
        #Load solution
        sol_files = retrive_sol_files (dataset_selected)
        solution_file = None
        for sol in sol_files:
            if solution_selected in sol:
                solution_file = sol
        if solution_file:
            Data_reconstruct = pd.read_csv(f'../ModelResults/AE_Reconstruction/{dataset_selected}/{solution_file}')
            #Filter Base on Range Slider
            Data_filtered = Data_reconstruct[Data_reconstruct['MSE'].between(mse_range[0], mse_range[1])]
            Data_NegativeFilter = Data_reconstruct[~Data_reconstruct['MSE'].between(mse_range[0], mse_range[1])]

            #Calculating number of points highlighted (HP)
            filtered_points = len(Data_filtered.index)
            percentage_fp = (filtered_points / len(Data_reconstruct.index)) * 100
            if percentage_fp >= 1:
                percentage_fp = np.round(percentage_fp, 0)
            else:
                percentage_fp =  np.round(percentage_fp , -int(np.floor(np.log10(abs(percentage_fp)))))

            fig = go.Figure()
            # Show points out of range in grey colour, for reference
            fig.add_trace(go.Scattergl(
                x = Data_NegativeFilter['UMAP_V1'],
                y = Data_NegativeFilter['UMAP_V2'],
                mode='markers',
                hoverinfo='skip',
                marker=dict(
                        color= 'rgba(100,100,100, 0.7)',
                    )
                )
            )

            # Add points based on MSE
            fig.add_trace(go.Scattergl(
                x = Data_filtered['UMAP_V1'],
                y = Data_filtered['UMAP_V2'],
                mode='markers',
                customdata = np.stack((Data_filtered['MSE'], Data_filtered['short_ID'], Data_filtered['window_ID']), axis=-1),
                hovertemplate ='<b>MSE: %{customdata[0]}</b><br>ID: %{customdata[1]}<br>Window: %{customdata[2]}<extra></extra>',
                marker=dict(
                        color= Data_filtered['MSE'],
                        cmax = Data_reconstruct['MSE'].max(),
                        cmin = Data_reconstruct['MSE'].min(),
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
        else:
            return {}

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

######################################################
# Update Line Plot
@app.callback(
    Output('line-graph', 'figure'),
    Input('dropdown-Dataset', 'value'),    
    Input('dropdown-SolNames', 'value'),
    Input('id-input', 'value'),
    Input('window-input', 'value'),
    Input('scatter-graph', 'clickData'),
)
def update_line_graph(dataset_selected, solution_selected, input_id, input_win, clickData):
    
    if (clickData is None):
        return placeholder_fig('Select Point on Graph.')
    else:
        if input_id is None:
            return placeholder_fig('Type a valid ID.')
        else:
            if input_win is None:
                return placeholder_fig('Type a valid Window.')
            else:
                #Load solution
                sol_files = retrive_sol_files (dataset_selected)
                solution_file = None
                for sol in sol_files:
                    if solution_selected in sol:
                        solution_file = sol
                if solution_file:
                    df_reconstruct = pd.read_csv(f'../ModelResults/AE_Reconstruction/{dataset_selected}/{solution_file}')
                  

                # Retriving Original Data
                orig_ts = Data_orig[(Data_orig['short_ID'] == input_id) & (Data_orig['window_ID'] == input_win)]
                df_orig = pd.DataFrame()
                df_orig['days'] = window_col_names
                df_orig['values'] = orig_ts[window_col_names].values[0]

                # Selected window data
                recons_ts = df_reconstruct[(df_reconstruct['short_ID'] == input_id) & (df_reconstruct['window_ID'] == input_win)]
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
    Input('dropdown-Dataset', 'value'),        
    Input('dropdown-SolNames', 'value'),
    Input('scatter-graph', 'clickData'),
    Input('mode-linegraph', 'value'),
)
def update_input(id_input, window_input, dataset_selected, solution_selected, clickData, mode_option):
    
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
        # Load solution
        sol_files = retrive_sol_files (dataset_selected)
        solution_file = None
        for sol in sol_files:
            if solution_selected in sol:
                solution_file = sol
        df_reconstruct = pd.read_csv(f'../ModelResults/AE_Reconstruction/{dataset_selected}/{solution_file}')

        win_range = [
            df_reconstruct[df_reconstruct['short_ID']==user_id].window_ID.min(),
            df_reconstruct[df_reconstruct['short_ID']==user_id].window_ID.max()]

        if mode_option == 'manual_mode':
            message = f'Window range: {[win_range[0], win_range[1]]}'

        return message, win_range[0], win_range[1], False, user_id, window
    

######################################################
# Running Dashboard
if __name__ == '__main__':
    app.run_server(debug=True)
