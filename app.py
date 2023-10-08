from dash import Dash, html, dcc, Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import base64
import pickle
import logging
from analyser import AccountStatementAnalysis  # Ensure this import works
from readers import read_any

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the app
# Define the layout of the app
app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H1("Welcome to Finance Analyzer", className="animate__animated animate__fadeInUp animate__faster text-center"),
                width={"size": 6}), className="mb-4 justify-content-center"
    ),
    dbc.Row(
        dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H4('File Management',
                                className='card-title text-center'),
                        dcc.Upload(
                            id='upload-data',
                            children=dbc.Button(
                                'Upload File', color='primary', className='d-block mx-auto'),
                            multiple=False
                        ),
                        html.Div([  # Replace FormGroup with a Div
                                    dbc.Label(
                                        'Enter PDF Password (if applicable):'),
                                    dbc.Input(id='pdf-password', type='password',
                                              placeholder='Enter password'),
                        ], className='form-group text-center'),  # Add 'form-group' class name
                        dbc.Button('Submit', id='submit-button',
                                   color='success', className='d-block mx-auto'),  # Center the button
                        html.Div(id='output-data-upload'),
                        dcc.Loading(id="loading-0",
                                    type="circle",
                                    children=html.Div(id='analysis-storage',
                                                      style={'display': 'none'})
                                    ),
                    ])
                ),
                ], width={"size": 6, "offset": 3}, className="mb-4"),
    ),
    dbc.Row(
        dbc.Col(
            # Center content
            html.Div(id='eda', className='card text-center'),
            width={"size": 8}
        ), className="mb-4 justify-content-center"  # Center row
    ),
    dbc.Row(
        dbc.Col([
                html.Div([  # Replace FormGroup with a Div
                            dbc.Label("Select Qualitative Insights:"),
                            dcc.Checklist(id='checkbox-qual', inline=True),
                            dbc.Label("Select Quantitative Insights:"),
                            dcc.Checklist(id='checkbox-quant', inline=True),
                ], className='form-group'),  # Add 'form-group' class name
                dcc.Loading(
                    id="loading-1",
                    type="default",
                    children=html.Div(id='output-graph')
                ),
                ], width=12), className="mb-4"
    ),
], fluid=True)

# Serialization and deserialization utility functions


def serialize_object(obj):
    return base64.b64encode(pickle.dumps(obj)).decode()


def deserialize_object(encoded):
    return pickle.loads(base64.b64decode(encoded.encode()))

# Function to handle file upload and analysis


def handle_upload(contents, filename, password):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    file_type = filename.split('.')[-1]
    df = read_any(decoded, file_type, password)
    if df is None:
        return None

    analysis = AccountStatementAnalysis(df)
    # analysis.run()
    return analysis

# Callback for file upload


@app.callback(
    [Output('output-data-upload', 'children'),
     Output('analysis-storage', 'children'),
     Output('checkbox-qual', 'options'),  # New Output
     Output('checkbox-quant', 'options'),
     Output('eda', 'children')],  # New Output],
    [Input('upload-data', 'contents'),
     Input('submit-button', 'n_clicks')],
    [State('pdf-password', 'value'),
     State('upload-data', 'filename')]
)
def upload_and_store(contents, n_clicks, password, filename):
    if contents is None or n_clicks is None or n_clicks == 0:
        raise PreventUpdate

    analysis = handle_upload(contents, filename, password)
    if analysis is None:
        return "Invalid data uploaded.", None

    # Update checklist options
    qual_options = [{'label': k, 'value': k}
                    for k in analysis.qual_plots.keys()]
    quant_options = [{'label': k, 'value': k}
                     for k in analysis.quant_plots.keys()]

    return f"File {filename} uploaded and analyzed.", serialize_object(analysis), qual_options, quant_options, analysis.eda(returns=True)

# Callback to display plots based on checkbox selections


@app.callback(
    Output('output-graph', 'children'),
    [Input('checkbox-qual', 'value'),
     Input('checkbox-quant', 'value')],
    [State('analysis-storage', 'children')]
)
def display_selected_plots(selected_qual, selected_quant, stored_analysis):
    if stored_analysis is None:
        raise PreventUpdate

    analysis = deserialize_object(stored_analysis)

    selected_graphs = []
    if selected_qual:
        for key in selected_qual:
            selected_graphs.append(analysis.qual_plots.get(key))

    if selected_quant:
        for key in selected_quant:
            selected_graphs.append(analysis.quant_plots.get(key))

    return selected_graphs


if __name__ == '__main__':
    app.run_server(debug=True, port=7777)
