import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json
from datetime import datetime
import os

# Initialisation de l'app Dash avec thème Bootstrap et styles personnalisés
app = dash.Dash(__name__, 
               external_stylesheets=[dbc.themes.LUX],
               meta_tags=[{'name': 'viewport',
                         'content': 'width=device-width, initial-scale=1.0'}])
server = app.server

# Chargement et préparation des données
def load_and_prepare_data():
    # Chargement des données (simulé - remplacer par vos fichiers réels)
    data = {
        'commune': ['Dakar', 'Pikine', 'Rufisque', 'Guediawaye', 'Keur Massar'] * 20,
        'arrondissement': ['Arrondissement ' + str(i) for i in range(1, 11)] * 10,
        'quartier': ['Quartier ' + str(i) for i in range(1, 101)],
        'concessions': np.random.randint(50, 500, 100),
        'menages': np.random.randint(100, 1000, 100),
        'hommes': np.random.randint(200, 2000, 100),
        'femmes': np.random.randint(200, 2000, 100),
        'population': np.random.randint(400, 4000, 100)
    }
    df = pd.DataFrame(data)
    
    # Calcul de ratios
    df['ratio_hf'] = df['hommes'] / (df['hommes'] + df['femmes'])
    df['menage_par_concession'] = df['menages'] / df['concessions']
    df['population_par_menage'] = df['population'] / df['menages']
    df['date_import'] = datetime.now().strftime("%Y-%m-%d")
    
    return df

df = load_and_prepare_data()

# Couleurs personnalisées
colors = {
    'background': '#f8f9fa',
    'text': '#343a40',
    'primary': '#2c3e50',
    'secondary': '#6c757d',
    'success': '#27ae60',
    'info': '#2980b9',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'light': '#ecf0f1',
    'dark': '#2c3e50'
}

# Création des options pour les dropdowns
commune_options = [{'label': commune, 'value': commune} for commune in sorted(df['commune'].unique())]
arrondissement_options = [{'label': arr, 'value': arr} for arr in sorted(df['arrondissement'].unique())]
quartier_options = [{'label': quartier, 'value': quartier} for quartier in sorted(df['quartier'].unique())]

# Layout du tableau de bord
app.layout = dbc.Container([
    # En-tête
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            html.Img(
                                src="https://upload.wikimedia.org/wikipedia/commons/f/fd/Flag_of_Senegal.svg",
                                style={'height': '60px', 'width': 'auto'},
                                className="mr-3"
                            ),
                            width="auto"
                        ),
                        dbc.Col([
                            html.H1("Tableau de Bord - Recensement Dakar", 
                                   className="mb-1",
                                   style={'color': colors['primary'], 'font-weight': 'bold'}),
                            html.P("Visualisation interactive des données démographiques", 
                                  className="text-muted mb-0")
                        ], align="center"),
                        dbc.Col(
                            html.Div(
                                id='last-update',
                                className="text-right small",
                                style={'color': colors['secondary']}
                            ),
                            width=3
                        )
                    ], align="center")
                ])
            ], className="border-0 shadow-sm", style={'border-top': f'5px solid {colors["primary"]}'})
        ], width=12)
    ], className="mb-4"),

    # Filtres
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    dbc.Tabs([
                        dbc.Tab(label="Filtres de Base", tab_id="basic-filters"),
                        dbc.Tab(label="Filtres Avancés", tab_id="advanced-filters"),
                    ], id="filter-tabs", active_tab="basic-filters")
                ]),
                dbc.CardBody([
                    dbc.Container(id="filter-tab-content", fluid=True)
                ])
            ], className="shadow-sm mb-4")
        ], width=12)
    ]),

    # KPI Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div(
                            html.I(className="fas fa-users fa-2x", 
                                 style={'color': 'white'}),
                            style={
                                'width': '50px', 
                                'height': '50px', 
                                'background-color': colors['primary'], 
                                'border-radius': '50%', 
                                'display': 'flex',
                                'align-items': 'center', 
                                'justify-content': 'center',
                                'margin-right': '15px'
                            }
                        ),
                        html.Div([
                            html.H5("Population Totale", className="card-title mb-1"),
                            html.H3(id='total-population', className="mb-0"),
                            html.Small("Variation depuis dernier recensement", 
                                     className="text-muted")
                        ])
                    ], className="d-flex align-items-center")
                ])
            ], className="h-100 border-0 shadow-sm")
        ], md=3, className="mb-3"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div(
                            html.I(className="fas fa-map-marked-alt fa-2x", 
                                 style={'color': 'white'}),
                            style={
                                'width': '50px', 
                                'height': '50px', 
                                'background-color': colors['success'], 
                                'border-radius': '50%', 
                                'display': 'flex',
                                'align-items': 'center', 
                                'justify-content': 'center',
                                'margin-right': '15px'
                            }
                        ),
                        html.Div([
                            html.H5("Nombre de Quartiers", className="card-title mb-1"),
                            html.H3(id='total-quartiers', className="mb-0"),
                            html.Small("Répartition par commune", 
                                     className="text-muted")
                        ])
                    ], className="d-flex align-items-center")
                ])
            ], className="h-100 border-0 shadow-sm")
        ], md=3, className="mb-3"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div(
                            html.I(className="fas fa-home fa-2x", 
                                 style={'color': 'white'}),
                            style={
                                'width': '50px', 
                                'height': '50px', 
                                'background-color': colors['info'], 
                                'border-radius': '50%', 
                                'display': 'flex',
                                'align-items': 'center', 
                                'justify-content': 'center',
                                'margin-right': '15px'
                            }
                        ),
                        html.Div([
                            html.H5("Moyenne Ménages", className="card-title mb-1"),
                            html.H3(id='avg-menage', className="mb-0"),
                            html.Small("Par concession", 
                                     className="text-muted")
                        ])
                    ], className="d-flex align-items-center")
                ])
            ], className="h-100 border-0 shadow-sm")
        ], md=3, className="mb-3"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div(
                            html.I(className="fas fa-venus-mars fa-2x", 
                                 style={'color': 'white'}),
                            style={
                                'width': '50px', 
                                'height': '50px', 
                                'background-color': colors['warning'], 
                                'border-radius': '50%', 
                                'display': 'flex',
                                'align-items': 'center', 
                                'justify-content': 'center',
                                'margin-right': '15px'
                            }
                        ),
                        html.Div([
                            html.H5("Ratio H/F", className="card-title mb-1"),
                            html.H3(id='gender-ratio', className="mb-0"),
                            html.Small("Équilibre démographique", 
                                     className="text-muted")
                        ])
                    ], className="d-flex align-items-center")
                ])
            ], className="h-100 border-0 shadow-sm")
        ], md=3, className="mb-3")
    ], className="mb-4"),

    # Première ligne de graphiques
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Span("Analyse en Composantes Principales", className="h5"),
                    dbc.Button(
                        html.I(className="fas fa-info-circle"),
                        id="pca-info-btn",
                        color="link",
                        className="float-right p-0"
                    )
                ]),
                dbc.CardBody([
                    dcc.Loading(
                        dcc.Graph(id='pca-plot'),
                        type="circle"
                    )
                ]),
                dbc.Tooltip(
                    "L'ACP réduit la dimensionalité des données pour visualiser les patterns",
                    target="pca-info-btn",
                    placement="left"
                )
            ], className="h-100 shadow-sm")
        ], lg=6, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Span("Répartition Spatiale", className="h5"),
                    dbc.DropdownMenu(
                        label="Options",
                        children=[
                            dbc.DropdownMenuItem("Population", id="map-option-population"),
                            dbc.DropdownMenuItem("Densité", id="map-option-density"),
                        ],
                        right=True,
                        className="float-right",
                        color="link"
                    )
                ]),
                dbc.CardBody([
                    dcc.Loading(
                        dcc.Graph(id='population-map'),
                        type="circle"
                    )
                ])
            ], className="h-100 shadow-sm")
        ], lg=6, className="mb-4")
    ]),

    # Deuxième ligne de graphiques
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H5("Pyramide des Âges", className="mb-0")
                ),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Tranche d'âge:"),
                            dcc.Dropdown(
                                options=[
                                    {'label': '0-14 ans', 'value': '0-14'},
                                    {'label': '15-64 ans', 'value': '15-64'},
                                    {'label': '65+ ans', 'value': '65+'}
                                ],
                                value='0-14',
                                id='age-group-dropdown',
                                clearable=False
                            )
                        ], md=6)
                    ], className="mb-3"),
                    dcc.Graph(id='population-pyramid')
                ])
            ], className="h-100 shadow-sm")
        ], lg=6, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H5("Relation Ménages/Concessions", className="mb-0")
                ),
                dbc.CardBody([
                    dcc.Graph(id='menage-concession-scatter')
                ]),
                dbc.CardFooter(
                    html.Small("Double-cliquez sur une légende pour isoler une commune", 
                             className="text-muted")
                )
            ], className="h-100 shadow-sm")
        ], lg=6, className="mb-4")
    ]),

    # Onglets d'analyse avancée
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H4("Analyse Avancée", className="mb-0")
                ),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab(
                            dbc.Container([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Variable à analyser:"),
                                        dcc.Dropdown(
                                            id='distribution-variable',
                                            options=[
                                                {'label': 'Population', 'value': 'population'},
                                                {'label': 'Ménages', 'value': 'menages'},
                                                {'label': 'Concessions', 'value': 'concessions'},
                                                {'label': 'Hommes', 'value': 'hommes'},
                                                {'label': 'Femmes', 'value': 'femmes'},
                                                {'label': 'Ratio H/F', 'value': 'ratio_hf'},
                                                {'label': 'Ménage par concession', 'value': 'menage_par_concession'},
                                                {'label': 'Population par ménage', 'value': 'population_par_menage'}
                                            ],
                                            value='population',
                                            clearable=False
                                        )
                                    ], md=6),
                                    dbc.Col([
                                        dbc.Label("Type de visualisation:"),
                                        dbc.RadioItems(
                                            options=[
                                                {"label": "Histogramme", "value": "hist"},
                                                {"label": "Boxplot", "value": "box"},
                                            ],
                                            value="hist",
                                            id="dist-plot-type",
                                            inline=True
                                        )
                                    ], md=6)
                                ], className="mb-3"),
                                dcc.Graph(id='distribution-plot')
                            ], fluid=True),
                            label="Distribution"
                        ),
                        
                        dbc.Tab(
                            dcc.Graph(id='correlation-heatmap'),
                            label="Corrélations"
                        ),
                        
                        dbc.Tab(
                            dcc.Graph(id='top-quartiers'),
                            label="Top Quartiers"
                        ),
                        
                        dbc.Tab(
                            dcc.Graph(id='cluster-analysis'),
                            label="Analyse par Cluster"
                        )
                    ])
                ])
            ], className="shadow-sm")
        ], width=12, className="mb-4")
    ]),

    # Pied de page
    dbc.Row([
        dbc.Col([
            html.Footer([
                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.Img(
                                src="https://upload.wikimedia.org/wikipedia/commons/f/fd/Flag_of_Senegal.svg",
                                height="40px",
                                className="mr-2"
                            ),
                            html.Span("Ministère de l'Urbanisme et de l'Habitat", 
                                    className="align-middle")
                        ], md=4, className="d-flex align-items-center"),
                        
                        dbc.Col([
                            html.Div([
                                html.Small("© 2023 Gouvernement du Sénégal", className="d-block"),
                                html.Small("Données mises à jour le: ", id='data-update')
                            ], className="text-center")
                        ], md=4),
                        
                        dbc.Col([
                            html.Div([
                                dbc.Button(
                                    html.I(className="fas fa-question-circle mr-1"),
                                    " Aide",
                                    id="help-button",
                                    color="link",
                                    className="float-right"
                                )
                            ])
                        ], md=4)
                    ])
                ], fluid=True)
            ], className="mt-4 p-3 bg-light rounded")
        ], width=12)
    ]),

    # Modal d'aide
    dbc.Modal([
        dbc.ModalHeader("Guide d'Utilisation"),
        dbc.ModalBody([
            html.H5("Navigation dans le tableau de bord:"),
            html.Ul([
                html.Li("Utilisez les onglets pour basculer entre différentes vues"),
                html.Li("Cliquez sur les icônes d'information pour obtenir des explications"),
                html.Li("Les filtres en haut contrôlent toutes les visualisations")
            ]),
            html.Hr(),
            html.H5("Interactivité:"),
            html.Ul([
                html.Li("Survolez les graphiques pour voir les détails"),
                html.Li("Cliquez et faites glisser pour zoomer"),
                html.Li("Double-cliquez pour réinitialiser la vue")
            ]),
            html.Hr(),
            html.H5("Export de données:"),
            html.Ul([
                html.Li("Cliquez sur le bouton de téléchargement dans les graphiques"),
                html.Li("Sélectionnez le format souhaité (PNG, JPEG, SVG)")
            ])
        ]),
        dbc.ModalFooter(
            dbc.Button("Fermer", id="close-help", className="ml-auto")
        )
    ], id="help-modal", size="lg", scrollable=True)
], fluid=True, style={'backgroundColor': '#f5f7fa'})

# Callbacks pour les filtres
@app.callback(
    Output("filter-tab-content", "children"),
    [Input("filter-tabs", "active_tab")]
)
def render_filter_tab(active_tab):
    if active_tab == "basic-filters":
        return dbc.Row([
            dbc.Col([
                dbc.Label("Communes:"),
                dcc.Dropdown(
                    id='commune-filter',
                    options=commune_options,
                    multi=True,
                    placeholder="Toutes les communes"
                )
            ], md=4),
            
            dbc.Col([
                dbc.Label("Arrondissements:"),
                dcc.Dropdown(
                    id='arrondissement-filter',
                    options=arrondissement_options,
                    multi=True,
                    placeholder="Tous les arrondissements"
                )
            ], md=4),
            
            dbc.Col([
                dbc.Label("Plage de Population:"),
                dcc.RangeSlider(
                    id='population-slider',
                    min=df['population'].min(),
                    max=df['population'].max(),
                    value=[df['population'].min(), df['population'].max()],
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], md=4)
        ])
    else:
        return dbc.Row([
            dbc.Col([
                dbc.Label("Ratio H/F:"),
                dcc.RangeSlider(
                    id='gender-ratio-slider',
                    min=0, max=1, step=0.01,
                    value=[0.3, 0.7],
                    marks={i/10: f"{i*10}%" for i in range(0, 11)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], md=6),
            
            dbc.Col([
                dbc.Label("Nombre de Clusters:"),
                dcc.Slider(
                    id='cluster-slider',
                    min=2, max=6, step=1,
                    value=3,
                    marks={i: str(i) for i in range(2, 7)}
                )
            ], md=6)
        ])

# Callback pour les KPI
@app.callback(
    [Output('total-population', 'children'),
     Output('total-quartiers', 'children'),
     Output('avg-menage', 'children'),
     Output('gender-ratio', 'children'),
     Output('last-update', 'children'),
     Output('data-update', 'children')],
    [Input('commune-filter', 'value'),
     Input('arrondissement-filter', 'value'),
     Input('population-slider', 'value'),
     Input('gender-ratio-slider', 'value')]
)
def update_kpis(selected_communes, selected_arrondissements, population_range, gender_ratio):
    filtered_df = df.copy()
    
    # Application des filtres
    if selected_communes:
        filtered_df = filtered_df[filtered_df['commune'].isin(selected_communes)]
    
    if selected_arrondissements:
        filtered_df = filtered_df[filtered_df['arrondissement'].isin(selected_arrondissements)]
    
    filtered_df = filtered_df[
        (filtered_df['population'] >= population_range[0]) & 
        (filtered_df['population'] <= population_range[1])
    ]
    
    filtered_df = filtered_df[
        (filtered_df['ratio_hf'] >= gender_ratio[0]) & 
        (filtered_df['ratio_hf'] <= gender_ratio[1])
    ]
    
    # Calcul des KPI
    total_pop = filtered_df['population'].sum()
    total_quartiers = filtered_df['quartier'].nunique()
    avg_menage = filtered_df['menage_par_concession'].mean()
    gender_ratio_val = filtered_df['hommes'].sum() / (filtered_df['hommes'].sum() + filtered_df['femmes'].sum())
    
    # Formatage des résultats
    last_update = f"Dernière actualisation: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    data_update = f"Données mises à jour le: {filtered_df['date_import'].iloc[0]}"
    
    return (
        f"{total_pop:,.0f}",
        f"{total_quartiers}",
        f"{avg_menage:,.1f}",
        f"{gender_ratio_val:.2%}",
        last_update,
        data_update
    )

# Callback pour l'ACP
@app.callback(
    Output('pca-plot', 'figure'),
    [Input('commune-filter', 'value'),
     Input('arrondissement-filter', 'value'),
     Input('population-slider', 'value'),
     Input('gender-ratio-slider', 'value'),
     Input('cluster-slider', 'value')]
)
def update_pca(selected_communes, selected_arrondissements, population_range, gender_ratio, n_clusters):
    filtered_df = df.copy()
    
    # Application des filtres
    if selected_communes:
        filtered_df = filtered_df[filtered_df['commune'].isin(selected_communes)]
    
    if selected_arrondissements:
        filtered_df = filtered_df[filtered_df['arrondissement'].isin(selected_arrondissements)]
    
    filtered_df = filtered_df[
        (filtered_df['population'] >= population_range[0]) & 
        (filtered_df['population'] <= population_range[1])
    ]
    
    filtered_df = filtered_df[
        (filtered_df['ratio_hf'] >= gender_ratio[0]) & 
        (filtered_df['ratio_hf'] <= gender_ratio[1])
    ]
    
    # Préparation des données pour ACP
    numeric_cols = ['concessions', 'menages', 'hommes', 'femmes', 'population']
    X = filtered_df[numeric_cols].dropna()
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ACP
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Création du DataFrame pour visualisation
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['commune'] = filtered_df['commune'].values[:len(pca_df)]
    pca_df['quartier'] = filtered_df['quartier'].values[:len(pca_df)]
    pca_df['population'] = filtered_df['population'].values[:len(pca_df)]
    pca_df['cluster'] = clusters.astype(str)
    
    # Variance expliquée
    explained_var = pca.explained_variance_ratio_
    
    # Visualisation
    fig = px.scatter(
        pca_df, x='PC1', y='PC2', 
        color='cluster',
        symbol='commune',
        size='population',
        hover_data=['quartier', 'population'],
        title=f"ACP avec Clustering (K={n_clusters}) - Variance expliquée: PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%}",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_layout(
        xaxis_title="Première Composante Principale",
        yaxis_title="Deuxième Composante Principale",
        legend_title="Cluster/Commune",
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest'
    )
    
    return fig

# Callback pour la carte
@app.callback(
    Output('population-map', 'figure'),
    [Input('commune-filter', 'value'),
     Input('arrondissement-filter', 'value')]
)
def update_map(selected_communes, selected_arrondissements):
    filtered_df = df.copy()
    
    if selected_communes:
        filtered_df = filtered_df[filtered_df['commune'].isin(selected_communes)]
    
    if selected_arrondissements:
        filtered_df = filtered_df[filtered_df['arrondissement'].isin(selected_arrondissements)]
    
    # Agrégation par commune
    commune_data = filtered_df.groupby(['commune', 'arrondissement']).agg({
        'population': 'sum',
        'hommes': 'sum',
        'femmes': 'sum',
        'menages': 'sum',
        'concessions': 'sum'
    }).reset_index()
    
    # Carte choroplèthe simulée
    fig = px.choropleth(
        commune_data,
        locations='commune',
        color='population',
        hover_name='commune',
        hover_data=['arrondissement', 'hommes', 'femmes', 'menages'],
        title="Répartition de la Population par Commune",
        color_continuous_scale="Viridis",
        scope="africa",
        center={"lat": 14.7167, "lon": -17.4677}  # Coordonnées de Dakar
    )
    
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_scale=15
    )
    
    fig.update_layout(
        margin={"r":0,"t":30,"l":0,"b":0},
        geo=dict(bgcolor='rgba(0,0,0,0)')
    )
    
    return fig

# Callback pour la pyramide des âges
@app.callback(
    Output('population-pyramid', 'figure'),
    [Input('commune-filter', 'value'),
     Input('arrondissement-filter', 'value'),
     Input('age-group-dropdown', 'value')]
)
def update_pyramid(selected_communes, selected_arrondissements, age_group):
    filtered_df = df.copy()
    
    if selected_communes:
        filtered_df = filtered_df[filtered_df['commune'].isin(selected_communes)]
    
    if selected_arrondissements:
        filtered_df = filtered_df[filtered_df['arrondissement'].isin(selected_arrondissements)]
    
    # Agrégation des données
    agg_data = filtered_df.groupby('commune').agg({
        'hommes': 'sum',
        'femmes': 'sum'
    }).reset_index()
    
    # Création de la pyramide des âges
    fig = go.Figure()
    
    # Barres pour les hommes (à gauche)
    fig.add_trace(go.Bar(
        y=agg_data['commune'],
        x=-agg_data['hommes'],
        name='Hommes',
        orientation='h',
        marker_color=colors['primary'],
        hoverinfo='x'
    ))
    
    # Barres pour les femmes (à droite)
    fig.add_trace(go.Bar(
        y=agg_data['commune'],
        x=agg_data['femmes'],
        name='Femmes',
        orientation='h',
        marker_color=colors['danger'],
        hoverinfo='x'
    ))
    
    fig.update_layout(
        title="Pyramide de Population par Commune",
        barmode='relative',
        xaxis_title="Population",
        yaxis_title="Commune",
        legend_title="Genre",
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            tickvals=[-max(agg_data['hommes'].max(), agg_data['femmes'].max()), 
                      0, 
                      max(agg_data['hommes'].max(), agg_data['femmes'].max())],
            ticktext=[str(max(agg_data['hommes'].max(), agg_data['femmes'].max())), 
                      '0', 
                      str(max(agg_data['hommes'].max(), agg_data['femmes'].max()))]
        )
    )
    
    return fig

# Callback pour le scatter plot
@app.callback(
    Output('menage-concession-scatter', 'figure'),
    [Input('commune-filter', 'value'),
     Input('arrondissement-filter', 'value')]
)
def update_scatter(selected_communes, selected_arrondissements):
    filtered_df = df.copy()
    
    if selected_communes:
        filtered_df = filtered_df[filtered_df['commune'].isin(selected_communes)]
    
    if selected_arrondissements:
        filtered_df = filtered_df[filtered_df['arrondissement'].isin(selected_arrondissements)]
    
    fig = px.scatter(
        filtered_df,
        x='concessions',
        y='menages',
        color='commune',
        size='population',
        hover_name='quartier',
        title="Relation entre Ménages et Concessions",
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_layout(
        xaxis_title="Nombre de Concessions",
        yaxis_title="Nombre de Ménages",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend_title="Commune"
    )
    
    return fig

# Callback pour la distribution
@app.callback(
    Output('distribution-plot', 'figure'),
    [Input('distribution-variable', 'value'),
     Input('dist-plot-type', 'value'),
     Input('commune-filter', 'value'),
     Input('arrondissement-filter', 'value')]
)
def update_distribution(variable, plot_type, selected_communes, selected_arrondissements):
    filtered_df = df.copy()
    
    if selected_communes:
        filtered_df = filtered_df[filtered_df['commune'].isin(selected_communes)]
    
    if selected_arrondissements:
        filtered_df = filtered_df[filtered_df['arrondissement'].isin(selected_arrondissements)]
    
    if plot_type == "hist":
        fig = px.histogram(
            filtered_df, x=variable, color='commune',
            marginal="box", nbins=30,
            title=f"Distribution de {variable}",
            color_discrete_sequence=px.colors.qualitative.Plotly,
            hover_data=['quartier', 'arrondissement']
        )
    else:
        fig = px.box(
            filtered_df, x='commune', y=variable,
            color='commune', title=f"Distribution de {variable} par commune",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
    
    fig.update_layout(
        xaxis_title=variable.capitalize(),
        yaxis_title="Fréquence",
        legend_title="Commune",
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.1
    )
    
    return fig

# Callback pour la heatmap de corrélation
@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('commune-filter', 'value'),
     Input('arrondissement-filter', 'value')]
)
def update_heatmap(selected_communes, selected_arrondissements):
    filtered_df = df.copy()
    
    if selected_communes:
        filtered_df = filtered_df[filtered_df['commune'].isin(selected_communes)]
    
    if selected_arrondissements:
        filtered_df = filtered_df[filtered_df['arrondissement'].isin(selected_arrondissements)]
    
    numeric_cols = ['concessions', 'menages', 'hommes', 'femmes', 'population', 
                   'ratio_hf', 'menage_par_concession', 'population_par_menage']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Matrice de Corrélation entre Variables",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        labels=dict(color="Corrélation")
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_side="top",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

# Callback pour le top des quartiers
@app.callback(
    Output('top-quartiers', 'figure'),
    [Input('commune-filter', 'value'),
     Input('arrondissement-filter', 'value')]
)
def update_top_quartiers(selected_communes, selected_arrondissements):
    filtered_df = df.copy()
    
    if selected_communes:
        filtered_df = filtered_df[filtered_df['commune'].isin(selected_communes)]
    
    if selected_arrondissements:
        filtered_df = filtered_df[filtered_df['arrondissement'].isin(selected_arrondissements)]
    
    # Top 20 des quartiers par population
    top_quartiers = filtered_df.sort_values('population', ascending=False).head(20)
    
    fig = px.bar(
        top_quartiers,
        x='population',
        y='quartier',
        color='commune',
        orientation='h',
        title="Top 20 des Quartiers par Population",
        hover_data=['arrondissement', 'menages', 'concessions'],
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_title="Population",
        yaxis_title="Quartier",
        legend_title="Commune",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

# Callback pour l'analyse par cluster
@app.callback(
    Output('cluster-analysis', 'figure'),
    [Input('commune-filter', 'value'),
     Input('arrondissement-filter', 'value'),
     Input('cluster-slider', 'value')]
)
def update_cluster_analysis(selected_communes, selected_arrondissements, n_clusters):
    filtered_df = df.copy()
    
    if selected_communes:
        filtered_df = filtered_df[filtered_df['commune'].isin(selected_communes)]
    
    if selected_arrondissements:
        filtered_df = filtered_df[filtered_df['arrondissement'].isin(selected_arrondissements)]
    
    # Préparation des données
    numeric_cols = ['concessions', 'menages', 'hommes', 'femmes', 'population']
    X = filtered_df[numeric_cols].dropna()
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Ajout des clusters aux données
    clustered_df = filtered_df.copy().dropna(subset=numeric_cols)
    clustered_df['cluster'] = clusters.astype(str)
    
    # Analyse des clusters
    cluster_profile = clustered_df.groupby('cluster')[numeric_cols].mean().reset_index()
    
    # Visualisation radar
    fig = go.Figure()
    
    for i in range(n_clusters):
        fig.add_trace(go.Scatterpolar(
            r=cluster_profile.iloc[i][numeric_cols].values,
            theta=numeric_cols,
            fill='toself',
            name=f'Cluster {i}',
            line_color=px.colors.qualitative.Plotly[i]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, cluster_profile[numeric_cols].values.max()]
            )
        ),
        title=f"Profil des Clusters (K={n_clusters})",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend_title="Cluster"
    )
    
    return fig

# Callback pour la modal d'aide
@app.callback(
    Output("help-modal", "is_open"),
    [Input("help-button", "n_clicks"), 
     Input("close-help", "n_clicks")],
    [State("help-modal", "is_open")],
)
def toggle_help(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Exécution de l'application
if __name__ == '__main__':
    app.run(debug=True)
