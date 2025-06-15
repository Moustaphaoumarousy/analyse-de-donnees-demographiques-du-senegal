import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, dash_table

# --------- Configuration ---------
fichiers = {
    "Dakar": "data-recensement(dakar).csv",
    "Pikine": "data-recensement (pikine).csv",
    "Rufisque": "data-recensement(rufisque).csv",
    "Guediawaye": "data-recensement(guediawaye).csv",
    "Keur Massar": "data-recensement(keur massar).csv"
}

# --------- Fonction de chargement ---------
def charger_donnees(nom):
    df = pd.read_csv(fichiers[nom])
    df.dropna(how='all', axis=1, inplace=True)
    return df

# --------- Initialisation App ---------
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container([
    html.H2("Tableau de bord du recensement des zones urbaines", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Zone"),
            dcc.Dropdown(
                id="zone-dropdown",
                options=[{"label": z, "value": z} for z in fichiers],
                value="Keur Massar",  # ➤ Valeur par défaut
                placeholder="Sélectionner une zone"
            ),
        ], md=4),
        dbc.Col([
            html.Label("Commune"),
            dcc.Dropdown(id="commune-dropdown", placeholder="Sélectionner une commune")
        ], md=4),
        dbc.Col([
            html.Label("Quartier"),
            dcc.Dropdown(id="quartier-dropdown", placeholder="Sélectionner un quartier")
        ], md=4)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card([html.H5("Communes", className="card-title"), html.H2(id="kpi-communes")], body=True, color="primary", inverse=True), md=3),
        dbc.Col(dbc.Card([html.H5("Quartiers", className="card-title"), html.H2(id="kpi-quartiers")], body=True, color="info", inverse=True), md=3),
        dbc.Col(dbc.Card([html.H5("Concessions", className="card-title"), html.H2(id="kpi-concessions")], body=True, color="success", inverse=True), md=3)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="bar-quartiers"), md=6),
        dbc.Col(dcc.Graph(id="bar-concessions"), md=6)
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id="bar-population"), md=6),
        dbc.Col(dcc.Graph(id="bar-sexe"), md=6)
    ]),

    html.Hr(),

    html.H4("Détails des données", className="mt-4"),
    dbc.Row([
        dbc.Col(dash_table.DataTable(id="table-donnees", page_size=10, style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left'}))
    ])
])

# --------- Callbacks dynamiques ---------
@app.callback(
    Output("commune-dropdown", "options"),
    Output("commune-dropdown", "value"),
    Input("zone-dropdown", "value")
)
def maj_communes(zone):
    if zone:
        df = charger_donnees(zone)
        communes = df["COMMUNE"].dropna().unique()
        options = [{"label": c, "value": c} for c in communes]
        return options, options[0]["value"] if options else None
    return [], None

@app.callback(
    Output("quartier-dropdown", "options"),
    Output("quartier-dropdown", "value"),
    Input("commune-dropdown", "value"),
    Input("zone-dropdown", "value")
)
def maj_quartiers(commune, zone):
    if zone and commune:
        df = charger_donnees(zone)
        quartiers = df[df["COMMUNE"] == commune]["QUARTIER_VILLAGE_HAMEAU"].dropna().unique()
        options = [{"label": q, "value": q} for q in quartiers]
        return options, options[0]["value"] if options else None
    return [], None

@app.callback(
    Output("kpi-communes", "children"),
    Output("kpi-quartiers", "children"),
    Output("kpi-concessions", "children"),
    Output("bar-quartiers", "figure"),
    Output("bar-concessions", "figure"),
    Output("bar-population", "figure"),
    Output("bar-sexe", "figure"),
    Output("table-donnees", "data"),
    Output("table-donnees", "columns"),
    Input("zone-dropdown", "value"),
    Input("commune-dropdown", "value"),
    Input("quartier-dropdown", "value")
)
def maj_dashboard(zone, commune, quartier):
    if not zone:
        return 0, 0, 0, px.bar(title="Quartiers"), px.bar(title="Concessions"), px.bar(title="Population"), px.pie(title="Sexe"), [], []

    df = charger_donnees(zone)

    # Filtrage dynamique
    if commune:
        df = df[df["COMMUNE"] == commune]
    if quartier:
        df = df[df["QUARTIER_VILLAGE_HAMEAU"] == quartier]

    nb_communes = df['COMMUNE'].nunique()
    nb_quartiers = df['QUARTIER_VILLAGE_HAMEAU'].nunique()
    nb_concessions = df['CONCESSION'].nunique() if 'CONCESSION' in df.columns else 'N/A'

    # Graphique quartiers
    grp_quartiers = df.groupby("COMMUNE")["QUARTIER_VILLAGE_HAMEAU"].nunique().reset_index()
    fig_q = px.bar(grp_quartiers, x="COMMUNE", y="QUARTIER_VILLAGE_HAMEAU", title="Nombre de quartiers par commune")

    # Graphique concessions
    if 'CONCESSION' in df.columns:
        grp_conc = df.groupby("COMMUNE")["CONCESSION"].nunique().reset_index()
        fig_c = px.bar(grp_conc, x="COMMUNE", y="CONCESSION", title="Nombre de concessions par commune")
    else:
        fig_c = px.bar(title="Pas de données sur les concessions")

    # Graphique population par commune
    if 'POPULATION' in df.columns:
        grp_pop = df.groupby("COMMUNE")["POPULATION"].sum().reset_index()
        fig_pop = px.bar(grp_pop, x="COMMUNE", y="POPULATION", title="Distribution de la population par commune")
    else:
        fig_pop = px.bar(title="Pas de données sur la population")

    # Graphique par sexe basé sur les colonnes HOMMES et FEMMES
    if 'HOMMES' in df.columns and 'FEMMES' in df.columns:
        total_HOMMES = df['HOMMES'].sum()
        total_FEMMES = df['FEMMES'].sum()
        grp_sexe = pd.DataFrame({
            'Sexe': ['HOMMES', 'FEMMES'],
            'Effectif': [total_HOMMES, total_FEMMES]
        })
        fig_sexe = px.pie(grp_sexe, names='Sexe', values='Effectif', title='Répartition par sexe')
    else:
        fig_sexe = px.pie(title="Pas de données sur le genre (HOMMES/FEMMES)")

    # Tableau de données
    table_data = df.to_dict("records")
    table_columns = [{"name": i, "id": i} for i in df.columns]

    return nb_communes, nb_quartiers, nb_concessions, fig_q, fig_c, fig_pop, fig_sexe, table_data, table_columns

# --------- Exécution ---------
if __name__ == '__main__':
    app.run(port=8000, debug=True)
