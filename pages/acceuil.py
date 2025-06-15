import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path="/")

# Charger les données
df = pd.read_csv("data/data-recensement(dakar).csv")

# Nettoyage simplifié (à adapter selon structure réelle)
df["Commune"] = df["Commune"].astype(str)
df["Population"] = df["Population"].fillna(0).astype(int)
df["Ménages"] = df["Ménages"].fillna(0).astype(int)

# Résumés
total_pop = df["Population"].sum()
total_menages = df["Ménages"].sum()
n_communes = df["Commune"].nunique()

# Carte
fig_map = px.choropleth_mapbox(df, geojson=None, locations="Commune",
                                color="Population",
                                center={"lat": 14.6928, "lon": -17.4467},
                                mapbox_style="carto-darkmatter", zoom=10,
                                color_continuous_scale="Viridis",
                                title="Répartition de la population par commune")

layout = dbc.Container([
    html.H2("📍 Aperçu du Recensement - Dakar"),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("👥 Population Totale"),
            dbc.CardBody(html.H4(f"{total_pop:,}", className="card-title"))
        ], color="dark", inverse=True), width=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("🏘️ Ménages Totaux"),
            dbc.CardBody(html.H4(f"{total_menages:,}", className="card-title"))
        ], color="secondary", inverse=True), width=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("📍 Communes"),
            dbc.CardBody(html.H4(f"{n_communes}", className="card-title"))
        ], color="info", inverse=True), width=4),
    ]),
    html.Br(),
    dcc.Graph(figure=fig_map)
])
