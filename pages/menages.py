import dash
import pandas as pd
from dash import dcc, html
import plotly.express as px
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/menages")

df = pd.read_csv("data/data-recensement(dakar).csv")

fig_hist = px.histogram(df, x="Commune", y="Ménages", title="Nombre de ménages par commune")

fig_scatter = px.scatter(df, x="Ménages", y="Population", color="Commune",
                         title="Corrélation entre Ménages et Population")

layout = dbc.Container([
    html.H2("🏠 Analyse des Ménages et Concessions"),
    dcc.Graph(figure=fig_hist),
    dcc.Graph(figure=fig_scatter),
])
