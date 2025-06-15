import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/tendances")

layout = dbc.Container([
    html.H2("📈 Tendances et Projections (à venir)"),
    html.P("Si les données historiques sont disponibles, des séries temporelles et ACP y seront ajoutées.")
])
