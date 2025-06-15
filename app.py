import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

# Initialise l'application
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SLATE])
app.title = "Recensement Dakar"
server = app.server

# Mise en page générale
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="Dashboard Recensement - Dakar",
        color="primary",
        dark=True,
        brand_href="/",
        children=[
            dbc.NavItem(dcc.Link("Accueil", href="/", className="nav-link")),
            dbc.NavItem(dcc.Link("Répartition Démographique", href="/repartition", className="nav-link")),
            dbc.NavItem(dcc.Link("Ménages & Concessions", href="/menages", className="nav-link")),
            dbc.NavItem(dcc.Link("Comparaison Communes", href="/comparaison", className="nav-link")),
            dbc.NavItem(dcc.Link("Tendances & Projections", href="/tendances", className="nav-link")),
        ]
    ),
    dash.page_container
], fluid=True)

if __name__ == "__main__":
    app.run_(debug=True)
