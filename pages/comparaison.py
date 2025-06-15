import dash
import pandas as pd
from dash import html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/comparaison")

df = pd.read_csv("data/data-recensement(dakar).csv")

communes = df["Commune"].unique()

layout = dbc.Container([
    html.H2("📌 Comparaison entre Communes"),
    dcc.Dropdown(communes, communes[0], id="commune-1", placeholder="Commune 1"),
    dcc.Dropdown(communes, communes[1], id="commune-2", placeholder="Commune 2"),
    dcc.Graph(id="radar"),
    dcc.Graph(id="violin")
])

@dash.callback(
    Output("radar", "figure"),
    Output("violin", "figure"),
    Input("commune-1", "value"),
    Input("commune-2", "value")
)
def update_charts(commune1, commune2):
    data1 = df[df["Commune"] == commune1].sum()
    data2 = df[df["Commune"] == commune2].sum()

    fig_radar = go.Figure()
    for name, data in zip([commune1, commune2], [data1, data2]):
        fig_radar.add_trace(go.Scatterpolar(
            r=[data["Population"], data["Ménages"]],
            theta=["Population", "Ménages"],
            fill='toself',
            name=name
        ))

    fig_radar.update_layout(title="Radar des indicateurs")

    fig_violin = px.violin(df, y="Ménages", x="Commune", box=True, points="all")
    return fig_radar, fig_violin
