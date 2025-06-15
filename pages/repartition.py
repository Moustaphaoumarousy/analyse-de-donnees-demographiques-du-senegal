import dash
import pandas as pd
import plotly.express as px
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/repartition")

df = pd.read_csv("data/data-recensement(dakar).csv")

# Exemple de colonnes : Commune, Hommes, Femmes, Quartier, Population
fig_density = px.choropleth(df, locations="Commune", locationmode="geojson-id",
                            color="Population", title="Densité de population")

fig_gender = px.bar(df, x="Commune", y=["Hommes", "Femmes"], barmode="group",
                    title="Répartition Hommes / Femmes")

df_quartiers = df.groupby("Quartier").agg({"Population": "sum"}).reset_index()
fig_treemap = px.treemap(df_quartiers.sort_values("Population", ascending=False).head(20),
                         path=["Quartier"], values="Population",
                         title="Quartiers les plus peuplés")

layout = dbc.Container([
    html.H2("📊 Répartition Démographique"),
    dcc.Graph(figure=fig_density),
    dcc.Graph(figure=fig_gender),
    dcc.Graph(figure=fig_treemap)
])
