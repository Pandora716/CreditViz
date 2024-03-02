import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, kelbow_visualizer
from modified_dendrogram import create_dendrogram

@st.cache_data
def heatmap(df):
    fig = px.imshow(round(df.corr(), 2), color_continuous_scale='RdBu_r', text_auto=True)
    fig.update_layout(title=dict(text="Correlation Heatmap", font=dict(size=24), x=0.55, xanchor='center'),
                      margin={"r": 0, "t": 100, "l": 0, "b": 0}, coloraxis_showscale=False, width=1400, height=900)
    return fig

@st.cache_data
def variables(df):
    fig = make_subplots(rows=17, cols=1, subplot_titles=(df.columns))
    for i, j in enumerate(df.columns):
        fig.add_trace(go.Histogram(x=df[j], marker_color='#9dcbe1', opacity=0.75), i + 1, 1)
        fig.update_yaxes(nticks=3, row=i + 1, col=1)
    fig.update_layout(
        title=dict(text="Variable Distribution", font=dict(size=24), x=0.5, xanchor='center'),
        margin={"r": 0, "t": 100, "l": 0, "b": 0}, showlegend=False, width=200, height=3000
    )
    return fig

@st.cache_data
def balance_limit_by_tenure(df):
    subtitles = [None]
    for i in df.TENURE.unique():
        subtitles.append(f"Tenure {i} - ({len(df[df.TENURE == i])})")
    fig = make_subplots(rows=7, cols=8, subplot_titles=subtitles,
                        horizontal_spacing=0.02,
                        vertical_spacing=0.03,
                        specs=[[{'rowspan': 7, 'colspan': 7}, None, None, None, None, None, None, {}],
                               [None, None, None, None, None, None, None, {}],
                               [None, None, None, None, None, None, None, {}],
                               [None, None, None, None, None, None, None, {}],
                               [None, None, None, None, None, None, None, {}],
                               [None, None, None, None, None, None, None, {}],
                               [None, None, None, None, None, None, None, {}]
                               ])
    marker_colors = px.colors.qualitative.T10
    for i, j in enumerate(df.TENURE.unique()):
        fig.add_trace(go.Scatter(x=df[df.TENURE == j].CREDIT_LIMIT, y=df[df.TENURE == j].BALANCE, mode='markers',
                                 marker=dict(color=marker_colors[i])), 1, 1)
        fig.update_xaxes(title='CREDIT_LIMIT',title_font=dict(size=20), row=1, col=1)
        fig.update_yaxes(title='BALANCE',title_font=dict(size=20), row=1, col=1)

        fig.add_trace(go.Scatter(x=df[df.TENURE != j].CREDIT_LIMIT, y=df[df.TENURE != j].BALANCE, mode='markers',
                                 marker=dict(size=2, color='#f5f5f5')), i + 1, 8)
        fig.add_trace(go.Scatter(x=df[df.TENURE == j].CREDIT_LIMIT, y=df[df.TENURE == j].BALANCE, mode='markers',
                                 marker=dict(size=2, color=marker_colors[i])), i + 1, 8)
        fig.update_xaxes(showticklabels=False, row=i + 1, col=8)
        fig.update_yaxes(showticklabels=False, row=i + 1, col=8)

    fig.update_annotations(font_size=14)
    fig.update_layout(title=dict(text="Credit Limit vs. Balance by Tenure", font=dict(size=24), x=0.5, xanchor='center'),
                      showlegend=False, height=700)
    return fig

@st.cache_data
def purchase(df):
    d1 = df[['PURCHASES', 'TENURE']]
    d1 = d1.groupby('TENURE').agg(MIN=('PURCHASES', 'min'), AVG=('PURCHASES', 'mean'),
                                  MAX=('PURCHASES', 'max')).reset_index()
    d2 = df[['PURCHASES_TRX', 'TENURE']]
    d2 = d2.groupby('TENURE').agg(MIN=('PURCHASES_TRX', 'min'), AVG=('PURCHASES_TRX', 'mean'),
                                  MAX=('PURCHASES_TRX', 'max')).reset_index()

    data1 = {"line_x": [], "line_y": [], "min": [], "max": [], "avg": []}
    data2 = {"line_x": [], "line_y": [], "min": [], "max": [], "avg": []}
    for i in d1.TENURE.unique():
        data1['min'].extend([d1.loc[d1.TENURE == i]['MIN'].values[0]])
        data1['max'].extend([d1.loc[d1.TENURE == i]['MAX'].values[0]])
        data1['avg'].extend([d1.loc[d1.TENURE == i]['AVG'].values[0]])
        data1['line_x'].extend([d1.loc[d1.TENURE == i]['MIN'].values[0], d1.loc[d1.TENURE == i]['MAX'].values[0], None])
        data1['line_y'].extend([i, i, None])

        data2['min'].extend([d2.loc[d2.TENURE == i]['MIN'].values[0]])
        data2['max'].extend([d2.loc[d2.TENURE == i]['MAX'].values[0]])
        data2['avg'].extend([d2.loc[d2.TENURE == i]['AVG'].values[0]])
        data2['line_x'].extend([d2.loc[d2.TENURE == i]['MIN'].values[0], d2.loc[d2.TENURE == i]['MAX'].values[0], None])
        data2['line_y'].extend([i, i, None])

    fig = make_subplots(rows=1, cols=2, subplot_titles=['Account Purchases Amount', 'Purchase Total Transactions'])
    fig.add_trace(go.Scatter(x=data1['line_x'], y=data1['line_y'], mode="lines", showlegend=False,
                             marker=dict(color="#cac9cd", line=dict(width=14))), 1, 1)
    fig.add_trace(go.Scatter(x=data1['min'], y=d1.TENURE.unique(), mode="markers+text", texttemplate='%{x}',
                             textfont=dict(color="#ffbb00"),
                             textposition="top center", showlegend=False, marker=dict(color="#ffbb00", size=18)), 1, 1)
    fig.add_trace(go.Scatter(x=data1['avg'], y=d1.TENURE.unique(), mode="markers+text", texttemplate='%{x}',
                             textfont=dict(color="#9a7ec8"),
                             textposition="bottom center", showlegend=False, marker=dict(color="#9a7ec8", size=18)), 1,
                  1)
    fig.add_trace(go.Scatter(x=data1['max'], y=d1.TENURE.unique(), mode="markers+text", texttemplate='%{x}',
                             textfont=dict(color="#6600a5"),
                             textposition="top center", showlegend=False, marker=dict(color="#6600a5", size=18)), 1, 1)
    fig.update_xaxes(title='PURCHASES', row=1, col=1)
    fig.update_yaxes(title='TENURE', row=1, col=1)

    fig.add_trace(go.Scatter(x=data2['line_x'], y=data2['line_y'], mode="lines", showlegend=False,
                             marker=dict(color="#cac9cd", line=dict(width=14))), 1, 2)
    fig.add_trace(go.Scatter(x=data2['min'], y=d2.TENURE.unique(), mode="markers+text", texttemplate='%{x}',
                             textfont=dict(color="#ffbb00"),
                             textposition="top center", showlegend=False, marker=dict(color="#ffbb00", size=18)), 1, 2)
    fig.add_trace(go.Scatter(x=data2['avg'], y=d2.TENURE.unique(), mode="markers+text", texttemplate='%{x}',
                             textfont=dict(color="#9a7ec8"),
                             textposition="bottom center", showlegend=False, marker=dict(color="#9a7ec8", size=18)), 1,
                  2)
    fig.add_trace(go.Scatter(x=data2['max'], y=d2.TENURE.unique(), mode="markers+text", texttemplate='%{x}',
                             textfont=dict(color="#6600a5"),
                             textposition="top center", showlegend=False, marker=dict(color="#6600a5", size=18)), 1, 2)
    fig.update_xaxes(title='PURCHASES_TRX', row=1, col=2)

    fig.update_layout(title=dict(text="Purchase Total Transactions", font=dict(size=24), x=0.5, xanchor='center'),
                      height=600)
    return fig

@st.cache_data
def installment_by_tenure(df):
    subtitles = [None]
    for i in df.TENURE.unique():
        subtitles.append(f"Tenure {i} - ({len(df[df.TENURE == i])})")
    fig = make_subplots(rows=8, cols=7, subplot_titles=subtitles,
                        horizontal_spacing=0.02,
                        vertical_spacing=0.13,
                        specs=[[{'rowspan': 6, 'colspan': 7}, None, None, None, None, None, None],
                               [None, None, None, None, None, None, None],
                               [None, None, None, None, None, None, None],
                               [None, None, None, None, None, None, None],
                               [None, None, None, None, None, None, None],
                               [None, None, None, None, None, None, None],
                               [{'rowspan': 2}, {'rowspan': 2}, {'rowspan': 2}, {'rowspan': 2}, {'rowspan': 2},
                                {'rowspan': 2}, {'rowspan': 2}],
                               [None, None, None, None, None, None, None]
                               ])
    marker_colors = px.colors.qualitative.T10
    for i, j in enumerate(df.TENURE.unique()):
        fig.add_trace(
            go.Scatter(x=df[df.TENURE == j].CREDIT_LIMIT, y=df[df.TENURE == j].INSTALLMENTS_PURCHASES, mode='markers',
                       marker=dict(color=marker_colors[i])), 1, 1)
        fig.update_xaxes(title='CREDIT_LIMIT', row=1, col=1)
        fig.update_yaxes(title='INSTALLMENTS_PURCHASES', row=1, col=1)

        fig.add_trace(
            go.Scatter(x=df[df.TENURE != j].CREDIT_LIMIT, y=df[df.TENURE != j].INSTALLMENTS_PURCHASES, mode='markers',
                       marker=dict(size=2, color='#f5f5f5')), 7, i + 1)
        fig.add_trace(
            go.Scatter(x=df[df.TENURE == j].CREDIT_LIMIT, y=df[df.TENURE == j].INSTALLMENTS_PURCHASES, mode='markers',
                       marker=dict(size=2, color=marker_colors[i])), 7, i + 1)
        fig.update_xaxes(showticklabels=False, row=7, col=i + 1)
        fig.update_yaxes(showticklabels=False, row=7, col=i + 1)

    fig.update_annotations(font_size=10)
    fig.update_layout(title=dict(text="Credit Limit vs. Installment Purchases by Tenure", font=dict(size=24), x=0.5,
                                 xanchor='center'),
                      margin={"r": 0, "t": 50, "l": 0, "b": 0}, showlegend=False, height=700)
    return fig

@st.cache_data
def kmeans_elbow(X):
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        specs=[[{"secondary_y": True}, {}]],
                        horizontal_spacing=0.15,
                        subplot_titles=['Distortion Score Elbow\n', 'Calinski-Harabasz Score Elbow\n'])

    elbow_score = KElbowVisualizer(KMeans(random_state=32, max_iter=500), k=(2, 10))
    elbow_score.fit(X)
    elbow_score_ch = KElbowVisualizer(KMeans(random_state=32, max_iter=500), k=(2, 10), metric='calinski_harabasz',
                                      timings=False)
    elbow_score_ch.fit(X)

    fig.add_trace(go.Scatter(x=elbow_score.k_values_, y=elbow_score.k_scores_, mode="markers+lines",
                             marker=dict(color="#ffcc00", size=7, symbol='diamond'), line_dash='dash',
                             showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=elbow_score.k_values_, y=elbow_score.k_timers_, mode="markers+lines",
                             marker=dict(color="#9a7ec8", size=7), showlegend=False), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=elbow_score_ch.k_values_, y=elbow_score_ch.k_scores_, mode="markers+lines",
                             marker=dict(color="#ffcc00", size=7, symbol='diamond'), line_dash='dash',
                             showlegend=False), 1, 2)

    fig.update_xaxes(title='K Values', row=1, col=1)
    fig.update_yaxes(title='Distortion Scores', row=1, col=1)
    fig.update_yaxes(title=dict(text='Fit Time (seconds)', font=dict(color="#9a7ec8")), showgrid=False,
                     tickfont=dict(color="#9a7ec8"), row=1, col=1, secondary_y=True)
    fig.add_vline(x=elbow_score.elbow_value_, line_color='#fafafa', line_dash="dash", row=1, col=1)
    fig.update_xaxes(title='K Values', row=1, col=2)
    fig.update_yaxes(title='Calinski-Harabasz Score', row=1, col=2)
    fig.add_vline(x=elbow_score_ch.elbow_value_, line_color='#fafafa', line_dash="dash", row=1, col=2)
    fig.update_layout(
        title=dict(text="Credit Card Customer Clustering using K-Means", font=dict(size=24), x=0.5, xanchor='center'),
        margin={"r": 0}, showlegend=False, height=400)
    return fig

@st.cache_data
def kmeans_dist(X):
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        specs=[[{"secondary_y": True}, {}]],
                        horizontal_spacing=0.05,
                        subplot_titles=['Silhouette Plots of Clusters\n', 'Scatter Plot Clusters Distributions\n'])

    kmeans = KMeans(n_clusters=4, random_state=32, max_iter=500)
    y_kmeans = kmeans.fit_predict(X)
    s_viz = SilhouetteVisualizer(kmeans, colors=px.colors.qualitative.T10)
    s_viz.fit(X)
    cluster_colors = px.colors.qualitative.T10
    kmeans_values = pd.DataFrame({"score": s_viz.silhouette_samples_, "cluster": y_kmeans}).sort_values(
        ['cluster', 'score']).reset_index()
    kmeans_dist = pd.concat([pd.DataFrame(X, columns=['x', 'y']), pd.DataFrame(y_kmeans, columns=["cluster"])], axis=1)
    centers = s_viz.cluster_centers_

    for label in range(4):
        df_imp1 = kmeans_values[kmeans_values['cluster'] == label]
        df_imp2 = kmeans_dist[kmeans_dist['cluster'] == label]
        fig.add_trace(go.Scatter(x=df_imp1.score, y=df_imp1.index, fill='tozerox', mode='none', orientation='h',
                                 fillcolor=cluster_colors[label],
                                 textposition='middle center', text=label, showlegend=False), 1, 1)
        fig.add_trace(go.Scatter(x=df_imp2.x, y=df_imp2.y, mode="markers",
                                 marker=dict(color=cluster_colors[label], line=dict(width=1, color='DarkSlateGrey')),
                                 name=f'Cluster {label}'), 1, 2)
    fig.add_trace(go.Scatter(x=s_viz.cluster_centers_[:, 0], y=s_viz.cluster_centers_[:, 1], mode='markers',
                             marker=dict(color='#2a3f5f', line=dict(width=1, color='DarkSlateGrey')), name='Centroids'),
                  1, 2)
    fig.update_xaxes(title="Coefficient Values", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.add_vline(x=s_viz.silhouette_score_, line_dash='dash', line_color='#fafafa', row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.add_annotation(x=0.52, y=4300, xref="x", yref="y", text="Average Silhouette Score", showarrow=False,
                       font=dict(size=18, color="#ffffff"), row=1, col=1)

    fig.update_layout(
        title=dict(text="Credit Card Customer Clustering using K-Means", font=dict(size=24), x=0.5, xanchor='center'),
        legend=dict(orientation='h', xanchor="right", x=0.94, yanchor="top", y=0.02),
        margin={"l": 30, "r": 0}, height=500)
    db_index, s_score, ch_index = evaluate_clustering(X, y_kmeans)
    return fig, db_index, s_score, ch_index

@st.cache_data
def evaluate_clustering(X, y):
    db_index = round(davies_bouldin_score(X, y), 3)
    s_score = round(silhouette_score(X, y), 3)
    ch_index = round(calinski_harabasz_score(X, y), 3)
    return db_index, s_score, ch_index

@st.cache_data
def epsilon(X):
    neighbors = NearestNeighbors(n_neighbors=2)
    nbrs = neighbors.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)

    fig = go.Figure(go.Scatter(y=distances[:, 1], mode='lines', marker=dict(color='#9a7ec8')))
    fig.update_xaxes(title='Total', range=[0, 10000])
    fig.update_yaxes(title="Oldpeak")
    fig.add_shape(type="rect", xref="x", yref="y", x0=8750, y0=-0.4, x1=9100, y1=1.58, line=dict(color="#ffcc00"), )
    fig.add_annotation(x=8750, y=1.58, xref="x", yref="y",
                       text='From the plot, the maximum curvature<br> of the curve is about 2, and thus<br>we picked our Eps as 2.',
                       bgcolor="#9a7ec8", showarrow=True, borderwidth=2, borderpad=6,
                       ax=-180, ay=-50)
    fig.update_layout(title=dict(text="DBSCAN Epsilon Value", font=dict(size=24),x=0.5, xanchor='center'),
                      margin={'l':200, 'r':150},)
    return fig

@st.cache_data
def dbscan_dist(X):
    fig = go.Figure()

    dbscan = DBSCAN(eps=2, min_samples=4)
    y_dbscan = dbscan.fit_predict(X)
    dbscan_dist = pd.concat([pd.DataFrame(X, columns=['x', 'y']), pd.DataFrame(y_dbscan, columns=["cluster"])], axis=1)
    cluster_colors = ['#9a7ec8', '#ffcc00']

    for label in range(2):
        df_imp = dbscan_dist[dbscan_dist['cluster'] == label]
        fig.add_trace(go.Scatter(x=df_imp.x, y=df_imp.y, mode="markers",
                                 marker=dict(color=cluster_colors[label], line=dict(width=1, color='DarkSlateGrey')),
                                 name=f'Cluster {label} - ({round(df_imp.shape[0] / dbscan_dist.shape[0] * 100, 2)}%)'))
    fig.add_trace(
        go.Scatter(x=dbscan_dist[dbscan_dist['cluster'] == -1].x, y=dbscan_dist[dbscan_dist['cluster'] == -1].y,
                   mode="markers",
                   marker=dict(color='#fafafa', line=dict(width=1, color='DarkSlateGrey')), showlegend=False))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.add_shape(type="rect", xref="x", yref="y", x0=11.5, y0=12.5, x1=12.5, y1=14, line=dict(color="#262626"))
    fig.add_annotation(x=12.5, y=13.5, xref="x", yref="y",
                       text='Outliers', font=dict(color='#262626'),
                       bgcolor="#ffcc00", showarrow=True, borderwidth=2, borderpad=6,
                       ax=40, ay=-50)
    fig.update_layout(
        title=dict(text="Credit Card Customer Clustering using DBSCAN", font=dict(size=24), x=0.5, xanchor='center'),
        margin={'l': 150, 'r': 150},
        legend=dict(orientation='h', xanchor="center", x=0.5, yanchor='top', y=0.0), height=500)

    db_dbscan, ss_dbscan, ch_dbscan = evaluate_clustering(X, y_dbscan)

    return fig, db_dbscan, ss_dbscan, ch_dbscan

@st.cache_data
def hierarchy_elbow(X):
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        subplot_titles=['Dendrogram', 'Calinski-Harabasz Score Elbow'])

    elbow_score_ch = KElbowVisualizer(AgglomerativeClustering(), metric='calinski_harabasz', timings=False)
    elbow_score_ch.fit(X)

    dent = create_dendrogram(X, colorscale=px.colors.qualitative.T10, truncate_mode="level", p=7)
    for i in dent['data']:
        fig.add_trace(i, 1, 1)
    fig.add_trace(go.Scatter(x=elbow_score_ch.k_values_, y=elbow_score_ch.k_scores_, mode="markers+lines",
                             marker=dict(color="#9a7ec8", size=7, symbol='diamond'), line_dash='dash'), 1, 2)

    fig.update_yaxes(title='Euclidean Distances\n', row=1, col=1)
    fig.update_yaxes(title='Calinski-Harabasz Score\n', row=1, col=2)
    fig.add_hline(y=20, line_color='#fafafa', annotation_text="Horizontal Cut Line",
                  annotation_position="top right", annotation_font_color="#fafafa", row=1, col=1)
    fig.add_vline(x=elbow_score_ch.elbow_value_, line_color='#fafafa', line_dash="dash", row=1, col=2)

    fig.update_layout(
        title=dict(text="Credit Card Customer Clustering using Hierarchical Clustering", font=dict(size=24), x=0.5,
                   xanchor='center'),
        height=500, showlegend=False)
    return fig

@st.cache_data
def hierarchy_dist(X):
    fig = go.Figure()

    agg_cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    y_agg_cluster = agg_cluster.fit_predict(X)
    agg_cluster_dist = pd.concat(
        [pd.DataFrame(X, columns=['x', 'y']), pd.DataFrame(y_agg_cluster, columns=["cluster"])], axis=1)
    cluster_colors = px.colors.qualitative.T10

    for label in range(len(agg_cluster_dist.cluster.unique())):
        df_imp = agg_cluster_dist[agg_cluster_dist['cluster'] == label]
        fig.add_trace(go.Scatter(x=df_imp.x, y=df_imp.y, mode="markers",
                                 marker=dict(color=cluster_colors[label], line=dict(width=1, color='DarkSlateGrey')),
                                 name=f'Cluster {label} - ({round(df_imp.shape[0] / agg_cluster_dist.shape[0] * 100, 2)}%)'))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig.update_layout(
        title=dict(text="Credit Card Customer Clustering using Hierarchical Clustering", font=dict(size=24), x=0.5,xanchor='center'),
        margin={'l': 150, 'r': 150},
        legend=dict(orientation='h', xanchor="center", x=0.5, yanchor='top', y=0.0), height=500)

    db_agg, ss_agg, ch_agg = evaluate_clustering(X, y_agg_cluster)
    return fig, db_agg, ss_agg, ch_agg

@st.cache_data
def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]
@st.cache_data
def highlight_min(s):
    '''
    highlight the minimum in a Series blue.
    '''
    is_min = s == s.min()
    return ['background-color: blue' if v else '' for v in is_min]

@st.cache_data
def by_cluster(x, y, df_clustered):
    subtitles = [None]
    for i in ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']:
        subtitles.append(
            f"{i} - ({round(len(df_clustered[df_clustered.cluster_result == i]) / len(df_clustered) * 100, 2)}%)")
    fig = make_subplots(rows=4, cols=5, subplot_titles=subtitles,
                        horizontal_spacing=0.02,
                        vertical_spacing=0.08,
                        specs=[[{'rowspan': 4, 'colspan': 4}, None, None, None, {}],
                               [None, None, None, None, {}],
                               [None, None, None, None, {}],
                               [None, None, None, None, {}],
                               ])
    marker_colors = ['#e8bd42', '#8d7ee9', '#60b4d2', '#d8414a']
    for i, j in enumerate(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']):
        fig.add_trace(go.Scatter(x=df_clustered[df_clustered.cluster_result == j][x],
                                 y=df_clustered[df_clustered.cluster_result == j][y], mode='markers',
                                 marker=dict(color=marker_colors[i])), 1, 1)
        fig.update_xaxes(title=x, title_font=dict(size=20), row=1, col=1)
        fig.update_yaxes(title=y, title_font=dict(size=20), row=1, col=1)

        fig.add_trace(go.Scatter(x=df_clustered[df_clustered.cluster_result != j][x],
                                 y=df_clustered[df_clustered.cluster_result != j][y], mode='markers',
                                 marker=dict(size=2, color='#f5f5f5')), i + 1, 5)
        fig.add_trace(go.Scatter(x=df_clustered[df_clustered.cluster_result == j][x],
                                 y=df_clustered[df_clustered.cluster_result == j][y], mode='markers',
                                 marker=dict(size=2, color=marker_colors[i])), i + 1, 5)
        fig.update_xaxes(showticklabels=False, row=i + 1, col=5)
        fig.update_yaxes(showticklabels=False, row=i + 1, col=5)

    fig.update_annotations(font_size=14)
    if (x=='CREDIT_LIMIT') & (y=='BALANCE'):
        text = "Credit Limit vs. Balance by Cluster"
    elif (x=='CREDIT_LIMIT')&(y=='ONEOFF_PURCHASES'):
        text = 'One-off Purchase vs. Credit Limit by Cluster'
    elif (x=='CREDIT_LIMIT')&(y=='INSTALLMENTS_PURCHASES'):
        text = 'Installments Purchases vs. Credit Limit by Clusters'
    fig.update_layout(
        title=dict(text=text, font=dict(size=24), x=0.5, xanchor='center'),
        showlegend=False, height=700)
    return fig


    return fig

@st.cache_data
def payment_tenure_cluster(df_clustered):
    marker_colors = ['#e8bd42', '#8d7ee9', '#60b4d2', '#d8414a']
    fig = px.strip(df_clustered.sort_values('cluster_result'), x="TENURE", y="PAYMENTS", color='cluster_result',
                   color_discrete_sequence=marker_colors)
    fig.update_layout(title=dict(text="Tenure vs. Payments by Clusters", font=dict(size=24), x=0.5, xanchor='center'),
                      legend=dict(title='', font=dict(size=18), xanchor="left", x=0.02, yanchor='top', y=0.97), height=700)
    return fig