import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import viz_functions as vf

from sklearn.cluster import KMeans

st.set_page_config(
    page_title="CreditViz",
    page_icon='💳️',
    layout="wide",
)

st.title('Credit Card Customer Profile')

@st.cache_data
def load_data():
    data = pd.read_csv("CC GENERAL.csv")
    data = data.set_index("CUST_ID")
    data.TENURE = data.TENURE.astype('object')
    dd = pd.read_csv('Data Description.csv').iloc[:, :3]
    return data, dd
@st.cache_data
def clustering_process(df):
    # --- Dropping `CUST_ID` Column ---
    df = df.reset_index()
    # --- List Null Columns ---
    null_columns = df.drop(['CUST_ID'], axis=1).columns[df.drop(['CUST_ID'], axis=1).isnull().any()].tolist()

    # --- Perform Imputation ---
    imputer = KNNImputer()
    df_imp = pd.DataFrame(imputer.fit_transform(df[null_columns]), columns=null_columns)
    df = df.fillna(df_imp)

    # --- Scaling Dataset w/ Standard Scaler ---
    X = pd.DataFrame(StandardScaler().fit_transform(df.drop(['CUST_ID'], axis=1)))

    # --- Transform into Array ---
    X = np.asarray(X)

    # --- Applying PCA ---
    pca = PCA(n_components=2, random_state=24)
    X = pca.fit_transform(X)
    return X
# --- declare functions ---
# @st.cache_data
# def heatmap(df):
#     return vf.heatmap(df)
# @st.cache_data
# def variables(df):
#     return vf.variables(df)
# @st.cache_data
# def balance_limit_by_tenure(df):
#     return vf.balance_limit_by_tenure(df)
# @st.cache_data
# def purchase(df):
#     return vf.purchase(df)
# @st.cache_data
# def installment_by_tenure(df):
#     return installment_by_tenure(df)
# @st.cache_data
# def kmeans_elbow(X):
#     return vf.kmeans_elbow(X)
# @st.cache_data
# def kmeans_dist(X):
#     return vf.kmeans_dist(X)
# @st.cache_data
# def epsilon(X):
#     return vf.epsilon(X)
# @st.cache_data
# def dbscan_dist(X):
#     return vf.dbscan_dist(X)
# @st.cache_data
# def hierarchy_elbow(X):
#     return vf.hierarchy_elbow(X)
# @st.cache_data
# def hierarchy_dist(X):
#     return vf.hierarchy_dist(X)
# @st.cache_data
# def by_cluster(x, y, df_clustered):
#     return vf.by_cluster(x, y, df_clustered)
# @st.cache_data
# def payment_tenure_cluster(df_clustered):
#     return vf.payment_tenure_cluster(df_clustered)
# --- end ---

df, dd = load_data()
X = clustering_process(df)

st.divider()
st.markdown("The dataset is retrieved from: https://www.kaggle.com/datasets/arjunbhasin2013/ccdata")
st.markdown("The dataset that will be used contains the usage behavior of around 9000 credit card users for the last six months. "
            "It is required to group credit card customers into several groups according to customer behavior to get an effective and efficient credit card marketing strategy.")
describe, image = st.columns(2)
with describe:
    st.subheader("Objectives")
    st.markdown('This app aims to:\n'
                '- :orange[Perform dataset exploration] using various types of data visualization.\n'
                '- :orange[Perform data preprocessing] before using models.\n'
                '- :orange[Grouping customers into clusters] using various clustering models.\n'
                '- :orange[Perform interpretation and analysis of the groups (profiling)] that have been created.\n')
    st.subheader("Clustering Models")
    st.markdown("Three clustering models will be included:\n"
                "1. Partition based (K-Means)\n"
                "2. Density based (DBSCAN)\n"
                "3. Hierarchical Clustering (Agglomerative)")
image.image("https://ts1.cn.mm.bing.net/th/id/R-C.efbf9cc200e120cd8ef1dee30191893e?rik=7jko4hIFsbzvrw&riu=http%3a%2f%2fpaymaxxpro.com%2fwp-content%2fuploads%2f2017%2f04%2fcreditcards.jpg"
            "&ehk=rqrvy9kPsvrwozoRdbfEPtTXnjhOL%2bXaR2nCVfbyzb4%3d&risl=&pid=ImgRaw&r=0")
st.divider()

# --- Dataset Overview ---
st.header("Dataset Overview")
with st.expander("Raw data"):
    st.write(df)
with st.expander("Dataset Description"):
    st.dataframe(dd)

col1, col2 = st.columns([3, 2], gap='large')
with col1:
    with st.container(height=900, border=False):
        st.plotly_chart(vf.heatmap(df), theme="streamlit", use_container_width=True)

with col2:
    with st.container(height=800,border=False):
        st.plotly_chart(vf.variables(df), theme="streamlit", use_container_width=True)

st.markdown("- There are large amount of 0 value in most variables. "
            "They are right-skewed distribution as histogram shown.\n"    
            "- Some variables have high correlation: The correlation value between `ONEOFF_PURCHASES` and `PURCHASES` is 0.92; "
            "The correlation value between `CASH_ADVANCE_TRX` and `CASH_ADVANCE_FREQUENCY` is 0.8\n"
            "- Most of customers prefer credit cards which tenure is 12. In the long run, customers are more likely to repay their credit cards, thereby obtaining higher interest rates\n"
            "- In `BALANCE`, many credit card balances are 0; There are also many 0 values in `PURCHASES`. "
            "Some users may intentionally maintain low account balances to obtain high credit limits, which can affect credit utilization and credit score improvement"
            "- Most `BALANCE_FREQUENCY` values are 1, indicating that most customers frequently use credit cards; "
            "However, from the perspectives of `ONEOFF_PURCHASES` and `PURCHASES_INSTALLMENT_FREQUESCY`, most users do not use credit cards for one-time transactions or installment payments"
            )

st.divider()
# --- EDA ---
st.header("Exploratory Data Analysis")
eda1, eda2, eda3 = st.tabs(['Credit Limit vs. Balance by Tenure', 'Purchase Total Transactions', 'Credit Limit vs. Installment Purchases'])
with eda1:
    st.plotly_chart(vf.balance_limit_by_tenure(df), theme="streamlit", use_container_width=True)
    st.markdown("The scatter plot shows that as `CREDIT-LIMIT` increases, `BALANCE` also increases. Furthermore, it can be seen that most users prefer tenure of 12 months")
with eda2:
    st.plotly_chart(vf.purchase(df), theme="streamlit", use_container_width=True)
    st.markdown("- AS dumbbell chart shows, users with 12 tenure are relatively more willing to use their account to purchase the amount and generate more transaction volume. "
                "This also corresponds to mentioned above, **users are more likely to repay debts in this way to increase interest rates.** "
                "In addition, **some customers intentionally do not engage in transactions (zero consumption and transactions) to obtain high credit limits**, "
                "which **affects the improvement of credit scores and credit utilization rates**\n"
                "- Customers' `PURCHASES AMOUNT` with 10 tenure are more than those with 11 tenure, but the total `TRANSACTION' volume is the opposite; "
                "The purchase amount of customers whose tenure is 7 is lower than that of customers who are 8, but the total transaction volume is higher")
with eda3:
    st.plotly_chart(vf.installment_by_tenure(df), theme="streamlit", use_container_width=True)
    st.markdown("- The scatter plot shows a random distribution and does not show much correlation")

st.divider()
# --- Clustering ---
st.header("Clustering Models")
st.markdown("- In this section, ***Principal component analysis (PCA)*** was used to reduce number of features to 2 dimensions which is convenient to visualize the clustering results.\n"
            "- Before implementing each algorithm, elbow plot will be used to determine the best number of clusters.\n"
            "- After implementing the algorithm, use three metrics to ***evaluate clustering quality***: Davies-Bouldin index, silhouette score, and Calinski-Harabasz index.\n"
            "> - ***Davis-Bouldin Index*** is a metric for evaluating clustering algorithms. It is defined as a ratio between the cluster scatter and the cluster's separation. "
            "Scores range from 0 and up. ***0 indicates better clustering***\n"
            "> - ***Silhouette Coefficient/Score*** is a metric used to calculate the goodness of a clustering technique. Its value ranges from -1 to 1. ***The higher the score, the better.*** "
            "1 means clusters are well apart from each other and clearly distinguished. 0 means clusters are indifferent/the distance between clusters is not significant. "
            "-1 means clusters are assigned in the wrong way.\n"
            "> - ***Calinski-Harabasz Index*** (also known as the Variance Ratio Criterion), is the ratio of the sum of between-clusters dispersion and of inter-cluster dispersion for all clusters, "
            "***the higher the score, the better the performances.***")
model = st.selectbox(
    label='Select clustering model:',
    options=('K-Means', 'DBSCAN', 'Hierarchical Clustering(Agglomerative)'),
    placeholder="Select clustering model..."
)
elbow, clustering = st.tabs(['Elbow Plots', 'Clustering'])
with elbow:
    if model=="K-Means":
        st.plotly_chart(vf.kmeans_elbow(X), theme="streamlit", use_container_width=True)
    elif model=='DBSCAN':
        st.plotly_chart(vf.epsilon(X), theme="streamlit", use_container_width=True)
    elif model=='Hierarchical Clustering(Agglomerative)':
        st.plotly_chart(vf.hierarchy_elbow(X), theme="streamlit", use_container_width=True)
with clustering:
    if model=="K-Means":
        st.plotly_chart(vf.kmeans_dist(X)[0], theme="streamlit", use_container_width=True)
    elif model=='DBSCAN':
        st.plotly_chart(vf.dbscan_dist(X)[0], theme="streamlit", use_container_width=True)
    elif model=='Hierarchical Clustering(Agglomerative)':
        st.plotly_chart(vf.hierarchy_dist(X)[0], theme="streamlit", use_container_width=True)
    m1, m2, m3, m4, m5 = st.columns([3,1,1,1,3])
    if model=='K-Means':
        db_index, s_score, ch_index = vf.kmeans_dist(X)[1:]
    elif model=='DBSCAN':
        db_index, s_score, ch_index = vf.dbscan_dist(X)[1:]
    elif model=='Hierarchical Clustering(Agglomerative)':
        db_index, s_score, ch_index = vf.hierarchy_dist(X)[1:]
    m2.metric("Davies-Bouldin Index", db_index)
    m3.metric("Silhouette Score", s_score)
    m4.metric("Calinski Harabasz Index", ch_index)

st.subheader("Model Evaluation")

compare = pd.DataFrame()
compare['K-Means'] = vf.kmeans_dist(X)[1:]
compare['DBSCAN'] = vf.dbscan_dist(X)[1:]
compare['Hierarchical Clustering'] = vf.hierarchy_dist(X)[1:]
compare = compare.T.reset_index()
compare.columns = ['Model','Davies-Bouldin Index(-)','Silhouette Score(+)','Calinski-Harabasz Index(+)']

eva1, eva2 = st.columns([2, 3], gap='large')
eva1.dataframe(compare.style.apply(vf.highlight_max, subset=['Silhouette Score(+)','Calinski-Harabasz Index(+)'])
               .apply(vf.highlight_min, subset=['Davies-Bouldin Index(-)']), hide_index=True)
eva1.markdown("> **(+)** : higher is better;  **(-)** : lower is better")
eva2.markdown("- The dataframe shows that the DBI of K-Means is the lowest, and it can be concluded that the clustering quality of K-Means is higher than the other two algorithms. "
              "However, based on the silhouette score, K-Means is the second highest, indicating some overlap in the clustering\n"
              "- In addition, using hierarchical clustering has similar clustering quality to K-Means, with slightly higher DBI and slightly lower silhouette score. "
              "Finally, compared to others, the DBSCAN algorithm has the worst DBI but the best silhouette score.\n"
              "- From the Calinski-Harabasz-Index, it can be seen that K-Means has the highest score, indicating that K-Means performs the best and has a higher density\n"
              "- It can be concluded that due to the lowest DBI and slightly better clustering overlap than hierarchical clustering, ***K-Means has the best clustering quality*** among these three algorithms.\n")

st.divider()
# --- Conclusions ---
st.header("Conclusions")
st.subheader("Cluster Profile")
profile, profile_text = st.columns(2)
@st.cache_data
def profile_data(df):
    kmeans = KMeans(n_clusters=4, random_state=32, max_iter=500)
    y_kmeans = kmeans.fit_predict(X)
    df_clustered = df.copy(deep=True)
    df_clustered['cluster_result'] = y_kmeans + 1
    df_clustered['cluster_result'] = 'Cluster ' + df_clustered['cluster_result'].astype(str)

    df_profile_overall = pd.DataFrame()
    df_profile_overall['Overall'] = df_clustered.describe().loc[['mean']].T

    df_cluster_summary = df_clustered.groupby('cluster_result').describe().T.reset_index().rename(columns={'level_0': 'Column Name', 'level_1': 'Metrics'})
    df_cluster_summary = df_cluster_summary[df_cluster_summary['Metrics'] == 'mean'].set_index('Column Name')
    df_profile = df_cluster_summary.join(df_profile_overall)
    return df_profile, df_clustered
df_profile, df_clustered = profile_data(df)
profile.dataframe(df_profile.style.background_gradient(cmap='YlOrBr'), height=600)
profile_text.markdown("- :orange[Cluster 1 (Full Payers Users):] Customers in this cluster are :orange[active users of the bank's credit card]. "
                      "This can be seen from the frequency of the balance which frequently changes and the balances amount is high enough compared to other clusters. "
                      "In addition, when compared to other clusters, this cluster has higher mean value in several aspects than other clusters. "
                      "Credit card customers in this cluster also actively use credit cards to facilitate transactions and installments. "
                      "Cash advances, transactions, and installments in this cluster also occur more frequently. "
                      "The relatively high tenure also shows that the credit scoring in this cluster is very good.\n"
                      "- :violet[Cluster 2 (Starter/Student users):] In contrast to cluster 1, :violet[customers rarely/almost never use credit cards for transactions and installments] in this cluster. "
                      "This is because the customer has a relatively small balance, the frequency of the balance rarely changes, and the installments are very low. "
                      "In addition, a low credit limit also shows that customers rarely/almost never use credit cards to process credit transactions, "
                      "and customers in this cluster also rarely make cash advances. So, :violet[it can be assumed that customers use credit cards for cash advance processes only with sufficient frequency]. "
                      "In addition, the low balance allows customers in this cluster :violet[to be students or new users who use credit cards at this bank].\n"
                      "- :blue[Cluster 3 (Installment Users):] In this cluster, customers use credit cards :blue[specifically for installment purposes]. "
                      "This is due to the relatively high level of transactions using installments in this cluster. "
                      "Moreover, customers in this cluster often make transactions with very large amounts per transaction and the frequency and transactions of cash in advance are very small. "
                      "Customers in this cluster very rarely make payments and cash in advance and have a relatively small cash-in-advance frequency and amount of payments. "
                      "It can be concluded that the _blue[customers in this cluster are very suitable for credit cards specifically for installment needs].\n"
                      "- :red[Cluster 4 (Cash Advance/Withdraw Users):] Customers in this cluster have high balances, the balances frequency are always changing, "
                      "and the frequency of cash in advance and cash in advance is high. "
                      "In addition, customers in this cluster have the lowest interest rates compared to other clusters and have the second highest credit limit and payments out of the four clusters. "
                      "However, credit card users in this cluster rarely make installments or one-off purchases and have the third-highest tenure of the four clusters. "
                      "Thus, it can be concluded that :red[customers in this cluster only use credit cards for the need to withdraw money or cash advances].")

prof1, prof2, prof3, prof4 = st.tabs(["Credit Limit vs. Balance by Cluster", "One-off Purchase vs. Credit Limit by Cluster",
                                      "Tenure vs. Payments by Clusters", "Installments Purchases vs. Credit Limit by Clusters"])
with prof1:
    st.plotly_chart(vf.by_cluster('CREDIT_LIMIT', 'BALANCE', df_clustered), theme="streamlit", use_container_width=True)
    st.markdown("From the figure above, it can be seen that :rainbow[clusters 1 and 4 have the highest balance and credit limit]. "
                "In addition, it can be seen that :rainbow[the more the balance increases, the more credit limits the customer gets]. "
                "This can be seen clearly in clusters 1 and 4 because these clusters have customers who are quite active in using credit cards. "
                "However, this is different from clusters 2 and 3, where there is a slight correlation between the two variables because customers in this cluster have a fairly rare frequency of balance updates. "
                "In addition, it can be seen that :rainbow[the most zero balance card holders are in clusters 2 and 3].")
with prof2:
    st.plotly_chart(vf.by_cluster('CREDIT_LIMIT', 'ONEOFF_PURCHASES', df_clustered), theme="streamlit", use_container_width=True)
    st.markdown(":rainbow[One-off purchase does not affect the additional credit limit obtained by the user]. "
                "In the figure above and as mentioned earlier, it can be seen that cluster 1 has a customer with the largest purchase amount for one transaction.")
with prof3:
    st.plotly_chart(vf.payment_tenure_cluster(df_clustered), theme="streamlit", use_container_width=True)
    st.markdown(":rainbow[Most customers in clusters 2 and 3 have zero payments] compared to other clusters in each tenure. As mentioned previously, "
                "it can be seen that most customers tend to choose 12-month tenure.")
with prof4:
    st.plotly_chart(vf.by_cluster('CREDIT_LIMIT', 'INSTALLMENTS_PURCHASES', df_clustered), theme="streamlit", use_container_width=True)
    st.markdown(":rainbow[It can be seen that clusters 1 and 3 have more installment purchases than clusters 2 and 4]. "
                "However, it can also be seen that a large number of installment purchases are not correlated with the credit limit increase.")
st.divider()