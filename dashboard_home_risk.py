# ------------ Libraries import ---------------------------
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import requests
import pickle
import shap
from PIL import Image

# ------------ Data Import --------------------------------
## Import passenger train data ##
df_test_sample = pd.read_csv(r'test_sample_data_home_risk.csv', index_col=0)
### last feature is a probability (will be used for visualization purpose) ###
features = df_test_sample.columns[: -1] 
loan_id = df_test_sample.index.sort_values()
treshold = 0.49
## import the illustration image ##
img = Image.open(r'logo_projet_fintech.png')
## loading the standard scaler to display normal values of client features ##
std_scaler = pickle.load(open(r'std_scaler_home_risk.pkl', 'rb'))
## loading shap tree explainer for our lgbm model (should be changed if our model is update!!) ##
shap_explainer = pickle.load(open(r'shap_tree_explainer_lgbm_model.pkl', 'rb'))
## create origin sample values ##
df_test_sample_origin = pd.DataFrame(std_scaler.inverse_transform(df_test_sample[features]), 
index = df_test_sample.index, columns=df_test_sample[features].columns)

# ------------ Function and class used in our dashboard -------------
## Class Object to use with shap waterfall plot ##
class ShapObject:
    
    def __init__(self, base_values, data, values, feature_names):
        ### Single value ###
        self.base_values = base_values
        ### Raw feature values for selected data (1 row of data) ###
        self.data = data
        ### SHAP values for the same row of data ###
        self.values = values
        ### features column name ###
        self.feature_names = feature_names

## Function to filter dataset with nearest neighbors in terms of probability result ##
def filter_near_customer(df, cust_id, n_near_cust, target):
    ''' Function to filter dataframe regarding the nearest neighbors of our customer in terms of probability.
    Note that the customer is included in the filtered DF
    --> df: dataframe with all customer data, must have an ID for customer credit request
    --> cust_id: Id of a customer request
    --> n_near_cust: number of nearest customer to the id request. It must be an even number!!!
    --> target: must be an str, name of the column contaigning the probability'''
    df_filter = df.sort_values(by=target, ascending=False).copy()
    ### getting true index value in the dataframe table ###
    index_cust = np.where(df_filter.index == cust_id)[0][0]
    ### Check if an enven number has been input, if not return non filtered dataset ###
    if n_near_cust%2 != 0:
        print('DataFrame has not been filtered just sorted, you have entered an odd number')
    else:
        ### calcul neighbours up and down our customer raw then balance if there is not enough up and down ###
        up_index = 0
        for t in range(1,(n_near_cust//2 + 1)):
            if len(df_filter.iloc[index_cust - t:index_cust, :]) == 0:
                break
        up_index = t
        down_index = 0
        for t in range(1,(n_near_cust//2 + 1)):
            if len(df_filter.iloc[index_cust:index_cust + t, :]) == 0:
                break
        down_index = t
        ### Balancing if there is not the same number up and down of the customer ###
        up_lift = n_near_cust//2 - down_index
        down_lift = n_near_cust//2 - up_index
        ### create filtered dataframe ###
        df_filter = df_filter.iloc[index_cust - (up_index - up_lift):index_cust + (down_index + (down_lift + 1)),:]
    return df_filter

## Function to fin in histogram in which bin is a value for visualization purpose
def bin_location(bins, value):
    '''Function to locate the bin were a single value is located in order to apply formatting to this specific bin
    bins --> list of bins return by plt.hist plot
    value --> the sepcific value to locate in a matplotlib histogramm
    it returns the index value in bins where value is located'''
    # set the index counter
    count = 0
    # playing for loop in bins list
    for b in bins:
        if value >= b:
            value_bin_ind = count
        count+=1
    return value_bin_ind

# ------------ Set base configuration for streamlit -------
st.set_page_config(layout="wide")

# ------------ Sidebar configuration ----------------------
## add side bar for user to interact ##

### Display the image with streamlit ###
st.sidebar.image(img)
### Add column for user input ###
st.sidebar.header('Sélectionner une demande de prêt:')
selected_credit = st.sidebar.selectbox('Prêt_ID', loan_id)
### Add checkbox for displaying different client informations ###
client_data = st.sidebar.checkbox('Données client')
client_pred_score = st.sidebar.checkbox('Résultat de la demande de prêt')
### Add checkbox for displaying score interpretation ###
score_interpret = st.sidebar.checkbox('Interprétations du score')
### Add checkbox for displaying client data analysis ###
client_analysis = st.sidebar.checkbox('Analyse des features client')


# ------------ Main display, part by part -----------------
## Generic title ##
st.write('# **SENECHAL Yannick: Projet 7 "Prêt à dépenser" / Formation OpenClassRooms DataScientist**')
st.write("## **Classification d'une demande de crédit**")

## Display input dataframe with multiselection of features for all the passenger list available (data are not standard scaled here!) ##
st.write('### Informations générales clients (index = ID de la demande de prêt):')
st.write('Dimension des données: ' + str(df_test_sample_origin.shape[0]) + ' lignes ' + str(df_test_sample_origin.shape[1]) + ' colonnes')
selections = st.multiselect('Vous pouvez ajouter ou enlever une donnée présente dans cette liste:', df_test_sample_origin.columns.tolist(),
 df_test_sample_origin.columns.tolist()[0:10])
st.dataframe(df_test_sample_origin.loc[:,selections])

## Display selected client data (checkbox condition: 'Données client') ##
if client_data:
    st.write(f'### Données du client, demande {selected_credit}')
    ### define values to display for the barchart and client data (with a maximum at 5) ###
    selections_client0 = st.multiselect('Vous pouvez afficher 5 données maximum parmi cette liste:', df_test_sample[features].columns.tolist(),
    df_test_sample[features].columns.tolist()[0:2])
    ### define columns to split some visual in two ###
    col1, col2 = st.columns(2)
    ### Display client informations regarding selected features ###
    col1.dataframe(df_test_sample_origin.loc[selected_credit, selections_client0])
    ### define pyplot for col2 barchart with selected passenger informations with condition of the number of selected features ###
    if len(selections_client0) <= 5:
        fig_client_info = plt.figure()
        plt.title(f'Diagramme bar données ID: {selected_credit}')
        sns.barplot(x=df_test_sample[features].loc[selected_credit, selections_client0].index, y=df_test_sample[features].loc[selected_credit, selections_client0].values)
        plt.xlabel('Features')
        plt.xticks(fontsize=8, rotation=45)
        plt.ylabel('Valeur normalisée')
        plt.yticks(fontsize=8)
        #### Display the graph ####
        col2.pyplot(fig_client_info, clear_figure=True)
    else:
        col2.write("Vous avez sélectionné trop de feature!!! Le graphique n'est pas affiché")

## Display loan answer regarding model probability calcul (path through API Flask to get the result / checbox condition : 'Résultat de la demande de prêt') ##
if client_pred_score:
    st.write('### Décision sur la demande de prêt')
    ### careful the url of the API should be change for serial deployment!! ###
    url_api_model_result = 'http://127.0.0.1:5000/scores'
    ### Be careful to the params, with must have a dict with index / ID loan value. It is how it is implemented in our API ###
    get_request = requests.get(url=url_api_model_result, params={'index': selected_credit})
    ### We get  the prediction information from the json format of the API model ###
    prediction_value = get_request.json()['Credit_score']
    ### We get the answer regardin loan acceptation ###
    answer_value = bool(get_request.json()['Answer'])
    ### Display results ###
    st.write(f'Demande de prêt ID: {selected_credit}')
    st.write(f'Probabilité de défauts de remboursement: {prediction_value*100:.2f} %')
    if answer_value:
        st.write('Demande de prêt acceptée!')
    else:
        #### add condition in function of the value of the prediction, if over the treshold but near should be discussed ####
        if prediction_value > treshold and prediction_value <= 0.55:
            st.write('Demande de prêt refusée --> à discuter avec le conseiller')
        else:
            st.write('Demande de prêt refusée!')
    ### add gauge for the prediction value with plotly library ###
    fig_gauge = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = float(f'{prediction_value*100:.1f}'),
    mode = "gauge+number+delta",
    title = {'text': "Score(%)"},
    delta = {'reference': treshold*100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
    gauge = {'axis': {'range': [0, 100]},
             'bar': {'color': 'black'},
             'steps' : [
                 {'range': [0, 30], 'color': "darkgreen"},
                 {'range': [30, (treshold*100)], 'color': "lightgreen"},
                 {'range': [(treshold*100),55], 'color': "orange"},
                 {'range': [55, 100], 'color':"red"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': treshold*100}}))
    st.plotly_chart(fig_gauge)

## Display interpretation about the score, global and local features importances (using SHAP library and SHAP model / checkbox: 'Interprétation du score' ) ##
if score_interpret:
    st.write('### Interprétations du score')
    ### calcul shap values with the explainer ###
    shap_values =shap_explainer.shap_values(df_test_sample[features])
    ### select between violin or bar plot for global features importance ###
    st.write('#### *Importance global des features*')
    selected_global_shap = st.selectbox("Sélectionner un graphique",
    ['Graphique_en_violon', 'Graphique_en_baton'])
    #### plot graphic in function of the selectbox ####
    if selected_global_shap == 'Graphique_en_violon':
        figure_shap_glob_v = plt.figure(figsize=(10,10))
        shap.summary_plot(shap_values[1], df_test_sample[features], feature_names=features, 
        show=False, plot_size=None)
        st.pyplot(figure_shap_glob_v, clear_figure=True)
        #### add expander for futher explanations on the graphic ####
        with st.expander('Informations complémentaires'):
            st.write(""" Explication sur le graphique violon avec high and low values """)
    elif selected_global_shap == 'Graphique_en_baton':
        figure_shap_glob_b = plt.figure(figsize=(10,10))
        shap.summary_plot(shap_values[1], df_test_sample[features], feature_names=features, 
        show=False, plot_size=None, plot_type = 'bar')
        st.pyplot(figure_shap_glob_b, clear_figure=True)
        #### add expander for futher explanations on the graphic ####
        with st.expander('Informations complémentaires'):
            st.write(""" Explication sur le graphique en baton """)
    ### Waterfall plot for local features importance ###
    st.write('#### *Importance local des features*')
    st.write('Graphique en cascade')
    #### define client raw with index of the ID and get specific shap values for it ####
    index_client0 = df_test_sample.index.get_loc(selected_credit)
    choosen_raw = df_test_sample.loc[df_test_sample.index == selected_credit][features]
    #### define ShapObject class to plot our waterfall for the selected client ####
    shap_object = ShapObject(base_values = shap_explainer.expected_value[1],
                         values = shap_explainer.shap_values(df_test_sample[features])[1][index_client0, :],
                         feature_names = features,
                         data = (choosen_raw.to_numpy().reshape(-1, 1)))
    #### plot graphic
    figure_loc_wtf = plt.figure(figsize=(10,10), facecolor='w')
    shap.waterfall_plot(shap_object)
    st.pyplot(figure_loc_wtf, clear_figure=True)
    #### add expander for further explanations on the graphic ####
    with st.expander('Informations complémentaires'):
            st.write(""" Explication sur le graphique en cascade """)

## Display comparison with all the client and the near client in score (using function created to filter near clients / checkbox: 'Analyse des features clients' ) ##
if client_analysis:
    st.write('### Analyse des features clients')
    ### add slider to select the number of near client that we want to select ###
    nearest_number = st.slider('Sélectionner le nombre de clients proche', 10, 40, None, 10)
    ### calculate the dataframe for near client ###
    df_nearest_client = filter_near_customer(df_test_sample, selected_credit, nearest_number, 'TARGET_PROB')
    ### bivariate analysis where we can choose the features to plot ###
    st.write('#### *Analyse bivariée*')
    #### define columns to split for several selection box ####
    col11, col12 = st.columns(2)
    feat1 = col11.selectbox('Feature 1', features, 0)
    feat2 = col12.selectbox('Feature 2', features, 1)
    #### Plot scatter plot with plotly ####
    figure_biv = go.Figure()
    #### all client scatter filtered with PREDICT_PROB column and treshold (accepted / denied) ####
    figure_biv.add_trace(go.Scatter(x=df_test_sample.loc[df_test_sample['TARGET_PROB'] < treshold][feat1], 
    y=df_test_sample.loc[df_test_sample['TARGET_PROB'] < treshold][feat2], 
    mode='markers', name='clients_prêt_acceptés', marker_symbol='circle', 
    marker={'color': df_test_sample.loc[df_test_sample['TARGET_PROB'] < treshold]['TARGET_PROB'], 
                            'coloraxis':'coloraxis'}))
    figure_biv.add_trace(go.Scatter(x=df_test_sample.loc[df_test_sample['TARGET_PROB'] >= treshold][feat1], 
    y=df_test_sample.loc[df_test_sample['TARGET_PROB'] >= treshold][feat2], 
    mode='markers', name='clients_prêt_refusés', marker_symbol='x', 
    marker={'color': df_test_sample.loc[df_test_sample['TARGET_PROB'] >= treshold]['TARGET_PROB'], 
                            'coloraxis':'coloraxis'}))
    #### neat customer scatter filtered with PREDICT_PROB column and treshold (accepted / denied) ####
    figure_biv.add_trace(go.Scatter(x=df_nearest_client.loc[df_nearest_client['TARGET_PROB'] < treshold][feat1], 
    y=df_nearest_client.loc[df_nearest_client['TARGET_PROB']< treshold][feat2], 
    mode='markers', name='clients_similaires_prêt_acceptés', marker_symbol='circle', 
    marker={'color': df_nearest_client.loc[df_nearest_client['TARGET_PROB'] < treshold]['TARGET_PROB'],
                             'coloraxis':'coloraxis'}))
    figure_biv.add_trace(go.Scatter(x=df_nearest_client.loc[df_nearest_client['TARGET_PROB'] >= treshold][feat1], 
    y=df_nearest_client.loc[df_nearest_client['TARGET_PROB'] >= treshold][feat2], 
    mode='markers', name='clients_similaires_prêt_refusés', marker_symbol='x', 
    marker={'color':df_nearest_client.loc[df_nearest_client['TARGET_PROB'] >= treshold]['TARGET_PROB'], 
                            'coloraxis':'coloraxis'}))
    #### plot selected client point ####
    figure_biv.add_trace(go.Scatter(x=[df_test_sample.loc[selected_credit, feat1]], y= [df_test_sample.loc[selected_credit, feat2]],
    mode='markers', name='ID_prêt_client_selectionné', 
    marker={'size':20, 'color':[df_test_sample.loc[selected_credit, 'TARGET_PROB']], 'coloraxis':'coloraxis', 
    'line':{'width':3, 'color':'black'}}))
    #### update legend localisation and add colorbar ####
    figure_biv.update_layout(legend={'orientation':"h", 'yanchor':'bottom','y':1.05, 'xanchor':'right','x':1, 'bgcolor':'DarkSlateGrey'},
                    xaxis={'title':feat1}, yaxis={'title':feat2}, coloraxis={'colorbar':{'title':'Score'}, 
                                                                                'colorscale':'RdYlGn_r', 'cmin':0, 'cmax':1, 'showscale':True})
    st.plotly_chart(figure_biv, use_container_width=True)
    ### Univariate analysis choose type of plot (boxplot or histogram/bargraph) ###
    st.write('#### *Analyse univariée*')
    #### select between boxplot or histogram/barplot distributions for univariate analysis ####
    selected_anaysis_gh = st.selectbox('Sélectionner un graphique', ['Boxplot', 'Histogramme/bâton'])
    if selected_anaysis_gh == 'Boxplot':
        ##### Add the possibility to display several features on the same plot #####
        selections_analysis = st.multiselect('Vous pouvez ajouter ou enlever une donnée présente dans cette liste:', df_test_sample[features].columns.tolist(),
        df_test_sample[features].columns.tolist()[0:5])
        ##### display boxplot #####
        ###### create in each df a columns to identifie them and use hue parameters ######
        df_test_sample['data_origin'] = 'clients'
        df_nearest_client['data_origin'] = 'clients_similaires'
        ###### concatenate two df before drawing boxplot ######
        cdf = pd.concat([df_test_sample[selections_analysis + ['data_origin']], 
        df_nearest_client[selections_analysis + ['data_origin']]])
        ###### Create DataFrame from the selected client loan ID series ######
        df_loan = pd.DataFrame([df_test_sample.loc[selected_credit, features].tolist()], columns=features)
        ###### using melt mehtod to adapt our concatenate dataframe to the format that we want (for displaying several features) with Seaborn ######
        cdf = pd.melt(cdf, id_vars='data_origin', var_name='Features')
        df_loan = pd.melt(df_loan[selections_analysis], var_name='Features')
        df_loan['data_origin'] = 'ID_prêt_client_selectionné'
        ###### plotting figure ######
        figure_boxplot = plt.figure(figsize=(4,2))
        ax = sns.boxplot(x = 'Features', y = 'value', hue='data_origin', data=cdf , showfliers=False, palette = 'tab10')
        sns.stripplot(x = 'Features', y = 'value', data = df_loan, hue = 'data_origin', palette=['yellow'], s=8, linewidth=1.5, edgecolor='black')
        plt.xticks(fontsize=6, rotation=45)
        plt.yticks(fontsize=6)
        plt.ylabel('Valeur normalisée')
        leg = plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ###### modify legend object for selected client loan ID to match graph style ######
        leg.legendHandles[-1].set_linewidth(1.5)
        leg.legendHandles[-1].set_edgecolor('black')
        st.pyplot(figure_boxplot, clear_figure=True)
    if selected_anaysis_gh == 'Histogramme/bâton':
        ##### Add the posibility to choose the distribution we want to see #####
        feat3 = st.selectbox('Feature', features,0)
        loan = df_test_sample.loc[selected_credit, :]
        figure_h=plt.figure(figsize=(10,4))
        figure_h.add_subplot(1,2,1)
        plt.title('Tous les clients', fontweight='bold')
        ###### careful, color used here for bins are maching seaborn previous ones used ######
        n, bins, patches = plt.hist(x = df_test_sample[feat3], color='#1f77b4', linewidth=1, edgecolor='black')
        ###### here we are setting the color bins for our selected loan customer ######
        patches[bin_location(bins, loan[feat3])].set_fc('yellow')
        plt.xlabel(f'{feat3} (Normalisé)')
        plt.xticks(bins, fontsize=8, rotation=45)
        plt.ylabel('Nombre total')
        plt.yticks(fontsize=8)
        figure_h.add_subplot(1,2,2)
        plt.title('Clients similaires', fontweight='bold')
        n, bins, patches = plt.hist(x = df_nearest_client[feat3], color='#ff7f0e', linewidth=1, edgecolor='black')
        patches[bin_location(bins, loan[feat3])].set_fc('yellow')
        plt.xlabel(f'{feat3} (Normalisé)')
        plt.xticks(bins, fontsize=8, rotation=45)
        plt.ylabel('Nombre Total')
        plt.yticks(fontsize=8)
        st.pyplot(figure_h, clear_figure=True)
    
