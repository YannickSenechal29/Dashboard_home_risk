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
    value_bin_ind = count
    for b in bins:
        if value > b:
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
st.sidebar.header('S??lectionner une demande de pr??t:')
selected_credit = st.sidebar.selectbox('Pr??t_ID', loan_id)
### Add checkbox for displaying different client informations ###
client_data = st.sidebar.checkbox('Donn??es client')
client_pred_score = st.sidebar.checkbox('R??sultat de la demande de pr??t')
### Add checkbox for displaying score interpretation ###
score_interpret = st.sidebar.checkbox('Interpr??tations du score')
### Add checkbox for displaying client data analysis ###
client_analysis = st.sidebar.checkbox('Analyse des features client')


# ------------ Main display, part by part -----------------
## Generic title ##
st.write('# **SENECHAL Yannick: Projet 7 "Pr??t ?? d??penser" / Formation OpenClassRooms DataScientist**')
st.write("## **Classification d'une demande de cr??dit**")

## Display input dataframe with multiselection of features for all the passenger list available (data are not standard scaled here!) ##
st.write('### Informations g??n??rales clients (index = ID de la demande de pr??t):')
st.write('Dimension des donn??es: ' + str(df_test_sample_origin.shape[0]) + ' lignes ' + str(df_test_sample_origin.shape[1]) + ' colonnes')
selections = st.multiselect('Vous pouvez ajouter ou enlever une donn??e pr??sente dans cette liste:', df_test_sample_origin.columns.tolist(),
 df_test_sample_origin.columns.tolist()[0:10])
st.dataframe(df_test_sample_origin.loc[:,selections].sort_index())
### add expander for further explanations on the data ###
with st.expander('Informations compl??mentaires'):
    st.write(""" Ici vous trouvez les informations disponibles pour tous les clients.  \n"""
            """ Pour plus d'informations sur les features (variables) disponibles merci de contacter l'??quipe support. """)

## Display selected client data (checkbox condition: 'Donn??es client') ##
if client_data:
    st.write(f'### Donn??es du client, demande {selected_credit}')
    ### define values to display for the barchart and client data (with a maximum at 5) ###
    selections_client0 = st.multiselect('Vous pouvez afficher 5 donn??es maximum parmi cette liste:', df_test_sample[features].columns.tolist(),
    df_test_sample[features].columns.tolist()[0:2])
    ### define columns to split some visual in two ###
    col1, col2 = st.columns(2)
    ### Display client informations regarding selected features ###
    col1.dataframe(df_test_sample_origin.loc[selected_credit, selections_client0])
    ### define pyplot for col2 barchart with selected passenger informations with condition of the number of selected features ###
    if len(selections_client0) <= 5:
        fig_client_info = plt.figure()
        plt.title(f'Diagramme bar donn??es ID: {selected_credit}')
        sns.barplot(x=df_test_sample[features].loc[selected_credit, selections_client0].index, y=df_test_sample[features].loc[selected_credit, selections_client0].values)
        plt.xlabel('Features')
        plt.xticks(fontsize=8, rotation=45)
        plt.ylabel('Valeur normalis??e')
        plt.yticks(fontsize=8)
        #### Display the graph ####
        col2.pyplot(fig_client_info, clear_figure=True)
    else:
        col2.write("Vous avez s??lectionn?? trop de feature!!! Le graphique n'est pas affich??")
    ### add expander for further explanations on the selected client data ###
    with st.expander('Informations compl??mentaires'):
        st.write(""" Ici vous trouvez les informations client disponibles pour la demande de pr??t s??lectionn??e.  \n"""
            """ La graphique en b??ton donne les valeurs de features (variables) normalis??es pour pouvoir les afficher sur la m??me ??chelle. """)

## Display loan answer regarding model probability calcul (path through API Flask to get the result / checbox condition : 'R??sultat de la demande de pr??t') ##
if client_pred_score:
    st.write('### D??cision sur la demande de pr??t')
    ### careful the url of the API should be change for serial deployment!! ###
    url_api_model_result = 'https://api-home-risk-oc-7.herokuapp.com/scores'
    ### Be careful to the params, with must have a dict with index / ID loan value. It is how it is implemented in our API ###
    get_request = requests.get(url=url_api_model_result, params={'index': selected_credit})
    ### We get  the prediction information from the json format of the API model ###
    prediction_value = get_request.json()['Credit_score']
    ### We get the answer regardin loan acceptation ###
    answer_value = bool(get_request.json()['Answer'])
    ### Display results ###
    st.write(f'Demande de pr??t ID: {selected_credit}')
    st.write(f'Probabilit?? de d??faut de remboursement: {prediction_value*100:.2f} %')
    if answer_value:
        st.write('Demande de pr??t accept??e!')
    else:
        #### add condition in function of the value of the prediction, if over the treshold but near should be discussed ####
        if prediction_value > treshold and prediction_value <= 0.52:
            st.write('Demande de pr??t refus??e --> ?? discuter avec le conseiller')
        else:
            st.write('Demande de pr??t refus??e!')
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
                 {'range': [(treshold*100),52], 'color': "orange"},
                 {'range': [52, 100], 'color':"red"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': treshold*100}}))
    st.plotly_chart(fig_gauge)
    ### add expander for further explanations on the prediction r??sult ###
    with st.expander('Informations compl??mentaires'):
        st.write(""" Le retour de l'API de pr??diction donne un score entre 0 et 100% qui repr??sente la probabilit?? de refus de pr??t.  \n"""
            """ Trois cas de figure sont alors possibles:  \n """
            """ 1) Le score est en dessous de 49% ??? la demande de pr??t est accept??e.  \n """
            """ 2) Le score est entre 49 et 52% ??? la demande de pr??t est refus??e 
            mais peut ??tre discut??e avec le conseiller pour ??ventuellement l'accepter 
            (gr??ce notamment aux onglets 'interpr??tations du score' et 'analyse des features clients').  \n"""
            """3) Le score est au dessus de 52% ??? la demande de pr??t est refus??e. """)

## Display interpretation about the score, global and local features importances (using SHAP library and SHAP model / checkbox: 'Interpr??tation du score' ) ##
if score_interpret:
    st.write('### Interpr??tations du score')
    ### calcul shap values with the explainer ###
    shap_values =shap_explainer.shap_values(df_test_sample[features])
    ### select between violin or bar plot for global features importance ###
    st.write('#### *Importance globale des features*')
    selected_global_shap = st.selectbox("S??lectionner un graphique",
    ['Graphique_en_violon', 'Graphique_en_baton'])
    #### plot graphic in function of the selectbox ####
    if selected_global_shap == 'Graphique_en_violon':
        figure_shap_glob_v = plt.figure(figsize=(10,10))
        shap.summary_plot(shap_values[1], df_test_sample[features], feature_names=features, 
        show=False, plot_size=None)
        st.pyplot(figure_shap_glob_v, clear_figure=True)
        ##### add expander for futher explanations on the graphic #####
        with st.expander('Informations compl??mentaires'):
            st.write(""" Dans ce graphique en violon, on affiche par ordre d'importance les 20 features 
            qui ont le plus d'influence globale dans la valeur du score avec leur distribution.  \n""" 
            """A gauche du traie elles vont dans le sens de r??duire le score (d'accepter un pr??t),
             en revanche ?? droite elles vont dans le sens d'augmenter le score (de refuser un pr??t).  \n """ 
             """ Le code couleur indique la valeur de la feature. Une valeur ??lev??e en rouge et une valeur faible en bleu. """)
    elif selected_global_shap == 'Graphique_en_baton':
        figure_shap_glob_b = plt.figure(figsize=(10,10))
        shap.summary_plot(shap_values[1], df_test_sample[features], feature_names=features, 
        show=False, plot_size=None, plot_type = 'bar')
        st.pyplot(figure_shap_glob_b, clear_figure=True)
        #### add expander for futher explanations on the graphic ####
        with st.expander('Informations compl??mentaires'):
            st.write(""" Dans ce graphique en b??ton, on affiche par ordre d'importance les 20 features qui ont
            le plus d'influence globale dans la valeur du score.  \n """
            """ L'influence allant dans le sens de refuser une demande de pr??t. """)
    ### Waterfall plot for local features importance ###
    st.write('#### *Importance locale des features*')
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
    with st.expander('Informations compl??mentaires'):
            st.write(""" Dans ce graphique en cascade, on affiche par ordre d'importance les features 
            qui ont le plus d'influence dans la valeur du score ?? l'??chelle de la demande client que l'on regarde (locale).  \n"""
            """ Le code couleur indique dans quel sens elles influes. En bleu dans le sens de r??duire le score (d'accepter le pr??t),
             en rouge dans le sens d'augmenter le score (de refuser le pr??t).""" 
            )

## Display comparison with all the client and the near client in score (using function created to filter near clients / checkbox: 'Analyse des features clients' ) ##
if client_analysis:
    st.write('### Analyse des features clients')
    ### add slider to select the number of near client that we want to select ###
    nearest_number = st.slider('S??lectionner le nombre de clients proche', 10, 40, None, 10)
    ### calculate the dataframe for near client ###
    df_nearest_client = filter_near_customer(df_test_sample, selected_credit, nearest_number, 'TARGET_PROB')
    ### bivariate analysis where we can choose the features to plot ###
    st.write('#### *Analyse bivari??e*')
    #### define columns to split for several selection box ####
    col11, col12 = st.columns(2)
    feat1 = col11.selectbox('Feature 1', features, 0)
    feat2 = col12.selectbox('Feature 2', features, 1)
    #### Plot scatter plot with plotly ####
    figure_biv = go.Figure()
    #### all client scatter filtered with PREDICT_PROB column and treshold (accepted / denied) ####
    figure_biv.add_trace(go.Scatter(x=df_test_sample.loc[df_test_sample['TARGET_PROB'] < treshold][feat1], 
    y=df_test_sample.loc[df_test_sample['TARGET_PROB'] < treshold][feat2], 
    mode='markers', name='Tous les clients_pr??t_accept??s', marker_symbol='circle', 
    marker={'color': df_test_sample.loc[df_test_sample['TARGET_PROB'] < treshold]['TARGET_PROB'], 
                            'coloraxis':'coloraxis'}))
    figure_biv.add_trace(go.Scatter(x=df_test_sample.loc[df_test_sample['TARGET_PROB'] >= treshold][feat1], 
    y=df_test_sample.loc[df_test_sample['TARGET_PROB'] >= treshold][feat2], 
    mode='markers', name='Tous les clients_pr??t_refus??s', marker_symbol='x', 
    marker={'color': df_test_sample.loc[df_test_sample['TARGET_PROB'] >= treshold]['TARGET_PROB'], 
                            'coloraxis':'coloraxis'}))
    #### neat customer scatter filtered with PREDICT_PROB column and treshold (accepted / denied) ####
    figure_biv.add_trace(go.Scatter(x=df_nearest_client.loc[df_nearest_client['TARGET_PROB'] < treshold][feat1], 
    y=df_nearest_client.loc[df_nearest_client['TARGET_PROB']< treshold][feat2], 
    mode='markers', name='clients_similaires_pr??t_accept??s', marker_symbol='circle', 
    marker={'color': df_nearest_client.loc[df_nearest_client['TARGET_PROB'] < treshold]['TARGET_PROB'],
                             'coloraxis':'coloraxis'}))
    figure_biv.add_trace(go.Scatter(x=df_nearest_client.loc[df_nearest_client['TARGET_PROB'] >= treshold][feat1], 
    y=df_nearest_client.loc[df_nearest_client['TARGET_PROB'] >= treshold][feat2], 
    mode='markers', name='clients_similaires_pr??t_refus??s', marker_symbol='x', 
    marker={'color':df_nearest_client.loc[df_nearest_client['TARGET_PROB'] >= treshold]['TARGET_PROB'], 
                            'coloraxis':'coloraxis'}))
    #### plot selected client point ####
    figure_biv.add_trace(go.Scatter(x=[df_test_sample.loc[selected_credit, feat1]], y= [df_test_sample.loc[selected_credit, feat2]],
    mode='markers', name='ID_pr??t_client_selectionn??', 
    marker={'size':20, 'color':[df_test_sample.loc[selected_credit, 'TARGET_PROB']], 'coloraxis':'coloraxis', 
    'line':{'width':3, 'color':'black'}}))
    #### update legend localisation and add colorbar ####
    figure_biv.update_layout(legend={'orientation':"h", 'yanchor':'bottom','y':1.05, 'xanchor':'right','x':1, 
    'bgcolor':'white', 'font':{'color':'black'}}, xaxis={'title':feat1}, 
    yaxis={'title':feat2}, coloraxis={'colorbar':{'title':'Score'}, 
                                        'colorscale':'RdYlGn_r', 'cmin':0, 'cmax':1, 'showscale':True})
    st.plotly_chart(figure_biv, use_container_width=True)
    #### add expander for further explanations on the scatterplot ####
    with st.expander('Informations compl??mentaires'):
            st.write(""" Ce graphique permet d'afficher un nuage de points en fonction de deux features s??lectionnables.  \n"""
            """ Notez qu'il est possible de cliquer dans la l??gende pour ne s??lectionner que le groupe de clients qui nous int??ressent pour comparer
             au client que l'on regarde.  \n """
             """ Le code couleur indique la valeur du score client. """ )
    ### Univariate analysis choose type of plot (boxplot or histogram/bargraph) ###
    st.write('#### *Analyse univari??e*')
    #### select between boxplot or histogram/barplot distributions for univariate analysis ####
    selected_anaysis_gh = st.selectbox('S??lectionner un graphique', ['Boxplot', 'Histogramme/b??ton'])
    if selected_anaysis_gh == 'Boxplot':
        ##### Add the possibility to display several features on the same plot #####
        selections_analysis = st.multiselect('Vous pouvez ajouter ou enlever une donn??e pr??sente dans cette liste:', df_test_sample[features].columns.tolist(),
        df_test_sample[features].columns.tolist()[0:5])
        ##### display boxplot #####
        ###### create in each df a columns to identifie them and use hue parameters ######
        df_test_sample['data_origin'] = 'Tous les clients'
        df_nearest_client['data_origin'] = 'clients_similaires'
        ###### concatenate two df before drawing boxplot ######
        cdf = pd.concat([df_test_sample[selections_analysis + ['data_origin']], 
        df_nearest_client[selections_analysis + ['data_origin']]])
        ###### Create DataFrame from the selected client loan ID series ######
        df_loan = pd.DataFrame([df_test_sample.loc[selected_credit, features].tolist()], columns=features)
        ###### using melt mehtod to adapt our concatenate dataframe to the format that we want (for displaying several features) with Seaborn ######
        cdf = pd.melt(cdf, id_vars='data_origin', var_name='Features')
        df_loan = pd.melt(df_loan[selections_analysis], var_name='Features')
        df_loan['data_origin'] = 'ID_pr??t_client_selectionn??'
        ###### plotting figure ######
        figure_boxplot = plt.figure(figsize=(4,2))
        ax = sns.boxplot(x = 'Features', y = 'value', hue='data_origin', data=cdf , showfliers=False, palette = 'tab10')
        sns.stripplot(x = 'Features', y = 'value', data = df_loan, hue = 'data_origin', palette=['yellow'], s=8, linewidth=1.5, edgecolor='black')
        plt.xticks(fontsize=6, rotation=45)
        plt.yticks(fontsize=6)
        plt.ylabel('Valeur normalis??e')
        leg = plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ###### modify legend object for selected client loan ID to match graph style ######
        leg.legendHandles[-1].set_linewidth(1.5)
        leg.legendHandles[-1].set_edgecolor('black')
        st.pyplot(figure_boxplot, clear_figure=True)
        ###### add expander for further explanations on the scatterplot ######
        with st.expander('Informations compl??mentaires'):
            st.write(""" Ce boxplot permet d'afficher les distributions des groupes de clients en fonction de la valeur du client s??lectionn??.  \n"""
            """ Notez que les variables sont normalis??es afin d'avoir une image de la situation de notre client par rapport aux autres groupes de clients.""")
    if selected_anaysis_gh == 'Histogramme/b??ton':
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
        plt.xlabel(f'{feat3} (Normalis??)')
        plt.xticks(bins, fontsize=8, rotation=45)
        plt.ylabel('Nombre total')
        plt.yticks(fontsize=8)
        figure_h.add_subplot(1,2,2)
        plt.title('Clients similaires', fontweight='bold')
        n, bins, patches = plt.hist(x = df_nearest_client[feat3], color='#ff7f0e', linewidth=1, edgecolor='black')
        patches[bin_location(bins, loan[feat3])].set_fc('yellow')
        plt.xlabel(f'{feat3} (Normalis??)')
        plt.xticks(bins, fontsize=8, rotation=45)
        plt.ylabel('Nombre Total')
        plt.yticks(fontsize=8)
        st.pyplot(figure_h, clear_figure=True)
        ###### add expander for further explanations on the scatterplot ######
        with st.expander('Informations compl??mentaires'):
            st.write(""" Cette histogramme permet d'afficher les distributions des groupes de clients.  \n"""
            """ Notez que la barre en jaune indique dans quel population de nos groupes de clients 
            se trouve notre client s??lectionn??.  \n """
            """ Les variables sont ??galement normalis??es. """)
    
