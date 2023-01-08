import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import io
import re
import geopandas as gpd
import plotly.express as px
from neuralprophet import NeuralProphet

from collections import defaultdict
#Creating Map
import folium
from folium.plugins import MarkerCluster
from folium import IFrame
from streamlit_folium import st_folium
from datetime import datetime

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

import pickle

import hdbscan

cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
        '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 
        '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', 
        '#000075', '#808080']*100

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .row_heading.level0 {display:none}
            .blank {display:none}
            .dataframe {text-align: left !important}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



def data_visualization():
    col1, col2 = st.columns(2)
    with col1:
        st.image("tomorrow_university.png")
    with col2:
        st.image("kfw_logo.png")

    data = pd.read_csv('Ladesaeulenregister_CSV.csv', skiprows=10, sep=';', decimal=',',
                     encoding="ISO-8859-1", engine='python')
    data2=data.copy()
    data_shape=data.shape
    data_columns=data.columns.tolist()
    
    st.header("Data Source:")
    st.subheader("Public charging infrastructure")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image("Bundesnetzagentur_logo.svg.png")

    st.header("Data Analysis")
    List_states=list(data['Bundesland'].unique()) #["All"]+
    State=st.selectbox('Pick State', List_states)
    if(State!="All"):
        data=data[data['Bundesland']==State]
        
    List_cities=list(data['Ort'].unique())
    selected_cities=st.multiselect('Choose cities', List_cities)
    if(selected_cities):
        data=data[data['Ort'].isin(selected_cities)]
        
    data.drop(['Public Key1','Public Key2','Public Key3'],axis=1,inplace=True)
    data.drop_duplicates(subset = ["Breitengrad","Längengrad"],keep = 'last',inplace=True)
    st.write(data)

    st.success("(Before filtering) The dataset contains "+str(data_shape[0])+" rows")  
    st.success("(After filtering) The dataset contains "+str(data.shape[0])+" rows (or chargers)")  
    st.success("There are "+str(data_shape[1])+" columns")  

    st.subheader("Summary of a DataFrame")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()

    st.text(s)

    st.subheader("Null values per column")
    data_null_values=data.isnull().sum()
    data_null_values=data_null_values.rename("# null values per column")
    st.write(data_null_values.sort_values(ascending=False))

    dataD = data.describe(include='all')
    for i,j in zip(dataD.dtypes.tolist(),dataD.dtypes.index.tolist()):
        if(i=="object"):
            dataD[j] = dataD[j].astype(str)

    st.subheader("Descriptive statistics of the dataframe")
    st.write(dataD)

    if(selected_cities):
        st.subheader("Top operators in "+str(selected_cities)+","+str(State))
    else:
        st.subheader("Top operators in "+str(State))
        
    s1=data['Betreiber'].value_counts()

    s2=(data['Betreiber'].value_counts(normalize=True).rename("Betreiber %")).apply(lambda x:x*100)
    df_concat=pd.concat([s1, s2], axis=1)
    st.write(df_concat)

    sum_betreiber=df_concat['Betreiber'].sum()-df_concat['Betreiber'].head(7).sum()
    
    fig,ax=plt.subplots(figsize=(11,7))
    ax.pie(df_concat['Betreiber'].head(7).tolist()+[sum_betreiber], labels=df_concat.head(7).index.values.tolist()+["others"], autopct='%1.1f%%',shadow=True, startangle=90)
    st.pyplot(fig)

    sum_betreiber=df_concat['Betreiber'].sum()-df_concat['Betreiber'].head(5).sum()

    fig1,ax1=plt.subplots(figsize=(11,7))
    ax1.bar(df_concat.head(5).index.values.tolist()+["others"], df_concat['Betreiber'].head(5).tolist()+[sum_betreiber],color='g', label='cos')
    st.pyplot(fig1)

    if(selected_cities):
        st.subheader("# charging points in each station "+str(selected_cities)+","+str(State))
    else:
        st.subheader("# charging points in each station "+str(State))
        
    fig,ax=plt.subplots(figsize=(11,7))
    ax.hist(data['Anzahl Ladepunkte'], data['Anzahl Ladepunkte'].nunique()*2)
    st.pyplot(fig)

    st.subheader("# Chargers per state")

    n1 = [0]+data2['Bundesland'].value_counts(ascending=True).tolist()
    w1 = [""]+data2['Bundesland'].value_counts(ascending=True).index.tolist()
    
    fig,ax=plt.subplots(figsize=(11,7))
    ax.barh(width=range(len(w1)),y=w1)
    ax.set_xticks(range(len(n1)),n1)
    st.pyplot(fig)

    st.subheader("# new chargers per year")

    year_list=[]

    for i in data2['Inbetriebnahmedatum'].tolist():
       year_list.append(int(i.split('.')[2]))


    data2['year']=year_list
    n = data2[data2['year']>2000]['year'].value_counts(ascending=True).tolist()
    w = data2[data2['year']>2000]['year'].value_counts(ascending=True).index.tolist()
            
    fig,ax=plt.subplots(figsize=(11,7))
    ax.plot(w, n,'.')
    ax.set_xlabel("Year")
    ax.set_ylabel("# Chargers per year (from 2000)")
    st.pyplot(fig)

    st.subheader("Outliers Identification")
    fig = px.box(data2, y="year")
    st.plotly_chart(fig)
            
    st.subheader("Correlation between features")
    
    corr = data2.corr()
    fig = plt.figure(figsize=(11, 7))
    sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,annot=True,cmap='Blues', fmt='g')
    st.pyplot(fig)
    st.success("P1&P2 are strongly positively correlated")
    st.success("Both, P1 & P2 are positively correlated to Charging power (Anschlusleistung)")

    

    table_MN = pd.read_html('https://www.citypopulation.de/en/germany/cities/')

    df_state=table_MN[0][['Name','Area A (km²)','Population Estimate (E) 2021-12-31']]
    df_city=table_MN[2][['Name','Population Estimate (E) 2021-12-31','Area']]
    
    df_state['Name'] =  df_state['Name'].apply(lambda x:x.split(' (')[0])
    df_state['Name'] =  df_state['Name'].apply(lambda x:x.split(' [')[0])

    df_city['Name'] = df_city['Name'].apply(lambda x:x.split(' (')[0])
    df_city['Name'] = df_city['Name'].apply(lambda x:x.split(' [')[0])

    data3_state=data2['Bundesland'].value_counts().rename_axis('Bundesland').reset_index(name='counts').merge(df_state, how='inner', left_on="Bundesland", right_on="Name")
    data3_ort=data2['Ort'].value_counts().rename_axis('Ort').reset_index(name='counts').merge(df_city, how='inner', left_on="Ort", right_on="Name")
            
    st.subheader("How many person per charger")
    data3_state['people/charger']=data3_state["Population Estimate (E) 2021-12-31"]/data3_state["counts"]
    data3_ort['people/charger']=data3_ort["Population Estimate (E) 2021-12-31"]/data3_ort["counts"]

    st.subheader("Per State")
    st.write(data3_state[['Bundesland','people/charger']].sort_values(['people/charger'],ascending=True))
    st.subheader("Per City")
    st.write(data3_ort[['Ort','people/charger']].sort_values(['people/charger'],ascending=True))

            
    st.subheader("How many charger per km²")
    data3_state['charger/km2']=data3_state["counts"]/data3_state["Area A (km²)"]
    data3_ort['charger/km2']=data3_ort["counts"]/data3_ort["Area"]

    st.subheader("Per State")
    st.write(data3_state[['Bundesland','charger/km2']].sort_values(['charger/km2'],ascending=False))
    st.subheader("Per City")
    st.write(data3_ort[['Ort','charger/km2']].sort_values(['charger/km2'],ascending=False))
            
    if(selected_cities):
        m = folium.Map(location=[51.104138, 10.180465], zoom_start=5.3, tiles="CartoDB positron")

        tooltip = "Click me!"

        for index,row in enumerate(data.drop_duplicates(subset = ["Breitengrad","Längengrad"],keep = 'last').itertuples()):
            html="<b>Operator:</b> {0} |<b>Commissioning date:</b> {6}| <b>Adress:</b> {1} {2}, {3} {4} {5}".format(row.Betreiber, row.Straße, row.Hausnummer, row.Postleitzahl,row.Ort,row.Bundesland,row.Inbetriebnahmedatum)
            try:
                folium.Marker([float(row.Breitengrad[:-1].replace(",",".")), float(row.Längengrad[:-1].replace(",","."))], popup=html, tooltip=tooltip).add_to(m)
            except:
                print("")

        st_folium(m, width=700, height=500)

def create_map(df, cluster_column):
    m = folium.Map(location=[df.Breitengrad.mean(), df.Längengrad.mean()], zoom_start=9, tiles='OpenStreet Map')

    for _, row in df.iterrows():

        if row[cluster_column] == -1:
            cluster_colour = '#000000'
        else:
            cluster_colour = cols[int(row[cluster_column])]

        folium.CircleMarker(
            location= [row['Breitengrad'], row['Längengrad']],
            radius=5,
            popup= row[cluster_column],
            color=cluster_colour,
            fill=True,
            fill_color=cluster_colour
        ).add_to(m)
        
    return(m)

def prediction():
    st.header("Prediction")

    data = pd.read_csv('Ladesaeulenregister_CSV.csv', skiprows=10, sep=';', decimal=',',
                     encoding="ISO-8859-1", engine='python')
            

  
    List_states=["All"]+list(data['Bundesland'].unique()) #
    State=st.selectbox('Pick State', List_states)
    if(State!="All"):
       data=data[data['Bundesland']==State]

       xx_list=[]
       yy_list=[]
       for index,row in enumerate(data.drop_duplicates(subset = ["Breitengrad","Längengrad"],keep = 'last').itertuples()):
           try:
               xx=float(row.Breitengrad.replace(",","."))
               yy=float(row.Längengrad.replace(",","."))
       
               xx_list.append(xx)
               yy_list.append(yy)
           except:
               print("")

       data_dict = {'Breitengrad': xx_list, 'Längengrad': yy_list}
       data_location=pd.DataFrame.from_dict(data_dict)
            
    year_list=[]

    for i in data['Inbetriebnahmedatum'].tolist():
        year_list.append(datetime.strptime(i.split('.')[2]+"-01-01","%Y-%m-%d"))



    data['year']=year_list
            
    future_filter = st.number_input('How many year to predict', 2, 23)

    df=pd.DataFrame.from_dict({"chargers per year":data['year'].value_counts(ascending=True).tolist(),"year":data['year'].value_counts(ascending=True).index.tolist()})
    new_column = df[["year","chargers per year"]]
    new_column=new_column[new_column["year"]!="2022-01-01"]
    new_column=new_column[new_column['year']>"2011-01-01"]
    new_column=new_column.sort_values(['year'],ascending=True)
        
        
    if(State!="All"):
        st.header("Clustering")
        Clustering_options=st.selectbox('Clustering Method', ['DBSCAN','HDBSCAN','HDBSCAN+KNN'])

        X = np.array(data_location[['Breitengrad', 'Längengrad']], dtype='float64')
        if(Clustering_options=='DBSCAN'):
            model = DBSCAN(eps=0.01, min_samples=3).fit(X)
            class_predictions = model.labels_

            data_location['CLUSTERS_DBSCAN'] = class_predictions
            m = create_map(data_location, 'CLUSTERS_DBSCAN')
                
        elif(Clustering_options=='HDBSCAN'):
            model = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, cluster_selection_epsilon=0.01)
            class_predictions = model.fit_predict(X)
                
            data_location['CLUSTER_HDBSCAN'] = class_predictions
            m = create_map(data_location, 'CLUSTER_HDBSCAN')
                
        elif(Clustering_options=='HDBSCAN+KNN'):
            model = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, cluster_selection_epsilon=0.01)
            class_predictions = model.fit_predict(X)
            
            data_location['CLUSTER_HDBSCAN'] = class_predictions
            classifier = KNeighborsClassifier(n_neighbors=1)
            df_train = data_location[data_location.CLUSTER_HDBSCAN!=-1]
            df_predict = data_location[data_location.CLUSTER_HDBSCAN==-1]
        
            X_train = np.array(df_train[['Breitengrad', 'Längengrad']], dtype='float64')
            y_train = np.array(df_train['CLUSTER_HDBSCAN'])

            X_predict = np.array(df_predict[['Breitengrad', 'Längengrad']], dtype='float64')  
                
            classifier.fit(X_train, y_train)
        
            predictions = classifier.predict(X_predict)
        
            data_location['CLUSTER_hybrid'] = data_location['CLUSTER_HDBSCAN']
            data_location.loc[data_location.CLUSTER_HDBSCAN==-1, 'CLUSTER_hybrid'] = predictions  
        
            m = create_map(data_location, 'CLUSTER_hybrid')     
                
        try:
            st_folium(m, width=700, height=500)
        except:
            print("")      
                


    st.header("Historical Data")        
    st.write(new_column)
            
    new_column.dropna(inplace=True)
    new_column.columns = ['ds', 'y']
            
    clicked=st.button('Expected # of chargers')
    if clicked:
        n = NeuralProphet()
        model = n.fit(new_column, freq='Y')
        future = n.make_future_dataframe(new_column, periods=future_filter)
        forecast = n.predict(future)
        forecast['yhat1']=forecast['yhat1'].apply(lambda x:int(x))
        forecast=forecast[['ds','yhat1']]
        forecast.columns=['year',"chargers per year"]
        st.header("Expected # chargers for the next years")
        st.write(forecast)
        
    st.header("Anschlussleistung/Normalladeeinrichtung Prediction")        
    st.header("ML Model Workflow")
    st.markdown('**Dropped these columns:** \n Betreiber,Straße,Hausnummer,Adresszusatz,Postleitzahl,Ort,Bundesland,Kreis/kreisfreie Stadt,Public Key1,Public Key2,Public Key3,Public Key4')
    st.markdown('**Converting:** \n longitute and latitude from string to float | type of charging; 1 if Normalladeeinrichtung 0 if Schnellladeeinrichtung')
    st.markdown('**New feature created:** \n Extract year value from Inbetriebnahmedatum column')
    st.markdown('**Using longitute and latitude and Kmeans (K=5):** \n deutschland was divided into 5 areas like in the photo')
    st.markdown('**One hot encoding applied on:** \n Plug type | Cluster Group ')

    st.image("kmeans_result.PNG")
    st.markdown('**_This is the final dataframe used for building the 2 models_**')
    st.image("dataframe_After_dataPreprocessing.PNG")
    st.success("Normalladeeinrichtung Prediction(Binary classification): Algorithm used: Logistic Regression | Accuracy: 98.856% | F1 Score: 98.853% ")
    st.success("Anschlussleistung Prediction(Multiclass classification, 13 class): Algorithm used: SVM | Accuracy: 58.256% | F1 Score: 51.33% ")

    st.header("ML model Demo")        
    a = st.radio('Select features to predict:', ["Anschlussleistung", "Normalladeeinrichtung"])

    predicted_label=[3.7,11.0,22.0,26.4,30.0,33.0,39.6,44.0,50.0,93.0,150.0,300.0,350.0]

    if a=="Normalladeeinrichtung":
        Anschlussleistung = st.selectbox('Anschlussleistung :', predicted_label)
    else:
        Normalladeeinrichtung = st.selectbox('Type of Charging :', ["Normalladeeinrichtung", "Schnellladeeinrichtung"])

    Anzahl_Ladepunkte = st.number_input('Nbre Charging Points :',1,4)
    year = st.number_input('Year :',2000,2050)
    Plug_Type=st.multiselect('Plug Type :', ['AC Schuko','AC Kupplung Typ 2',  'AC CEE 5 polig', 'DC Kupplung Combo', 'AC Steckdose Typ 2','AC CEE 3 polig','DC CHAdeMO'])
    Langengrad = st.number_input('Longitute :',-180,180)
    Breitengrad = st.number_input('Latitude :',-90,90)
     
    clicked2=st.button(str(a)+' prediction')
        
    row_topredict=[]
    if clicked2:
        if a=="Normalladeeinrichtung":  
            row_topredict.append(Anschlussleistung)
        else:
            if Normalladeeinrichtung=="Normalladeeinrichtung":
                row_topredict.append(1)
            else:
                row_topredict.append(0)  
                
        row_topredict.append(Anzahl_Ladepunkte)
        
        plug_list=[0,0,0,0,0,0,0]
        
        if Plug_Type=="AC CEE 3 polig":
                plug_list[0]=1
                
        elif Plug_Type=="AC Kupplung Typ 2":
                plug_list[1]=1
                
        elif Plug_Type=="DC CHAdeMO":
                plug_list[2]=1

        elif Plug_Type=="AC Steckdose Typ 2":
                plug_list[3]=1
                
        elif Plug_Type=="DC Kupplung Combo":
                plug_list[4]=1   
                
        elif Plug_Type=="AC Schuko":
                plug_list[5]=1
                
        elif Plug_Type=="AC CEE 5 polig":
                plug_list[6]=1
                
        row_topredict=row_topredict+plug_list
        
        row_topredict.append(year)
        
        
        pickled_model = pickle.load(open('kmeans_model.pkl', 'rb'))
        class_predictions = pickled_model.predict([[float(Langengrad),float(Breitengrad)]])
        
        cluster_list=[0,0,0,0,0]
        cluster_list[class_predictions[0]]=1
        row_topredict=row_topredict+cluster_list
                
        if a=="Normalladeeinrichtung":
                pickled_model_2 = pickle.load(open('model_LG.pkl', 'rb'))
                class_predictions = pickled_model_2.predict([row_topredict])
                if class_predictions[0]==1:
                        st.success(str("Normalladeeinrichtung"))        
                else:
                        st.success(str("Schnellladeeinrichtung"))        

        else:
                pickled_model_2 = pickle.load(open('model.pkl', 'rb'))
                class_predictions = pickled_model_2.predict([row_topredict])        
                st.success(str(predicted_label[class_predictions[0]]))        

def recommendations():

    data = pd.read_csv('Ladesaeulenregister_CSV.csv', skiprows=10, sep=';', decimal=',',
                     encoding="ISO-8859-1", engine='python')
    data2=data.copy()


    st.subheader("# new chargers per year")

    year_list=[]

    for i in data2['Inbetriebnahmedatum'].tolist():
       year_list.append(int(i.split('.')[2]))


    data2['year']=year_list
    n = data2[data2['year']>2000]['year'].value_counts(ascending=True).tolist()
    w = data2[data2['year']>2000]['year'].value_counts(ascending=True).index.tolist()

    table_MN = pd.read_html('https://www.citypopulation.de/en/germany/cities/')

    df_state=table_MN[0][['Name','Area A (km²)','Population Estimate (E) 2021-12-31']]
    df_city=table_MN[2][['Name','Population Estimate (E) 2021-12-31','Area']]
    
    df_state['Name'] =  df_state['Name'].apply(lambda x:x.split(' (')[0])
    df_state['Name'] =  df_state['Name'].apply(lambda x:x.split(' [')[0])

    df_city['Name'] = df_city['Name'].apply(lambda x:x.split(' (')[0])
    df_city['Name'] = df_city['Name'].apply(lambda x:x.split(' [')[0])

    data3_state=data2['Bundesland'].value_counts().rename_axis('Bundesland').reset_index(name='counts').merge(df_state, how='inner', left_on="Bundesland", right_on="Name")
    data3_ort=data2['Ort'].value_counts().rename_axis('Ort').reset_index(name='counts').merge(df_city, how='inner', left_on="Ort", right_on="Name")
            
    st.subheader("How many person per charger")
    data3_ort['people/charger']=data3_ort["Population Estimate (E) 2021-12-31"]/data3_ort["counts"]

    st.write(data3_ort[['Ort','people/charger']].sort_values(['people/charger'],ascending=True))
    st.success(df["people/charger"].mean())
        
    st.subheader("How many charger per km²")
    data3_ort['charger/km2']=data3_ort["counts"]/data3_ort["Area"]

    st.write(data3_ort[['Ort','charger/km2']].sort_values(['charger/km2'],ascending=False))


page_names_to_funcs = {
"Data Visualization": data_visualization,
"Prediction": prediction,
"Recommendations": recommendations    
}

demo_name = st.sidebar.selectbox("Choose the App", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
