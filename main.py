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

from collections import defaultdict
#Creating Map
import folium
from folium.plugins import MarkerCluster
from folium import IFrame
from streamlit_folium import st_folium

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


def prediction():
    st.header("Prediction")

    data = pd.read_csv('Ladesaeulenregister_CSV.csv', skiprows=10, sep=';', decimal=',',
                     encoding="ISO-8859-1", engine='python')
  

    year_list=[]

    for i in data['Inbetriebnahmedatum'].tolist():
       year_list.append(int(i.split('.')[2]))


    data['year']=year_list
            
    future_filter = st.number_input('How many year to predict', 2, 23)

    df=pd.DataFrame({"year":data['year'].value_counts(ascending=True).index.tolist(),"chargers per year":data['year'].value_counts(ascending=True).tolist()})
                     
    df=df[["year","chargers per year"]].sort_values("year").reset_index().drop("index",axis=1)
    new_column = df[["year","chargers per year"]]
    new_column=new_column[new_column["year"]!=2022]
    new_column=new_column[new_column['year']>2000]['year']
            
    new_column.dropna(inplace=True)
    new_column.columns = ['ds', 'y']
    clicked=st.button('Predict')
    if clicked:
        n = NeuralProphet()
        model = n.fit(new_column, freq='Y')
        future = n.make_future_dataframe(new_column, periods=future_filter+1)
        forecast = n.predict(future)
        forecast['yhat1']=forecast['yhat1'].apply(lambda x:int(x))
        forecast=forecast[['ds','yhat1']]
        forecast.columns=['year',column_to_predict]
        forecast['year']=forecast['year'].apply(lambda x:str(x).split("-")[0])
        st.write(forecast[1:])

    
page_names_to_funcs = {
"Data Visualization": data_visualization,
"Prediction": prediction
}

demo_name = st.sidebar.selectbox("Choose the App", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
