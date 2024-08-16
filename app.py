# import streamlit as st
# import pickle
# from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
# import pandas
# import numpy as np
# # import the model
# with open('pipe1.pkl', 'rb') as file:
#     # Load the object from the file
#     pipe = pickle.load(file)
# import pandas as pd
#
# # load the df
# import pickle
#
# # try:
# #     # Attempt to load the pickled object from the file
# #     with open('df2.pkl', 'rb') as file:
# #         df = pickle.load(file)
# #     print("Pickled object loaded successfully!")
# # except Exception as e:
# #     print("Error occurred while loading the pickled object:", e)
#
#
# # #load the df
# # df=pickle.load('df1.pkl', 'rb')
# with open('df1.pkl', 'rb') as file:
#     df = pickle.load(file)
# st.title('Laptop Price Predictor')
#
# # i will ask konse brand ka laptop
# company=st.selectbox('Brand', df['Company'].unique())
# # select box m ek box ayga jiski heading Brand hoga or usme values uski unique
#
#
# # type of laptop
# type=st.selectbox('Type of Laptop',df['TypeName'].unique())
#
#
# # Ram
# ram=st.selectbox('Ram(in GB', [2,4,5,12,16,32,64])
#
# #weight in numbers
# weight=st.number_input('Weight of the laptop ')
#
# # touchscreen
# touchscreen=st.selectbox('TouchScreen', ['No', 'Yes'])
#
# # ips display
# ips=st.selectbox('Ips_Display', ['No','Yes'])
#
# # ppi for ppi we wna to input scrren res and size
# screen_size=screen_size=st.number_input('Screen_size')
#
# # res see yhi kuch def sizes hai to just give options
# resolution=resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
#
# # cpu
# cpu=st.selectbox('CPU', df['Cpu_Brand'].unique())
#
# # harddrive
# hdd=st.selectbox('HDD(in GB)',[0,128,256,512, 1024, 2048])
#
#
# ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
#
# gpu = st.selectbox('GPU',df['Gpu_Brand'].unique())
#
# os = st.selectbox('OS',df['OS'].unique())
#
# if st.button('Predict Price'):
#     ppi=None
#     # cpu yes or no means 1 or 0 so convertthem
#     if touchscreen=='Yes':
#         touchscreen=1
#     else:
#         touchscreen=0
#     if ips=='Yes':
#         ips=1
#     else:
#         ips=0
#     X_res=int(resolution.split('x')[0])
#     y_res=int(resolution.split('x')[1])
#     ppi=(X_res**2)+(y_res**2)/screen_size
#     # we will do differentiate resolutin on x and y liem we did in jup
#
#
#     query=np.array([company, type, ram,weight, touchscreen, ips, ppi, cpu, hdd,ssd,gpu, os])
#     query=query.reshape(1,12)
#
#     st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
#     # np.exp bcox hmne log m convert kra tha
#
import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe_final1.pkl','rb'))
df = pickle.load(open('df_final.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',[0,1])

# IPS
ips = st.selectbox('IPS',[0,1])

# screen size
screen_size = st.slider('Screensize in inches', 10.0, 18.0, 13.0)
#
# # resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
#ppi=st.number_input('PPI', df['PPI'],[226.983005,127.677940,141.211998,220.534624,226.983005])
#cpu
cpu = st.selectbox('CPU',df['Cpu_Brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu_Brand'].unique())

os = st.selectbox('OS',df['OS'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1, 12)  # Reshape to 2D array



    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
