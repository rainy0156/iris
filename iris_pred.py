import streamlit as st
import joblib
st.title('IRIS品種預測')
svc = joblib.load('svc_model.joblib')
rf = joblib.load('rf_model.joblib')
lr = joblib.load('lr_model.joblib')
knn = joblib.load('knn_model.joblib')
s1 = st.sidebar.selectbox('請選擇模型:',('svc','RandomForest','logisticRegression','knn'))
if s1 == 'svc':
    model = svc
elif s1 == 'RandomForest':
    model = rf
elif s1 == 'logisticRegression':
    model = lr
elif s1 == 'knn':
    model = knn
se1 = st.slider('花萼長度:',3.0,8.0,5.8)
se2 = st.slider('花萼寬度:',1.8,5.0,3.8)
se3 = st.slider('花瓣長度:',0.7,7.2,5.0)
se4 = st.slider('花瓣寬度:',0.0,3.5,1.8)
st.image('iris.png')
labels = ['setosa', 'versicolor', 'virginica']
if st.button('進行預測'):
    X = [[se1,se2,se3,se4]]
    y = model.predict(X)
    st.write(y[0])
    st.write('### 預測結果,品種名稱應是:',labels[y[0]])