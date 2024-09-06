import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# تحميل النموذج
with open('log_reg_model.pkl', 'rb') as file:
    log_reg = pickle.load(file)

# عنوان التطبيق
st.title('Diabetes Prediction App')

# إضافة صورة للتطبيق
st.image('https://example.com/your-image.jpg', caption='Diabetes Prediction', use_column_width=True)

# وصف التطبيق
st.markdown("""
Welcome to the Diabetes Prediction App! Use this tool to predict whether a person is likely to have diabetes based on their health metrics. 

Please input the following details:
""")

# إدخال بيانات من المستخدم
glucose = st.number_input('Enter Glucose Level', min_value=0, max_value=300, value=120)
bmi = st.number_input('Enter BMI', min_value=0.0, max_value=60.0, value=25.0, format="%.1f")
insulin = st.number_input('Enter Insulin Level', min_value=0, max_value=900, value=80)
dpf = st.number_input('Enter Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, format="%.2f")

# عند الضغط على الزر
if st.button('Predict'):
    # تحويل المدخلات إلى مصفوفة
    input_data = np.array([[glucose, bmi, insulin, dpf]])
    
    # التعامل مع PolynomialFeatures كما في التدريب
    poly = PolynomialFeatures(degree=2)
    input_data_poly = poly.fit_transform(input_data)
    
    # التنبؤ
    prediction = log_reg.predict(input_data_poly)
    
    # عرض النتيجة
    if prediction[0] == 1:
        st.success("The person is likely to have diabetes.")
    else:
        st.success("The person is unlikely to have diabetes.")

# إضافة ملاحظة أسفل الصفحة
st.markdown("""
*This app is designed to help you assess diabetes risk based on health metrics. Please consult a medical professional for accurate diagnosis and treatment.*
""")
