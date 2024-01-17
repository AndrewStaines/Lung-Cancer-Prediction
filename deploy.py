# Import necessary libraries
import streamlit as st
from catboost import  CatBoostClassifier

import pandas as pd

method = CatBoostClassifier()

# Load your model
model = method.load_model('Cancer_model', format='cbm')

st.set_page_config(initial_sidebar_state="collapsed")
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)
# Streamlit app
st.title('Lung Cancer Prediction',anchor='False')

# st.dataframe('cancer patient data sets.csv')

tab1, tab2, tab3 = st.tabs(["About Disease", "Risk", "Prevention"])

with tab1:
   
   st.subheader("Lung Cancer",divider='red',anchor=False)
   st.write('''Lung cancer is a type of cancer that begins in the lungs, usually in the cells lining the air passages. It is one of the most common cancers worldwide and is a leading cause of cancer-related deaths. There are two main types of lung cancer: non-small cell lung cancer (NSCLC) and small cell lung cancer (SCLC). These types are based on the microscopic appearance of the cancer cells and their behavior.

Non-Small Cell Lung Cancer (NSCLC): This is the more common type, accounting for about 85% of lung cancers. NSCLC includes several subtypes, such as adenocarcinoma, squamous cell carcinoma, and large cell carcinoma.

Small Cell Lung Cancer (SCLC): This type is less common but tends to grow and spread more quickly than NSCLC. SCLC is often associated with a history of smoking.

Risk factors for lung cancer include smoking (the leading cause), exposure to secondhand smoke, exposure to certain occupational carcinogens (such as asbestos and radon), a family history of lung cancer, and a history of certain lung diseases.

Symptoms of lung cancer can vary but may include persistent cough, chest pain, shortness of breath, hoarseness, unexplained weight loss, and coughing up blood. Early-stage lung cancer may not cause noticeable symptoms, making early detection challenging.

Diagnosis often involves imaging studies (like chest X-rays and CT scans), biopsy, and other tests to determine the type and stage of lung cancer. Treatment options may include surgery, chemotherapy, radiation therapy, targeted therapy, and immunotherapy, depending on the specific characteristics of the cancer and its stage.

Prevention is a key aspect of lung cancer control, and quitting smoking is the most effective way to reduce the risk. Early detection and advancements in treatment have improved outcomes, but the prognosis for lung cancer can vary widely depending on factors such as the stage at diagnosis and overall health.''')

with tab2:
   st.subheader("Risk",divider='red',anchor=False)
   st.write('''Lung cancer can have severe and life-threatening effects on the body. The impact of lung cancer can vary depending on factors such as the type of lung cancer, its stage at diagnosis, and the overall health of the individual. Here are some of the dangerous effects associated with lung cancer:

Metastasis: Lung cancer has a high potential to spread (metastasize) to other parts of the body, such as the bones, liver, brain, and other organs. This can lead to the formation of secondary tumors and complicate treatment.

Respiratory Complications: Lung cancer can affect the normal functioning of the lungs, leading to respiratory complications. This may result in difficulty breathing, chronic cough, and an increased risk of infections.

Pleural Effusion: Lung cancer can cause the accumulation of fluid in the pleural space, the area between the lung and the chest wall. This condition, known as pleural effusion, can cause breathing difficulties and discomfort.

Cachexia: Advanced lung cancer can lead to cachexia, a condition characterized by severe weight loss, muscle wasting, and weakness. Cachexia negatively affects overall health and can be challenging to manage.

Pain: Lung cancer can cause pain as it progresses. This may be due to the tumor pressing on nerves, invading nearby tissues, or causing inflammation.

Hemoptysis: Some individuals with lung cancer may experience hemoptysis, which is the coughing up of blood. This can be alarming and may indicate that the cancer has invaded blood vessels in the lungs.

Paraneoplastic Syndromes: Lung cancer can sometimes trigger paraneoplastic syndromes, which are rare disorders caused by the immune system's response to the cancer. These syndromes can affect various organs and systems in the body.

Weakened Immune System: Cancer, including lung cancer, can weaken the immune system, making the individual more susceptible to infections and other illnesses.

Thrombosis: Lung cancer increases the risk of blood clots, which can lead to conditions such as deep vein thrombosis (DVT) or pulmonary embolism.

Impact on Quality of Life: Beyond the physical effects, lung cancer and its treatments can have a significant impact on an individual's quality of life, affecting emotional well-being, relationships, and daily activities.

It's crucial to note that advancements in medical treatments and early detection can improve outcomes and manage some of the complications associated with lung cancer. However, the dangerous effects highlight the importance of prevention, early diagnosis, and comprehensive cancer care. Individuals at risk should seek regular medical check-ups and adopt lifestyle changes, such as quitting smoking, to reduce their risk of developing lung cancer.''')

with tab3:
   st.subheader("Prevention",divider='red',anchor=False)
   st.write('''Prevention plays a crucial role in reducing the risk of lung cancer, and many strategies focus on minimizing exposure to known risk factors. Here are some key measures for the prevention of lung cancer:

Quit Smoking: The most effective way to prevent lung cancer is to quit smoking. If you don't smoke, don't start, and if you do smoke, seek help to quit. Smoking is the leading cause of lung cancer and is responsible for the majority of cases.

Avoid Secondhand Smoke: Exposure to secondhand smoke also increases the risk of lung cancer. Encourage a smoke-free environment in your home and workplaces, and support smoke-free public spaces.

Radon Mitigation: Radon is a naturally occurring radioactive gas that can seep into homes. Test your home for radon, and if levels are high, take steps to mitigate the risk.

Occupational Safety: If you work in an industry where exposure to carcinogens like asbestos, arsenic, or certain chemicals is common, follow recommended safety guidelines and use protective equipment to reduce exposure.

Limit Exposure to Carcinogens: Be aware of and minimize exposure to environmental and occupational carcinogens, such as industrial pollutants and certain chemicals.

Healthy Lifestyle Choices: Adopting a healthy lifestyle can contribute to overall well-being and may reduce the risk of cancer. This includes maintaining a balanced diet rich in fruits and vegetables, staying physically active, and managing stress.

Screening for High-Risk Individuals: High-risk individuals, such as current or former heavy smokers, may benefit from lung cancer screening using low-dose computed tomography (LDCT). Talk to your healthcare provider about the appropriateness of screening for your specific situation.

Immunizations: Some viruses, such as the human papillomavirus (HPV), can increase the risk of lung cancer. Staying up-to-date on vaccinations can help prevent infections that may contribute to cancer risk.

Regular Health Check-ups: Attend regular health check-ups and screenings. Early detection and treatment of lung cancer can significantly improve outcomes.

Education and Awareness: Stay informed about the risks of lung cancer and the importance of prevention. Encourage friends, family, and community members to adopt healthy behaviors and make informed choices.

It's important to note that while these preventive measures can significantly reduce the risk of lung cancer, they do not provide an absolute guarantee. Genetic factors and other unknown variables can also play a role. However, adopting a healthy lifestyle and avoiding known risk factors remain crucial in the effort to prevent lung cancer and promote overall health. If you have concerns about your risk of lung cancer, consult with a healthcare professional for personalized advice and guidance.''')
   
df = pd.read_csv(r'bar.csv')  
st.subheader("Death Rate",divider='red',anchor=False)
st.bar_chart(data=df,x='Year',y='Total Number of Deaths')

with st.expander("Predicting Cancer"):
    st.write('''
We have performed data preprocessing and builds a predictive model for cancer patient data using the CatBoostClassifier. It begins by importing essential libraries such as NumPy, Pandas, and Matplotlib. The dataset, presumably containing information about cancer patients, is read from a CSV file into a Pandas DataFrame. The code then inspects the dataset using the info() method to understand its structure and identifies missing values using isnull().sum(). The 'Patient Id' column is dropped as it likely serves as an identifier and does not contribute to predictive modeling. Categorical values in the 'Level' column (representing severity levels like 'Low', 'Medium', 'High') are mapped to numerical values for machine learning compatibility. The resulting DataFrame is then divided into features (x) and the target variable (y). The dataset is split into training and testing sets using the train_test_split function. Finally, a CatBoostClassifier model is instantiated and trained on the training set, with the resulting model saved for future use. The code aims to prepare the data and create a predictive model to classify cancer severity levels based on the provided features.          
    ''')
    st.divider()
    st.write('''
The cancer prediction model implemented in the provided Python code leverages the CatBoostClassifier, a powerful machine learning algorithm designed for categorical feature handling. The model is trained on a dataset containing patient information, specifically focusing on predicting cancer severity levels. Through careful preprocessing steps, including the removal of an unnecessary identifier column ('Patient Id') and the mapping of categorical severity levels ('Low', 'Medium', 'High') to numerical representations, the dataset is prepared for modeling. The CatBoostClassifier is known for its ability to handle categorical variables efficiently and requires minimal feature engineering. With a learning rate set at 0.01, the model is trained on a subset of the data, aiming to learn patterns and relationships that can aid in predicting the severity levels of cancer. By saving the trained model as 'Cancer_model.cbm,' the code ensures that the predictive capabilities of the model can be readily utilized in future scenarios, contributing to early diagnosis and risk assessment in the context of cancer care.
             ''')
    if st.button("Predict",type='primary'):
        st.switch_page("pages/page.py")