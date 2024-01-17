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
st.title('Lung Cancer Prediction',anchor=False)

df = pd.read_csv(r'cancer patient data sets.csv')

st.header("For the prediction of Lung Cancer, please enter the following details:",anchor=False)
st.caption("Note: All the fields are mandatory and set the values as per your medical reports.")

x1=st.slider("Age", 1, 100, 1)
st.divider()
x2=st.radio("Gender",[1,2])
st.write('''1 - Male
         \n2 - Female''')
st.divider()
x3=st.slider("Level of Air Pollution", df['Air Pollution'].min(), df['Air Pollution'].max(), 1)
st.caption("Air pollution is a mixture of solid particles and gases in the air. Car emissions, chemicals from factories, dust, pollen and mold spores may be suspended as particles. Ozone, a gas, is a major part of air pollution in cities. When ozone forms air pollution, it's also called smog.")
st.divider()
x4=st.slider("Level of Alcohol use", df['Alcohol use'].min(), df['Alcohol use'].max(), 1)
st.caption("Alcohol use disorder (which includes a level that's sometimes called alcoholism) is a pattern of alcohol use that involves problems controlling your drinking, being preoccupied with alcohol, continuing to use alcohol even when it causes problems, having to drink more to get the same effect, or having withdrawal symptoms when you rapidly decrease or stop drinking.")
st.divider()
x5=st.slider("Level of Dust Allergy", df['Dust Allergy'].min(), df['Dust Allergy'].max(), 1)
st.caption("Dust mite allergy is an allergic reaction to tiny bugs that commonly live in house dust. Signs of dust mite allergy include those common to hay fever, such as sneezing and runny nose. Many people with dust mite allergy also experience signs of asthma, such as wheezing and difficulty breathing.")
st.divider()
x6=st.slider("Level of OccuPational Hazards", df['OccuPational Hazards'].min(), df['OccuPational Hazards'].max(), 1)
st.caption("Occupational hazards are any dangers that are posed to a worker while performing a prescribed task. These hazards include exposure to chemicals, noise, extreme temperatures or radiation and repetitive motions.")
st.divider()
x7=st.slider("Level of Genetic Risk", df['Genetic Risk'].min(), df['Genetic Risk'].max(), 1)
st.caption("Genetic risk factors are changes or differences in genes that can influence the chance of getting a disease. Having a genetic risk factor does not mean that you will definitely get a disease. Instead, it means that you have a higher chance of getting that disease than someone who doesn't have that risk factor.")
st.divider()
x8=st.slider("Level of chronic Lung Disease", df['chronic Lung Disease'].min(), df['chronic Lung Disease'].max(), 1)
st.caption("Chronic lung disease is the term for a range of conditions that affect the lungs. These conditions include asthma and chronic obstructive pulmonary disease (COPD). COPD includes emphysema and chronic bronchitis. These conditions can affect the airways, lung tissues, and circulation of blood through your lungs.")
st.divider()
x9=st.slider("Level of Balanced Diet", df['Balanced Diet'].min(), df['Balanced Diet'].max(), 1)
st.caption("A balanced diet is a diet that contains differing kinds of foods in certain quantities and proportions so that the requirement for calories, proteins, minerals, vitamins and alternative nutrients is adequate and a small provision is reserved for additional nutrients to endure the short length of leanness.")
st.divider()
x10=st.slider("Level of Obesity", df['Obesity'].min(), df['Obesity'].max(), 1)
st.caption("Obesity is a complex disease involving an excessive amount of body fat. Obesity isn't just a cosmetic concern. It is a medical problem that increases your risk of other diseases and health problems, such as heart disease, diabetes, high blood pressure and certain cancers.")
st.divider()
x11=st.slider("Level of Smoking", df['Smoking'].min(), df['Smoking'].max(), 1)
st.caption("Smoking is a practice in which a substance is burned and the resulting smoke is breathed in to be tasted and absorbed into the bloodstream. Most commonly, the substance used is the dried leaves of the tobacco plant, which have been rolled into a small rectangle of rolling paper to create a small, round cylinder called a cigarette.")
st.divider()
x12=st.slider("Level of Passive Smoker", df['Passive Smoker'].min(), df['Passive Smoker'].max(), 1)
st.caption("Passive smoking means breathing in other people's tobacco smoke.")
st.divider()
x13=st.slider("Level of Chest Pain", df['Chest Pain'].min(), df['Chest Pain'].max(), 1)
st.caption("Chest pain is discomfort, typically in the front of the chest. It may be described as sharp, dull, pressure, heaviness or squeezing. Associated symptoms may include pain in the shoulder, arm, upper abdomen or jaw or nausea, sweating or shortness of breath.")
st.divider()
x14=st.slider("Level of Coughing of Blood", df['Coughing of Blood'].min(), df['Coughing of Blood'].max(), 1)
st.caption("Coughing up blood can be alarming, but isn't usually a sign of a serious problem if you're young and otherwise healthy. It's more a cause for concern in older people, particularly those who smoke.")
st.divider()
x15=st.slider("Level of Fatigue", df['Fatigue'].min(), df['Fatigue'].max(), 1)
st.caption("Fatigue is a term used to describe an overall feeling of tiredness or lack of energy. It isn't the same as simply feeling drowsy or sleepy. When you're fatigued, you have no motivation and no energy. Being sleepy may be a symptom of fatigue, but it's not the same thing.")
st.divider()
x16=st.slider("Level of Weight Loss", df['Weight Loss'].min(), df['Weight Loss'].max(), 1)
st.caption("Weight loss is a decrease in body weight resulting from either voluntary (diet, exercise) or involuntary (illness) circumstances. Most instances of weight loss arise due to the loss of body fat, but in cases of extreme or severe weight loss, protein and other substances in the body can also be depleted.")
st.divider()
x17=st.slider("Level of Shortness of Breath", df['Shortness of Breath'].min(), df['Shortness of Breath'].max(), 1)
st.caption("Shortness of breath means you feel like you can't get enough air. You may feel as if your breathing is forced, labored, or that you have to work harder to breathe.")
st.divider()
x18=st.slider("Level of Wheezing", df['Wheezing'].min(), df['Wheezing'].max(), 1)
st.caption("Wheezing is a high-pitched whistling sound made while you breathe. It's heard most clearly when you exhale, but in severe cases, it can be heard when you inhale. It's caused by narrowed airways or inflammation.")
st.divider()
x19=st.slider("Level of Swallowing Difficulty", df['Swallowing Difficulty'].min(), df['Swallowing Difficulty'].max(), 1)
st.caption("Difficulty swallowing (dysphagia) means it takes more time and effort to move food or liquid from your mouth to your stomach. Dysphagia may also be associated with pain. In some cases, swallowing may be impossible.")
st.divider()
x20=st.slider("Level of Clubbing of Finger Nails", df['Clubbing of Finger Nails'].min(), df['Clubbing of Finger Nails'].max(), 1)
st.caption("Clubbing is a thickening and widening of the fingertips or toes due to an increase in the tissue below the nail. Clubbing occurs in stages. Early stages involve swelling of the tissue, which makes the nail and skin look red. Later, the nail shape changes, making the nail curve downward.")
st.divider()
x21=st.slider("Level of Frequent Cold", df['Frequent Cold'].min(), df['Frequent Cold'].max(), 1)
st.caption("A cold is a viral infection of the upper respiratory tract — the nose and throat. A cold is often called a viral upper respiratory tract infection. It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold.")
st.divider()
x22=st.slider("Level of Dry Cough", df['Dry Cough'].min(), df['Dry Cough'].max(), 1)
st.caption("A dry cough is a type of cough that doesn't bring up phlegm. Dry coughs are often short-lived and rarely a cause for concern. However, a chronic dry cough can be a symptom of an underlying condition.")
st.divider()
x23=st.slider("Level of Snoring", df['Snoring'].min(), df['Snoring'].max(), 1)
st.caption("Snoring is the vibration of respiratory structures and the resulting sound due to obstructed air movement during breathing while sleeping. In some cases, the sound may be soft, but in most cases, it can be loud and unpleasant. Snoring during sleep may be a sign, or first alarm, of obstructive sleep apnea (OSA).")
st.divider()
x=[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23]
if st.button("Predict") :
    res = model.predict(x)
    
    st.subheader("The predicted level of Lung Cancer is:",anchor=False)
    if res==1:
        st.write("Low")
    elif res==2:
        st.write("Medium")
    else:
        st.write("High")