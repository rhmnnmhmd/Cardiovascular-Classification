# LIBRARY IMPORTS-------------------------------------------------------
import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# CARDIO PREDICTOR------------------------------------------------------------------------------
# import trained model
MODEL = os.path.join(os.getcwd(), 'model', 'best_pipe_cardio.pkl')
with open(MODEL, 'rb') as file:
    best_pipe = pickle.load(file)

# read cleaned data
CLEANED_DATA = os.path.join(os.getcwd(), 'data', 'df_cleaned.csv')
df = pd.read_csv(CLEANED_DATA, 
                 usecols=['age', 'gender', 'height', 'weight', 'sys', 'dia', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio'])

TRANS_DATA = os.path.join(os.getcwd(), 'data', 'df_transformed.csv')
df_2 = pd.read_csv(TRANS_DATA, 
                   usecols=['age', 'gender', 'height', 'weight', 'sys', 'dia', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio'])

st.title('Cardiovascular Disease Detector')

st.header('Background')

st.markdown('''As of 2020, according to the data published by the World Health Organization (WHO), 
coronary heart disease (an example of cardiovascular disease) is the #1 leading cause of death in Malaysia. 
The number of death caused by it reached 36,729 deaths amounting to 21.86% of the total deaths in Malaysia. 
This puts Malaysia as the rank #61 country with the highest age-adjusted death rate (136 per 100,000 population).
[Source](https://www.worldlifeexpectancy.com/malaysia-coronary-heart-disease
)''')

st.markdown('''Due to this, experts from various fields especially the medical field have been studying this particular 
disease extensively to improve on the current prevention/treatment measures or to find the 'cure' for it. In terms of the 'detection' 
of the disease itself, patients need to be diagnosed by a medical professional especially specializing in cardiovascular diseases. 
However, this method of detection might be inaccessible or unaffordable for some people.''')

st.markdown('''Because of the aforementioned problem, the **Cardiovascular Disease Predictor** in this website tries to provide 
an additional (NOT A REPLACEMENT) method for cardiovascular disease detection using one of the most widely-known machine learning 
(classification) model, Logistic Regression. The model was trained with 59,000+ patient data before deployment 
[Original data source](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset). 
The user needs to input some of their personal/medical details such as age, gender, and blood pressure. As the output, 
the user will be presented with a sentence stating whether they are "highly likely" or "less likely" to get cardiovascular 
disease alongside its probability/likelihood.''')

st.markdown('''In the section below, you can click the **Show Me!** button below to display some information (e.g. data table and charts) 
about the data that was used for training the logistic regression model. If your're not interested in that, you can just leave it 
unchecked and just use Cardiovascular Disease Detector (CDD) tool.''')

st.header('Patients Health Data')

# to show the dataframe
checkbox = st.checkbox('Show me data table and the charts!')

if checkbox:
    df_2

    # basic figure, axes creation
    fig1, ax1 = plt.subplots(2, 2, sharex=False, figsize=(10, 10))

    # set title of the Fig object
    fig1.suptitle('Distribution of Continuous Data')

    # histogram of height
    ax1[0, 0].set_title('Distribution of Height')
    sns.distplot(df['height'], ax=ax1[0, 0])

    # histogram of weight
    ax1[0, 1].set_title('Distribution of Weight')
    sns.distplot(df['weight'], ax=ax1[0, 1])

    # histogram of age
    ax1[1, 0].set_title('Distribution of Age')
    sns.distplot(df['age'], ax=ax1[1, 0])

    # histogram of sys
    ax1[1, 1].set_title('Distribution of Systolic/Diastolic Blood Pressure')
    sns.histplot(data=df, x='sys', y='dia', ax=ax1[1, 1], bins=40)
    plt.style.use('dark_background')

    # to plot/show the figure (with the axes)
    st.subheader(
        'Distribution of Height, Weight, Age, and Systolic Blood Pressure')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(2, 2, figsize=(10, 10))

    # set title of the Fig object
    fig2.suptitle('Distribution of Categorical Data')

    sns.countplot(data=df_2, x='cardio', order=df_2['cardio'].value_counts(
        ascending=False).index, ax=ax2[0, 0])
    abs_values1 = df['cardio'].value_counts(ascending=False)
    rel_values1 = df['cardio'].value_counts(
        ascending=False, normalize=True).values*100
    lbls1 = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values1, rel_values1)]
    ax2[0, 0].bar_label(container=ax2[0, 0].containers[0], labels=lbls1)

    sns.countplot(data=df_2, x='gender', order=df_2['gender'].value_counts(
        ascending=False).index, ax=ax2[0, 1])
    abs_values2 = df['gender'].value_counts(ascending=False)
    rel_values2 = df['gender'].value_counts(
        ascending=False, normalize=True).values*100
    lbls2 = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values2, rel_values2)]
    ax2[0, 1].bar_label(container=ax2[0, 1].containers[0], labels=lbls2)

    sns.countplot(data=df_2, x='cholesterol', order=df_2['cholesterol'].value_counts(
        ascending=False).index, ax=ax2[1, 0])
    abs_values3 = df['cholesterol'].value_counts(ascending=False)
    rel_values3 = df['cholesterol'].value_counts(
        ascending=False, normalize=True).values*100
    lbls3 = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values3, rel_values3)]
    ax2[1, 0].bar_label(container=ax2[1, 0].containers[0], labels=lbls3)

    sns.countplot(data=df_2, x='gluc', order=df_2['gluc'].value_counts(
        ascending=False).index, ax=ax2[1, 1])
    abs_values4 = df['gluc'].value_counts(ascending=False)
    rel_values4 = df['gluc'].value_counts(
        ascending=False, normalize=True).values*100
    lbls4 = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values4, rel_values4)]
    ax2[1, 1].bar_label(container=ax2[1, 1].containers[0], labels=lbls4)

    # to plot/show the figure (with the axes)
    st.subheader('Distribution of Categorical Features')
    st.pyplot(fig2)

st.header("Cardiovascular Disease Detector (CDD)")

with st.form("form_1"):
    st.write("Please Insert Relevant Details")
    age = int(st.slider('Select your age', min_value=0,
              max_value=100, value=50, step=1))

    gender = st.selectbox('Select your gender',
                          options=df_2['gender'].unique())

    height = float(st.number_input('Input your height (cm)'))

    weight = float(st.number_input('Input your weight (kg)'))

    sys = int(st.number_input('Insert your systolic blood pressure (mm Hg)',
              help='Usually the larger number from your BP tool'))

    dia = int(st.number_input('Insert your diastolic blood pressure (mm Hg)',
              help='Usually the smaller number from your BP tool'))

    cholesterol = st.selectbox(
        'Select your cholesterol level', options=df_2['cholesterol'].unique())

    gluc = st.selectbox('Select your sugar level',
                        options=df_2['gluc'].unique())

    smoke = st.selectbox('Do you smoke?', options=df_2['smoke'].unique())

    alco = st.selectbox('Do you drink alcohol?', options=df_2['alco'].unique())

    active = st.selectbox('Do you consider yourself active?',
                          options=df_2['active'].unique())

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")

    if submitted:
        st.write('Your details are:')
        st.write('age', age, 'gender', gender, 'height', height,
                 'weight', weight, 'sys', sys, 'dia', dia,
                 'cholesterol', cholesterol, 'glucose', gluc, 'smoke', smoke,
                 'alcohol', alco, 'active', active)

        gender_mapped = 1 if gender == 'female' else 2

        if cholesterol == 'normal':
            chol_mapped = 1.0
        elif cholesterol == 'above normal':
            chol_mapped = 2.0
        elif cholesterol == 'well above normal':
            chol_mapped = 3.0

        if gluc == 'normal':
            gluc_mapped = 1.0
        elif gluc == 'above normal':
            gluc_mapped = 2.0
        elif gluc == 'well above normal':
            gluc_mapped = 3.0

        smoke_mapped = 1 if smoke == 'yes' else 0

        alco_mapped = 1.0 if alco == 'yes' else 0.0

        active_mapped = 1 if active == 'yes' else 0

        input_row = np.array([age, gender_mapped, height, weight,
                             sys, dia, chol_mapped, gluc_mapped, smoke_mapped, alco_mapped, active_mapped])
        input_data = input_row.reshape(1, -1)

        y_pred = best_pipe.predict(input_data)

        proba = best_pipe.predict_proba(input_data)

        result = {0: 'Less likely to have CV disease',
                  1: 'Highly likely to have CV disease'}

        st.write(
            f'Prediction: {result[y_pred[0]]}. Probability of having CV: {proba[0, -1]*100: .4f}\%')

# TEMPLATE 1-------------------------------------------------------
# st.markdown("# Main page 🎈")
# # st.sidebar.markdown("# mavwdsvsZVin page 🎈")

# # header title
# st.title('Uber pickups in New York City')

# # get data
# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#             'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# @st.cache
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     def lowercase(x): return str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# # Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')
# # Load 10,000 rows of data into the dataframe.
# data = load_data(10000)
# # Notify the reader that the data was successfully loaded.
# data_load_state.text("Done! (using st.cache)")

# # inspect data
# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)

# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(
#     data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
# st.bar_chart(hist_values)

# st.subheader('Map of all pickups')
# st.map(data)

# # hour_to_filter = 17
# # min: 0h, max: 23h, default: 17h
# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
# st.subheader(f'Map of all pickups at {hour_to_filter}:00')
# st.map(filtered_data)


# TEMPLATE 2-----------------------------------------------------------------------
# if st.checkbox('Show dataframe'):
#     chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
#     chart_data

# df = pd.DataFrame({'first column': [1, 2, 3, 4],
#                    'second column': [10, 20, 30, 40]})
# option = st.selectbox('Which number do you like best?', df['second column'])
# 'You selected: ', option

# Add a selectbox to the sidebar:
# add_selectbox = st.sidebar.selectbox(
#     'How would you like to be contacted?', ('Email', 'Home phone', 'Mobile phone'))
# # Add a slider to the sidebar:
# add_slider = st.sidebar.slider(
#     'Select a range of values', 0.0, 100.0, (25.0, 75.0))

# left_column, right_column = st.columns(2)
# # You can use a column just like st.sidebar:
# left_column.button('Press me!')
# # Or even better, call Streamlit functions inside a "with" block:
# with right_column:
#     chosen = st.radio('Sorting hat', ("Gryffindor",
#                       "Ravenclaw", "Hufflepuff", "Slytherin"))
#     st.write(f"You are in {chosen} house!")

# 'Starting a long computation...'
# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)
# for i in range(5):
#     # Update the progress bar with each iteration.
#     latest_iteration.text(f'Iteration {i+1}')
#     bar.progress(i + 1)
#     time.sleep(0.1)
# '...and now we\'re done!'

# dataframe with 20 rows, 3 columns. values filled with random numbers
# chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])

# show 'Area Chart'
# st.write('Area Chart')

# to display area chart
# st.area_chart(chart_data)


# create a button
# if st.button('Say hello'):          # if press button, will display the text and create falling snow
#     st.write('Why hello there')
#     st.snow()
# else:           # else case
#     st.write('Goodbye')

# create a form
# with st.form("my_form"):
#     st.write("Inside the form")
#     slider_val = st.slider("Form slider")
#     checkbox_val = st.checkbox("Form checkbox")
#     number = st.number_input('Insert a number')

#     # Every form must have a submit button.
#     submitted = st.form_submit_button("Submit")
#     if submitted:
#         st.write("slider", slider_val, "checkbox", checkbox_val, 'age', number)

# st.write("Outside the form")