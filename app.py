# Core Packages
import streamlit as st
import altair as alt
import plotly.express as px

# EDA Packages
import pandas as pd
import numpy as np
from datetime import datetime

# Load Models
import joblib 
pipe_lr1 = joblib.load(open("./models/japan_svm.pkl","rb"))
pipe_lr2 = joblib.load(open("./models/korean_svm.pkl","rb"))
pipe_lr3 = joblib.load(open("./models/emotion_classifier_pipe_lr.pkl","rb"))
pipe_lr4 = joblib.load(open("./models/filipino svm.pkl","rb"))
pipe_lr5 = joblib.load(open("./models/spanish svm.pkl","rb"))

# Track Utils
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table

# Function
def predict_emotions(docx, model):
    results = model.predict([docx])
    probabilities = model.predict_proba([docx])[0] * 100  # Convert probabilities to percentage
    max_probability_index = np.argmax(probabilities)
    return results[0], probabilities[max_probability_index]

def get_prediction_proba(docx, model):
    results = model.predict_proba([docx])
    return results

# Main Application
def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    create_emotionclf_table()
    if choice == "Home":
        add_page_visited_details("Home", datetime.now())
        st.subheader("Emotion Detection in Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            model_choice = st.selectbox("Select Model", ["Japanese", "Korean", "English","Filipino", "Spanish"])
            submit_text = st.form_submit_button(label = 'Submit')

        if submit_text:
            col1, col2  = st.columns(2)

            # Apply Function Here
            if model_choice == "Japanese":
                model = pipe_lr1
            elif model_choice == "Korean":
                model = pipe_lr2
            elif model_choice == "Filipino":
                model = pipe_lr4
            elif model_choice == "Spanish":
                model = pipe_lr5        
            else:
                model = pipe_lr3

            pred, confidence = predict_emotions(raw_text, model)
            probability = get_prediction_proba(raw_text, model)

            add_prediction_details(raw_text, pred, confidence, datetime.now())

            with col1:
                st.success("Original Text")
                st.header(raw_text)

                st.success("Prediction")
                st.header(pred)
                st.header("Confidence:{:.2f}%".format(confidence))

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns = model.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                text_size = 18 # Set the text size here

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x = alt.X('emotions', axis=alt.Axis(labelFontSize=text_size)),
                    y = alt.Y('probability', axis=alt.Axis(labelFontSize=text_size)),
                    color = 'emotions',
                    text = alt.Text('probability', format=".2f")
                )
                st.altair_chart(fig,use_container_width = True)

    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now())
        st.subheader("Monitor App")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns = ['Page Name', 'Time of Visit'])
            st.dataframe(page_visited_details)  

            pg_count =page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name = 'Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x = 'Page Name', y = 'Counts', color = 'Page Name')
            st.altair_chart(c,use_container_width = True)   

        with st.expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns = ['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name = 'Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x = 'Prediction', y = 'Counts', color = 'Prediction')
            st.altair_chart(pc, use_container_width = True) 

    else:
        add_page_visited_details("About", datetime.now())

        st.write("Welcome to the Emotion Detection in Text App! This application utilizes the power of natural language processing and machine learning to analyze and identify emotions in textual data.")
        
        st.subheader("Our Mission")

        st.write("At Emotion Detection in Text, our mission is to provide a user-friendly and efficient tool that helps individuals and organizations understand the emotional content hidden within text. We believe that emotions play a crucial role in communication, and by uncovering these emotions, we can gain valuable insights into the underlying sentiments and attitudes expressed in written text.")
        
        st.subheader("How It Works")
        
        st.write("When you input text into the app, our system processes it and applies advanced natural language processing algorithms to extract meaningful features from the text. These features are then fed into the trained model, which predicts the emotions associated with the input text. The app displays the detected emotions, along with a confidence score, providing you with valuable insights into the emotional content of your text.")
        
        st.subheader("Key Features:")
        
        st.markdown("##### 1. Real-time Emotion Detection")
        
        st.write("Our app offers real-time emotion detection, allowing you to instantly analyze the emotions expressed in any given text. Whether you're analyzing customer feedback, social media posts, or any other form of text, our app provides you with immediate insights into the emotions underlying the text.")
        
        st.markdown("##### 2. Confidence Score")
        
        st.write("Alongside the detected emotions, our app provides a confidence score, indicating the model's certainty in its predictions. This score helps you gauge the reliability of the emotion detection results and make more informed decisions based on the analysis.")
        
        st.markdown("##### 3. User-friendly Interface")
        
        st.write("We've designed our app with simplicity and usability in mind. The intuitive user interface allows you to effortlessly input text, view the results, and interpret the emotions detected. Whether you're a seasoned data scientist or someone with limited technical expertise, our app is accessible to all.")
        
        st.subheader("Applications")
        
        st.markdown("""
          The Emotion Detection in Text App has a wide range of applications across various industries and domains. Some common use cases include:
          - Social media sentiment analysis
          - Customer feedback analysis
          - Market research and consumer insights
          - Brand monitoring and reputation management
          - Content analysis and recommendation systems
          """)
if __name__ == '__main__':
    main()
