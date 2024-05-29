import streamlit as st
from streamlit_lottie import st_lottie
import json
import pygame
from gtts import gTTS
import speech_recognition as sr
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import base64
import pickle

with open('C:/Users/thesc/OneDrive/Desktop/MARC/final/models/svm_model.pkl', 'rb') as svm_model_file: #Replace path with your svm model file path
    svm_classifier = pickle.load(svm_model_file)

with open('C:/Users/thesc/OneDrive/Desktop/MARC/final/models/svm_vectorizer.pkl', 'rb') as vectorizer_file: #Replace path with your svm vectorizer file path
    vectorizer = pickle.load(vectorizer_file)

st.set_page_config(
    page_title="Medical Assistant - MARC",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.markdown("<h1 style='text-align: center; font-size: 3em;'>M.A.R.C</h1>", unsafe_allow_html=True)

custom_css = """
<style>
    .stButton>button {
        color: white;
        padding: 10px 15px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 20px;
        margin: 4px 75px;
        cursor: pointer;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

gif_path = "C:/Users/thesc/OneDrive/Desktop/MARC/final/assets/imp.gif" #Replace path with the imp.gif path from assets folder
with open(gif_path, "rb") as f:
    gif_data = f.read()
    encoded_gif = base64.b64encode(gif_data).decode()

st.sidebar.markdown(
    f"""
    <style>
        .center {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 75vh; /* 75% of the viewport height */
        }}
    </style>
    <div class="center">
        <img src="data:image/gif;base64,{encoded_gif}" alt="gif" width="600" height="400">
    </div>
    """,
    unsafe_allow_html=True,
)

model_path = "C:/Users/thesc/Downloads/llama-2-7b-chat.Q8_0.gguf" #Replace path with the llama2 model file path
callback = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = 50
n_batch = 1024
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.5,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    max_tokens=100,
    top_p=1,
    callback_manager=callback,
    verbose=True,
)

template = """
Answer the following text delimited by triple backticks.
Return your response to the text in brief in a single paragraph.
{text}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
waiting_for_input = True

def get_user_input_from_button(recognizer, button_key="start_button"):
    if st.button("Press to Speak", key=button_key):
        st.write("Listening...")
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source, timeout=8)
                user_input = recognizer.recognize_google(audio)
                return user_input
            except sr.WaitTimeoutError:
                return 0 
            except sr.UnknownValueError:
                return 1
            except sr.RequestError as e:
                return 2
    else:
        return None

def classify_user_input(user_input):
    user_input_vector = vectorizer.transform([user_input])
    
    prediction = svm_classifier.predict(user_input_vector)

    return prediction[0]

def process_user_request(recognizer, button_key):
    global waiting_for_input
    if waiting_for_input:
        user_input = get_user_input_from_button(recognizer, button_key)
        print(user_input)
        
        if user_input:
            st.write(user_input)

            if str(user_input).lower() == "exit":
                waiting_for_input = False
                st.write("Goodbye!")
        
            elif user_input == 0:
                waiting_for_input = False
                st.write("Sorry, I didn't hear anything. Please try again.")
                waiting_for_input = True
            
            elif user_input == 1:
                waiting_for_input = False
                st.write("Sorry, I couldn't understand the audio. Please try again.")
                waiting_for_input = True
            
            elif user_input == 2:
                waiting_for_input = False
                st.write("Could not request results from Google Speech Recognition service") 

            else:
                with st.spinner("Processing..."):
                    waiting_for_input = False
                    classification = classify_user_input(user_input)

                    if classification == 1:
                        response = llm_chain(user_input)
                        actual_text = response['text']
                        st.write(actual_text)
                        convert_text_to_speech(actual_text)
                    else:
                        response = "Sorry, I only answer queries related to lung cancer"
                        st.write(response)
                        convert_text_to_speech(response)

                    st.success("Processing complete!")  # Display a success message

                waiting_for_input = True

        return None



def convert_text_to_speech(text, output_file="C:/Users/thesc/OneDrive/Desktop/MARC/output.mp3", language="en"): #Replace path with the path where you want to store the generated output file
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(output_file)
        st.audio(output_file, format='audio/mp3')
        pygame.mixer.init()
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(3)

        pygame.mixer.quit()

    except Exception as e:
        st.error(f"Error during text-to-speech conversion: {str(e)}")

def load_lottie_file(file_path):
    """
    Load Lottie animation from a local file.
    """
    with open(file_path, "r") as file:
        lottie_json = json.load(file)
    return lottie_json

def main():
    try:
        lottie_file_path = 'C:/Users/thesc/OneDrive/Desktop/MARC/final/assets/docs.json' #Replace with the docs.json file path from assets folder
        lottie_json = load_lottie_file(lottie_file_path)
        st.markdown("<h1 style='text-align: center; font-size: 3em;'>Hi ! I'm your medical assistant    </h1>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])

        with col1:
            st_lottie(lottie_json, speed=1, width=250, height=300)

        with col2:
            st.markdown("<h1 style='text-align: center; font-size: 3em;'>🌐 M.A.R.C 🌐</h1>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; font-size: 2.5em;'>Medical Assistance and Response Companion</h2>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; font-size: 2em;'>You can press the button below to ask a question or say 'exit' to end the conversation.</h3>", unsafe_allow_html=True)

        recognizer = sr.Recognizer()
        button_key = "start_button"

        process_user_request(recognizer, button_key)

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()