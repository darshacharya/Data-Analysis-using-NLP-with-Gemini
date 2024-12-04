import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini
from pandasai.responses.response_parser import ResponseParser
from PIL import Image


class SimpleOutputParser(ResponseParser):
    def parse(self, response):
        if isinstance(response, pd.DataFrame):
            return {"type": "dataframe", "value": response}
        elif isinstance(response, dict) and response.get('type') == 'plot':
            return {"type": "plot", "value": response['value']}
        else:
            return {"type": "string", "value": str(response)}


# Set up the app page layout and title
st.set_page_config(layout="centered")
st.title("🤖 DataFrame Chatbot - Gemini")

# Initialize session state for chat history and DataFrame
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None


# Function to read the uploaded file
def read_data(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xls"):
            return pd.read_excel(file, engine='xlrd')
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file, engine='openpyxl')
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


# File uploader component
uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls"])

# Load DataFrame if a file is uploaded
if uploaded_file:
    st.session_state.df = read_data(uploaded_file)
    if st.session_state.df is not None:
        st.write("DataFrame Preview:")
        st.dataframe(st.session_state.df.head())
    else:
        st.stop()  # Stop if there's an error in loading the file

# Initialize Google Gemini model
if st.session_state.df is not None:
    llm = GoogleGemini(api_key="YOUR_API_KEY")
    sdf = SmartDataframe(st.session_state.df, config={"llm": llm, "conversational": False})

    # Display all previous messages in chat format
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], pd.DataFrame):
                st.dataframe(message["content"])
            else:
                st.markdown(message["content"])

    # Chat input for user queries
    user_prompt = st.chat_input("Ask about your data:")

    if user_prompt:
        # Display user's query immediately
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Add user's query to the chat history
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Analyze data using Google Gemini
        response = sdf.chat(f"As a data analyst, please analyze the following data query: {user_prompt}")

        # Display assistant's response immediately
        with st.chat_message("assistant"):
            if isinstance(response, pd.DataFrame):
                st.dataframe(response)
            elif isinstance(response, dict) and response.get("type") == "plot":
                try:
                    image = Image.open(response["value"])
                    st.image(image, caption="Generated Plot")
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    st.write(f"Image path: {response['value']}")
            else:
                st.write(response)

        # Add assistant's response to the chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
else:
    st.info("Please upload a valid file to start chatting with your data.")
