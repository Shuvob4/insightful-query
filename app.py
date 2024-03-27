__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from transformers import  pipeline
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain import OpenAI, VectorDBQA

OPENAI_API_KEY = st.secrets["openai_api_key"]

# Function for creating relevant context
def get_text_snippet(text, question, window_size=500):
    # Load a pre-trained QA model and tokenizer
    qa_pipeline = pipeline("question-answering")

    # Prepare the inputs for the model
    inputs = {
        "question": question,
        "context": text
    }

    # Get the answer from the model
    result = qa_pipeline(inputs)

    # Check if the model found an answer within the context
    if not result['answer']:
        return "The model could not find an answer in the context provided."

    # Find the end positions of the answer in the context
    end_position = result['end']

    # Calculate the end snippet position, expanding around the answer based on the window size
    end_snippet = min(len(text), end_position + window_size)

    # Set the start of the snippet to the beginning of the text
    start_snippet = 0

    # Extract and return the snippet containing the answer
    snippet = text[start_snippet:end_snippet]
    print("Length of Given Context : ", len(text))
    print("Length of Initial Generated Relevant Text : ", len(snippet))

    # checking if given context and generated context length is same or not
    if len(text) == len(snippet):
        start_position = result['start']
        end_position = result['end']
        
        # Adjust the start and end snippet positions to center around the answer
        snippet_length = window_size // 2  # Half before, half after the answer
        start_snippet = max(0, start_position - snippet_length)
        end_snippet = min(len(text), end_position + snippet_length)

        # Ensure the snippet doesn't start in the middle of a word
        if start_snippet > 0 and text[start_snippet - 1].isalnum():
            start_snippet = text.rfind(" ", 0, start_snippet) + 1

        snippet = text[start_snippet:end_snippet]
        print('Length of final Snippet : ', len(snippet))
        return snippet                                  
    else:
        print('Length of final Snippet : ', len(snippet))
        return snippet


# Function for getting the final answer
def get_answer_from_context(context, question):
    text_splitter = CharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )

    texts = text_splitter.create_documents([context])
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    docsearch = Chroma.from_documents(texts, embeddings)
    eqa = VectorDBQA.from_chain_type(
        llm = OpenAI(openai_api_key = OPENAI_API_KEY, temperature=0),
        chain_type = 'stuff',
        vectorstore = docsearch,
        return_source_documents = True
    )
    
    answer = eqa.invoke(question)
    return answer

# setting up the web page
st.set_page_config(page_title="Extractive Question Answering System",
                   layout='centered',
                   initial_sidebar_state='collapsed')


st.header("Extractive Question Answering System")

question_input = st.text_area("Enter Your Question here", key="question_input")
context_input = st.text_area("Enter Your Context here",  height=300, key="context_input")

generate_relevant_text_button = st.button("Get Relevant Text")

# Initialize or ensure session state variables exist
if 'output_text' not in st.session_state:
    st.session_state['output_text'] = ""
if 'display_final_answer' not in st.session_state:
    st.session_state['display_final_answer'] = False

# Action for "Get Relevant Text" button
if generate_relevant_text_button:
    # Generate and store the relevant text in session state
    st.session_state['output_text'] = get_text_snippet(context_input, question_input)
    st.session_state['display_final_answer'] = False  # Reset this flag each time "Get Relevant Text" is pressed

# Always display the output text area if there's something to show
if st.session_state['output_text']:
    st.text_area("Result", value=st.session_state['output_text'], height=250, disabled=True, key="output_result")

    # Condition for displaying the "Generate Final Answer" button
    if "The model could not find an answer in the context provided." not in st.session_state['output_text']:
        if st.button('Generate Final Answer', key='generate_final_answer'):
            st.session_state['display_final_answer'] = True  # Flag to display the final answer and score

# Display final answer and score if the flag is set
if st.session_state['display_final_answer']:
    result = get_answer_from_context(st.session_state['output_text'], question_input)
    answer = result['result']  # Safely get the 'answer' if it exists
    st.text_area("Final Answer:", value=str(answer), height=100, disabled=True, key="final_answer")
    
