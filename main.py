import streamlit as st
from st_keyup import st_keyup

st.warning("This application is for trial purposes alone. Personally identifying information should never be entered. The application may not be used in the real world. The application does not replace assessment by healthcare professionals.")

text = st_keyup("Enter a value", key="0", value="enter a text", debounce=500, label_visibility="collapsed") + " "

from deep_translator import GoogleTranslator

# Use any translator you like, in this example GoogleTranslator
text = GoogleTranslator(source='auto', target='en').translate(text)

st.write("Translation:", text)

from transformers import pipeline

# Load the NER pipeline with the camembert/ner-medical model
ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all")

# Example medical text
# text = "The patient was diagnosed with diabetes and prescribed insulin."

# Perform named entity recognition
medical_entities = ner_pipeline(text)

# Extract relevant information from the entities
extracted_entities = [
    {"text": entity["word"], "start": entity["start"], "end": entity["end"], "label": entity["entity"]}
    for entity in medical_entities
]

medical_entities = " ".join([ent["text"] for ent in extracted_entities])

# Print the extracted medical entities
st.write("Extracted Medical Entities:", medical_entities)

import clarifai

from langchain.llms import Clarifai
from langchain import PromptTemplate, LLMChain

USER_ID="meta"
APP_ID="Llama-2"
MODEL_ID="llama2-7b-chat"

llm = Clarifai(pat=st.secrets.CLARIFAI_PAT, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)

template = """Given the following case vignette: {medical_entities}\n
What additional questions would you ask to differentiate possible underlying diagnoses?\n
Show the suggested questions as bullet list."""

prompt = PromptTemplate(
    template=template, 
    input_variables=["medical_entities"]
) 

llm_chain = LLMChain(prompt=prompt, llm=llm)

if st.button("Submit"):
    st.write(llm_chain.run(medical_entities))