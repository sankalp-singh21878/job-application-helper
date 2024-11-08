import streamlit as st
from utils.document_extraction import DocumentExtractor
from utils.cold_email_generation import ColdEmailGenerator
import pandas as pd
import chromadb
import uuid

# Initialize ChromaDB client
client = chromadb.Client("vectorstore")

# Streamlit page configuration
st.set_page_config(page_title="Job Application Helper", layout="wide")

# App Title
st.title("Job Application Helper")

# User Inputs
page_url = st.text_input("Enter the job posting URL:")
resume_file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])
skills_file = st.file_uploader("Upload your skills CSV file", type=["csv"])

# Check that all inputs are provided
if page_url and resume_file and skills_file:
    # Initialize document extractor
    extractor = DocumentExtractor()

    # Load and parse documents
    try:
        page_data, resume_text = extractor.extract_documents(page_url, resume_file)
        jd_data, resume_data = extractor.extract_json(page_data, resume_text)
    except Exception as e:
        st.error(f"Failed to process documents: {e}")

    # Load skills data from uploaded CSV
    try:
        skills = pd.read_csv(skills_file)

        # Populate ChromaDB collection with skills if it's empty
        collection = client.get_or_create_collection("portfolio")
        if not collection.count():
            for _, row in skills.iterrows():
                collection.add(
                    documents=row["Skills"],
                    metadatas={"links": row["Portfolio URL"]},
                    ids=[str(uuid.uuid4())]
                )
        
        # Retrieve links from collection based on job description skills
        links_list = collection.query(query_texts=jd_data["skills"], n_results=2).get("metadatas")
    except Exception as e:
        st.error(f"Failed to load skills CSV: {e}")

    # Email generation
    email_generator = ColdEmailGenerator()
    cold_email = email_generator.generate_email(jd_data, resume_text, links_list)
    
    st.subheader("Generated Cold Email")
    st.write(cold_email)

else:
    st.info("Please enter the job URL, upload your resume, and upload your skills CSV file.")
