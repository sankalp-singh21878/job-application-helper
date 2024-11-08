import tempfile
from langchain_community.document_loaders import WebBaseLoader, PDFPlumberLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

class DocumentExtractor:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.2-90b-text-preview",
            temperature=0,
            groq_api_key="YOUR_API_KEY",
        )
        self.json_parser = JsonOutputParser()

    def extract_documents(self, page_url, resume_file):
        # Load job description content from URL
        page = WebBaseLoader(page_url)
        page_data = page.load().pop().page_content

        # Temporarily save the uploaded resume file and load content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(resume_file.getbuffer())
            temp_pdf_path = temp_pdf.name

        resume = PDFPlumberLoader(temp_pdf_path)
        resume_data = resume.load()
        resume_text = "\n".join([doc.page_content for doc in resume_data])

        return page_data, resume_text

    def extract_json(self, page_data, resume_text):
        # Ensure that both inputs contain data
        if not page_data.strip():
            raise ValueError("The page data is empty. Please provide a valid URL with content.")
        if not resume_text.strip():
            raise ValueError("The resume text is empty. Please upload a valid resume file.")

        jd_data = self._extract_job_description_json(page_data)
        resume_data = self._extract_resume_json(resume_text)
        return jd_data, resume_data

    def _extract_job_description_json(self, page_data):
        prompt = PromptTemplate.from_template("""
        ### SCRAPED TEXT FROM WEBSITE
        {page_data}
        ### INSTRUCTIONS
        The scraped text is from a job description.
        Extract profile information and return only in JSON format with the following keys:
        'Profile Summary', 'skills', 'experience', 'description'.
        Strictly output valid JSON format without additional text.
        """)
        chain = prompt | self.llm
        response = chain.invoke(input={'page_data': page_data})

        # Parse and validate the JSON output
        try:
            return self.json_parser.parse(response.content)
        except Exception as e:
            raise ValueError(f"Failed to parse job description JSON output: {e}")

    def _extract_resume_json(self, resume_text):
        prompt = PromptTemplate.from_template("""
        ### SCRAPED TEXT FROM RESUME
        {resume_text}
        ### INSTRUCTIONS
        The scraped text is from a resume.
        Extract profile summary and return only in JSON format with the following keys:
        'profile', 'skills', 'experience', 'projects'.
        Strictly output valid JSON format without additional text.
        """)
        chain = prompt | self.llm
        response = chain.invoke(input={'resume_text': resume_text})

        # Parse and validate the JSON output
        try:
            return self.json_parser.parse(response.content)
        except Exception as e:
            raise ValueError(f"Failed to parse resume JSON output: {e}")
