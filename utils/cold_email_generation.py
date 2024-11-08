from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

class ColdEmailGenerator:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.2-90b-text-preview",
            temperature=0,
            groq_api_key=st.secrets["api_key"],
        )

    def generate_email(self, job_description, resume_text, links_list):
        prompt = PromptTemplate.from_template(
            """
                ### JOB DESCRIPTION
                {job_description}

                ### INSTRUCTION:
                Your are an employee who is looking to switch jobs. Generate a professional cold email for a potential employer in the AI/ML field based on the following {resume_text} data. The email should introduce the sender (the employee), highlight their relevant skills and experiences, and express interest in any potential job openings or collaborations.

                Include the following elements:

                A polite greeting and introduction (e.g., "Hello [Hiring Manager]")
                A brief personal introduction (e.g., name, current role, and industry experience)
                Mention of key skills related to AI/ML (e.g., Python, machine learning frameworks, etc.)
                A summary of relevant accomplishments or projects (e.g., portfolio or past achievements)
                A request to discuss potential opportunities or contribute to the company's goals
                A closing that encourages follow-up, along with polite thanks

                The tone should be formal, concise, and tailored to the AI/ML field

                Also add the most relevant links from the following list of links {links_list} and add it as a way of showcasing your skills for the given jd

                You have to write the email from the perspective of the employee who has submitted the resume
            """
        )
        chain = prompt | self.llm
        response = chain.invoke(input={'job_description': str(job_description), 'resume_text': resume_text, 'links_list': links_list})
        return response.content
