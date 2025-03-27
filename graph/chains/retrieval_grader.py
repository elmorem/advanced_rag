from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

llm = ChatOpenAI(temperature=.8)

class GradeDocuments(BaseModel):
    """ 
    Binary score for relevance check on retrieved docs
    """

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """ You are a grader assessing relevance of a retrieved document to a user question \n
        If the document contains keywords or semantic meaning related to the question, grade it as relevant.
        give a binary score or 'ye' or 'no' to indicated whether the document is relevant to the question.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system, system"),
        ("human", "retrieved document: \n\n{document}\n\n user question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader