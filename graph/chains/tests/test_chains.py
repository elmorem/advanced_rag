import os
import pytest
from dotenv import load_dotenv
from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments


load_dotenv(override=True)

ai_api_key = os.environ["OPENAI_API_KEY"] 

if not ai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
else:
    print(f"{ai_api_key = }")
def test_retrieval_grader_yes_no():
    # Test with a relevant document
    question = "What is the capital of France?"
    document = "The capital of France is Paris."
    
    result = retrieval_grader.invoke({"question": question, "document": document})
    
    assert isinstance(result, GradeDocuments)
    assert result.binary_score == "yes"