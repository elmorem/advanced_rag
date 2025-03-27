from typing import List, TypedDict

class GraphState(TypedDict):
    """
    GraphState is a TypedDict that represents the state of a graph.
    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to do web search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]