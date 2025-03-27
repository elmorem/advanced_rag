from typing import Any, Dict
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from graph.state import GraphState

load_dotenv(override=True)

web_search_tool = TavilySearchResults(max_resuts=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Perform a web search using the TavilySearchResults tool.
    """
    question = state["question"]
    documents = state["documents"]
    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_result = "\n".join([doc["content"] for doc in tavily_results])

    web_results = Document(page_content=joined_tavily_result)
    if documents is None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {
        "documents": documents,
        "question": question,
    }

if __name__ == "__main__":
    # Example usage
    web_search(state={"question": "agent memory", "documents": None})

