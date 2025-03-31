from langchain_community.tools.tavily_search import TavilySearchResults
from src.AdaptiveRag.tools.tool_manager import ToolManager

def get_tool_mananger() -> ToolManager:
    """Creates and configures a ToolManager

    Returns:
        ToolManager: Configured tool manager
    """
    tool_manager = ToolManager()
    
    search_tool = TavilySearchResults(max_results=5)
    
    tool_manager.add_tool("web_search", search_tool)
    
    return tool_manager