from langgraph.prebuilt import ToolNode
from typing import Dict, Any, Optional

class ToolManager:
    """Manages tools for the chatbot, allowing dynamic addition of multiple tools."""

    def __init__(self):
        """Initialize the ToolManager with an empty tools dictionary."""
        self.tools: Dict[str, Any] = {}

    def add_tool(self, tool_name: str, tool: Any) -> None:
        """Add a single tool to the tools dictionary.
        
        Args:
            tool_name: Name identifier for the tool
            tool: A tool instance that implements the required API
            
        Raises:
            ValueError: If the provided tool is None
        """
        if tool is not None:
            self.tools[tool_name] = tool
        else:
            raise ValueError("Tool cannot be None.")

    def add_tools(self, tools_dict: Dict[str, Any]) -> None:
        """Add multiple tools at once from a dictionary.
        
        Args:
            tools_dict: Dictionary mapping tool names to tool instances
            
        Raises:
            ValueError: If no tools are provided
        """
        if not tools_dict:
            raise ValueError("No tools provided. Please add at least one tool.")
        
        for tool_name, tool in tools_dict.items():
            self.add_tool(tool_name, tool)

    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Retrieve a specific tool by its name.
        
        Args:
            tool_name: The name of the tool to retrieve
            
        Returns:
            The tool instance if found, None otherwise
        """
        return self.tools.get(tool_name)

    def get_tools(self) -> Dict[str, Any]:
        """Retrieve the dictionary of all registered tools.
        
        Returns:
            Dictionary mapping tool names to tool instances
        """
        return list(self.tools.values())

    def create_tool_node(self) -> ToolNode:
        """Create a ToolNode for the graph using all registered tools.
        
        Returns:
            A LangGraph ToolNode containing all registered tools
            
        Raises:
            ValueError: If no tools have been registered
        """
        if not self.tools:
            raise ValueError("No tools registered. Please add tools before creating a ToolNode.")
        
        # ToolNode expects a list of tools, so we extract the values from our dictionary
        return ToolNode(tools=list(self.tools.values()))