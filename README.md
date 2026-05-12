🚀 What This Project Does

This project creates a basic chat interface where:

The user sends a message
The LLM decides whether a tool should be called
The MCP client connects to the MCP server
Tool results are returned back to the LLM
The final response is shown in the UI

In this project, I connected an Expense Tracker MCP server as a learning example.


Tech Stack
Python
Streamlit
LangChain
MCP (Model Context Protocol)
OpenRouter API
Asyncio

⚙️ How It Works

The Streamlit app:

initializes the LLM
connects to MCP servers
fetches available tools
binds tools to the LLM
handles tool execution flow
maintains conversation history

The MCP client communicates with the MCP server using stdio transport.
