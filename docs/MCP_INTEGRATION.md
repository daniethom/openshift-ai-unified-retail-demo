# MCP (Model Context Protocol) Integration
## 1. Purpose of MCP
The Model Context Protocol (MCP) is a critical architectural component in this project. It serves as a standardized communication layer that decouples the AI agents from their tools.

Instead of an agent needing to know the specific client library and implementation details for Tavily, Milvus, or our internal analytics, it only needs to know how to make a simple, standardized API call. This provides several key advantages:

- **Modularity:** Tools (like the search backend) can be swapped out with minimal to no changes to the agents themselves.

- **Simplicity:** Agents are simpler because they don't contain complex client-specific logic.

- **Scalability:** Each MCP server is a separate microservice that can be developed, deployed, and scaled independently.

- **Extensibility:** Adding a new tool or capability to the system is as simple as creating a new MCP server and registering it with the agents.

## 2. Server Overview
The project implements four distinct MCP servers, each running as a FastAPI application:

- **llm_server.py:** Provides a standardized interface for agents to access the base Large Language Model for tasks like text generation and understanding.

- **rag_server.py:** Connects to the Milvus vector database to retrieve relevant documents and context.

- **search_server.py:** Acts as a gateway to the external Tavily API to perform real-time web searches.

- **analytics_server.py:** Executes business logic and calculations using the project's internal JSON data files.

## 3. Standard API Contract
All MCP servers expose a single, consistent endpoint for tool invocation.

```http
Endpoint: POST /invoke

Generic Request Body
The request body follows a simple ToolInput model:

{
  "tool_name": "name_of_the_tool_to_run",
  "input_data": {
    "parameter1": "value1",
    "parameter2": "value2"
  }
}

Generic Response Body
The response body follows a standard ToolOutput model:

{
  "status": "success",
  "result": {
    "output_key_1": "output_value_1",
    "output_key_2": "output_value_2"
  }
}
```

## 4. Server-Specific Examples
### Analytics Server
```http
Request:

{
  "tool_name": "get_total_inventory_value",
  "input_data": {}
}

Response:

{
  "status": "success",
  "result": {
    "total_stock_value_zar": 2375.0,
    "total_product_count": 3,
    "average_value_per_product": 791.67
  }
}
```
### Search Server
```http
Request:

{
  "tool_name": "web_search",
  "input_data": {
    "query": "latest fashion trends in south africa"
  }
}

Response:

{
  "status": "success",
  "result": [
    {
      "title": "Latest Winter Fashion Trends in South Africa - Fashion Weekly",
      "url": "[https://fake-fashion-weekly.com/trends-sa-winter-2025](https://fake-fashion-weekly.com/trends-sa-winter-2025)",
      "content": "This winter in South Africa, expect to see a rise in 'utilitarian chic'..."
    }
  ]
}
```

### RAG Server
```http
Request:

{
  "tool_name": "retrieve_documents",
  "input_data": {
    "query": "internal reports on winter fashion sales"
  }
}

Response:

{
  "status": "success",
  "result": [
    {
      "source": "docs/Q4_2024_Sales_Report.pdf",
      "content": "Sales of outerwear, particularly wool coats, increased by 45% in Q4...",
      "score": 0.91
    }
  ]
}
```
### LLM Server
```http
Request:

{
  "tool_name": "generate_text",
  "input_data": {
    "prompt": "Summarize the following report for a marketing executive: ..."
  }
}

Response:

{
  "status": "success",
  "result": {
    "text": "The Q4 2024 Sales Report highlights a significant 45% increase in outerwear sales, driven primarily by strong performance in the wool coat category..."
  }
}
```
