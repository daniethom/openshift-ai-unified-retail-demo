System Architecture
1. Introduction
This document provides a high-level technical overview of the Meridian Retail AI Demo system. The system is designed as a multi-agent AI platform that leverages a standardized protocol (MCP) to solve complex retail challenges. It is built to run on Red Hat OpenShift AI, utilizing modern cloud-native principles for scalability and maintainability.

The core purpose is to demonstrate how a crew of specialized AI agents can collaborate to provide deep, cross-functional business insights, moving beyond simple single-query responses.

2. Architecture Diagram
graph TB
    subgraph "Web Interface"
        UI[Streamlit Dashboard]
    end
    
    subgraph "Multi-Agent Layer (CrewAI)"
        HOME[Home Agent - Orchestrator]
        TREND[Trend Agent]
        INV[Inventory Agent]
        CUST[Customer Agent]
        PRICE[Pricing Agent]
    end
    
    subgraph "MCP Protocol Layer (FastAPI)"
        MCP_LLM[LLM MCP Server]
        MCP_RAG[RAG MCP Server]
        MCP_SEARCH[Search MCP Server]
        MCP_ANALYTICS[Analytics MCP Server]
    end
    
    subgraph "Backend Services & Data"
        LLM[Granite 3B / Llama 3.2]
        MILVUS[Milvus Vector DB]
        TAVILY[Tavily Search API]
        DATA[JSON Data Files]
    end
    
    subgraph "Platform (OpenShift AI)"
        KSERVE[kServe & vLLM]
        GPU[NVIDIA GPU Resources]
    end
    
    UI --> HOME
    HOME --> TREND & INV & CUST & PRICE
    TREND & INV & CUST & PRICE --> MCP_LLM & MCP_RAG & MCP_SEARCH & MCP_ANALYTICS
    MCP_LLM --> LLM
    MCP_RAG --> MILVUS
    MCP_SEARCH --> TAVILY
    MCP_ANALYTICS --> DATA
    LLM --> KSERVE
    KSERVE --> GPU

3. Component Breakdown
3.1. Streamlit UI
Technology: Streamlit

Role: Acts as the primary user interface for the demo. It provides a simple, chat-based dashboard where a user can input queries and view the synthesized responses from the multi-agent system.

3.2. Multi-Agent Layer
Technology: CrewAI

Role: This is the cognitive core of the system. It consists of a crew of specialized AI agents designed to mimic a retail operations team.

HomeAgent (Orchestrator): The "manager" of the crew. It receives all incoming queries, analyzes their intent and complexity, and delegates tasks to the appropriate specialist agents.

Specialized Agents (Inventory, Pricing, Customer, Trend): Each agent is an expert in its domain. They execute specific tasks using a set of tools and report their findings back to the HomeAgent.

3.3. MCP (Model Context Protocol) Protocol Layer
Technology: FastAPI

Role: This layer serves as a standardized bridge between the agents and their tools. It decouples the agents from the specific implementation of backend services. Each agent communicates with this layer via a simple, consistent API contract.

LLM Server: Provides access to the base Large Language Model for text generation and understanding.

RAG Server: Connects to the Milvus vector database to retrieve relevant documents and context.

Search Server: Connects to the external Tavily API to perform real-time web searches.

Analytics Server: Executes business logic and calculations using the project's internal JSON data files.

3.4. Backend Services & Data
Role: These are the foundational tools and data sources that power the system's intelligence.

LLM (Granite 3B / Llama 3.2): The base language model responsible for natural language understanding and generation.

Milvus: A vector database used to store and retrieve embeddings from internal documents, enabling the RAG system.

Tavily API: A third-party service providing real-time web search capabilities.

JSON Data Files: The project's synthetic "source of truth" for products, customers, and market data.

3.5. OpenShift AI Platform
Role: The enterprise-grade platform for deploying and scaling the entire system.

kServe & vLLM: Used to serve the LLM efficiently on GPU-accelerated hardware.

GPU Resources: Provides the necessary hardware acceleration for running the language model at scale.

Container Orchestration: Manages the lifecycle, networking, and scalability of all the microservices (UI and MCP servers).

4. Data Flow
A typical user query follows this path through the system:

A user enters a natural language query into the Streamlit UI.

The query is sent to the HomeAgent.

The HomeAgent analyzes the query's intent and complexity to determine which specialized agents are needed (e.g., a query about price and stock involves the PricingAgent and InventoryAgent).

The HomeAgent tasks the selected specialized agents with sub-queries.

The specialized agents use their tools by sending standardized requests to the relevant MCP Servers via the /invoke endpoint. For instance, the TrendAgent might call the Search MCP Server to get competitor trend data.

The MCP Servers execute their backend logic (e.g., querying the Tavily API or the Milvus database) and return a structured JSON response.

The specialized agents receive the data from the MCP servers, process it, and report their findings back to the HomeAgent.

The HomeAgent synthesizes the findings from all involved agents into a single, cohesive, and user-friendly response.

The final response is displayed back to the user in the Streamlit UI.