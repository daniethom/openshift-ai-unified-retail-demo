# System Architecture
## 1. Introduction
This document provides a high-level technical overview of the Meridian Retail AI Demo system. The system is designed as a multi-agent AI platform that leverages a standardized protocol (MCP) to solve complex retail challenges. It is built to run on Red Hat OpenShift AI, utilizing modern cloud-native principles for scalability and maintainability.

The core purpose is to demonstrate how a crew of specialized AI agents can collaborate to provide deep, cross-functional business insights, moving beyond simple single-query responses.

## 2. Architecture Diagram
[![](https://mermaid.ink/img/pako:eNp1VFFv2jAQ_iuWH6ZOAkaBDsjDpDShbdYAEQlMnakmNzFgNcTIcdhY1f--sxNKugqkJOe77-783R33gmORMGzhtaS7DYrcZZYXT-VhiX-wJ-RliskVjdkSLzNU_eYeCZVkdJtyhVyab54ElcnjMmNZUqLexRkXqeJNe80yhXx6YBJdOJL9tr3P9aB30_GI3IktQyWyiaYy3rBcSaqEfDwBo9lo4pJIQrISWrN5kwXxsj0ohTx8sDrzMCJOkSvIIj9Yg5nnjEggecyz9Zv1DCUnQIEUSsQiPXK6obmyg_ekAPfL98cEHi2jkMk9q5PRgJl9S-A5DwhH9sy5IyGjUJHzMHti-w-R54TEzmh6UDzOP4BrdPS7Rumaxs-6pBrNY5ajT9BaRetsNJNbSTOuGOpeoy_IT-mWom6rU7-J5y_mIRnzdF_kaMFiaAVyr-sNtBee_0AiuufpAVWsoHI1iGtHNvkeTifmDuiGpyyvzOcZBClVKyG36GK6Y1m44SuF_huy-3A0W4zIsykJMNwDpVra22BOJgvP9WwtohnLRSHjc6nnHmo2v5nBLRVaMiozohAexhHeeuzgY-arBJ63G_dqakBXjUcllXNQHd66XYY8-mh_-J6U2t0ENX056atg5rqmISfTW2hj1b0obcf4ZRVLXSkbNZQMN2CV8ARbShasgeFvtqX6iF80eonVhm1hlVggJmxFYTHo7ryC245mP4XYHj2lKNYbbK1omsOp2CVUMZdTaPUJAu1g0hFFprDVGfRMDGy94D_Y6g1a7X5nOGxf9npX7V77soEP2OpetTq94bD_tdvudvqDfue1gf-apO3WoH_VwCzhMK3jciea1fj6DxkRfcM?type=png)](https://mermaid.live/edit#pako:eNp1VFFv2jAQ_iuWH6ZOAkaBDsjDpDShbdYAEQlMnakmNzFgNcTIcdhY1f--sxNKugqkJOe77-783R33gmORMGzhtaS7DYrcZZYXT-VhiX-wJ-RliskVjdkSLzNU_eYeCZVkdJtyhVyab54ElcnjMmNZUqLexRkXqeJNe80yhXx6YBJdOJL9tr3P9aB30_GI3IktQyWyiaYy3rBcSaqEfDwBo9lo4pJIQrISWrN5kwXxsj0ohTx8sDrzMCJOkSvIIj9Yg5nnjEggecyz9Zv1DCUnQIEUSsQiPXK6obmyg_ekAPfL98cEHi2jkMk9q5PRgJl9S-A5DwhH9sy5IyGjUJHzMHti-w-R54TEzmh6UDzOP4BrdPS7Rumaxs-6pBrNY5ajT9BaRetsNJNbSTOuGOpeoy_IT-mWom6rU7-J5y_mIRnzdF_kaMFiaAVyr-sNtBee_0AiuufpAVWsoHI1iGtHNvkeTifmDuiGpyyvzOcZBClVKyG36GK6Y1m44SuF_huy-3A0W4zIsykJMNwDpVra22BOJgvP9WwtohnLRSHjc6nnHmo2v5nBLRVaMiozohAexhHeeuzgY-arBJ63G_dqakBXjUcllXNQHd66XYY8-mh_-J6U2t0ENX056atg5rqmISfTW2hj1b0obcf4ZRVLXSkbNZQMN2CV8ARbShasgeFvtqX6iF80eonVhm1hlVggJmxFYTHo7ryC245mP4XYHj2lKNYbbK1omsOp2CVUMZdTaPUJAu1g0hFFprDVGfRMDGy94D_Y6g1a7X5nOGxf9npX7V77soEP2OpetTq94bD_tdvudvqDfue1gf-apO3WoH_VwCzhMK3jciea1fj6DxkRfcM)

    
# 3. Component Breakdown
## 3.1. Streamlit UI
**Technology: Streamlit**

**Role:** Acts as the primary user interface for the demo. It provides a simple, chat-based dashboard where a user can input queries and view the synthesized responses from the multi-agent system.

## 3.2. Multi-Agent Layer
**Technology: CrewAI**

**Role:** This is the cognitive core of the system. It consists of a crew of specialized AI agents designed to mimic a retail operations team.

- **HomeAgent (Orchestrator):** The "manager" of the crew. It receives all incoming queries, analyzes their intent and complexity, and delegates tasks to the appropriate specialist agents.

- **Specialized Agents (Inventory, Pricing, Customer, Trend):** Each agent is an expert in its domain. They execute specific tasks using a set of tools and report their findings back to the HomeAgent.

## 3.3. MCP (Model Context Protocol) Protocol Layer
**Technology: FastAPI**

**Role:** This layer serves as a standardized bridge between the agents and their tools. It decouples the agents from the specific implementation of backend services. Each agent communicates with this layer via a simple, consistent API contract.

- **LLM Server:** Provides access to the base Large Language Model for text generation and understanding.

- **RAG Server:** Connects to the Milvus vector database to retrieve relevant documents and context.

- **Search Server:** Connects to the external Tavily API to perform real-time web searches.

- **Analytics Server:** Executes business logic and calculations using the project's internal JSON data files.

## 3.4. Backend Services & Data
**Role:** These are the foundational tools and data sources that power the system's intelligence.

- **LLM (Granite 3B / Llama 3.2):** The base language model responsible for natural language understanding and generation.

- **Milvus:** A vector database used to store and retrieve embeddings from internal documents, enabling the RAG system.

- **Tavily API:** A third-party service providing real-time web search capabilities.

- **JSON Data Files:** The project's synthetic "source of truth" for products, customers, and market data.

## 3.5. OpenShift AI Platform
**Role:** The enterprise-grade platform for deploying and scaling the entire system.

- **kServe & vLLM:** Used to serve the LLM efficiently on GPU-accelerated hardware.

- **GPU Resources:** Provides the necessary hardware acceleration for running the language model at scale.

- **Container Orchestration:** Manages the lifecycle, networking, and scalability of all the microservices (UI and MCP servers).

## 4. Data Flow
- A typical user query follows this path through the system:

- A user enters a natural language query into the Streamlit UI.

- The query is sent to the HomeAgent.

- The HomeAgent analyzes the query's intent and complexity to determine which specialized agents are needed (e.g., a query about price and stock involves the PricingAgent and InventoryAgent).

- The HomeAgent tasks the selected specialized agents with sub-queries.

- The specialized agents use their tools by sending standardized requests to the relevant MCP Servers via the /invoke endpoint. For instance, the TrendAgent might call the Search MCP Server to get competitor trend data.

- The MCP Servers execute their backend logic (e.g., querying the Tavily API or the Milvus database) and return a structured JSON response.

- The specialized agents receive the data from the MCP servers, process it, and report their findings back to the HomeAgent.

- The HomeAgent synthesizes the findings from all involved agents into a single, cohesive, and user-friendly response.

- The final response is displayed back to the user in the Streamlit UI.