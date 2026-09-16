"""Interactive MCP tool explorer for local development."""

import json

import streamlit as st

from config.settings import settings
from streamlit_app.page_utils import fetch_mcp_health, invoke_mcp, mcp_server_rows

st.title("🔧 MCP Tools")
st.caption("Health checks and direct `/invoke` calls against local MCP servers")

st.subheader("Server health")
health_cols = st.columns(4)
for idx, server in enumerate(mcp_server_rows()):
    with health_cols[idx]:
        health = fetch_mcp_health(server["url"])
        if health["ok"]:
            st.success(f"{server['name']} online")
            st.caption(json.dumps(health["body"]))
        else:
            st.error(f"{server['name']} offline")
            st.caption(health["error"])

st.divider()

llm_tab, rag_tab, search_tab, analytics_tab = st.tabs(
    ["LLM", "RAG", "Search", "Analytics"]
)

with llm_tab:
    st.markdown(f"**Endpoint:** `{settings.llm_mcp_url}`")
    prompt = st.text_area(
        "Prompt",
        value="Summarize Meridian Retail Group in one sentence.",
        height=120,
    )
    if st.button("Invoke LLM", type="primary", key="invoke_llm"):
        result = invoke_mcp(
            settings.llm_mcp_url,
            {"prompt": prompt},
            timeout=120.0,
        )
        if result["ok"]:
            st.write(result["body"].get("response", result["body"]))
        else:
            st.error(result["error"])

with rag_tab:
    st.markdown(f"**Endpoint:** `{settings.rag_mcp_url}`")
    query = st.text_input("Retrieval query", value="winter fashion Cape Town")
    top_k = st.number_input("Top K", min_value=1, max_value=10, value=3)
    if st.button("Retrieve documents", type="primary", key="invoke_rag"):
        result = invoke_mcp(
            settings.rag_mcp_url,
            {
                "tool_name": "retrieve_documents",
                "input_data": {"query": query, "top_k": int(top_k)},
            },
        )
        if result["ok"]:
            st.json(result["body"])
        else:
            st.error(result["error"])

with search_tab:
    st.markdown(f"**Endpoint:** `{settings.search_mcp_url}`")
    query = st.text_input(
        "Search query",
        value="winter fashion trends Cape Town professional women",
        key="search_query",
    )
    if st.button("Run web search", type="primary", key="invoke_search"):
        result = invoke_mcp(
            settings.search_mcp_url,
            {"tool_name": "web_search", "input_data": {"query": query}},
        )
        if result["ok"]:
            for item in result["body"].get("result", []):
                st.markdown(
                    f"**[{item.get('title', 'Result')}]({item.get('url', '')})**"
                )
                st.write(item.get("content", ""))
                st.divider()
        else:
            st.error(result["error"])

with analytics_tab:
    st.markdown(f"**Endpoint:** `{settings.analytics_mcp_url}`")
    tool = st.selectbox(
        "Analytics tool",
        [
            "get_total_inventory_value",
            "get_product_count_by_brand",
            "get_product_details",
            "get_demand_analytics",
            "get_customer_profile",
            "search_customers_by_name",
        ],
    )

    product_id = st.text_input("product_id", value="MF001")
    customer_id = st.text_input("customer_id", value="CUST001")
    brand_name = st.text_input("brand_name", value="Meridian Fashion")
    customer_name = st.text_input("customer_name", value="Sarah Johnson")

    input_data: dict = {}
    if tool == "get_product_details":
        input_data = {"product_id": product_id}
    elif tool == "get_demand_analytics":
        input_data = {"product_id": product_id}
    elif tool == "get_customer_profile":
        input_data = {"customer_id": customer_id}
    elif tool == "search_customers_by_name":
        input_data = {"name": customer_name}
    elif tool == "get_product_count_by_brand":
        input_data = {"brand_name": brand_name}

    if st.button("Invoke analytics tool", type="primary", key="invoke_analytics"):
        result = invoke_mcp(
            settings.analytics_mcp_url,
            {"tool_name": tool, "input_data": input_data},
        )
        if result["ok"]:
            st.json(result["body"])
        else:
            st.error(result["error"])
