"""Business analytics powered by the Analytics MCP server."""

import streamlit as st

from config.settings import settings
from streamlit_app.page_utils import ensure_project_root, invoke_mcp

ensure_project_root()

st.title("📈 Analytics")
st.caption("PostgreSQL-backed retail metrics via the Analytics MCP server")

col1, col2 = st.columns(2)
with col1:
    st.metric(
        "Data backend",
        "PostgreSQL" if not settings.use_json_fallback else "JSON fallback",
    )
with col2:
    st.metric("Analytics MCP", settings.analytics_mcp_url)

st.divider()

overview_tab, customer_tab, product_tab = st.tabs(
    ["Inventory Overview", "Customer Lookup", "Product Insights"]
)

with overview_tab:
    brand_name = st.selectbox(
        "Brand",
        ["Meridian Fashion", "Stratus", "Casa Living", "Vertex Sports"],
        index=0,
    )
    if st.button("Load inventory summary", type="primary"):
        with st.spinner("Querying analytics MCP..."):
            total = invoke_mcp(
                settings.analytics_mcp_url,
                {"tool_name": "get_total_inventory_value", "input_data": {}},
            )
            brands = invoke_mcp(
                settings.analytics_mcp_url,
                {
                    "tool_name": "get_product_count_by_brand",
                    "input_data": {"brand_name": brand_name},
                },
            )

        if total["ok"]:
            result = total["body"].get("result", {})
            value = result.get("total_stock_value_zar")
            st.metric(
                "Total inventory value (ZAR)",
                f"R{value:,.2f}" if isinstance(value, (int, float)) else result,
            )
        else:
            st.error(total["error"])

        if brands["ok"]:
            st.subheader(f"Products for {brand_name}")
            st.json(brands["body"].get("result", {}))
        else:
            st.error(brands["error"])

with customer_tab:
    customer_name = st.text_input("Customer name", value="Sarah Johnson")
    if st.button("Search customers", type="primary"):
        result = invoke_mcp(
            settings.analytics_mcp_url,
            {
                "tool_name": "search_customers_by_name",
                "input_data": {"name": customer_name},
            },
        )
        if result["ok"]:
            matches = result["body"].get("result", [])
            if not matches:
                st.info("No customers matched that name.")
            else:
                st.success(f"Found {len(matches)} customer(s)")
                st.json(matches)
        else:
            st.error(result["error"])

    customer_id = st.text_input("Customer ID for profile lookup", value="CUST001")
    if st.button("Load customer profile"):
        result = invoke_mcp(
            settings.analytics_mcp_url,
            {
                "tool_name": "get_customer_profile",
                "input_data": {"customer_id": customer_id},
            },
        )
        if result["ok"]:
            st.json(result["body"].get("result", {}))
        else:
            st.error(result["error"])

with product_tab:
    product_id = st.text_input("Product ID", value="MF001")
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Product details"):
            result = invoke_mcp(
                settings.analytics_mcp_url,
                {
                    "tool_name": "get_product_details",
                    "input_data": {"product_id": product_id},
                },
            )
            if result["ok"]:
                st.json(result["body"].get("result", {}))
            else:
                st.error(result["error"])

    with col_b:
        if st.button("Demand analytics"):
            result = invoke_mcp(
                settings.analytics_mcp_url,
                {
                    "tool_name": "get_demand_analytics",
                    "input_data": {"product_id": product_id},
                },
            )
            if result["ok"]:
                st.json(result["body"].get("result", {}))
            else:
                st.error(result["error"])
