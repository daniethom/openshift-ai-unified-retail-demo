import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from streamlit_app.page_utils import format_agent_insight, format_assistant_response


def test_format_agent_insight_uses_result_field():
    insight = {
        "status": "success",
        "result": {"primary_insight": "Stock is low in Cape Town"},
    }
    assert format_agent_insight(insight) == "Stock is low in Cape Town"


def test_format_agent_insight_shows_customer_cross_sell_products():
    insight = {
        "status": "success",
        "result": {
            "recommendations": [
                {
                    "product_id": "MF-SCF-101",
                    "name": "Merino Wool Scarf",
                    "price": 599,
                    "reason": "Completes the winter coat purchase",
                },
                {
                    "product_id": "MF-GLV-022",
                    "name": "Leather Touchscreen Gloves",
                    "price": 799,
                    "reason": "High attach rate with outerwear buyers",
                },
            ],
            "reasoning": "Based on your preferences for modern styles",
            "cross_sell_opportunities": [
                {"message": "Explore our homeware collection"},
            ],
        },
    }
    formatted = format_agent_insight(insight)
    assert formatted is not None
    assert "Merino Wool Scarf" in formatted
    assert "Leather Touchscreen Gloves" in formatted
    assert "Explore our homeware collection" in formatted
    assert formatted.index("Merino Wool Scarf") < formatted.index("Why:")


def test_format_agent_insight_uses_trend_analysis_payload():
    insight = {
        "status": "success",
        "analysis": {
            "key_trends": [
                {"name": "Power Suiting", "relevance": 0.9},
                {"name": "Luxe Knitwear", "relevance": 0.85},
            ]
        },
        "recommendations": [{"action": "Introduce capsule collection"}],
    }
    formatted = format_agent_insight(insight)
    assert formatted is not None
    assert "Power Suiting" in formatted
    assert "Luxe Knitwear" in formatted
    assert "Introduce capsule collection" in formatted


def test_format_assistant_response_includes_agent_insights():
    response = {
        "summary": "Based on the analysis, key insights have been generated.",
        "detailed_insights": {
            "CustomerAgent": {
                "status": "success",
                "result": {
                    "recommendations": [
                        {
                            "name": "Merino Wool Scarf",
                            "price": 599,
                            "reason": "Completes the winter coat purchase",
                        }
                    ],
                    "reasoning": "Based on Sarah's minimalist style",
                },
            },
            "InventoryAgent": {
                "status": "success",
                "result": {
                    "primary_insight": "Winter accessories in stock at Cape Town",
                },
            },
            "PricingAgent": {
                "status": "success",
                "result": {
                    "primary_insight": "15% accessory bundle available with coat purchase",
                },
            },
        },
        "recommendations": [],
    }

    formatted = format_assistant_response(response)

    assert "**Key Insights:**" in formatted
    assert "**CustomerAgent**" in formatted
    assert "Merino Wool Scarf" in formatted
    assert "**InventoryAgent**" in formatted
    assert "Cape Town" in formatted
    assert "**PricingAgent**" in formatted
    assert "15% accessory bundle" in formatted
