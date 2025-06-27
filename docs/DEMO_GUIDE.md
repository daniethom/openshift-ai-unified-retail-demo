# Meridian Retail AI - Live Demo Guide

## 1. Introduction Script

**(Presenter):** "Welcome, everyone. Today, we're going to demonstrate the Meridian Retail AI system, a powerful multi-agent platform running on Red Hat OpenShift AI.

What you'll see isn't just a chatbot. It's a collaborative crew of specialized AI agents—experts in pricing, inventory, customer service, and trends—working together to solve complex, real-world retail problems. We'll show how this system moves beyond simple answers to provide deep, cross-functional insights that can drive real business value. Let's get started."

---

## 2. Scenario Walkthroughs

### Scenario 1: Strategic Fashion Trend Analysis

* **Presenter Talking Points:**
    * "First, let's tackle a strategic planning challenge. Imagine I'm a retail manager for our Cape Town stores, and I need to plan our inventory for the upcoming winter season."
    * "A simple query to a normal chatbot might give me a generic list of trends. But our system understands the business context—it knows about our brands, our customer demographics, and our current stock. Let's ask it a complex question."

* **Query to Enter:**
    ```
    What winter fashion trends should our Cape Town stores focus on for professional women?
    ```

* **Expected Outcome:**
    * The system will provide a rich, multi-faceted response.
    * **Key Insight to Highlight:** "Notice how the response isn't just a list. The `HomeAgent` has coordinated multiple agents. The `TrendAgent` identified 'Power Suiting' and 'Luxe Knitwear'. The `InventoryAgent` has cross-referenced this with our current stock, and the `PricingAgent` has suggested an initial pricing strategy. This is a complete, actionable plan, not just data."

---

### Scenario 2: Personalized Cross-Sell Opportunity

* **Presenter Talking Points:**
    * "Now, let's move from strategic planning to a real-time customer interaction. Our system can help enhance the customer experience and drive sales."
    * "Imagine a customer, Sarah Johnson, has just purchased a winter coat. Our goal is to provide a relevant and personalized recommendation for her next purchase."

* **Query to Enter:**
    ```
    Customer Sarah Johnson bought a winter coat. What should we recommend?
    ```

* **Expected Outcome:**
    * The system will access Sarah's profile and purchase history via the RAG tool.
    * **Key Insight to Highlight:** "The system doesn't just suggest random accessories. It knows Sarah is a Gold-tier loyalty member and prefers the 'Meridian Fashion' brand. It recommends a 'Cashmere Scarf' and a 'Leather Tote Bag'—items that perfectly complement her recent purchase and match her known preferences. This is data-driven personalization in action."

---

### Scenario 3: Dynamic Inventory Optimization

* **Presenter Talking Points:**
    * "Next, let's look at an operational challenge: inventory management. Holding too much stock is expensive, but stocking out loses sales. Our system can help find the optimal balance."
    * "I'll ask the system to perform a broad optimization task for our Johannesburg stores for the summer."

* **Query to Enter:**
    ```
    Optimize inventory for the upcoming summer season across all Johannesburg stores.
    ```

* **Expected Outcome:**
    * The `HomeAgent` will engage the `InventoryAgent` and `TrendAgent`.
    * **Key Insight to Highlight:** "The system provides a detailed plan. It forecasts demand for key summer categories like 'Bold Prints', suggests specific reorder quantities, and recommends a stock redistribution plan to move inventory to stores where it's most likely to sell. This demonstrates how the AI can proactively manage the supply chain to maximize profitability."

---

### Scenario 4: Complex Customer Complaint Resolution

* **Presenter Talking Points:**
    * "Finally, let's see how the system handles a difficult situation: an unhappy customer. A quick, empathetic, and effective response is key to customer retention."
    * "Let's simulate a high-value customer complaining about a service issue."

* **Query to Enter:**
    ```
    A high-value customer is complaining about a delayed delivery and poor service.
    ```

* **Expected Outcome:**
    * The `HomeAgent` will immediately recognize the severity and orchestrate a multi-agent response.
    * **Key Insight to Highlight:** "This is where the multi-agent collaboration truly shines. The `CustomerAgent` assesses the customer's value and the complaint's severity. The `InventoryAgent` checks the order status. The `PricingAgent` calculates an appropriate compensation voucher. The `HomeAgent` synthesizes all of this into a single, empathetic response that includes an apology, a solution, and a retention offer. The system has solved a complex problem in seconds that would typically require multiple human touchpoints."
