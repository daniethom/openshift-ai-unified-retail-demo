Multi-Agent System Design
1. Orchestration Strategy
The heart of the Meridian Retail AI system is its multi-agent design, orchestrated by the HomeAgent. This agent acts as a central "manager" or "brain" for the entire crew. It does not perform specialized tasks itself; instead, its sole purpose is to receive user queries, analyze them, and delegate the work to the appropriate specialist agents.

This orchestration is dynamic. The HomeAgent uses its internal logic to classify the complexity and intent of each query and then chooses the most efficient collaboration pattern to generate a comprehensive response.

2. Collaboration Patterns
The HomeAgent employs several collaboration patterns depending on the query:

Simple Query (Direct Routing)
Trigger: A query with a clear, unambiguous intent that maps to a single agent's capability.

Example: "What is the stock level for product XYZ?"

Execution: The HomeAgent identifies this as an inventory question and routes it directly to the InventoryAgent. The response is passed back to the user with minimal synthesis.

Parallel Execution
Trigger: A query that requires input from multiple, independent domains simultaneously.

Example: "What is the price and stock level for our winter coats?"

Execution: The HomeAgent recognizes the need for both the PricingAgent and the InventoryAgent. It tasks both agents concurrently using asyncio.gather. This is highly efficient as the agents work in parallel. The HomeAgent then waits for both responses and synthesizes them into a single answer.

Hierarchical Delegation
Trigger: A complex, multi-step query where the output of one agent is required as the input for another.

Example: "Plan our marketing campaign for the top winter fashion trend."

Execution:

The HomeAgent first tasks the TrendAgent with identifying the top winter trend.

Once the TrendAgent responds (e.g., "The top trend is 'Luxe Knitwear'"), the HomeAgent uses this information to create a new, more specific task.

It then tasks the CustomerAgent or a marketing specialist agent to "Create a marketing campaign for 'Luxe Knitwear' targeting our Gold-tier members."

3. Agent Roster
The system is composed of a crew of five specialized agents:

Home Agent
Role: Chief AI Orchestrator

Goal: Coordinate all AI agents to deliver comprehensive, accurate, and actionable insights for retail operations.

Capabilities: Query analysis, multi-agent orchestration, response synthesis, system monitoring.

Inventory Agent
Role: Inventory Management Specialist

Goal: Optimize inventory levels to maximize availability while minimizing costs.

Capabilities: Checking stock levels, forecasting demand, optimizing inventory, and generating reorder recommendations.

Pricing Agent
Role: Revenue Optimization Specialist

Goal: Maximize revenue and profitability through intelligent pricing strategies.

Capabilities: Optimizing product pricing, analyzing competitor prices, and planning promotional campaigns.

Customer Agent
Role: Customer Experience Specialist

Goal: Deliver exceptional customer service and build lasting relationships through personalized interactions.

Capabilities: Handling customer queries, providing personalized recommendations, resolving complaints, and analyzing customer loyalty.

Trend Agent
Role: Fashion Trend Analyst

Goal: Identify and analyze fashion trends to guide inventory and marketing decisions.

Capabilities: Analyzing seasonal trends, forecasting trend adoption, and monitoring competitor trend strategies.