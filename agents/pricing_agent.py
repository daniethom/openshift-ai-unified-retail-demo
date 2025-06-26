"""
Dynamic Pricing and Revenue Optimization Agent for Meridian Retail Group
Specializes in pricing strategies, competitive analysis, and margin optimization
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

from agents.base_agent import BaseAgent, AgentCapability

logger = logging.getLogger(__name__)


class PricingAgent(BaseAgent):
    """
    Specialized agent for dynamic pricing and revenue optimization
    Handles pricing strategies, competitive analysis, and promotional planning
    """
    
    def __init__(self, mcp_servers: Dict[str, Any], data_store: Any):
        # Define agent capabilities
        capabilities = [
            AgentCapability(
                name="optimize_pricing",
                description="Optimize product pricing based on demand and competition",
                input_schema={
                    "product_id": "string",
                    "strategy": "string",
                    "constraints": "object"
                },
                output_schema={
                    "current_price": "number",
                    "recommended_price": "number",
                    "expected_impact": "object",
                    "confidence": "number"
                }
            ),
            AgentCapability(
                name="analyze_competition",
                description="Analyze competitive pricing landscape",
                input_schema={
                    "category": "string",
                    "brand": "string",
                    "include_online": "boolean"
                },
                output_schema={
                    "competitive_position": "string",
                    "price_gaps": "array",
                    "opportunities": "array"
                }
            ),
            AgentCapability(
                name="plan_promotions",
                description="Plan and optimize promotional campaigns",
                input_schema={
                    "products": "array",
                    "duration": "string",
                    "target_margin": "number"
                },
                output_schema={
                    "promotion_strategy": "object",
                    "expected_revenue": "number",
                    "margin_impact": "number"
                }
            )
        ]
        
        super().__init__(
            name="PricingAgent",
            role="Revenue Optimization Specialist",
            goal="Maximize revenue and profitability through intelligent pricing strategies while maintaining competitive position",
            backstory="""You are an expert in retail pricing strategy with deep understanding 
            of market dynamics, consumer psychology, and revenue optimization. You excel at 
            balancing profitability with competitiveness, considering factors like demand 
            elasticity, inventory levels, and seasonal trends. You understand the South African 
            retail market and can adapt global pricing strategies to local conditions.""",
            mcp_servers=mcp_servers,
            capabilities=capabilities
        )
        
        self.data_store = data_store
        
        # Pricing strategy parameters
        self.pricing_strategies = {
            "competitive": {
                "description": "Match or beat competitor prices",
                "margin_flexibility": 0.05,
                "price_adjustment_limit": 0.15
            },
            "premium": {
                "description": "Position above market for quality perception",
                "margin_target": 0.40,
                "price_premium": 0.20
            },
            "penetration": {
                "description": "Below market to gain market share",
                "margin_minimum": 0.15,
                "price_discount": 0.10
            },
            "value": {
                "description": "Optimal price-quality perception",
                "margin_target": 0.30,
                "price_flexibility": 0.10
            },
            "dynamic": {
                "description": "Adjust based on real-time demand",
                "margin_range": (0.20, 0.45),
                "adjustment_frequency": "daily"
            }
        }
        
        # Price elasticity models by category
        self.elasticity_models = {
            "fashion": {"base": -1.5, "seasonal_factor": 0.3},
            "electronics": {"base": -2.0, "seasonal_factor": 0.1},
            "homeware": {"base": -1.2, "seasonal_factor": 0.2},
            "sports": {"base": -1.8, "seasonal_factor": 0.25}
        }
        
        # Competitor mapping
        self.competitors = {
            "fashion": ["H&M", "Zara", "Cotton On", "Woolworths"],
            "electronics": ["Game", "Makro", "Incredible Connection"],
            "homeware": ["@Home", "Sheet Street", "Woolworths Home"],
            "sports": ["Sportsmans Warehouse", "Totalsports", "Cape Union Mart"]
        }
    
    def get_tools(self) -> List[Any]:
        """Return available tools for this agent"""
        return [
            self._pricing_optimization_tool,
            self._competitive_analysis_tool,
            self._promotion_planning_tool
        ]
    
    async def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a pricing-related query"""
        start_time = datetime.now()
        
        try:
            # Determine query type
            query_type = self._classify_pricing_query(query)
            
            if query_type == "optimization":
                result = await self._handle_optimization_query(query, context)
            elif query_type == "competition":
                result = await self._handle_competition_query(query, context)
            elif query_type == "promotion":
                result = await self._handle_promotion_query(query, context)
            elif query_type == "margin":
                result = await self._handle_margin_query(query, context)
            else:
                result = await self._handle_general_pricing_query(query, context)
            
            # Calculate response time and update metrics
            response_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(response_time, success=True)
            
            return {
                "status": "success",
                "query": query,
                "query_type": query_type,
                "result": result,
                "metadata": {
                    "agent": self.name,
                    "timestamp": datetime.now().isoformat(),
                    "response_time": response_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing pricing query: {e}")
            response_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(response_time, success=False)
            
            return {
                "status": "error",
                "error": str(e),
                "query": query,
                "metadata": {
                    "agent": self.name,
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def _classify_pricing_query(self, query: str) -> str:
        """Classify the type of pricing query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["optimize", "best price", "pricing strategy"]):
            return "optimization"
        elif any(word in query_lower for word in ["competitor", "competition", "market price"]):
            return "competition"
        elif any(word in query_lower for word in ["promotion", "sale", "discount", "campaign"]):
            return "promotion"
        elif any(word in query_lower for word in ["margin", "profit", "markup"]):
            return "margin"
        else:
            return "general"
    
    async def _handle_optimization_query(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle pricing optimization queries"""
        product_id = context.get("product_id")
        strategy = context.get("strategy", "value")
        constraints = context.get("constraints", {})
        
        # Get current pricing data
        current_data = await self._get_current_pricing(product_id)
        
        # Get market data
        market_data = await self._get_market_data(product_id)
        
        # Get demand data
        demand_data = await self._get_demand_data(product_id)
        
        # Calculate optimal price
        optimization_result = self._calculate_optimal_price(
            current_data,
            market_data,
            demand_data,
            strategy,
            constraints
        )
        
        # Simulate impact
        impact = self._simulate_pricing_impact(
            current_data,
            optimization_result["recommended_price"],
            demand_data
        )
        
        return {
            "product_id": product_id,
            "current_pricing": current_data,
            "optimization": optimization_result,
            "expected_impact": impact,
            "implementation": self._create_implementation_plan(
                product_id,
                optimization_result,
                strategy
            )
        }
    
    async def _handle_competition_query(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle competitive analysis queries"""
        category = context.get("category", "all")
        brand = context.get("brand", "all")
        include_online = context.get("include_online", True)
        
        # Get competitive landscape
        competitive_data = await self._analyze_competitive_landscape(
            category,
            brand,
            include_online
        )
        
        # Identify positioning
        positioning = self._determine_price_positioning(competitive_data)
        
        # Find opportunities
        opportunities = self._identify_pricing_opportunities(
            competitive_data,
            positioning
        )
        
        # Generate recommendations
        recommendations = self._generate_competitive_recommendations(
            positioning,
            opportunities
        )
        
        return {
            "market_analysis": competitive_data,
            "current_positioning": positioning,
            "opportunities": opportunities,
            "recommendations": recommendations,
            "competitive_metrics": self._calculate_competitive_metrics(competitive_data)
        }
    
    async def _handle_promotion_query(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle promotional planning queries"""
        products = context.get("products", [])
        duration = context.get("duration", "7_days")
        target_margin = context.get("target_margin", 0.20)
        promotion_type = context.get("type", "percentage_off")
        
        # Analyze products for promotion
        product_analysis = await self._analyze_products_for_promotion(products)
        
        # Design promotion strategy
        promotion_strategy = self._design_promotion_strategy(
            product_analysis,
            duration,
            target_margin,
            promotion_type
        )
        
        # Forecast impact
        forecast = self._forecast_promotion_impact(
            promotion_strategy,
            product_analysis
        )
        
        # Optimize promotion mix
        optimized_promotion = self._optimize_promotion_mix(
            promotion_strategy,
            forecast,
            target_margin
        )
        
        return {
            "promotion_plan": optimized_promotion,
            "forecast": forecast,
            "roi_analysis": self._calculate_promotion_roi(optimized_promotion, forecast),
            "implementation_timeline": self._create_promotion_timeline(
                optimized_promotion,
                duration
            )
        }
    
    async def _handle_margin_query(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle margin analysis queries"""
        product_id = context.get("product_id")
        category = context.get("category")
        time_period = context.get("time_period", "current")
        
        # Get margin data
        margin_data = await self._get_margin_data(product_id, category, time_period)
        
        # Analyze margin trends
        margin_trends = self._analyze_margin_trends(margin_data)
        
        # Identify margin optimization opportunities
        optimization_opportunities = self._identify_margin_opportunities(
            margin_data,
            margin_trends
        )
        
        # Generate margin improvement plan
        improvement_plan = self._create_margin_improvement_plan(
            optimization_opportunities,
            margin_data
        )
        
        return {
            "current_margins": margin_data,
            "trends": margin_trends,
            "opportunities": optimization_opportunities,
            "improvement_plan": improvement_plan,
            "projected_impact": self._project_margin_impact(improvement_plan)
        }
    
    async def _handle_general_pricing_query(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle general pricing queries"""
        return {
            "overview": await self._get_pricing_overview(),
            "key_metrics": self._get_pricing_kpis(),
            "active_strategies": self._get_active_strategies(),
            "alerts": self._get_pricing_alerts(),
            "recommendations": self._get_general_pricing_recommendations()
        }
    
    async def _get_current_pricing(self, product_id: str) -> Dict[str, Any]:
        """Get current pricing data for a product"""
        # In production, query actual pricing database
        # Mock data for demo
        pricing_data = {
            "MF-BLZ-001": {
                "product_id": "MF-BLZ-001",
                "name": "Executive Blazer",
                "current_price": 1899,
                "cost": 1140,
                "margin": 0.40,
                "list_price": 2299,
                "min_price": 1499,
                "price_history": [
                    {"date": "2024-01-01", "price": 2299},
                    {"date": "2024-03-15", "price": 1899}
                ]
            }
        }
        
        return pricing_data.get(product_id, {
            "product_id": product_id,
            "current_price": 999,
            "cost": 600,
            "margin": 0.40
        })
    
    async def _get_market_data(self, product_id: str) -> Dict[str, Any]:
        """Get market data for competitive analysis"""
        # Search for competitor prices using MCP search server
        if self.mcp_servers.get("search_server"):
            search_results = await self.mcp_servers["search_server"].search(
                f"price comparison {product_id} South Africa retailers"
            )
        else:
            search_results = {}
        
        # Mock enhanced market data
        return {
            "competitor_prices": [
                {"competitor": "H&M", "price": 1799, "in_stock": True},
                {"competitor": "Zara", "price": 2199, "in_stock": True},
                {"competitor": "Woolworths", "price": 2499, "in_stock": False}
            ],
            "market_average": 2166,
            "price_range": {"min": 1799, "max": 2499},
            "our_position": "below_average",
            "online_prices": search_results.get("prices", [])
        }
    
    async def _get_demand_data(self, product_id: str) -> Dict[str, Any]:
        """Get demand data for price optimization"""
        # Query analytics for demand patterns
        if self.mcp_servers.get("analytics_server"):
            demand_analytics = await self.mcp_servers["analytics_server"].get_demand_analytics(
                product_id
            )
        else:
            demand_analytics = {}
        
        # Mock demand data
        return {
            "current_demand": "high",
            "trend": "increasing",
            "seasonality_factor": 1.2,
            "price_sensitivity": -1.5,  # Price elasticity
            "stock_level": "medium",
            "days_of_supply": 21,
            "conversion_rate": 0.08,
            "analytics": demand_analytics
        }
    
    def _calculate_optimal_price(
        self,
        current_data: Dict[str, Any],
        market_data: Dict[str, Any],
        demand_data: Dict[str, Any],
        strategy: str,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate optimal price based on strategy and data"""
        current_price = current_data["current_price"]
        cost = current_data["cost"]
        market_avg = market_data["market_average"]
        elasticity = demand_data["price_sensitivity"]
        
        # Get strategy parameters
        strategy_params = self.pricing_strategies.get(strategy, self.pricing_strategies["value"])
        
        # Base calculation depends on strategy
        if strategy == "competitive":
            base_price = market_avg * (1 - strategy_params["price_adjustment_limit"])
        elif strategy == "premium":
            base_price = market_avg * (1 + strategy_params["price_premium"])
        elif strategy == "penetration":
            base_price = market_avg * (1 - strategy_params["price_discount"])
        elif strategy == "dynamic":
            # Dynamic pricing based on demand
            demand_factor = 1.0
            if demand_data["current_demand"] == "high":
                demand_factor = 1.1
            elif demand_data["current_demand"] == "low":
                demand_factor = 0.9
            
            base_price = current_price * demand_factor
        else:  # value strategy
            # Find optimal price point
            base_price = self._find_value_optimal_price(
                cost, market_avg, elasticity
            )
        
        # Apply constraints
        min_price = constraints.get("min_price", cost * 1.15)
        max_price = constraints.get("max_price", market_avg * 1.5)
        
        recommended_price = max(min_price, min(base_price, max_price))
        
        # Round to pricing points
        recommended_price = self._round_to_price_point(recommended_price)
        
        # Calculate confidence
        confidence = self._calculate_pricing_confidence(
            recommended_price,
            current_price,
            market_data,
            demand_data
        )
        
        return {
            "recommended_price": recommended_price,
            "price_change": recommended_price - current_price,
            "price_change_pct": (recommended_price - current_price) / current_price,
            "new_margin": (recommended_price - cost) / recommended_price,
            "strategy_used": strategy,
            "confidence": confidence,
            "reasoning": self._explain_pricing_decision(
                strategy, recommended_price, current_price, market_avg
            )
        }
    
    def _find_value_optimal_price(
        self,
        cost: float,
        market_avg: float,
        elasticity: float
    ) -> float:
        """Find optimal price for value strategy"""
        # Simple optimization: maximize (price - cost) * demand
        # Demand = base_demand * (price / market_avg) ^ elasticity
        
        # Optimal price formula for constant elasticity
        optimal_multiplier = -1 / (elasticity + 1)
        optimal_price = cost * optimal_multiplier
        
        # Constrain to reasonable range
        optimal_price = max(cost * 1.3, min(optimal_price, market_avg * 1.2))
        
        return optimal_price
    
    def _round_to_price_point(self, price: float) -> float:
        """Round price to psychological pricing points"""
        if price < 100:
            # Round to .99
            return int(price) + 0.99
        elif price < 1000:
            # Round to nearest 9
            return round(price, -1) - 1
        else:
            # Round to nearest 99
            return round(price, -2) - 1
    
    def _calculate_pricing_confidence(
        self,
        recommended_price: float,
        current_price: float,
        market_data: Dict[str, Any],
        demand_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence in pricing recommendation"""
        confidence = 0.7  # Base confidence
        
        # Adjust based on market position
        if market_data["price_range"]["min"] <= recommended_price <= market_data["price_range"]["max"]:
            confidence += 0.1
        
        # Adjust based on demand signals
        if demand_data["current_demand"] in ["high", "very_high"]:
            confidence += 0.1
        
        # Reduce confidence for large price changes
        price_change_pct = abs((recommended_price - current_price) / current_price)
        if price_change_pct > 0.2:
            confidence -= 0.2
        elif price_change_pct > 0.1:
            confidence -= 0.1
        
        return min(max(confidence, 0.3), 0.95)
    
    def _explain_pricing_decision(
        self,
        strategy: str,
        recommended_price: float,
        current_price: float,
        market_avg: float
    ) -> str:
        """Generate explanation for pricing decision"""
        price_change = recommended_price - current_price
        
        explanations = {
            "competitive": f"Pricing set to match market competition. Currently {((market_avg - recommended_price) / market_avg * 100):.1f}% below market average.",
            "premium": f"Premium pricing reflects superior quality and brand value. Positioned {((recommended_price - market_avg) / market_avg * 100):.1f}% above market.",
            "penetration": f"Aggressive pricing to capture market share. Offering {((market_avg - recommended_price) / market_avg * 100):.1f}% below competitors.",
            "value": f"Optimal price balances customer value perception with profitability.",
            "dynamic": f"Price adjusted based on current demand and inventory levels."
        }
        
        base_explanation = explanations.get(strategy, "Price optimized based on market conditions.")
        
        if price_change > 0:
            return f"{base_explanation} Increasing price by R{price_change:.0f} to capture value."
        elif price_change < 0:
            return f"{base_explanation} Reducing price by R{abs(price_change):.0f} to boost demand."
        else:
            return f"{base_explanation} Current price is optimal."
    
    def _simulate_pricing_impact(
        self,
        current_data: Dict[str, Any],
        new_price: float,
        demand_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate impact of price change"""
        current_price = current_data["current_price"]
        elasticity = demand_data["price_sensitivity"]
        
        # Calculate demand change
        price_ratio = new_price / current_price
        demand_multiplier = price_ratio ** elasticity
        
        # Estimate volume and revenue impact
        base_volume = 100  # Baseline units
        new_volume = base_volume * demand_multiplier
        
        current_revenue = current_price * base_volume
        new_revenue = new_price * new_volume
        
        # Calculate profit impact
        cost = current_data["cost"]
        current_profit = (current_price - cost) * base_volume
        new_profit = (new_price - cost) * new_volume
        
        return {
            "volume_change": (new_volume - base_volume) / base_volume,
            "revenue_change": (new_revenue - current_revenue) / current_revenue,
            "profit_change": (new_profit - current_profit) / current_profit,
            "new_margin": (new_price - cost) / new_price,
            "break_even_volume": cost * base_volume / (new_price - cost),
            "payback_period_days": 7 if new_profit > current_profit else 30
        }
    
    def _create_implementation_plan(
        self,
        product_id: str,
        optimization_result: Dict[str, Any],
        strategy: str
    ) -> Dict[str, Any]:
        """Create implementation plan for price change"""
        plan = {
            "product_id": product_id,
            "current_to_new": f"R{optimization_result['recommended_price']:.2f}",
            "strategy": strategy,
            "phases": []
        }
        
        # Determine implementation approach
        price_change_pct = abs(optimization_result["price_change_pct"])
        
        if price_change_pct < 0.05:
            # Small change - immediate
            plan["phases"].append({
                "phase": 1,
                "action": "Direct price update",
                "timing": "Immediate",
                "risk": "Low"
            })
        elif price_change_pct < 0.15:
            # Moderate change - phased
            plan["phases"].extend([
                {
                    "phase": 1,
                    "action": "Test in select stores",
                    "timing": "Week 1",
                    "risk": "Low"
                },
                {
                    "phase": 2,
                    "action": "Roll out to all locations",
                    "timing": "Week 2",
                    "risk": "Medium"
                }
            ])
        else:
            # Large change - gradual with communication
            plan["phases"].extend([
                {
                    "phase": 1,
                    "action": "Customer communication",
                    "timing": "Week 1",
                    "risk": "Medium"
                },
                {
                    "phase": 2,
                    "action": "Gradual price adjustment (50%)",
                    "timing": "Week 2",
                    "risk": "Medium"
                },
                {
                    "phase": 3,
                    "action": "Final price adjustment",
                    "timing": "Week 4",
                    "risk": "High"
                }
            ])
        
        plan["monitoring"] = [
            "Daily sales volume",
            "Customer feedback",
            "Competitor response",
            "Inventory levels"
        ]
        
        return plan
    
    async def _analyze_competitive_landscape(
        self,
        category: str,
        brand: str,
        include_online: bool
    ) -> Dict[str, Any]:
        """Analyze competitive pricing landscape"""
        # Get competitors for category
        relevant_competitors = self.competitors.get(category, self.competitors["fashion"])
        
        # Mock competitive data
        competitive_data = {
            "category": category,
            "competitors_analyzed": len(relevant_competitors),
            "price_comparison": [],
            "market_dynamics": {
                "trend": "stable",
                "volatility": "medium",
                "promotional_intensity": "high"
            }
        }
        
        # Generate price comparisons
        for competitor in relevant_competitors:
            competitive_data["price_comparison"].append({
                "competitor": competitor,
                "avg_price_index": 100 + np.random.randint(-20, 20),
                "price_range": "R" + str(np.random.randint(500, 2000)) + "-R" + str(np.random.randint(2000, 5000)),
                "promotional_frequency": np.random.choice(["high", "medium", "low"]),
                "market_share": np.random.randint(5, 25)
            })
        
        # Add online competition if requested
        if include_online:
            competitive_data["online_competition"] = {
                "amazon": {"present": False, "threat_level": "low"},
                "takealot": {"present": True, "threat_level": "high"},
                "local_online": {"present": True, "threat_level": "medium"}
            }
        
        return competitive_data
    
    def _determine_price_positioning(
        self,
        competitive_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine our price positioning in the market"""
        # Calculate our position
        competitor_indices = [
            comp["avg_price_index"] 
            for comp in competitive_data["price_comparison"]
        ]
        
        our_index = 100  # Baseline
        market_avg = np.mean(competitor_indices)
        
        if our_index > market_avg + 10:
            position = "premium"
        elif our_index < market_avg - 10:
            position = "value"
        else:
            position = "competitive"
        
        return {
            "current_position": position,
            "price_index": our_index,
            "market_average_index": market_avg,
            "percentile": self._calculate_percentile(our_index, competitor_indices),
            "gap_to_leader": max(competitor_indices) - our_index,
            "gap_to_value": our_index - min(competitor_indices)
        }
    
    def _calculate_percentile(self, value: float, population: List[float]) -> float:
        """Calculate percentile position"""
        below = sum(1 for x in population if x < value)
        return (below / len(population)) * 100
    
    def _identify_pricing_opportunities(
        self,
        competitive_data: Dict[str, Any],
        positioning: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify pricing opportunities"""
        opportunities = []
        
        # Check for price leadership opportunity
        if positioning["percentile"] > 70 and competitive_data["market_dynamics"]["trend"] == "growing":
            opportunities.append({
                "type": "price_leadership",
                "description": "Opportunity to establish price leadership in growing market",
                "potential_impact": "5-10% margin improvement",
                "risk": "medium",
                "action": "Gradual price increases on key products"
            })
        
        # Check for competitive gaps
        price_comparisons = competitive_data["price_comparison"]
        high_promo_competitors = [
            c for c in price_comparisons 
            if c["promotional_frequency"] == "high"
        ]
        
        if len(high_promo_competitors) > len(price_comparisons) / 2:
            opportunities.append({
                "type": "promotional_response",
                "description": "High promotional activity from competitors",
                "potential_impact": "Defend market share",
                "risk": "high",
                "action": "Implement targeted promotions"
            })
        
        # Check for value positioning opportunity
        if positioning["current_position"] == "premium" and positioning["gap_to_value"] > 30:
            opportunities.append({
                "type": "value_line",
                "description": "Introduce value product line",
                "potential_impact": "Capture price-sensitive segment",
                "risk": "low",
                "action": "Launch fighter brand or value SKUs"
            })
        
        return opportunities
    
    def _generate_competitive_recommendations(
        self,
        positioning: Dict[str, Any],
        opportunities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate competitive pricing recommendations"""
        recommendations = []
        
        # Position-based recommendations
        if positioning["current_position"] == "premium":
            recommendations.append({
                "strategy": "maintain_premium",
                "tactics": [
                    "Enhance value proposition communication",
                    "Limited exclusive offers for loyalty members",
                    "Focus on quality differentiation"
                ],
                "pricing_action": "Maintain current pricing with selective increases"
            })
        elif positioning["current_position"] == "value":
            recommendations.append({
                "strategy": "value_leadership",
                "tactics": [
                    "Aggressive promotional calendar",
                    "Bundle offers",
                    "Price match guarantees"
                ],
                "pricing_action": "Maintain aggressive pricing with volume focus"
            })
        else:
            recommendations.append({
                "strategy": "competitive_parity",
                "tactics": [
                    "Match key competitor prices",
                    "Differentiate through service",
                    "Selective promotions"
                ],
                "pricing_action": "Dynamic pricing to maintain market position"
            })
        
        # Add opportunity-based recommendations
        for opportunity in opportunities[:2]:  # Top 2 opportunities
            recommendations.append({
                "strategy": opportunity["type"],
                "tactics": [opportunity["action"]],
                "pricing_action": opportunity["description"],
                "expected_impact": opportunity["potential_impact"]
            })
        
        return recommendations
    
    def _calculate_competitive_metrics(
        self,
        competitive_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate key competitive metrics"""
        price_indices = [c["avg_price_index"] for c in competitive_data["price_comparison"]]
        
        return {
            "price_spread": max(price_indices) - min(price_indices),
            "market_concentration": self._calculate_herfindahl_index(
                [c["market_share"] for c in competitive_data["price_comparison"]]
            ),
            "competitive_intensity": len([
                c for c in competitive_data["price_comparison"]
                if c["promotional_frequency"] == "high"
            ]) / len(competitive_data["price_comparison"]),
            "price_volatility": np.std(price_indices) / np.mean(price_indices)
        }
    
    def _calculate_herfindahl_index(self, market_shares: List[float]) -> float:
        """Calculate Herfindahl index for market concentration"""
        return sum(share ** 2 for share in market_shares) / 10000
    
    async def _analyze_products_for_promotion(
        self,
        products: List[str]
    ) -> List[Dict[str, Any]]:
        """Analyze products for promotional suitability"""
        analyzed_products = []
        
        for product_id in products:
            # Get product data
            product_data = await self._get_current_pricing(product_id)
            
            # Analyze suitability
            suitability_score = self._calculate_promotion_suitability(product_data)
            
            analyzed_products.append({
                "product_id": product_id,
                "current_price": product_data.get("current_price", 0),
                "margin": product_data.get("margin", 0),
                "suitability_score": suitability_score,
                "promotion_history": self._get_promotion_history(product_id),
                "inventory_level": np.random.choice(["high", "medium", "low"])
            })
        
        return analyzed_products
    
    def _calculate_promotion_suitability(self, product_data: Dict[str, Any]) -> float:
        """Calculate how suitable a product is for promotion"""
        score = 0.5
        
        # High margin products are better for promotion
        if product_data.get("margin", 0) > 0.4:
            score += 0.2
        
        # Products with price history are safer
        if len(product_data.get("price_history", [])) > 2:
            score += 0.1
        
        # Random factors for demo
        score += np.random.uniform(-0.1, 0.2)
        
        return min(max(score, 0), 1)
    
    def _get_promotion_history(self, product_id: str) -> List[Dict[str, Any]]:
        """Get historical promotion data"""
        # Mock promotion history
        return [
            {
                "date": (datetime.now() - timedelta(days=60)).isoformat(),
                "type": "percentage",
                "discount": 20,
                "performance": "good"
            }
        ]
    
    def _design_promotion_strategy(
        self,
        product_analysis: List[Dict[str, Any]],
        duration: str,
        target_margin: float,
        promotion_type: str
    ) -> Dict[str, Any]:
        """Design comprehensive promotion strategy"""
        # Calculate duration in days
        duration_days = {
            "3_days": 3,
            "7_days": 7,
            "14_days": 14,
            "30_days": 30
        }.get(duration, 7)
        
        strategy = {
            "name": f"{duration} {promotion_type} promotion",
            "duration_days": duration_days,
            "type": promotion_type,
            "products": [],
            "target_margin": target_margin,
            "mechanics": self._determine_promotion_mechanics(promotion_type),
            "communication": self._plan_promotion_communication(duration_days)
        }
        
        # Design product-specific promotions
        for product in product_analysis:
            if product["suitability_score"] > 0.5:
                discount = self._calculate_optimal_discount(
                    product,
                    target_margin,
                    promotion_type
                )
                
                strategy["products"].append({
                    "product_id": product["product_id"],
                    "original_price": product["current_price"],
                    "discount": discount,
                    "promo_price": product["current_price"] * (1 - discount),
                    "expected_uplift": self._estimate_volume_uplift(discount)
                })
        
        return strategy
    
    def _determine_promotion_mechanics(self, promotion_type: str) -> Dict[str, Any]:
        """Determine promotion mechanics"""
        mechanics = {
            "percentage_off": {
                "display": "X% OFF",
                "calculation": "Simple percentage discount",
                "customer_appeal": "high"
            },
            "rand_off": {
                "display": "SAVE RX",
                "calculation": "Fixed rand amount off",
                "customer_appeal": "medium"
            },
            "bogo": {
                "display": "Buy One Get One",
                "calculation": "Second item free or discounted",
                "customer_appeal": "very_high"
            },
            "bundle": {
                "display": "Bundle Deal",
                "calculation": "Discount on multiple items",
                "customer_appeal": "high"
            }
        }
        
        return mechanics.get(promotion_type, mechanics["percentage_off"])
    
    def _plan_promotion_communication(self, duration_days: int) -> Dict[str, Any]:
        """Plan promotion communication strategy"""
        if duration_days <= 3:
            return {
                "urgency": "high",
                "channels": ["email", "sms", "push"],
                "frequency": "daily",
                "message": "Limited time flash sale!"
            }
        elif duration_days <= 7:
            return {
                "urgency": "medium",
                "channels": ["email", "social", "website"],
                "frequency": "every_2_days",
                "message": "This week only!"
            }
        else:
            return {
                "urgency": "low",
                "channels": ["email", "print", "in_store"],
                "frequency": "weekly",
                "message": "Month-long savings event"
            }
    
    def _calculate_optimal_discount(
        self,
        product: Dict[str, Any],
        target_margin: float,
        promotion_type: str
    ) -> float:
        """Calculate optimal discount for product"""
        current_margin = product["margin"]
        max_discount = current_margin - target_margin
        
        # Base discount on promotion type
        base_discounts = {
            "percentage_off": 0.20,
            "rand_off": 0.15,
            "bogo": 0.50,  # Effective 50% on second item
            "bundle": 0.25
        }
        
        base_discount = base_discounts.get(promotion_type, 0.20)
        
        # Adjust based on inventory
        if product["inventory_level"] == "high":
            base_discount *= 1.2
        elif product["inventory_level"] == "low":
            base_discount *= 0.8
        
        # Ensure we maintain target margin
        optimal_discount = min(base_discount, max_discount)
        
        # Round to nice percentages
        return round(optimal_discount * 20) / 20  # Round to nearest 5%
    
    def _estimate_volume_uplift(self, discount: float) -> float:
        """Estimate volume uplift from discount"""
        # Simple elasticity model
        base_elasticity = -2.0  # 2% volume increase per 1% price decrease
        uplift = discount * abs(base_elasticity)
        
        # Add some randomness for realism
        uplift *= np.random.uniform(0.8, 1.2)
        
        return uplift
    
    def _forecast_promotion_impact(
        self,
        promotion_strategy: Dict[str, Any],
        product_analysis: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Forecast financial impact of promotion"""
        total_revenue_baseline = 0
        total_revenue_promo = 0
        total_units_baseline = 0
        total_units_promo = 0
        
        for promo_product in promotion_strategy["products"]:
            # Find matching product analysis
            product = next(
                (p for p in product_analysis if p["product_id"] == promo_product["product_id"]),
                None
            )
            
            if product:
                # Baseline (without promotion)
                baseline_units = 100  # Daily baseline
                baseline_revenue = baseline_units * promo_product["original_price"]
                
                # With promotion
                uplift = promo_product["expected_uplift"]
                promo_units = baseline_units * (1 + uplift)
                promo_revenue = promo_units * promo_product["promo_price"]
                
                total_units_baseline += baseline_units * promotion_strategy["duration_days"]
                total_units_promo += promo_units * promotion_strategy["duration_days"]
                total_revenue_baseline += baseline_revenue * promotion_strategy["duration_days"]
                total_revenue_promo += promo_revenue * promotion_strategy["duration_days"]
        
        return {
            "baseline_revenue": total_revenue_baseline,
            "promotional_revenue": total_revenue_promo,
            "incremental_revenue": total_revenue_promo - total_revenue_baseline,
            "baseline_units": total_units_baseline,
            "promotional_units": total_units_promo,
            "incremental_units": total_units_promo - total_units_baseline,
            "revenue_impact_pct": (total_revenue_promo - total_revenue_baseline) / total_revenue_baseline if total_revenue_baseline > 0 else 0,
            "volume_impact_pct": (total_units_promo - total_units_baseline) / total_units_baseline if total_units_baseline > 0 else 0
        }
    
    def _optimize_promotion_mix(
        self,
        promotion_strategy: Dict[str, Any],
        forecast: Dict[str, Any],
        target_margin: float
    ) -> Dict[str, Any]:
        """Optimize the promotion mix for maximum impact"""
        optimized = promotion_strategy.copy()
        
        # Check if we're meeting targets
        if forecast["revenue_impact_pct"] < -0.1:  # Revenue dropping too much
            # Reduce discounts
            for product in optimized["products"]:
                product["discount"] *= 0.8
                product["promo_price"] = product["original_price"] * (1 - product["discount"])
        
        # Add tiering for better margin management
        optimized["tiering"] = {
            "tier_1": {
                "description": "Deep discounts on slow movers",
                "products": [p for p in optimized["products"] if p["discount"] > 0.25],
                "target_audience": "bargain_hunters"
            },
            "tier_2": {
                "description": "Moderate discounts on popular items",
                "products": [p for p in optimized["products"] if 0.15 <= p["discount"] <= 0.25],
                "target_audience": "regular_customers"
            },
            "tier_3": {
                "description": "Token discounts on premium items",
                "products": [p for p in optimized["products"] if p["discount"] < 0.15],
                "target_audience": "premium_shoppers"
            }
        }
        
        return optimized
    
    def _calculate_promotion_roi(
        self,
        optimized_promotion: Dict[str, Any],
        forecast: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate ROI for promotion"""
        # Calculate costs
        discount_cost = forecast["baseline_revenue"] - forecast["promotional_revenue"]
        marketing_cost = len(optimized_promotion["products"]) * 1000  # R1000 per product marketing
        operational_cost = optimized_promotion["duration_days"] * 500  # R500 per day ops cost
        
        total_cost = abs(discount_cost) + marketing_cost + operational_cost
        
        # Calculate benefits
        incremental_profit = forecast["incremental_revenue"] * 0.3  # 30% margin on incremental
        future_value = forecast["incremental_units"] * 50  # R50 future value per new customer
        
        total_benefit = incremental_profit + future_value
        
        return {
            "total_cost": total_cost,
            "total_benefit": total_benefit,
            "net_benefit": total_benefit - total_cost,
            "roi_percentage": (total_benefit - total_cost) / total_cost * 100 if total_cost > 0 else 0,
            "payback_days": total_cost / (total_benefit / optimized_promotion["duration_days"]) if total_benefit > 0 else float('inf'),
            "break_even_units": total_cost / (forecast["promotional_revenue"] / forecast["promotional_units"]) if forecast["promotional_units"] > 0 else 0
        }
    
    def _create_promotion_timeline(
        self,
        optimized_promotion: Dict[str, Any],
        duration: str
    ) -> List[Dict[str, Any]]:
        """Create detailed promotion timeline"""
        timeline = []
        duration_days = optimized_promotion["duration_days"]
        
        # Pre-promotion
        timeline.append({
            "phase": "pre_promotion",
            "days_before": 7,
            "actions": [
                "Finalize pricing",
                "Prepare marketing materials",
                "Train staff",
                "Update systems"
            ]
        })
        
        # Launch
        timeline.append({
            "phase": "launch",
            "day": 1,
            "actions": [
                "Email blast",
                "Social media announcement",
                "Update website",
                "In-store signage"
            ]
        })
        
        # Mid-promotion
        if duration_days > 7:
            timeline.append({
                "phase": "mid_promotion",
                "day": duration_days // 2,
                "actions": [
                    "Performance review",
                    "Adjustment if needed",
                    "Reminder communications"
                ]
            })
        
        # Final push
        timeline.append({
            "phase": "final_push",
            "day": duration_days - 1,
            "actions": [
                "Last chance messaging",
                "Urgency creation",
                "Stock clearance"
            ]
        })
        
        # Post-promotion
        timeline.append({
            "phase": "post_promotion",
            "days_after": 3,
            "actions": [
                "Performance analysis",
                "Customer feedback",
                "Lesson learned",
                "Plan next promotion"
            ]
        })
        
        return timeline
    
    async def _get_margin_data(
        self,
        product_id: Optional[str],
        category: Optional[str],
        time_period: str
    ) -> Dict[str, Any]:
        """Get margin data for analysis"""
        if product_id:
            # Product-specific margins
            product_data = await self._get_current_pricing(product_id)
            return {
                "level": "product",
                "id": product_id,
                "current_margin": product_data.get("margin", 0),
                "margin_trend": "stable",
                "historical_margins": [
                    {"date": "2024-Q1", "margin": 0.38},
                    {"date": "2024-Q2", "margin": 0.40},
                    {"date": "2024-Q3", "margin": 0.39}
                ]
            }
        elif category:
            # Category margins
            return {
                "level": "category",
                "category": category,
                "avg_margin": 0.35,
                "margin_range": {"min": 0.25, "max": 0.45},
                "top_performers": [
                    {"product": "Premium items", "margin": 0.45},
                    {"product": "Exclusive lines", "margin": 0.42}
                ],
                "low_performers": [
                    {"product": "Basic items", "margin": 0.25},
                    {"product": "Clearance", "margin": 0.15}
                ]
            }
        else:
            # Overall margins
            return {
                "level": "company",
                "overall_margin": 0.32,
                "by_brand": {
                    "meridian_fashion": 0.38,
                    "stratus": 0.28,
                    "casa_living": 0.35,
                    "vertex_sports": 0.30
                }
            }
    
    def _analyze_margin_trends(self, margin_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in margin data"""
        trends = {
            "direction": "stable",
            "volatility": "low",
            "seasonal_pattern": False,
            "key_drivers": []
        }
        
        if margin_data["level"] == "product":
            # Analyze historical margins
            margins = [m["margin"] for m in margin_data.get("historical_margins", [])]
            if margins:
                if margins[-1] > margins[0]:
                    trends["direction"] = "improving"
                elif margins[-1] < margins[0]:
                    trends["direction"] = "declining"
                
                trends["volatility"] = "high" if np.std(margins) > 0.05 else "low"
        
        # Identify key drivers
        if trends["direction"] == "improving":
            trends["key_drivers"] = ["Better sourcing", "Price optimization", "Product mix"]
        elif trends["direction"] == "declining":
            trends["key_drivers"] = ["Increased competition", "Rising costs", "Promotional pressure"]
        
        return trends
    
    def _identify_margin_opportunities(
        self,
        margin_data: Dict[str, Any],
        margin_trends: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify margin improvement opportunities"""
        opportunities = []
        
        if margin_data["level"] == "category" and margin_data.get("low_performers"):
            opportunities.append({
                "type": "product_optimization",
                "description": "Improve margins on low-performing products",
                "products": margin_data["low_performers"],
                "potential_impact": "2-3% margin improvement",
                "actions": [
                    "Renegotiate supplier terms",
                    "Reduce SKU count",
                    "Reposition pricing"
                ]
            })
        
        if margin_trends["volatility"] == "high":
            opportunities.append({
                "type": "stabilization",
                "description": "Reduce margin volatility",
                "potential_impact": "More predictable profitability",
                "actions": [
                    "Implement dynamic pricing",
                    "Hedge input costs",
                    "Standardize promotional calendar"
                ]
            })
        
        # Always suggest value engineering
        opportunities.append({
            "type": "value_engineering",
            "description": "Optimize product specifications",
            "potential_impact": "1-2% cost reduction",
            "actions": [
                "Review product specifications",
                "Identify over-engineering",
                "Supplier collaboration"
            ]
        })
        
        return opportunities
    
    def _create_margin_improvement_plan(
        self,
        opportunities: List[Dict[str, Any]],
        margin_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive margin improvement plan"""
        current_margin = margin_data.get("current_margin", 0.32)
        
        plan = {
            "current_state": {
                "margin": current_margin,
                "annual_profit": current_margin * 10000000  # Assumed revenue
            },
            "target_state": {
                "margin": current_margin + 0.03,  # 3% improvement
                "timeline": "6 months"
            },
            "initiatives": []
        }
        
        # Convert opportunities to initiatives
        for i, opp in enumerate(opportunities[:3]):  # Top 3
            plan["initiatives"].append({
                "id": i + 1,
                "name": opp["description"],
                "type": opp["type"],
                "actions": opp["actions"],
                "timeline": f"Month {i*2 + 1}-{i*2 + 2}",
                "expected_impact": opp["potential_impact"],
                "investment_required": "Low" if i == 0 else "Medium",
                "risk": "Low" if opp["type"] == "value_engineering" else "Medium"
            })
        
        return plan
    
    def _project_margin_impact(self, improvement_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Project financial impact of margin improvements"""
        current_margin = improvement_plan["current_state"]["margin"]
        target_margin = improvement_plan["target_state"]["margin"]
        annual_revenue = 10000000  # Assumed
        
        current_profit = annual_revenue * current_margin
        target_profit = annual_revenue * target_margin
        
        return {
            "annual_profit_increase": target_profit - current_profit,
            "roi_percentage": ((target_profit - current_profit) / current_profit) * 100,
            "break_even_months": 3,  # Assumed investment payback
            "5_year_value": (target_profit - current_profit) * 5,
            "risk_adjusted_value": (target_profit - current_profit) * 0.8  # 80% success probability
        }
    
    async def _get_pricing_overview(self) -> Dict[str, Any]:
        """Get overall pricing overview"""
        return {
            "total_products_managed": 5420,
            "active_price_changes": 23,
            "active_promotions": 8,
            "avg_margin": 0.32,
            "price_compliance": 0.96,
            "last_optimization_run": (datetime.now() - timedelta(days=2)).isoformat()
        }
    
    def _get_pricing_kpis(self) -> Dict[str, Any]:
        """Get key pricing KPIs"""
        return {
            "price_realization": 0.94,  # Actual vs list price
            "markdown_rate": 0.12,  # % of sales on markdown
            "promotion_effectiveness": 1.8,  # Lift factor
            "competitive_index": 98,  # vs market
            "margin_achievement": 0.96  # vs target
        }
    
    def _get_active_strategies(self) -> List[Dict[str, Any]]:
        """Get currently active pricing strategies"""
        return [
            {
                "brand": "meridian_fashion",
                "strategy": "premium",
                "status": "active",
                "performance": "on_target"
            },
            {
                "brand": "stratus",
                "strategy": "competitive",
                "status": "active",
                "performance": "exceeding"
            },
            {
                "brand": "casa_living",
                "strategy": "value",
                "status": "testing",
                "performance": "monitoring"
            }
        ]
    
    def _get_pricing_alerts(self) -> List[Dict[str, Any]]:
        """Get current pricing alerts"""
        return [
            {
                "level": "warning",
                "message": "Competitor price cut detected on key items",
                "action_required": "Review competitive response",
                "products_affected": 12
            },
            {
                "level": "info",
                "message": "Promotional calendar conflicts detected",
                "action_required": "Adjust promotion schedule",
                "date": "Next week"
            }
        ]
    
    def _get_general_pricing_recommendations(self) -> List[Dict[str, Any]]:
        """Get general pricing recommendations"""
        return [
            {
                "recommendation": "Implement surge pricing for high-demand periods",
                "impact": "3-5% revenue increase",
                "effort": "medium",
                "priority": "high"
            },
            {
                "recommendation": "Harmonize pricing across channels",
                "impact": "Improve customer satisfaction",
                "effort": "high",
                "priority": "medium"
            },
            {
                "recommendation": "Develop AI-driven price optimization",
                "impact": "5-8% margin improvement",
                "effort": "high",
                "priority": "high"
            }
        ]
    
    def _pricing_optimization_tool(
        self,
        product_id: str,
        strategy: str = "value"
    ) -> Dict[str, Any]:
        """Tool wrapper for price optimization"""
        return asyncio.run(self._handle_optimization_query(
            f"Optimize pricing for {product_id}",
            {"product_id": product_id, "strategy": strategy}
        ))
    
    def _competitive_analysis_tool(
        self,
        category: str,
        brand: str = "all"
    ) -> Dict[str, Any]:
        """Tool wrapper for competitive analysis"""
        return asyncio.run(self._handle_competition_query(
            f"Analyze competition in {category}",
            {"category": category, "brand": brand}
        ))
    
    def _promotion_planning_tool(
        self,
        products: List[str],
        duration: str = "7_days"
    ) -> Dict[str, Any]:
        """Tool wrapper for promotion planning"""
        return asyncio.run(self._handle_promotion_query(
            f"Plan promotion for products",
            {"products": products, "duration": duration}
        ))
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information for UI display"""
        return {
            "name": self.name,
            "role": self.role,
            "description": "Dynamic pricing and revenue optimization specialist",
            "capabilities": [cap.name for cap in self.capabilities],
            "status": self.get_status_dict(),
            "specialties": [
                "Price optimization",
                "Competitive analysis",
                "Promotional planning",
                "Margin management",
                "Revenue maximization"
            ],
            "active_strategies": len(self._get_active_strategies()),
            "current_alerts": len(self._get_pricing_alerts())
        }
