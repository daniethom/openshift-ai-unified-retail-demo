"""
Inventory Management Agent for Meridian Retail Group
Specializes in stock management, supply chain optimization, and demand forecasting
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from agents.base_agent import BaseAgent, AgentCapability
import numpy as np

logger = logging.getLogger(__name__)


class InventoryAgent(BaseAgent):
    """
    Specialized agent for inventory management and supply chain optimization
    Handles stock levels, reordering, and demand forecasting
    """
    
    def __init__(self, mcp_servers: Dict[str, Any], data_store: Any):
        # Define agent capabilities
        capabilities = [
            AgentCapability(
                name="check_stock_levels",
                description="Check current stock levels for products",
                input_schema={
                    "product_id": "string",
                    "location": "string",
                    "include_warehouse": "boolean"
                },
                output_schema={
                    "stock_level": "object",
                    "availability": "string",
                    "reorder_status": "string"
                }
            ),
            AgentCapability(
                name="optimize_inventory",
                description="Optimize inventory levels based on demand",
                input_schema={
                    "brand": "string",
                    "category": "string",
                    "timeframe": "string"
                },
                output_schema={
                    "recommendations": "array",
                    "potential_savings": "number",
                    "service_level": "number"
                }
            ),
            AgentCapability(
                name="forecast_demand",
                description="Forecast future demand for products",
                input_schema={
                    "product_id": "string",
                    "period": "string",
                    "include_seasonality": "boolean"
                },
                output_schema={
                    "forecast": "array",
                    "confidence_interval": "object",
                    "factors": "array"
                }
            )
        ]
        
        super().__init__(
            name="InventoryAgent",
            role="Inventory Management Specialist",
            goal="Optimize inventory levels to maximize availability while minimizing costs",
            backstory="""You are an expert inventory manager with deep knowledge of 
            retail supply chain operations. You excel at balancing stock levels to meet 
            customer demand while avoiding overstock. You understand seasonality, 
            regional preferences, and can predict demand patterns accurately. Your 
            recommendations always consider both cost efficiency and customer satisfaction.""",
            mcp_servers=mcp_servers,
            capabilities=capabilities
        )
        
        self.data_store = data_store
        
        # Inventory thresholds and parameters
        self.inventory_params = {
            "safety_stock_multiplier": 1.5,
            "reorder_point_multiplier": 2.0,
            "max_stock_multiplier": 4.0,
            "stockout_cost_multiplier": 3.0,
            "holding_cost_rate": 0.25,  # 25% annual holding cost
            "service_level_target": 0.95  # 95% service level
        }
        
        # Store locations
        self.locations = {
            "cape_town": {"type": "store", "capacity": 5000},
            "johannesburg": {"type": "store", "capacity": 7000},
            "durban": {"type": "store", "capacity": 4000},
            "pretoria": {"type": "store", "capacity": 4500},
            "central_warehouse": {"type": "warehouse", "capacity": 50000}
        }
    
    def get_tools(self) -> List[Any]:
        """Return available tools for this agent"""
        return [
            self._check_stock_tool,
            self._forecast_tool,
            self._optimize_tool
        ]
    
    async def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process an inventory-related query"""
        start_time = datetime.now()
        
        try:
            # Determine query type
            query_type = self._classify_query(query)
            
            if query_type == "stock_check":
                result = await self._handle_stock_check(query, context)
            elif query_type == "optimization":
                result = await self._handle_optimization(query, context)
            elif query_type == "forecast":
                result = await self._handle_forecast(query, context)
            elif query_type == "reorder":
                result = await self._handle_reorder(query, context)
            else:
                result = await self._handle_general_query(query, context)
            
            # Calculate response time and update metrics
            response_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(response_time, success=True)
            
            return {
                "status": "success",
                "query": query,
                "result": result,
                "metadata": {
                    "agent": self.name,
                    "timestamp": datetime.now().isoformat(),
                    "response_time": response_time,
                    "query_type": query_type
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing inventory query: {e}")
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
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of inventory query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["stock", "inventory", "available", "have"]):
            return "stock_check"
        elif any(word in query_lower for word in ["optimize", "improve", "reduce cost"]):
            return "optimization"
        elif any(word in query_lower for word in ["forecast", "predict", "future", "demand"]):
            return "forecast"
        elif any(word in query_lower for word in ["reorder", "replenish", "order more"]):
            return "reorder"
        else:
            return "general"
    
    async def _handle_stock_check(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle stock level checking queries"""
        product_id = context.get("product_id")
        location = context.get("location", "all")
        include_warehouse = context.get("include_warehouse", True)
        
        # Get current stock levels
        stock_data = await self._get_stock_levels(product_id, location, include_warehouse)
        
        # Calculate availability
        availability = self._calculate_availability(stock_data)
        
        # Check reorder status
        reorder_status = self._check_reorder_status(stock_data, product_id)
        
        # Get product details from data store
        product_info = await self._get_product_info(product_id)
        
        return {
            "product": product_info,
            "stock_levels": stock_data,
            "availability": availability,
            "reorder_status": reorder_status,
            "recommendations": self._generate_stock_recommendations(
                stock_data, product_id, availability
            )
        }
    
    async def _handle_optimization(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle inventory optimization queries"""
        brand = context.get("brand", "all")
        category = context.get("category", "all")
        timeframe = context.get("timeframe", "30_days")
        
        # Get current inventory data
        inventory_data = await self._get_inventory_data(brand, category)
        
        # Analyze current performance
        performance = self._analyze_inventory_performance(inventory_data)
        
        # Generate optimization recommendations
        optimizations = self._generate_optimizations(
            inventory_data, performance, timeframe
        )
        
        # Calculate potential savings
        savings = self._calculate_potential_savings(optimizations)
        
        return {
            "current_performance": performance,
            "optimizations": optimizations,
            "potential_savings": savings,
            "implementation_plan": self._create_implementation_plan(optimizations)
        }
    
    async def _handle_forecast(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle demand forecasting queries"""
        product_id = context.get("product_id")
        period = context.get("period", "30_days")
        include_seasonality = context.get("include_seasonality", True)
        
        # Get historical data
        historical_data = await self._get_historical_data(product_id)
        
        # Generate forecast
        forecast = self._generate_forecast(
            historical_data, period, include_seasonality
        )
        
        # Identify influencing factors
        factors = self._identify_demand_factors(historical_data, product_id)
        
        return {
            "product_id": product_id,
            "forecast_period": period,
            "forecast": forecast,
            "confidence_interval": self._calculate_confidence_interval(forecast),
            "influencing_factors": factors,
            "recommendations": self._generate_forecast_recommendations(
                forecast, product_id
            )
        }
    
    async def _handle_reorder(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle reorder recommendation queries"""
        urgent_only = context.get("urgent_only", False)
        location = context.get("location", "all")
        
        # Get items needing reorder
        reorder_items = await self._get_reorder_items(location, urgent_only)
        
        # Calculate optimal order quantities
        order_recommendations = []
        for item in reorder_items:
            optimal_qty = self._calculate_optimal_order_quantity(item)
            order_recommendations.append({
                "product_id": item["product_id"],
                "current_stock": item["current_stock"],
                "reorder_point": item["reorder_point"],
                "optimal_order_qty": optimal_qty,
                "urgency": item["urgency"],
                "estimated_stockout_date": item.get("stockout_date")
            })
        
        # Group by supplier for efficiency
        supplier_orders = self._group_by_supplier(order_recommendations)
        
        return {
            "reorder_recommendations": order_recommendations,
            "supplier_orders": supplier_orders,
            "total_items": len(order_recommendations),
            "urgent_items": len([r for r in order_recommendations if r["urgency"] == "high"]),
            "estimated_cost": self._calculate_order_cost(order_recommendations)
        }
    
    async def _handle_general_query(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle general inventory queries"""
        # Provide overview of inventory health
        inventory_health = await self._assess_inventory_health()
        
        return {
            "inventory_health": inventory_health,
            "key_metrics": {
                "total_sku_count": inventory_health["total_skus"],
                "stockout_rate": inventory_health["stockout_rate"],
                "overstock_rate": inventory_health["overstock_rate"],
                "inventory_turnover": inventory_health["turnover_rate"],
                "service_level": inventory_health["service_level"]
            },
            "alerts": self._get_inventory_alerts(),
            "recommendations": self._get_general_recommendations(inventory_health)
        }
    
    async def _get_stock_levels(
        self, 
        product_id: str, 
        location: str, 
        include_warehouse: bool
    ) -> Dict[str, Any]:
        """Get current stock levels for a product"""
        stock_data = {}
        
        if location == "all":
            # Get stock for all locations
            for loc_name, loc_info in self.locations.items():
                if not include_warehouse and loc_info["type"] == "warehouse":
                    continue
                
                stock = await self._query_stock(product_id, loc_name)
                stock_data[loc_name] = stock
        else:
            # Get stock for specific location
            stock = await self._query_stock(product_id, location)
            stock_data[location] = stock
        
        return stock_data
    
    async def _query_stock(self, product_id: str, location: str) -> Dict[str, Any]:
        """Query stock for a specific product and location"""
        # In production, this would query the actual database
        # For demo, return simulated data
        return {
            "on_hand": np.random.randint(0, 100),
            "allocated": np.random.randint(0, 20),
            "available": np.random.randint(0, 80),
            "incoming": np.random.randint(0, 50),
            "last_updated": datetime.now().isoformat()
        }
    
    def _calculate_availability(self, stock_data: Dict[str, Any]) -> str:
        """Calculate overall availability status"""
        total_available = sum(
            loc_data.get("available", 0) 
            for loc_data in stock_data.values()
        )
        
        if total_available > 100:
            return "high"
        elif total_available > 20:
            return "medium"
        elif total_available > 0:
            return "low"
        else:
            return "out_of_stock"
    
    def _check_reorder_status(
        self, 
        stock_data: Dict[str, Any], 
        product_id: str
    ) -> Dict[str, Any]:
        """Check if product needs reordering"""
        total_stock = sum(
            loc_data.get("on_hand", 0) 
            for loc_data in stock_data.values()
        )
        
        # Get reorder point (simplified calculation)
        avg_daily_demand = 10  # Would be calculated from historical data
        lead_time_days = 7
        safety_stock = avg_daily_demand * self.inventory_params["safety_stock_multiplier"]
        reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
        
        if total_stock <= reorder_point:
            days_until_stockout = max(0, total_stock / avg_daily_demand)
            return {
                "needs_reorder": True,
                "urgency": "high" if days_until_stockout < 3 else "medium",
                "reorder_point": reorder_point,
                "current_stock": total_stock,
                "days_until_stockout": days_until_stockout
            }
        
        return {
            "needs_reorder": False,
            "urgency": "none",
            "reorder_point": reorder_point,
            "current_stock": total_stock,
            "stock_coverage_days": total_stock / avg_daily_demand
        }
    
    async def _get_product_info(self, product_id: str) -> Dict[str, Any]:
        """Get product information"""
        # Query MCP Analytics server for product details
        if self.mcp_servers.get("analytics_server"):
            return await self.mcp_servers["analytics_server"].get_product_details(
                product_id
            )
        
        # Fallback to mock data
        return {
            "product_id": product_id,
            "name": "Product Name",
            "category": "Category",
            "price": 299.99
        }
    
    def _generate_stock_recommendations(
        self, 
        stock_data: Dict[str, Any], 
        product_id: str,
        availability: str
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on stock levels"""
        recommendations = []
        
        if availability == "out_of_stock":
            recommendations.append({
                "type": "urgent",
                "action": "immediate_reorder",
                "reason": "Product is out of stock",
                "priority": "critical"
            })
        elif availability == "low":
            recommendations.append({
                "type": "warning",
                "action": "reorder_soon",
                "reason": "Stock levels are low",
                "priority": "high"
            })
        
        # Check for imbalanced distribution
        if len(stock_data) > 1:
            stock_values = [loc.get("available", 0) for loc in stock_data.values()]
            if max(stock_values) > 3 * min(stock_values):
                recommendations.append({
                    "type": "optimization",
                    "action": "redistribute_stock",
                    "reason": "Stock is unevenly distributed across locations",
                    "priority": "medium"
                })
        
        return recommendations
    
    async def _get_reorder_items(
        self, 
        location: str, 
        urgent_only: bool
    ) -> List[Dict[str, Any]]:
        """Get items that need reordering"""
        reorder_items = []
        
        # In production, this would query the database
        # For demo, simulate some items needing reorder
        sample_items = [
            {
                "product_id": "MF-BLZ-001",
                "current_stock": 15,
                "reorder_point": 30,
                "urgency": "high",
                "stockout_date": (datetime.now() + timedelta(days=2)).isoformat()
            },
            {
                "product_id": "ST-DNM-045",
                "current_stock": 50,
                "reorder_point": 75,
                "urgency": "medium",
                "stockout_date": (datetime.now() + timedelta(days=7)).isoformat()
            }
        ]
        
        if urgent_only:
            reorder_items = [item for item in sample_items if item["urgency"] == "high"]
        else:
            reorder_items = sample_items
        
        return reorder_items
    
    def _calculate_optimal_order_quantity(self, item: Dict[str, Any]) -> int:
        """Calculate optimal order quantity using EOQ formula"""
        # Simplified EOQ calculation
        annual_demand = item.get("annual_demand", 3650)  # Default 10/day
        ordering_cost = 100  # Cost per order
        holding_cost = item.get("holding_cost", 5)  # Cost per unit per year
        
        # Economic Order Quantity formula
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        
        # Round to nearest 10
        return int(round(eoq, -1))
    
    def _group_by_supplier(
        self, 
        order_recommendations: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group order recommendations by supplier"""
        # In production, would lookup actual suppliers
        # For demo, assign suppliers based on product prefix
        supplier_groups = {
            "Fashion Supplier Co": [],
            "Stratus Imports": [],
            "Casa Living Partners": [],
            "Vertex Sports Direct": []
        }
        
        for item in order_recommendations:
            product_prefix = item["product_id"].split("-")[0]
            if product_prefix == "MF":
                supplier_groups["Fashion Supplier Co"].append(item)
            elif product_prefix == "ST":
                supplier_groups["Stratus Imports"].append(item)
            elif product_prefix == "CL":
                supplier_groups["Casa Living Partners"].append(item)
            elif product_prefix == "VS":
                supplier_groups["Vertex Sports Direct"].append(item)
        
        # Remove empty suppliers
        return {k: v for k, v in supplier_groups.items() if v}
    
    def _calculate_order_cost(
        self, 
        order_recommendations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate estimated cost of orders"""
        total_units = sum(item["optimal_order_qty"] for item in order_recommendations)
        avg_unit_cost = 150  # Average cost per unit
        
        subtotal = total_units * avg_unit_cost
        shipping = len(self._group_by_supplier(order_recommendations)) * 500
        
        return {
            "subtotal": subtotal,
            "shipping": shipping,
            "total": subtotal + shipping,
            "currency": "ZAR"
        }
    
    async def _assess_inventory_health(self) -> Dict[str, Any]:
        """Assess overall inventory health"""
        # In production, would analyze actual inventory data
        # For demo, return representative metrics
        return {
            "total_skus": 1247,
            "stockout_rate": 0.03,  # 3% stockout
            "overstock_rate": 0.12,  # 12% overstock
            "turnover_rate": 6.5,  # 6.5x per year
            "service_level": 0.97,  # 97% availability
            "health_score": 0.85,  # Overall health score
            "trend": "improving"
        }
    
    def _get_inventory_alerts(self) -> List[Dict[str, Any]]:
        """Get current inventory alerts"""
        return [
            {
                "level": "warning",
                "message": "15 items approaching reorder point",
                "action": "Review reorder recommendations",
                "timestamp": datetime.now().isoformat()
            },
            {
                "level": "info",
                "message": "Summer inventory optimization recommended",
                "action": "Run seasonal analysis",
                "timestamp": datetime.now().isoformat()
            }
        ]
    
    def _get_general_recommendations(
        self, 
        inventory_health: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate general inventory recommendations"""
        recommendations = []
        
        if inventory_health["stockout_rate"] > 0.02:
            recommendations.append({
                "type": "safety_stock",
                "priority": "high",
                "description": "Increase safety stock for high-velocity items",
                "expected_impact": "Reduce stockouts by 50%"
            })
        
        if inventory_health["overstock_rate"] > 0.15:
            recommendations.append({
                "type": "clearance",
                "priority": "medium",
                "description": "Implement clearance strategy for slow-moving items",
                "expected_impact": "Reduce holding costs by 20%"
            })
        
        if inventory_health["turnover_rate"] < 6:
            recommendations.append({
                "type": "optimization",
                "priority": "medium",
                "description": "Optimize order frequencies",
                "expected_impact": "Improve cash flow by 15%"
            })
        
        return recommendations
    
    def _check_stock_tool(self, product_id: str, location: str = "all") -> Dict[str, Any]:
        """Tool wrapper for stock checking"""
        return asyncio.run(self._handle_stock_check(
            f"Check stock for {product_id}",
            {"product_id": product_id, "location": location}
        ))
    
    def _forecast_tool(self, product_id: str, period: str = "30_days") -> Dict[str, Any]:
        """Tool wrapper for demand forecasting"""
        return asyncio.run(self._handle_forecast(
            f"Forecast demand for {product_id}",
            {"product_id": product_id, "period": period}
        ))
    
    def _optimize_tool(self, brand: str, category: str) -> Dict[str, Any]:
        """Tool wrapper for inventory optimization"""
        return asyncio.run(self._handle_optimization(
            f"Optimize inventory for {brand} {category}",
            {"brand": brand, "category": category}
        ))
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information for UI display"""
        return {
            "name": self.name,
            "role": self.role,
            "description": "Inventory management and supply chain optimization",
            "capabilities": [cap.name for cap in self.capabilities],
            "status": self.get_status_dict(),
            "specialties": [
                "Stock level management",
                "Demand forecasting",
                "Reorder optimization",
                "Supply chain efficiency",
                "Inventory analytics"
            ],
            "current_alerts": len(self._get_inventory_alerts())
        }
    
    async def _get_inventory_data(
        self, 
        brand: str, 
        category: str
    ) -> List[Dict[str, Any]]:
        """Get inventory data for analysis"""
        # In production, query actual database
        # For demo, return sample data
        return [
            {
                "product_id": f"{brand}_001",
                "category": category,
                "current_stock": 150,
                "avg_daily_sales": 10,
                "holding_cost": 5.0,
                "stockout_cost": 50.0
            }
            # More products would be included
        ]
    
    def _analyze_inventory_performance(
        self, 
        inventory_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze current inventory performance"""
        total_products = len(inventory_data)
        stockout_count = sum(1 for item in inventory_data if item["current_stock"] == 0)
        overstock_count = sum(
            1 for item in inventory_data 
            if item["current_stock"] > item["avg_daily_sales"] * 60
        )
        
        return {
            "total_skus": total_products,
            "stockout_rate": stockout_count / total_products if total_products > 0 else 0,
            "overstock_rate": overstock_count / total_products if total_products > 0 else 0,
            "turnover_rate": self._calculate_turnover_rate(inventory_data),
            "service_level": 1 - (stockout_count / total_products) if total_products > 0 else 1
        }
    
    def _calculate_turnover_rate(self, inventory_data: List[Dict[str, Any]]) -> float:
        """Calculate inventory turnover rate"""
        if not inventory_data:
            return 0
        
        total_sales = sum(item["avg_daily_sales"] * 365 for item in inventory_data)
        avg_inventory = sum(item["current_stock"] for item in inventory_data) / len(inventory_data)
        
        return total_sales / avg_inventory if avg_inventory > 0 else 0
    
    def _generate_optimizations(
        self, 
        inventory_data: List[Dict[str, Any]], 
        performance: Dict[str, Any],
        timeframe: str
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        optimizations = []
        
        # Address stockouts
        if performance["stockout_rate"] > 0.05:
            optimizations.append({
                "type": "increase_safety_stock",
                "target_products": "frequently_stocked_out",
                "adjustment": "+20%",
                "impact": "Reduce stockouts by 50%",
                "cost": "Increase holding cost by 10%"
            })
        
        # Address overstock
        if performance["overstock_rate"] > 0.2:
            optimizations.append({
                "type": "reduce_order_quantities",
                "target_products": "slow_moving",
                "adjustment": "-30%",
                "impact": "Reduce holding costs by 25%",
                "cost": "Minimal risk"
            })
        
        # Improve turnover
        if performance["turnover_rate"] < 4:
            optimizations.append({
                "type": "implement_dynamic_ordering",
                "description": "Adjust order quantities based on demand patterns",
                "impact": "Increase turnover by 40%",
                "implementation": "Use demand forecasting"
            })
        
        return optimizations
    
    def _calculate_potential_savings(
        self, 
        optimizations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate potential cost savings from optimizations"""
        # Simplified calculation for demo
        holding_cost_reduction = 50000  # Base estimate
        stockout_cost_reduction = 75000
        
        for opt in optimizations:
            if opt["type"] == "reduce_order_quantities":
                holding_cost_reduction *= 1.25
            elif opt["type"] == "increase_safety_stock":
                stockout_cost_reduction *= 1.5
        
        return {
            "annual_savings": holding_cost_reduction + stockout_cost_reduction,
            "holding_cost_reduction": holding_cost_reduction,
            "stockout_cost_reduction": stockout_cost_reduction,
            "payback_period_months": 3
        }
    
    def _create_implementation_plan(
        self, 
        optimizations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create implementation plan for optimizations"""
        plan = []
        
        for i, opt in enumerate(optimizations):
            plan.append({
                "phase": i + 1,
                "optimization": opt["type"],
                "timeline": f"Week {(i * 2) + 1}-{(i * 2) + 2}",
                "actions": [
                    "Analyze target products",
                    "Adjust system parameters",
                    "Monitor results",
                    "Fine-tune as needed"
                ],
                "success_metrics": [
                    "Stock availability > 95%",
                    "Holding costs reduced by target %",
                    "No increase in stockouts"
                ]
            })
        
        return plan
    
    async def _get_historical_data(self, product_id: str) -> List[Dict[str, Any]]:
        """Get historical sales data for forecasting"""
        # Generate sample historical data for demo
        history = []
        base_demand = 50
        
        for i in range(365):
            date = datetime.now() - timedelta(days=365-i)
            
            # Add seasonality
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365)
            
            # Add trend
            trend_factor = 1 + (i / 365) * 0.1
            
            # Add random variation
            random_factor = np.random.normal(1, 0.1)
            
            daily_sales = int(base_demand * seasonal_factor * trend_factor * random_factor)
            
            history.append({
                "date": date.isoformat(),
                "sales": max(0, daily_sales),
                "day_of_week": date.weekday(),
                "month": date.month
            })
        
        return history
    
    def _generate_forecast(
        self, 
        historical_data: List[Dict[str, Any]], 
        period: str,
        include_seasonality: bool
    ) -> List[Dict[str, Any]]:
        """Generate demand forecast"""
        # Simple moving average with seasonality for demo
        days_to_forecast = {
            "7_days": 7,
            "30_days": 30,
            "90_days": 90
        }.get(period, 30)
        
        recent_sales = [d["sales"] for d in historical_data[-30:]]
        avg_sales = sum(recent_sales) / len(recent_sales)
        
        forecast = []
        for i in range(days_to_forecast):
            date = datetime.now() + timedelta(days=i+1)
            
            # Base forecast
            forecast_value = avg_sales
            
            # Add seasonality if requested
            if include_seasonality:
                day_of_year = date.timetuple().tm_yday
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
                forecast_value *= seasonal_factor
            
            # Add some uncertainty
            forecast_value *= np.random.normal(1, 0.05)
            
            forecast.append({
                "date": date.isoformat(),
                "forecast": int(max(0, forecast_value)),
                "lower_bound": int(max(0, forecast_value * 0.8)),
                "upper_bound": int(forecast_value * 1.2)
            })
        
        return forecast
    
    def _calculate_confidence_interval(
        self, 
        forecast: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate confidence interval for forecast"""
        forecast_values = [f["forecast"] for f in forecast]
        
        return {
            "mean": np.mean(forecast_values),
            "std": np.std(forecast_values),
            "confidence_level": 0.95,
            "lower_bound": np.percentile(forecast_values, 2.5),
            "upper_bound": np.percentile(forecast_values, 97.5)
        }
    
    def _identify_demand_factors(
        self, 
        historical_data: List[Dict[str, Any]], 
        product_id: str
    ) -> List[Dict[str, Any]]:
        """Identify factors influencing demand"""
        factors = [
            {
                "factor": "seasonality",
                "impact": "high",
                "description": "Strong seasonal pattern detected",
                "peak_period": "Summer months"
            },
            {
                "factor": "trend",
                "impact": "medium",
                "description": "Gradual upward trend in demand",
                "growth_rate": "10% annually"
            },
            {
                "factor": "day_of_week",
                "impact": "low",
                "description": "Weekend sales slightly higher",
                "pattern": "15% increase on weekends"
            }
        ]
        
        return factors
    
    def _generate_forecast_recommendations(
        self, 
        forecast: List[Dict[str, Any]], 
        product_id: str
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on forecast"""
        total_forecast = sum(f["forecast"] for f in forecast)
        avg_daily_forecast = total_forecast / len(forecast)
        
        recommendations = []
        
        # Check if demand is increasing
        first_week_avg = sum(f["forecast"] for f in forecast[:7]) / 7
        last_week_avg = sum(f["forecast"] for f in forecast[-7:]) / 7
        
        if last_week_avg > first_week_avg * 1.1:
            recommendations.append({
                "type": "increase_stock",
                "reason": "Demand is trending upward",
                "action": "Increase safety stock by 20%",
                "timing": "immediate"
            })
        
        # Check for high variability
        forecast_std = np.std([f["forecast"] for f in forecast])
        if forecast_std / avg_daily_forecast > 0.3:
            recommendations.append({
                "type": "buffer_stock",
                "reason": "High demand variability detected",
                "action": "Maintain higher safety stock",
                "timing": "ongoing"
            })
        
        return recommendations
