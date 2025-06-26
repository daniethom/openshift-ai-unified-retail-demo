"""
Customer Service Agent for Meridian Retail Group
Specializes in customer interactions, support, and personalized recommendations
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from decimal import Decimal

from agents.base_agent import BaseAgent, AgentCapability

logger = logging.getLogger(__name__)


class CustomerAgent(BaseAgent):
    """
    Specialized agent for customer service and engagement
    Handles customer queries, complaints, recommendations, and loyalty programs
    """
    
    def __init__(self, mcp_servers: Dict[str, Any], data_store: Any):
        # Define agent capabilities
        capabilities = [
            AgentCapability(
                name="handle_customer_query",
                description="Handle customer questions and provide assistance",
                input_schema={
                    "customer_id": "string",
                    "query_type": "string",
                    "context": "object"
                },
                output_schema={
                    "response": "string",
                    "suggested_actions": "array",
                    "satisfaction_score": "number"
                }
            ),
            AgentCapability(
                name="personalized_recommendations",
                description="Provide personalized product recommendations",
                input_schema={
                    "customer_id": "string",
                    "category": "string",
                    "occasion": "string"
                },
                output_schema={
                    "recommendations": "array",
                    "reasoning": "string",
                    "confidence": "number"
                }
            ),
            AgentCapability(
                name="handle_complaint",
                description="Process and resolve customer complaints",
                input_schema={
                    "complaint_type": "string",
                    "severity": "string",
                    "details": "string"
                },
                output_schema={
                    "resolution": "object",
                    "compensation": "object",
                    "follow_up_required": "boolean"
                }
            ),
            AgentCapability(
                name="loyalty_analysis",
                description="Analyze customer loyalty and provide retention strategies",
                input_schema={
                    "customer_id": "string",
                    "time_period": "string"
                },
                output_schema={
                    "loyalty_score": "number",
                    "retention_risk": "string",
                    "strategies": "array"
                }
            )
        ]
        
        super().__init__(
            name="CustomerAgent",
            role="Customer Experience Specialist",
            goal="Deliver exceptional customer service and build lasting relationships through personalized interactions",
            backstory="""You are an expert customer service representative with deep empathy 
            and understanding of customer needs. You excel at resolving issues, providing 
            personalized recommendations, and creating positive experiences. You understand 
            South African customer preferences and cultural nuances. You always prioritize 
            customer satisfaction while balancing business objectives.""",
            mcp_servers=mcp_servers,
            capabilities=capabilities
        )
        
        self.data_store = data_store
        
        # Customer service parameters
        self.service_params = {
            "response_tone": "friendly_professional",
            "escalation_threshold": 0.3,  # Satisfaction score threshold
            "compensation_limits": {
                "minor": 100,  # R100 voucher
                "moderate": 500,  # R500 credit
                "major": 1000  # R1000 compensation
            },
            "loyalty_tiers": {
                "bronze": {"min_spend": 0, "discount": 0.05},
                "silver": {"min_spend": 5000, "discount": 0.10},
                "gold": {"min_spend": 15000, "discount": 0.15},
                "platinum": {"min_spend": 30000, "discount": 0.20}
            }
        }
        
        # Query type mappings
        self.query_types = {
            "product": ["availability", "sizing", "features", "comparison"],
            "order": ["status", "tracking", "modification", "cancellation"],
            "return": ["policy", "process", "refund", "exchange"],
            "payment": ["methods", "issues", "refund_status"],
            "general": ["store_hours", "locations", "policies", "promotions"]
        }
        
        # Complaint categories
        self.complaint_categories = {
            "product_quality": {"severity": "moderate", "resolution_time": "24h"},
            "delivery_issue": {"severity": "minor", "resolution_time": "48h"},
            "billing_error": {"severity": "major", "resolution_time": "immediate"},
            "staff_behavior": {"severity": "major", "resolution_time": "24h"},
            "website_issue": {"severity": "minor", "resolution_time": "72h"}
        }
    
    def get_tools(self) -> List[Any]:
        """Return available tools for this agent"""
        return [
            self._customer_query_tool,
            self._recommendation_tool,
            self._complaint_tool,
            self._loyalty_tool
        ]
    
    async def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a customer service query"""
        start_time = datetime.now()
        
        try:
            # Determine query type
            query_type = self._classify_customer_query(query)
            
            if query_type == "support":
                result = await self._handle_support_query(query, context)
            elif query_type == "recommendation":
                result = await self._handle_recommendation_query(query, context)
            elif query_type == "complaint":
                result = await self._handle_complaint_query(query, context)
            elif query_type == "loyalty":
                result = await self._handle_loyalty_query(query, context)
            else:
                result = await self._handle_general_customer_query(query, context)
            
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
                    "response_time": response_time,
                    "customer_id": context.get("customer_id")
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing customer query: {e}")
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
    
    def _classify_customer_query(self, query: str) -> str:
        """Classify the type of customer query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["help", "how", "where", "when", "status"]):
            return "support"
        elif any(word in query_lower for word in ["recommend", "suggest", "looking for", "need"]):
            return "recommendation"
        elif any(word in query_lower for word in ["complaint", "problem", "issue", "unhappy", "disappointed"]):
            return "complaint"
        elif any(word in query_lower for word in ["loyalty", "points", "rewards", "member"]):
            return "loyalty"
        else:
            return "general"
    
    async def _handle_support_query(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle customer support queries"""
        customer_id = context.get("customer_id", "guest")
        query_category = self._determine_query_category(query)
        
        # Get customer profile
        customer_profile = await self._get_customer_profile(customer_id)
        
        # Generate response based on query category
        response = await self._generate_support_response(
            query, query_category, customer_profile
        )
        
        # Check if escalation is needed
        needs_escalation = self._check_escalation_needed(
            query, customer_profile, response
        )
        
        # Get relevant information
        relevant_info = await self._gather_relevant_info(
            query_category, query, customer_profile
        )
        
        return {
            "response": response,
            "category": query_category,
            "relevant_information": relevant_info,
            "needs_escalation": needs_escalation,
            "suggested_actions": self._generate_suggested_actions(
                query_category, customer_profile
            ),
            "satisfaction_predicted": self._predict_satisfaction(
                response, customer_profile
            )
        }
    
    async def _handle_recommendation_query(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle product recommendation queries"""
        customer_id = context.get("customer_id", "guest")
        category = context.get("category", self._extract_category(query))
        occasion = context.get("occasion", self._extract_occasion(query))
        
        # Get customer profile and preferences
        customer_profile = await self._get_customer_profile(customer_id)
        preferences = await self._get_customer_preferences(customer_id)
        
        # Get purchase history
        purchase_history = await self._get_purchase_history(customer_id)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            customer_profile,
            preferences,
            purchase_history,
            category,
            occasion
        )
        
        # Personalize recommendations
        personalized_recs = self._personalize_recommendations(
            recommendations,
            customer_profile,
            preferences
        )
        
        # Add complementary items
        with_complements = self._add_complementary_items(personalized_recs)
        
        return {
            "recommendations": with_complements,
            "reasoning": self._explain_recommendations(
                with_complements, preferences, occasion
            ),
            "personalization_score": self._calculate_personalization_score(
                with_complements, preferences
            ),
            "cross_sell_opportunities": self._identify_cross_sell(
                with_complements, purchase_history
            )
        }
    
    async def _handle_complaint_query(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle customer complaints"""
        customer_id = context.get("customer_id", "guest")
        complaint_type = self._identify_complaint_type(query)
        severity = self._assess_complaint_severity(query, complaint_type)
        
        # Get customer profile and history
        customer_profile = await self._get_customer_profile(customer_id)
        complaint_history = await self._get_complaint_history(customer_id)
        
        # Generate resolution
        resolution = self._generate_resolution(
            complaint_type,
            severity,
            customer_profile,
            complaint_history
        )
        
        # Calculate compensation if needed
        compensation = self._calculate_compensation(
            complaint_type,
            severity,
            customer_profile
        )
        
        # Create follow-up plan
        follow_up_plan = self._create_follow_up_plan(
            complaint_type,
            severity,
            resolution
        )
        
        # Log complaint for tracking
        complaint_id = await self._log_complaint(
            customer_id,
            complaint_type,
            query,
            resolution
        )
        
        return {
            "complaint_id": complaint_id,
            "complaint_type": complaint_type,
            "severity": severity,
            "resolution": resolution,
            "compensation": compensation,
            "follow_up_plan": follow_up_plan,
            "estimated_resolution_time": self._estimate_resolution_time(
                complaint_type, severity
            ),
            "escalation_path": self._define_escalation_path(severity)
        }
    
    async def _handle_loyalty_query(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle loyalty program queries"""
        customer_id = context.get("customer_id", "guest")
        time_period = context.get("time_period", "12_months")
        
        # Get loyalty data
        loyalty_data = await self._get_loyalty_data(customer_id)
        
        # Calculate loyalty metrics
        loyalty_score = self._calculate_loyalty_score(loyalty_data)
        
        # Determine tier and benefits
        current_tier = self._determine_loyalty_tier(loyalty_data)
        tier_benefits = self._get_tier_benefits(current_tier)
        
        # Analyze retention risk
        retention_risk = self._analyze_retention_risk(
            loyalty_data, time_period
        )
        
        # Generate retention strategies
        retention_strategies = self._generate_retention_strategies(
            retention_risk,
            loyalty_score,
            current_tier
        )
        
        # Calculate next tier progress
        next_tier_progress = self._calculate_next_tier_progress(
            loyalty_data, current_tier
        )
        
        return {
            "customer_id": customer_id,
            "loyalty_score": loyalty_score,
            "current_tier": current_tier,
            "tier_benefits": tier_benefits,
            "points_balance": loyalty_data.get("points_balance", 0),
            "retention_risk": retention_risk,
            "retention_strategies": retention_strategies,
            "next_tier_progress": next_tier_progress,
            "personalized_offers": self._generate_loyalty_offers(
                current_tier, loyalty_data
            )
        }
    
    async def _handle_general_customer_query(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle general customer queries"""
        # Provide helpful information
        faq_matches = self._search_faq(query)
        
        return {
            "response": self._generate_general_response(query),
            "faq_matches": faq_matches,
            "helpful_links": self._get_helpful_links(query),
            "contact_options": self._get_contact_options(),
            "store_information": self._get_store_information(
                context.get("location", "national")
            )
        }
    
    async def _get_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        """Get customer profile data"""
        # In production, query actual customer database
        # Mock data for demo
        if customer_id == "guest":
            return {
                "customer_id": "guest",
                "tier": "none",
                "lifetime_value": 0,
                "join_date": None,
                "preferred_channels": ["web"],
                "communication_preferences": "email"
            }
        
        return {
            "customer_id": customer_id,
            "name": "Sarah Johnson",
            "tier": "gold",
            "lifetime_value": 25000,
            "join_date": "2022-03-15",
            "preferred_channels": ["app", "store"],
            "communication_preferences": "email",
            "location": "cape_town",
            "age_group": "25-35",
            "style_preference": "modern_chic"
        }
    
    def _determine_query_category(self, query: str) -> str:
        """Determine the category of support query"""
        query_lower = query.lower()
        
        for category, keywords in self.query_types.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return "general"
    
    async def _generate_support_response(
        self,
        query: str,
        category: str,
        customer_profile: Dict[str, Any]
    ) -> str:
        """Generate appropriate support response"""
        # Personalize based on tier
        tier = customer_profile.get("tier", "none")
        
        responses = {
            "product": f"I'd be happy to help you with product information. ",
            "order": f"Let me check your order status for you. ",
            "return": f"I understand you have a return query. Our return policy allows ",
            "payment": f"I'll assist you with your payment question. ",
            "general": f"Thank you for reaching out. "
        }
        
        base_response = responses.get(category, "I'm here to help. ")
        
        # Add tier-specific greeting
        if tier in ["gold", "platinum"]:
            base_response = f"Welcome back, valued {tier} member! " + base_response
        
        return base_response + "How can I assist you further?"
    
    def _check_escalation_needed(
        self,
        query: str,
        customer_profile: Dict[str, Any],
        response: str
    ) -> bool:
        """Check if query needs escalation"""
        # High-value customers get priority
        if customer_profile.get("lifetime_value", 0) > 50000:
            return True
        
        # Certain keywords trigger escalation
        escalation_keywords = ["legal", "lawsuit", "media", "urgent", "emergency"]
        if any(keyword in query.lower() for keyword in escalation_keywords):
            return True
        
        return False
    
    async def _gather_relevant_info(
        self,
        category: str,
        query: str,
        customer_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Gather relevant information for the query"""
        info = {}
        
        if category == "order":
            info["recent_orders"] = await self._get_recent_orders(
                customer_profile.get("customer_id")
            )
        elif category == "product":
            info["product_availability"] = await self._check_product_availability(query)
        elif category == "return":
            info["return_policy"] = self._get_return_policy()
            info["return_window"] = "30 days"
        
        return info
    
    def _generate_suggested_actions(
        self,
        category: str,
        customer_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate suggested actions for customer"""
        actions = []
        
        if category == "product":
            actions.append({
                "action": "browse_similar",
                "description": "View similar products",
                "link": "/browse/similar"
            })
        elif category == "order":
            actions.append({
                "action": "track_order",
                "description": "Track your order",
                "link": "/orders/track"
            })
        
        # Add loyalty action for members
        if customer_profile.get("tier") != "none":
            actions.append({
                "action": "check_points",
                "description": "Check loyalty points",
                "link": "/loyalty/points"
            })
        
        return actions
    
    def _predict_satisfaction(
        self,
        response: str,
        customer_profile: Dict[str, Any]
    ) -> float:
        """Predict customer satisfaction with response"""
        base_satisfaction = 0.7
        
        # Tier adjustment
        tier_adjustments = {
            "platinum": 0.1,
            "gold": 0.05,
            "silver": 0.02,
            "bronze": 0
        }
        
        tier = customer_profile.get("tier", "bronze")
        satisfaction = base_satisfaction + tier_adjustments.get(tier, 0)
        
        # Response quality adjustment (simplified)
        if len(response) > 100:
            satisfaction += 0.05
        
        return min(satisfaction, 0.95)
    
    def _extract_category(self, query: str) -> str:
        """Extract product category from query"""
        categories = ["fashion", "electronics", "homeware", "sports"]
        query_lower = query.lower()
        
        for category in categories:
            if category in query_lower:
                return category
        
        return "all"
    
    def _extract_occasion(self, query: str) -> Optional[str]:
        """Extract occasion from query"""
        occasions = {
            "wedding": ["wedding", "marriage", "bride", "groom"],
            "party": ["party", "celebration", "birthday"],
            "work": ["work", "office", "business", "professional"],
            "casual": ["casual", "weekend", "relaxed"],
            "sport": ["gym", "running", "fitness", "sport"]
        }
        
        query_lower = query.lower()
        for occasion, keywords in occasions.items():
            if any(keyword in query_lower for keyword in keywords):
                return occasion
        
        return None
    
    async def _get_customer_preferences(self, customer_id: str) -> Dict[str, Any]:
        """Get customer preferences"""
        # Mock data for demo
        return {
            "colors": ["blue", "black", "white"],
            "styles": ["modern", "minimalist"],
            "brands": ["meridian_fashion", "stratus"],
            "price_sensitivity": "medium",
            "size": {"clothing": "M", "shoes": "9"}
        }
    
    async def _get_purchase_history(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get customer purchase history"""
        # Mock data for demo
        return [
            {
                "order_id": "ORD-2024-001",
                "date": "2024-11-15",
                "items": [
                    {"product_id": "MF-BLZ-001", "category": "fashion", "price": 1899},
                    {"product_id": "MF-SHT-023", "category": "fashion", "price": 599}
                ],
                "total": 2498
            }
        ]
    
    async def _generate_recommendations(
        self,
        customer_profile: Dict[str, Any],
        preferences: Dict[str, Any],
        purchase_history: List[Dict[str, Any]],
        category: str,
        occasion: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Generate product recommendations"""
        recommendations = []
        
        # Base recommendations on category
        if category == "fashion":
            recommendations.extend([
                {
                    "product_id": "MF-DRS-045",
                    "name": "Elegant Evening Dress",
                    "price": 2499,
                    "match_score": 0.92,
                    "reason": "Matches your style preference"
                },
                {
                    "product_id": "MF-BLZ-002",
                    "name": "Classic Blazer",
                    "price": 1999,
                    "match_score": 0.88,
                    "reason": "Similar to previous purchase"
                }
            ])
        
        # Filter by occasion if specified
        if occasion:
            recommendations = [
                rec for rec in recommendations 
                if self._matches_occasion(rec, occasion)
            ]
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _personalize_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        customer_profile: Dict[str, Any],
        preferences: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Personalize recommendations based on customer data"""
        personalized = []
        
        for rec in recommendations:
            # Adjust match score based on preferences
            if any(color in rec.get("colors", []) for color in preferences.get("colors", [])):
                rec["match_score"] *= 1.1
            
            # Add personalized messaging
            if customer_profile.get("tier") in ["gold", "platinum"]:
                rec["exclusive_discount"] = 0.10  # 10% off for premium members
            
            personalized.append(rec)
        
        # Sort by match score
        return sorted(personalized, key=lambda x: x["match_score"], reverse=True)
    
    def _add_complementary_items(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add complementary items to recommendations"""
        enhanced_recs = []
        
        for rec in recommendations:
            enhanced_recs.append(rec)
            
            # Add complements
            if "blazer" in rec.get("name", "").lower():
                rec["complements"] = [
                    {
                        "product_id": "MF-TRS-012",
                        "name": "Matching Trousers",
                        "price": 1299,
                        "bundle_discount": 0.15
                    }
                ]
        
        return enhanced_recs
    
    def _explain_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        preferences: Dict[str, Any],
        occasion: Optional[str]
    ) -> str:
        """Explain why items were recommended"""
        explanation = "Based on your preferences for "
        explanation += f"{', '.join(preferences.get('styles', ['modern']))} styles"
        
        if occasion:
            explanation += f" and your need for {occasion} wear"
        
        explanation += ", I've selected items that match your taste and previous purchases."
        
        return explanation
    
    def _calculate_personalization_score(
        self,
        recommendations: List[Dict[str, Any]],
        preferences: Dict[str, Any]
    ) -> float:
        """Calculate how well recommendations match preferences"""
        if not recommendations:
            return 0.0
        
        total_score = sum(rec.get("match_score", 0) for rec in recommendations)
        return total_score / len(recommendations)
    
    def _identify_cross_sell(
        self,
        recommendations: List[Dict[str, Any]],
        purchase_history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify cross-sell opportunities"""
        opportunities = []
        
        # Check for category gaps
        purchased_categories = set()
        for order in purchase_history:
            for item in order.get("items", []):
                purchased_categories.add(item.get("category"))
        
        # Suggest items from unpurchased categories
        all_categories = {"fashion", "homeware", "electronics", "sports"}
        missing_categories = all_categories - purchased_categories
        
        for category in missing_categories:
            opportunities.append({
                "category": category,
                "message": f"Explore our {category} collection",
                "potential_value": 2000
            })
        
        return opportunities
    
    def _matches_occasion(self, product: Dict[str, Any], occasion: str) -> bool:
        """Check if product matches occasion"""
        occasion_keywords = {
            "wedding": ["elegant", "formal", "dress", "suit"],
            "party": ["party", "cocktail", "evening"],
            "work": ["business", "professional", "blazer", "shirt"],
            "casual": ["casual", "relaxed", "jeans", "t-shirt"],
            "sport": ["active", "sport", "gym", "athletic"]
        }
        
        product_name = product.get("name", "").lower()
        keywords = occasion_keywords.get(occasion, [])
        
        return any(keyword in product_name for keyword in keywords)
    
    def _identify_complaint_type(self, query: str) -> str:
        """Identify the type of complaint"""
        query_lower = query.lower()
        
        for complaint_type, config in self.complaint_categories.items():
            keywords = complaint_type.replace("_", " ").split()
            if any(keyword in query_lower for keyword in keywords):
                return complaint_type
        
        return "general_complaint"
    
    def _assess_complaint_severity(self, query: str, complaint_type: str) -> str:
        """Assess severity of complaint"""
        # Check for severity indicators
        high_severity_words = ["urgent", "unacceptable", "legal", "terrible", "worst"]
        
        if any(word in query.lower() for word in high_severity_words):
            return "major"
        
        # Use default severity for complaint type
        return self.complaint_categories.get(complaint_type, {}).get("severity", "minor")
    
    async def _get_complaint_history(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get customer complaint history"""
        # Mock data for demo
        return [
            {
                "complaint_id": "CMP-2024-001",
                "date": "2024-10-01",
                "type": "delivery_issue",
                "resolved": True,
                "satisfaction": 4
            }
        ]
    
    def _generate_resolution(
        self,
        complaint_type: str,
        severity: str,
        customer_profile: Dict[str, Any],
        complaint_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate complaint resolution"""
        resolution = {
            "immediate_action": self._get_immediate_action(complaint_type),
            "long_term_solution": self._get_long_term_solution(complaint_type),
            "apology": self._craft_apology(severity, customer_profile),
            "preventive_measures": self._define_preventive_measures(complaint_type)
        }
        
        # Escalate for repeat complaints
        if len(complaint_history) > 2:
            resolution["escalated_to"] = "Customer Relations Manager"
        
        return resolution
    
    def _calculate_compensation(
        self,
        complaint_type: str,
        severity: str,
        customer_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate appropriate compensation"""
        base_compensation = self.service_params["compensation_limits"][severity]
        
        # Adjust for customer tier
        tier_multipliers = {
            "platinum": 1.5,
            "gold": 1.3,
            "silver": 1.1,
            "bronze": 1.0
        }
        
        tier = customer_profile.get("tier", "bronze")
        adjusted_compensation = base_compensation * tier_multipliers.get(tier, 1.0)
        
        return {
            "type": "store_credit" if severity == "minor" else "refund",
            "amount": adjusted_compensation,
            "additional": "20% off next purchase" if tier in ["gold", "platinum"] else None,
            "validity": "6 months"
        }
    
    def _create_follow_up_plan(
        self,
        complaint_type: str,
        severity: str,
        resolution: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create follow-up plan for complaint"""
        plan = []
        
        # Immediate follow-up
        plan.append({
            "timing": "24 hours",
            "action": "Call customer to confirm resolution satisfaction",
            "responsible": "Customer Service Team"
        })
        
        # Based on severity
        if severity in ["moderate", "major"]:
            plan.append({
                "timing": "1 week",
                "action": "Email survey on resolution experience",
                "responsible": "Quality Assurance"
            })
        
        if severity == "major":
            plan.append({
                "timing": "1 month",
                "action": "Personal call from management",
                "responsible": "Store Manager"
            })
        
        return plan
    
    async def _log_complaint(
        self,
        customer_id: str,
        complaint_type: str,
        query: str,
        resolution: Dict[str, Any]
    ) -> str:
        """Log complaint for tracking"""
        complaint_id = f"CMP-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"
        
        # In production, save to database
        logger.info(f"Logged complaint {complaint_id} for customer {customer_id}")
        
        return complaint_id
    
    def _estimate_resolution_time(self, complaint_type: str, severity: str) -> str:
        """Estimate time to resolve complaint"""
        base_times = {
            "minor": "24-48 hours",
            "moderate": "2-3 business days",
            "major": "Within 24 hours"
        }
        
        return base_times.get(severity, "48 hours")
    
    def _define_escalation_path(self, severity: str) -> List[str]:
        """Define escalation path for complaint"""
        paths = {
            "minor": ["Customer Service Agent", "Team Lead"],
            "moderate": ["Customer Service Agent", "Team Lead", "Department Manager"],
            "major": ["Customer Service Agent", "Department Manager", "Regional Manager", "Executive Team"]
        }
        
        return paths.get(severity, ["Customer Service Agent"])
    
    def _get_immediate_action(self, complaint_type: str) -> str:
        """Get immediate action for complaint type"""
        actions = {
            "product_quality": "Initiate quality check and offer immediate replacement",
            "delivery_issue": "Contact courier service and provide tracking update",
            "billing_error": "Reverse incorrect charge and issue corrected invoice",
            "staff_behavior": "Report to store management for immediate investigation",
            "website_issue": "Log technical issue and provide alternative ordering method"
        }
        
        return actions.get(complaint_type, "Address concern immediately")
    
    def _get_long_term_solution(self, complaint_type: str) -> str:
        """Get long-term solution for complaint type"""
        solutions = {
            "product_quality": "Review supplier quality control processes",
            "delivery_issue": "Evaluate delivery partner performance",
            "billing_error": "Audit billing system for systematic issues",
            "staff_behavior": "Implement additional customer service training",
            "website_issue": "Schedule comprehensive website maintenance"
        }
        
        return solutions.get(complaint_type, "Implement process improvements")
    
    def _craft_apology(self, severity: str, customer_profile: Dict[str, Any]) -> str:
        """Craft appropriate apology message"""
        tier = customer_profile.get("tier", "bronze")
        name = customer_profile.get("name", "Valued Customer")
        
        if severity == "major":
            apology = f"Dear {name}, we sincerely apologize for this unacceptable experience. "
            apology += "This falls far short of our standards and your expectations as a "
            if tier in ["gold", "platinum"]:
                apology += f"valued {tier} member. "
            apology += "We take full responsibility and are committed to making this right."
        elif severity == "moderate":
            apology = f"We apologize for the inconvenience, {name}. "
            apology += "We understand how frustrating this must be and appreciate your patience."
        else:
            apology = "We apologize for any inconvenience this may have caused."
        
        return apology
    
    def _define_preventive_measures(self, complaint_type: str) -> List[str]:
        """Define preventive measures for complaint type"""
        measures = {
            "product_quality": [
                "Enhanced quality control checks",
                "Supplier audit program",
                "Customer feedback integration"
            ],
            "delivery_issue": [
                "Real-time tracking updates",
                "Multiple delivery options",
                "Delivery partner SLA monitoring"
            ],
            "billing_error": [
                "Automated billing verification",
                "Double-check process for manual entries",
                "Regular system audits"
            ]
        }
        
        return measures.get(complaint_type, ["Process review", "Staff training", "System improvements"])
    
    async def _get_loyalty_data(self, customer_id: str) -> Dict[str, Any]:
        """Get customer loyalty data"""
        # Mock data for demo
        return {
            "customer_id": customer_id,
            "points_balance": 2500,
            "lifetime_points": 15000,
            "current_year_spend": 12000,
            "member_since": "2022-03-15",
            "last_purchase": "2024-11-20",
            "purchase_frequency": "monthly",
            "average_order_value": 1800
        }
    
    def _calculate_loyalty_score(self, loyalty_data: Dict[str, Any]) -> float:
        """Calculate customer loyalty score"""
        score = 0.0
        
        # Tenure score (max 0.3)
        member_since = datetime.strptime(loyalty_data.get("member_since", "2024-01-01"), "%Y-%m-%d")
        years_member = (datetime.now() - member_since).days / 365
        score += min(years_member / 10, 0.3)  # Max at 10 years
        
        # Spending score (max 0.3)
        annual_spend = loyalty_data.get("current_year_spend", 0)
        score += min(annual_spend / 50000, 0.3)  # Max at R50,000
        
        # Frequency score (max 0.2)
        frequency_scores = {
            "weekly": 0.2,
            "monthly": 0.15,
            "quarterly": 0.1,
            "rarely": 0.05
        }
        score += frequency_scores.get(loyalty_data.get("purchase_frequency", "rarely"), 0.05)
        
        # Engagement score (max 0.2)
        points_earned = loyalty_data.get("lifetime_points", 0)
        score += min(points_earned / 50000, 0.2)  # Max at 50,000 points
        
        return round(score, 2)
    
    def _determine_loyalty_tier(self, loyalty_data: Dict[str, Any]) -> str:
        """Determine customer loyalty tier"""
        annual_spend = loyalty_data.get("current_year_spend", 0)
        
        for tier, config in reversed(list(self.service_params["loyalty_tiers"].items())):
            if annual_spend >= config["min_spend"]:
                return tier
        
        return "bronze"
    
    def _get_tier_benefits(self, tier: str) -> Dict[str, Any]:
        """Get benefits for loyalty tier"""
        benefits = {
            "bronze": {
                "discount": "5% on all purchases",
                "points_multiplier": 1.0,
                "special_perks": ["Birthday discount", "Early sale access"]
            },
            "silver": {
                "discount": "10% on all purchases",
                "points_multiplier": 1.5,
                "special_perks": ["Birthday discount", "Early sale access", "Free shipping over R500"]
            },
            "gold": {
                "discount": "15% on all purchases",
                "points_multiplier": 2.0,
                "special_perks": ["Birthday discount", "VIP sale preview", "Free shipping", "Personal shopper"]
            },
            "platinum": {
                "discount": "20% on all purchases",
                "points_multiplier": 3.0,
                "special_perks": ["Birthday month discounts", "Exclusive events", "Free shipping", "Dedicated concierge", "Custom orders"]
            }
        }
        
        return benefits.get(tier, benefits["bronze"])
    
    def _analyze_retention_risk(
        self,
        loyalty_data: Dict[str, Any],
        time_period: str
    ) -> str:
        """Analyze customer retention risk"""
        # Days since last purchase
        last_purchase = datetime.strptime(loyalty_data.get("last_purchase", "2024-01-01"), "%Y-%m-%d")
        days_inactive = (datetime.now() - last_purchase).days
        
        # Purchase frequency expectation
        frequency_days = {
            "weekly": 7,
            "monthly": 30,
            "quarterly": 90,
            "rarely": 180
        }
        
        expected_days = frequency_days.get(loyalty_data.get("purchase_frequency", "rarely"), 90)
        
        if days_inactive > expected_days * 3:
            return "high"
        elif days_inactive > expected_days * 2:
            return "medium"
        else:
            return "low"
    
    def _generate_retention_strategies(
        self,
        retention_risk: str,
        loyalty_score: float,
        current_tier: str
    ) -> List[Dict[str, Any]]:
        """Generate customer retention strategies"""
        strategies = []
        
        if retention_risk == "high":
            strategies.extend([
                {
                    "strategy": "win_back_campaign",
                    "action": "Send personalized win-back offer with 25% discount",
                    "timing": "immediate",
                    "channel": "email + sms"
                },
                {
                    "strategy": "personal_outreach",
                    "action": "Personal call from customer relations",
                    "timing": "within 48 hours",
                    "channel": "phone"
                }
            ])
        elif retention_risk == "medium":
            strategies.extend([
                {
                    "strategy": "re_engagement",
                    "action": "Exclusive preview of new collection",
                    "timing": "next campaign",
                    "channel": "email"
                },
                {
                    "strategy": "bonus_points",
                    "action": "Double points on next purchase",
                    "timing": "30 days",
                    "channel": "app notification"
                }
            ])
        
        # Tier-specific strategies
        if current_tier in ["gold", "platinum"]:
            strategies.append({
                "strategy": "vip_experience",
                "action": "Invite to exclusive shopping event",
                "timing": "next event",
                "channel": "personal invitation"
            })
        
        return strategies
    
    def _calculate_next_tier_progress(
        self,
        loyalty_data: Dict[str, Any],
        current_tier: str
    ) -> Dict[str, Any]:
        """Calculate progress to next tier"""
        current_spend = loyalty_data.get("current_year_spend", 0)
        
        # Find next tier
        tier_order = ["bronze", "silver", "gold", "platinum"]
        current_index = tier_order.index(current_tier)
        
        if current_index < len(tier_order) - 1:
            next_tier = tier_order[current_index + 1]
            next_tier_threshold = self.service_params["loyalty_tiers"][next_tier]["min_spend"]
            
            return {
                "next_tier": next_tier,
                "current_spend": current_spend,
                "required_spend": next_tier_threshold,
                "remaining_spend": max(0, next_tier_threshold - current_spend),
                "progress_percentage": min(100, (current_spend / next_tier_threshold) * 100)
            }
        
        return {
            "next_tier": None,
            "message": "You've reached our highest tier! Thank you for being a valued platinum member."
        }
    
    def _generate_loyalty_offers(
        self,
        tier: str,
        loyalty_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate personalized loyalty offers"""
        offers = []
        
        # Tier-based offers
        if tier == "platinum":
            offers.append({
                "offer_id": "PLT-001",
                "description": "Exclusive 30% off new designer collection",
                "valid_until": (datetime.now() + timedelta(days=14)).isoformat(),
                "code": "PLATINUM30"
            })
        elif tier == "gold":
            offers.append({
                "offer_id": "GLD-001",
                "description": "20% off your favorite brand",
                "valid_until": (datetime.now() + timedelta(days=21)).isoformat(),
                "code": "GOLD20FAV"
            })
        
        # Points-based offers
        points_balance = loyalty_data.get("points_balance", 0)
        if points_balance > 1000:
            offers.append({
                "offer_id": "PTS-001",
                "description": f"Redeem {points_balance} points for R{points_balance // 10} off",
                "valid_until": (datetime.now() + timedelta(days=30)).isoformat(),
                "action": "redeem_points"
            })
        
        return offers
    
    def _search_faq(self, query: str) -> List[Dict[str, Any]]:
        """Search FAQ database for relevant answers"""
        # Mock FAQ data
        faqs = [
            {
                "question": "What is your return policy?",
                "answer": "We offer a 30-day return policy on all items in original condition.",
                "category": "returns"
            },
            {
                "question": "How do I track my order?",
                "answer": "You can track your order using the tracking number sent to your email.",
                "category": "orders"
            },
            {
                "question": "What are your store hours?",
                "answer": "Our stores are open Monday-Saturday 9AM-8PM, Sunday 10AM-6PM.",
                "category": "general"
            }
        ]
        
        # Simple keyword matching
        matches = []
        query_lower = query.lower()
        for faq in faqs:
            if any(word in faq["question"].lower() for word in query_lower.split()):
                matches.append(faq)
        
        return matches[:3]  # Top 3 matches
    
    def _generate_general_response(self, query: str) -> str:
        """Generate general helpful response"""
        return (
            "I'm here to help with any questions about our products, orders, or services. "
            "Please feel free to ask about specific items, check order status, or learn about "
            "our policies and programs."
        )
    
    def _get_helpful_links(self, query: str) -> List[Dict[str, Any]]:
        """Get helpful links based on query"""
        return [
            {"title": "Store Locator", "url": "/stores"},
            {"title": "Order Tracking", "url": "/orders/track"},
            {"title": "Return Portal", "url": "/returns"},
            {"title": "Customer Support", "url": "/support"}
        ]
    
    def _get_contact_options(self) -> Dict[str, Any]:
        """Get customer contact options"""
        return {
            "phone": "0800 123 456",
            "email": "support@meridianretail.co.za",
            "chat": "Available 24/7",
            "whatsapp": "+27 123 456 789",
            "store_visit": "Find nearest store"
        }
    
    def _get_store_information(self, location: str) -> Dict[str, Any]:
        """Get store information"""
        stores = {
            "cape_town": {
                "address": "V&A Waterfront, Cape Town",
                "phone": "021 123 4567",
                "hours": "Mon-Sat 9AM-8PM, Sun 10AM-6PM"
            },
            "johannesburg": {
                "address": "Sandton City, Johannesburg",
                "phone": "011 123 4567",
                "hours": "Mon-Sat 9AM-8PM, Sun 10AM-6PM"
            }
        }
        
        if location == "national":
            return {"total_stores": 25, "find_store_link": "/stores"}
        
        return stores.get(location, stores["cape_town"])
    
    async def _get_recent_orders(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get recent orders for customer"""
        # Mock data
        return [
            {
                "order_id": "ORD-2024-1234",
                "date": "2024-11-20",
                "status": "delivered",
                "total": 2499
            }
        ]
    
    async def _check_product_availability(self, query: str) -> Dict[str, Any]:
        """Check product availability"""
        # Mock availability check
        return {
            "in_stock": True,
            "locations": ["Cape Town", "Johannesburg"],
            "online_available": True,
            "expected_restock": None
        }
    
    def _get_return_policy(self) -> Dict[str, Any]:
        """Get return policy details"""
        return {
            "return_window": "30 days",
            "condition": "Original condition with tags",
            "refund_method": "Original payment method",
            "processing_time": "5-7 business days"
        }
    
    # Tool wrapper methods
    def _customer_query_tool(self, customer_id: str, query: str) -> Dict[str, Any]:
        """Tool wrapper for customer queries"""
        return asyncio.run(self._handle_support_query(
            query,
            {"customer_id": customer_id}
        ))
    
    def _recommendation_tool(
        self,
        customer_id: str,
        category: str = "all"
    ) -> Dict[str, Any]:
        """Tool wrapper for recommendations"""
        return asyncio.run(self._handle_recommendation_query(
            f"Recommend {category} products",
            {"customer_id": customer_id, "category": category}
        ))
    
    def _complaint_tool(self, complaint: str, customer_id: str) -> Dict[str, Any]:
        """Tool wrapper for complaints"""
        return asyncio.run(self._handle_complaint_query(
            complaint,
            {"customer_id": customer_id}
        ))
    
    def _loyalty_tool(self, customer_id: str) -> Dict[str, Any]:
        """Tool wrapper for loyalty analysis"""
        return asyncio.run(self._handle_loyalty_query(
            "Check loyalty status",
            {"customer_id": customer_id}
        ))
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information for UI display"""
        return {
            "name": self.name,
            "role": self.role,
            "description": "Customer service and personalized shopping assistance",
            "capabilities": [cap.name for cap in self.capabilities],
            "status": self.get_status_dict(),
            "specialties": [
                "Customer support",
                "Personalized recommendations",
                "Complaint resolution",
                "Loyalty program management",
                "Customer retention"
            ],
            "metrics": {
                "avg_satisfaction_score": 0.92,
                "resolution_rate": 0.96,
                "response_time": "< 2 minutes"
            }
        }