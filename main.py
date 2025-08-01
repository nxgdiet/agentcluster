import requests
import json
import os
import time
from typing import Dict, Any, Optional, List
from openai import OpenAI
from dotenv import load_dotenv
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nftgaming import NFTGamingAgent
from nftpriceEstimate import NFTPriceEstimateAgent
from nftbrand import NFTBrandAgent
from nftdefi import NFTDeFiAgent
from nftfungible import NFTFungibleAgent
from nftwallet import NFTWalletAgent
from nfttoken import NFTTokenAgent
from portfolio import PortfolioAgent

# FastAPI app initialization
app = FastAPI(
    title="NFT Orchestrator API",
    description="A comprehensive NFT analytics orchestrator that routes queries to specialized agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    verbose: bool = True

class ChatResponse(BaseModel):
    response: str
    agent_used: str
    query: str
    reason: str
    success: bool
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    agents_available: List[str]

class NFTOrchestrator:
    def __init__(self, verbose: bool = True):
        """
        Initialize the NFT Orchestrator that routes queries to appropriate agents
        
        Args:
            verbose (bool): Enable verbose logging to see orchestrator's thinking process
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API keys from environment variables
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.client = OpenAI(api_key=openai_api_key)
        self.verbose = verbose
        
        # Initialize all agents with verbose=True to see detailed tool execution
        self.gaming_agent = NFTGamingAgent(verbose=True)  # Set to True to see detailed tool execution
        self.price_agent = NFTPriceEstimateAgent(verbose=True)
        self.brand_agent = NFTBrandAgent(verbose=True)
        self.defi_agent = NFTDeFiAgent(verbose=True)
        self.fungible_agent = NFTFungibleAgent(verbose=True)
        self.wallet_agent = NFTWalletAgent(verbose=True)
        self.token_agent = NFTTokenAgent(verbose=True)
        self.portfolio_agent = PortfolioAgent(verbose=True)
        
        # Define the routing tools for OpenAI function calling
        self.routing_tools = [
            {
                "type": "function",
                "function": {
                    "name": "route_to_gaming_agent",
                    "description": "Route the query to the NFT Gaming Agent. Use this for queries about gaming metrics, game contracts, gaming collections, player activity, gaming performance, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The user's original query to be processed by the gaming agent"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief explanation of why this query should go to the gaming agent"
                            }
                        },
                        "required": ["query", "reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "route_to_price_agent",
                    "description": "Route the query to the NFT Price Estimate Agent. Use this for queries about price predictions, price estimates, NFT valuations, collection pricing, token pricing, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The user's original query to be processed by the price estimation agent"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief explanation of why this query should go to the price estimation agent"
                            }
                        },
                        "required": ["query", "reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "route_to_brand_agent",
                    "description": "Route the query to the NFT Brand Agent. Use this for queries about brand NFTs, brand metrics, brand categories, specific brands like Starbucks, Nike, Adidas, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The user's original query to be processed by the brand agent"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief explanation of why this query should go to the brand agent"
                            }
                        },
                        "required": ["query", "reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "route_to_defi_agent",
                    "description": "Route the query to the NFT DeFi Agent. Use this for queries about DeFi pools, DEX protocols, pair addresses, Uniswap, Sushiswap, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The user's original query to be processed by the DeFi agent"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief explanation of why this query should go to the DeFi agent"
                            }
                        },
                        "required": ["query", "reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "route_to_fungible_agent",
                    "description": "Route the query to the NFT Fungible Token Agent. Use this for queries about fungible tokens, ERC-20 tokens, historical prices, price estimates, token prices.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The user's original query to be processed by the fungible token agent"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief explanation of why this query should go to the fungible token agent"
                            }
                        },
                        "required": ["query", "reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "route_to_wallet_agent",
                    "description": "Route the query to the NFT Wallet Analytics Agent. Use this for queries about wallet analytics, wallet scores, wallet profiles, wallet performance, wallet ratings.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The user's original query to be processed by the wallet analytics agent"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief explanation of why this query should go to the wallet analytics agent"
                            }
                        },
                        "required": ["query", "reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "route_to_token_agent",
                    "description": "Route the query to the NFT Token Analytics Agent. Use this for queries about token metrics, token price predictions, DEX prices, token performance, or token market data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The user's original query to be processed by the token analytics agent"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief explanation of why this query should go to the token analytics agent"
                            }
                        },
                        "required": ["query", "reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "route_to_portfolio_agent",
                    "description": "Route the query to the Portfolio Analysis Agent. Use this for queries about wallet portfolios, DeFi holdings, NFT holdings, ERC20 tokens, wallet labels, wallet scores, wallet metrics, comprehensive wallet analysis, or portfolio analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The user's original query to be processed by the portfolio analysis agent"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief explanation of why this query should go to the portfolio analysis agent"
                            }
                        },
                        "required": ["query", "reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "route_to_both_agents",
                    "description": "Route the query to both agents when it contains elements of both gaming and pricing. Use this for complex queries that need both gaming metrics and price information. ALWAYS split the query into separate parts for each agent.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "gaming_query": {
                                "type": "string",
                                "description": "The part of the query related to gaming metrics (e.g., 'game contracts', 'gaming metrics', 'player activity')"
                            },
                            "price_query": {
                                "type": "string",
                                "description": "The part of the query related to price estimation (e.g., 'collection metadata', 'price estimates', 'supported collections')"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief explanation of why this query needs both agents"
                            }
                        },
                        "required": ["gaming_query", "price_query", "reason"]
                    }
                }
            }
        ]

    def route_to_gaming_agent(self, query: str, reason: str) -> Dict[str, Any]:
        """Route query to the gaming agent"""
        if self.verbose:
            print(f"🎮 ROUTING TO GAMING AGENT")
            print(f"📝 Query: {query}")
            print(f"💭 Reason: {reason}")
        
        try:
            if self.verbose:
                print(f"🎮 CALLING GAMING AGENT WITH QUERY: '{query}'")
            
            result = self.gaming_agent.chat(query)
            
            if self.verbose:
                print(f"🎮 GAMING AGENT RAW OUTPUT:")
                print(f"📄 {result}")
                print(f"🎮 GAMING AGENT EXECUTION COMPLETED")
            
            return {
                "agent": "gaming",
                "query": query,
                "reason": reason,
                "response": result
            }
        except Exception as e:
            error_msg = f"Gaming agent error: {str(e)}"
            if self.verbose:
                print(f"❌ GAMING AGENT ERROR: {error_msg}")
                print(f"❌ ERROR TYPE: {type(e).__name__}")
                import traceback
                print(f"❌ FULL TRACEBACK:")
                traceback.print_exc()
            return {
                "agent": "gaming",
                "query": query,
                "reason": reason,
                "error": error_msg
            }

    def route_to_price_agent(self, query: str, reason: str) -> Dict[str, Any]:
        """Route query to the price estimation agent"""
        if self.verbose:
            print(f"💰 ROUTING TO PRICE ESTIMATION AGENT")
            print(f"📝 Query: {query}")
            print(f"💭 Reason: {reason}")
        
        try:
            if self.verbose:
                print(f"💰 CALLING PRICE AGENT WITH QUERY: '{query}'")
            
            result = self.price_agent.chat(query)
            
            if self.verbose:
                print(f"💰 PRICE AGENT RAW OUTPUT:")
                print(f"📄 {result}")
                print(f"💰 PRICE AGENT EXECUTION COMPLETED")
            
            return {
                "agent": "price_estimation",
                "query": query,
                "reason": reason,
                "response": result
            }
        except Exception as e:
            error_msg = f"Price agent error: {str(e)}"
            if self.verbose:
                print(f"❌ PRICE AGENT ERROR: {error_msg}")
                print(f"❌ ERROR TYPE: {type(e).__name__}")
                import traceback
                print(f"❌ FULL TRACEBACK:")
                traceback.print_exc()
            return {
                "agent": "price_estimation",
                "query": query,
                "reason": reason,
                "error": error_msg
            }

    def route_to_brand_agent(self, query: str, reason: str) -> Dict[str, Any]:
        """Route query to the brand agent"""
        if self.verbose:
            print(f"🏷️  ROUTING TO BRAND AGENT")
            print(f"📝 Query: {query}")
            print(f"💭 Reason: {reason}")
        
        try:
            if self.verbose:
                print(f"🏷️  CALLING BRAND AGENT WITH QUERY: '{query}'")
            
            result = self.brand_agent.chat(query)
            
            if self.verbose:
                print(f"🏷️  BRAND AGENT RAW OUTPUT:")
                print(f"📄 {result}")
                print(f"🏷️  BRAND AGENT EXECUTION COMPLETED")
            
            return {
                "agent": "brand",
                "query": query,
                "reason": reason,
                "response": result
            }
        except Exception as e:
            error_msg = f"Brand agent error: {str(e)}"
            if self.verbose:
                print(f"❌ BRAND AGENT ERROR: {error_msg}")
                print(f"❌ ERROR TYPE: {type(e).__name__}")
                import traceback
                print(f"❌ FULL TRACEBACK:")
                traceback.print_exc()
            return {
                "agent": "brand",
                "query": query,
                "reason": reason,
                "error": error_msg
            }

    def route_to_defi_agent(self, query: str, reason: str) -> Dict[str, Any]:
        """Route query to the DeFi agent"""
        if self.verbose:
            print(f"🔄 ROUTING TO DEFI AGENT")
            print(f"📝 Query: {query}")
            print(f"💭 Reason: {reason}")
        
        try:
            if self.verbose:
                print(f"🔄 CALLING DEFI AGENT WITH QUERY: '{query}'")
            
            result = self.defi_agent.chat(query)
            
            if self.verbose:
                print(f"🔄 DEFI AGENT RAW OUTPUT:")
                print(f"📄 {result}")
                print(f"🔄 DEFI AGENT EXECUTION COMPLETED")
            
            return {
                "agent": "defi",
                "query": query,
                "reason": reason,
                "response": result
            }
        except Exception as e:
            error_msg = f"DeFi agent error: {str(e)}"
            if self.verbose:
                print(f"❌ DEFI AGENT ERROR: {error_msg}")
                print(f"❌ ERROR TYPE: {type(e).__name__}")
                import traceback
                print(f"❌ FULL TRACEBACK:")
                traceback.print_exc()
            return {
                "agent": "defi",
                "query": query,
                "reason": reason,
                "error": error_msg
            }

    def route_to_fungible_agent(self, query: str, reason: str) -> Dict[str, Any]:
        """Route query to the fungible token agent"""
        if self.verbose:
            print(f"🪙 ROUTING TO FUNGIBLE TOKEN AGENT")
            print(f"📝 Query: {query}")
            print(f"💭 Reason: {reason}")
        
        try:
            if self.verbose:
                print(f"🪙 CALLING FUNGIBLE TOKEN AGENT WITH QUERY: '{query}'")
            
            result = self.fungible_agent.chat(query)
            
            if self.verbose:
                print(f"🪙 FUNGIBLE TOKEN AGENT RAW OUTPUT:")
                print(f"📄 {result}")
                print(f"🪙 FUNGIBLE TOKEN AGENT EXECUTION COMPLETED")
            
            return {
                "agent": "fungible",
                "query": query,
                "reason": reason,
                "response": result
            }
        except Exception as e:
            error_msg = f"Fungible token agent error: {str(e)}"
            if self.verbose:
                print(f"❌ FUNGIBLE TOKEN AGENT ERROR: {error_msg}")
                print(f"❌ ERROR TYPE: {type(e).__name__}")
                import traceback
                print(f"❌ FULL TRACEBACK:")
                traceback.print_exc()
            return {
                "agent": "fungible",
                "query": query,
                "reason": reason,
                "error": error_msg
            }

    def route_to_wallet_agent(self, query: str, reason: str) -> Dict[str, Any]:
        """Route query to the wallet analytics agent"""
        if self.verbose:
            print(f"💼 ROUTING TO WALLET ANALYTICS AGENT")
            print(f"📝 Query: {query}")
            print(f"💭 Reason: {reason}")
        
        try:
            if self.verbose:
                print(f"💼 CALLING WALLET ANALYTICS AGENT WITH QUERY: '{query}'")
            
            result = self.wallet_agent.chat(query)
            
            if self.verbose:
                print(f"💼 WALLET ANALYTICS AGENT RAW OUTPUT:")
                print(f"📄 {result}")
                print(f"💼 WALLET ANALYTICS AGENT EXECUTION COMPLETED")
            
            return {
                "agent": "wallet",
                "query": query,
                "reason": reason,
                "response": result
            }
        except Exception as e:
            error_msg = f"Wallet analytics agent error: {str(e)}"
            if self.verbose:
                print(f"❌ WALLET ANALYTICS AGENT ERROR: {error_msg}")
                print(f"❌ ERROR TYPE: {type(e).__name__}")
                import traceback
                print(f"❌ FULL TRACEBACK:")
                traceback.print_exc()
            return {
                "agent": "wallet",
                "query": query,
                "reason": reason,
                "error": error_msg
            }

    def route_to_token_agent(self, query: str, reason: str) -> Dict[str, Any]:
        """Route query to the token analytics agent"""
        if self.verbose:
            print(f"🪙 ROUTING TO TOKEN ANALYTICS AGENT")
            print(f"📝 Query: {query}")
            print(f"💭 Reason: {reason}")
        
        try:
            if self.verbose:
                print(f"🪙 CALLING TOKEN ANALYTICS AGENT WITH QUERY: '{query}'")
            
            result = self.token_agent.chat(query)
            
            if self.verbose:
                print(f"🪙 TOKEN ANALYTICS AGENT RAW OUTPUT:")
                print(f"📄 {result}")
                print(f"🪙 TOKEN ANALYTICS AGENT EXECUTION COMPLETED")
            
            return {
                "agent": "token",
                "query": query,
                "reason": reason,
                "response": result
            }
        except Exception as e:
            error_msg = f"Token analytics agent error: {str(e)}"
            if self.verbose:
                print(f"❌ TOKEN ANALYTICS AGENT ERROR: {error_msg}")
                print(f"❌ ERROR TYPE: {type(e).__name__}")
                import traceback
                print(f"❌ FULL TRACEBACK:")
                traceback.print_exc()
            return {
                "agent": "token",
                "query": query,
                "reason": reason,
                "error": error_msg
            }

    def route_to_portfolio_agent(self, query: str, reason: str) -> Dict[str, Any]:
        """Route query to the portfolio analysis agent"""
        if self.verbose:
            print(f"💼 ROUTING TO PORTFOLIO ANALYSIS AGENT")
            print(f"📝 Query: {query}")
            print(f"💭 Reason: {reason}")
        
        try:
            if self.verbose:
                print(f"💼 CALLING PORTFOLIO ANALYSIS AGENT WITH QUERY: '{query}'")
            
            result = self.portfolio_agent.chat(query)
            
            if self.verbose:
                print(f"💼 PORTFOLIO ANALYSIS AGENT RAW OUTPUT:")
                print(f"📄 {result}")
                print(f"💼 PORTFOLIO ANALYSIS AGENT EXECUTION COMPLETED")
            
            return {
                "agent": "portfolio",
                "query": query,
                "reason": reason,
                "response": result
            }
        except Exception as e:
            error_msg = f"Portfolio analysis agent error: {str(e)}"
            if self.verbose:
                print(f"❌ PORTFOLIO ANALYSIS AGENT ERROR: {error_msg}")
                print(f"❌ ERROR TYPE: {type(e).__name__}")
                import traceback
                print(f"❌ FULL TRACEBACK:")
                traceback.print_exc()
            return {
                "agent": "portfolio",
                "query": query,
                "reason": reason,
                "error": error_msg
            }

    def route_to_both_agents(self, gaming_query: str, price_query: str, reason: str) -> Dict[str, Any]:
        """Route query to both agents"""
        if self.verbose:
            print(f"🔄 ROUTING TO BOTH AGENTS")
            print(f"🎮 Gaming Query: {gaming_query}")
            print(f"💰 Price Query: {price_query}")
            print(f"💭 Reason: {reason}")
        
        try:
            # Execute gaming agent FIRST and wait for completion
            if self.verbose:
                print(f"\n🎮 EXECUTING GAMING AGENT (FIRST)...")
                print(f"⏳ WAITING FOR GAMING AGENT TO COMPLETE...")
                print(f"🎮 GAMING QUERY: '{gaming_query}'")
            
            gaming_result = self.gaming_agent.chat(gaming_query)
            
            if self.verbose:
                print(f"✅ GAMING AGENT COMPLETED")
                print(f"🎮 GAMING AGENT RAW OUTPUT:")
                print(f"📄 {gaming_result}")
                print(f"🎮 GAMING AGENT EXECUTION COMPLETED")
            
            # Small delay between agent executions for clarity
            if self.verbose:
                print(f"⏳ PAUSING BEFORE NEXT AGENT...")
            time.sleep(1)
            
            # Execute price agent SECOND and wait for completion
            if self.verbose:
                print(f"\n💰 EXECUTING PRICE AGENT (SECOND)...")
                print(f"⏳ WAITING FOR PRICE AGENT TO COMPLETE...")
                print(f"💰 PRICE QUERY: '{price_query}'")
            
            price_result = self.price_agent.chat(price_query)
            
            if self.verbose:
                print(f"✅ PRICE AGENT COMPLETED")
                print(f"💰 PRICE AGENT RAW OUTPUT:")
                print(f"📄 {price_result}")
                print(f"💰 PRICE AGENT EXECUTION COMPLETED")
            
            return {
                "agent": "both",
                "gaming_query": gaming_query,
                "price_query": price_query,
                "reason": reason,
                "gaming_response": gaming_result,
                "price_response": price_result
            }
        except Exception as e:
            error_msg = f"Both agents error: {str(e)}"
            if self.verbose:
                print(f"❌ BOTH AGENTS ERROR: {error_msg}")
                print(f"❌ ERROR TYPE: {type(e).__name__}")
                import traceback
                print(f"❌ FULL TRACEBACK:")
                traceback.print_exc()
            return {
                "agent": "both",
                "gaming_query": gaming_query,
                "price_query": price_query,
                "reason": reason,
                "error": error_msg
            }

    def execute_routing_call(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the appropriate routing function"""
        if self.verbose:
            print(f"\n🎯 EXECUTING ROUTING FUNCTION: {function_name}")
            print(f"🔍 FUNCTION ARGUMENTS: {json.dumps(arguments, indent=2)}")
        
        if function_name == "route_to_gaming_agent":
            return self.route_to_gaming_agent(**arguments)
        elif function_name == "route_to_price_agent":
            return self.route_to_price_agent(**arguments)
        elif function_name == "route_to_brand_agent":
            return self.route_to_brand_agent(**arguments)
        elif function_name == "route_to_defi_agent":
            return self.route_to_defi_agent(**arguments)
        elif function_name == "route_to_fungible_agent":
            return self.route_to_fungible_agent(**arguments)
        elif function_name == "route_to_wallet_agent":
            return self.route_to_wallet_agent(**arguments)
        elif function_name == "route_to_token_agent":
            return self.route_to_token_agent(**arguments)
        elif function_name == "route_to_portfolio_agent":
            return self.route_to_portfolio_agent(**arguments)
        elif function_name == "route_to_both_agents":
            return self.route_to_both_agents(**arguments)
        else:
            error_result = {"error": f"Unknown routing function: {function_name}"}
            if self.verbose:
                print(f"❌ ROUTING ERROR: {error_result}")
            return error_result

    def format_final_response(self, routing_result: Dict[str, Any]) -> str:
        """Format the final response based on routing results"""
        if routing_result.get("agent") == "both":
            # Combine responses from both agents
            gaming_response = routing_result.get("gaming_response", "No gaming data available")
            price_response = routing_result.get("price_response", "No price data available")
            
            formatted_response = f"""🤖 **Multi-Agent Response**

🎮 **Gaming Metrics:**
{gaming_response}

💰 **Price Estimation:**
{price_response}

---
*This response combines data from both the NFT Gaming Agent and NFT Price Estimation Agent.*"""
            
        elif routing_result.get("agent") == "gaming":
            formatted_response = f"""🎮 **NFT Gaming Response**

{routing_result.get('response', 'No response available')}

---
*Processed by NFT Gaming Agent*"""
            
        elif routing_result.get("agent") == "price_estimation":
            formatted_response = f"""💰 **NFT Price Estimation Response**

{routing_result.get('response', 'No response available')}

---
*Processed by NFT Price Estimation Agent*"""
            
        elif routing_result.get("agent") == "brand":
            formatted_response = f"""🏷️ **NFT Brand Response**

{routing_result.get('response', 'No response available')}

---
*Processed by NFT Brand Agent*"""
            
        elif routing_result.get("agent") == "defi":
            formatted_response = f"""🔄 **NFT DeFi Response**

{routing_result.get('response', 'No response available')}

---
*Processed by NFT DeFi Agent*"""
            
        elif routing_result.get("agent") == "fungible":
            formatted_response = f"""🪙 **NFT Fungible Token Response**

{routing_result.get('response', 'No response available')}

---
*Processed by NFT Fungible Token Agent*"""
            
        elif routing_result.get("agent") == "wallet":
            formatted_response = f"""💼 **NFT Wallet Analytics Response**

{routing_result.get('response', 'No response available')}

---
*Processed by NFT Wallet Analytics Agent*"""
            
        elif routing_result.get("agent") == "token":
            formatted_response = f"""🪙 **NFT Token Analytics Response**

{routing_result.get('response', 'No response available')}

---
*Processed by NFT Token Analytics Agent*"""
            
        elif routing_result.get("agent") == "portfolio":
            formatted_response = f"""💼 **Portfolio Analysis Response**

{routing_result.get('response', 'No response available')}

---
*Processed by Portfolio Analysis Agent*"""
            
        else:
            formatted_response = f"❌ Error: {routing_result.get('error', 'Unknown error')}"
        
        return formatted_response

    def chat(self, user_message: str) -> str:
        """
        Process a natural language query and route to appropriate agent(s)
        
        Args:
            user_message (str): Natural language query from the user
            
        Returns:
            str: Formatted response with the requested data
        """
        if self.verbose:
            print(f"\n" + "="*60)
            print(f"🧠 ORCHESTRATOR THINKING PROCESS")
            print(f"="*60)
            print(f"💬 USER QUERY: {user_message}")
            print(f"🤖 Analyzing query and determining which agent(s) to route to...")
        
        try:
            # Create the initial conversation with system prompt
            messages = [
                {
                    "role": "system",
                    "content": """You are an NFT Query Orchestrator. You analyze user queries and route them to the appropriate specialized agent:
NOTE: Since you are connected to a frontend, you will also receive information about the wallet which is connected, so please don't stress too much about it until and unless you are asked something related to it

**Available Agents:**
1. **NFT Gaming Agent** - Handles gaming metrics, game contracts, player activity, gaming performance, gaming collections
2. **NFT Price Estimation Agent** - Handles price predictions, price estimates, NFT valuations, collection pricing, token pricing
3. **NFT Brand Agent** - Handles brand NFTs, brand metrics, brand categories, specific brands like Starbucks, Nike, Adidas, etc.
4. **NFT DeFi Agent** - Handles DeFi pools, DEX protocols, pair addresses, Uniswap, Sushiswap, PancakeSwap, etc.
5. **NFT Fungible Token Agent** - Handles fungible tokens, ERC-20 tokens, historical prices, price estimates, token prices
6. **NFT Wallet Analytics Agent** - Handles wallet analytics, wallet scores, wallet profiles, wallet performance, wallet ratings
7. **NFT Token Analytics Agent** - Handles token metrics, token price predictions, DEX prices, token performance, token market data
8. **Portfolio Analysis Agent** - Handles wallet portfolios, DeFi holdings, NFT holdings, ERC20 tokens, wallet labels, wallet scores, wallet metrics, comprehensive wallet analysis

**Routing Rules:**
- If the query mentions gaming, games, players, gaming metrics, game contracts → route_to_gaming_agent
- If the query mentions price, pricing, estimates, valuations, predictions → route_to_price_agent  
- If the query mentions brands, brand NFTs, specific brands (Starbucks, Nike, Adidas, etc.), brand categories → route_to_brand_agent
- If the query mentions DeFi, DEX, pools, protocols, pair addresses, Uniswap, Sushiswap, PancakeSwap → route_to_defi_agent
- If the query mentions fungible tokens, ERC-20, historical prices, token prices, price history → route_to_fungible_agent
- If the query mentions wallet, wallet analytics, wallet scores, wallet profiles, wallet performance, wallet ratings → route_to_wallet_agent
- If the query mentions token metrics, token price predictions, DEX prices, token performance, token market data → route_to_token_agent
- If the query mentions portfolio, DeFi holdings, NFT holdings, ERC20 tokens, wallet labels, wallet scores, wallet metrics, comprehensive wallet analysis → route_to_portfolio_agent
- If the query mentions both gaming AND pricing/price → route_to_both_agents
- If the query contains multiple parts that need different agents, SPLIT THE QUERY and use route_to_both_agents

**IMPORTANT: When splitting queries:**
- "collection metadata" → goes to price agent (get_supported_collections)
- "game contracts" → goes to gaming agent (get_game_contracts_info)
- "brand details", "brand metrics", "brand categories" → goes to brand agent
- "DeFi pools", "DEX protocols", "pair addresses" → goes to DeFi agent
- "fungible tokens", "ERC-20", "historical prices", "token prices" → goes to fungible token agent
- "token metrics", "token price predictions", "DEX prices", "token performance" → goes to token analytics agent

**Keywords for Gaming Agent:** game, gaming, player, contract, metrics, activity, performance, collection (in gaming context)
**Keywords for Price Agent:** price, pricing, estimate, prediction, valuation, cost, worth, value, metadata, collections
**Keywords for Brand Agent:** brand, brands, Starbucks, Nike, Adidas, Coca-Cola, McDonald's, Gucci, Louis Vuitton, category, categories
**Keywords for DeFi Agent:** defi, dex, pool, pools, protocol, protocols, pair, address, uniswap, sushiswap, pancakeswap, curve, balancer, aave, compound
**Keywords for Fungible Token Agent:** fungible, token, tokens, erc-20, erc20, historical, history, price history, token price, usdc, eth, dai
**Keywords for Wallet Analytics Agent:** wallet, analytics, scores, profile, performance, rating, ratings, portfolio, trading, metrics, trends
**Keywords for Token Analytics Agent:** token metrics, token price predictions, DEX prices, token performance, token market data, price forecasts, volatility trends
**Keywords for Portfolio Analysis Agent:** portfolio, defi holdings, nft holdings, erc20 tokens, wallet labels, wallet scores, wallet metrics, comprehensive analysis, wallet analysis, holdings, balance

**ALWAYS split complex queries that mention both collection metadata AND game contracts into separate parts for each agent.**"""
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]

            if self.verbose:
                print(f"🔄 Making routing decision with GPT-4o...")

            # Make the initial API call to GPT-4o for routing
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.routing_tools,
                tool_choice="auto"
            )

            if self.verbose:
                print(f"📤 GPT-4o ROUTING RESPONSE RECEIVED")
                if response.choices[0].message.tool_calls:
                    print(f"🛠️  GPT-4o wants to call {len(response.choices[0].message.tool_calls)} routing function(s)")
                else:
                    print(f"💭 GPT-4o provided direct response (no routing needed)")

            # Check if the model wants to call a routing function
            if response.choices[0].message.tool_calls:
                # Add the assistant's response to messages
                messages.append(response.choices[0].message)
                
                # Process each routing call SEQUENTIALLY (one at a time)
                routing_results = []
                for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                    if self.verbose:
                        print(f"\n📞 ROUTING CALL #{i+1}:")
                        print(f"🔧 Function: {tool_call.function.name}")
                        print(f"🆔 Call ID: {tool_call.id}")
                    
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if self.verbose:
                        print(f"⏳ EXECUTING ROUTING FUNCTION #{i+1} - WAITING FOR COMPLETION...")
                    
                    # Execute the routing function and WAIT for completion
                    routing_result = self.execute_routing_call(function_name, function_args)
                    routing_results.append(routing_result)
                    
                    if self.verbose:
                        print(f"✅ ROUTING FUNCTION #{i+1} COMPLETED")
                        print(f"🔄 Adding result to conversation context...")
                    
                    # Add the routing result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(routing_result)
                    })
                    
                    # Wait a moment before processing next routing call (if any)
                    if i < len(response.choices[0].message.tool_calls) - 1:
                        if self.verbose:
                            print(f"⏳ MOVING TO NEXT ROUTING CALL...")

                if self.verbose:
                    print(f"\n🔄 Sending results back to GPT-4o for final formatting...")

                # Get the final response from GPT-4o for formatting
                final_response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                
                final_content = final_response.choices[0].message.content
                
                if self.verbose:
                    print(f"\n🎯 ORCHESTRATOR FINAL SUMMARY:")
                    print(f"📊 Routing Results: {len(routing_results)} agent(s) executed")
                    for i, result in enumerate(routing_results):
                        agent_type = result.get("agent", "unknown")
                        print(f"   {i+1}. {agent_type.upper()} AGENT")
                        if agent_type == "both":
                            print(f"      🎮 Gaming Query: {result.get('gaming_query', 'N/A')}")
                            print(f"      💰 Price Query: {result.get('price_query', 'N/A')}")
                        else:
                            print(f"      📝 Query: {result.get('query', 'N/A')}")
                        print(f"      💭 Reason: {result.get('reason', 'N/A')}")
                        if agent_type == "brand":
                            print(f"      🏷️  Brand Agent Used")
                        elif agent_type == "gaming":
                            print(f"      🎮 Gaming Agent Used")
                        elif agent_type == "price_estimation":
                            print(f"      💰 Price Agent Used")
                        elif agent_type == "defi":
                            print(f"      🔄 DeFi Agent Used")
                        elif agent_type == "fungible":
                            print(f"      🪙 Fungible Token Agent Used")
                        elif agent_type == "wallet":
                            print(f"      💼 Wallet Analytics Agent Used")
                        elif agent_type == "token":
                            print(f"      🪙 Token Analytics Agent Used")
                        elif agent_type == "portfolio":
                            print(f"      💼 Portfolio Analysis Agent Used")
                
                if self.verbose:
                    print(f"✅ FINAL RESPONSE GENERATED")
                    print(f"📝 Response length: {len(final_content)} characters")
                    print(f"="*60)
                    print(f"🎯 ORCHESTRATOR FINAL RESPONSE:")
                    print(f"="*60)
                
                return final_content
            else:
                # No routing needed, return the direct response
                direct_response = response.choices[0].message.content
                
                if self.verbose:
                    print(f"✅ DIRECT RESPONSE (no routing needed)")
                    print(f"📝 Response length: {len(direct_response)} characters")
                    print(f"="*60)
                    print(f"🎯 ORCHESTRATOR FINAL RESPONSE:")
                    print(f"="*60)
                
                return direct_response

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            if self.verbose:
                print(f"❌ CRITICAL ERROR: {error_msg}")
            return error_msg

# Global orchestrator instance
orchestrator_instance = None

def get_orchestrator():
    """Get or create the orchestrator instance"""
    global orchestrator_instance
    if orchestrator_instance is None:
        try:
            orchestrator_instance = NFTOrchestrator(verbose=True)
        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize orchestrator: {str(e)}")
    return orchestrator_instance

# API Endpoints

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check and API information"""
    return HealthResponse(
        status="healthy",
        message="NFT Orchestrator API is running",
        agents_available=[
            "NFT Gaming Agent",
            "NFT Price Estimation Agent", 
            "NFT Brand Agent",
            "NFT DeFi Agent",
            "NFT Fungible Token Agent",
            "NFT Wallet Analytics Agent",
            "NFT Token Analytics Agent",
            "Portfolio Analysis Agent"
        ]
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test orchestrator initialization
        get_orchestrator()
        return HealthResponse(
            status="healthy",
            message="All systems operational",
            agents_available=[
                "NFT Gaming Agent",
                "NFT Price Estimation Agent", 
                "NFT Brand Agent",
                "NFT DeFi Agent",
                "NFT Fungible Token Agent",
                "NFT Wallet Analytics Agent",
                "NFT Token Analytics Agent",
                "Portfolio Analysis Agent"
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that routes queries to appropriate agents
    
    Args:
        request: ChatRequest containing the user message and verbose flag
        
    Returns:
        ChatResponse with the agent response and metadata
    """
    try:
        orchestrator = get_orchestrator()
        
        # Set verbose mode based on request
        orchestrator.verbose = request.verbose
        
        # Process the chat request
        response = orchestrator.chat(request.message)
        
        # Extract agent information from the response
        agent_used = "unknown"
        reason = "Query processed by orchestrator"
        
        # Try to determine which agent was used based on response content
        if "🎮" in response and "Gaming" in response:
            agent_used = "gaming"
            reason = "Query routed to NFT Gaming Agent"
        elif "💰" in response and "Price" in response:
            agent_used = "price_estimation"
            reason = "Query routed to NFT Price Estimation Agent"
        elif "🏷️" in response and "Brand" in response:
            agent_used = "brand"
            reason = "Query routed to NFT Brand Agent"
        elif "🔄" in response and "DeFi" in response:
            agent_used = "defi"
            reason = "Query routed to NFT DeFi Agent"
        elif "🪙" in response and "Fungible" in response:
            agent_used = "fungible"
            reason = "Query routed to NFT Fungible Token Agent"
        elif "💼" in response and "Wallet" in response:
            agent_used = "wallet"
            reason = "Query routed to NFT Wallet Analytics Agent"
        elif "🪙" in response and "Token" in response:
            agent_used = "token"
            reason = "Query routed to NFT Token Analytics Agent"
        elif "💼" in response and "Portfolio" in response:
            agent_used = "portfolio"
            reason = "Query routed to Portfolio Analysis Agent"
        elif "Multi-Agent" in response:
            agent_used = "multiple"
            reason = "Query processed by multiple agents"
        
        return ChatResponse(
            response=response,
            agent_used=agent_used,
            query=request.message,
            reason=reason,
            success=True
        )
        
    except Exception as e:
        return ChatResponse(
            response="",
            agent_used="error",
            query=request.message,
            reason="Error occurred during processing",
            success=False,
            error=str(e)
        )

@app.get("/agents")
async def list_agents():
    """List all available agents and their capabilities"""
    return {
        "agents": [
            {
                "name": "NFT Gaming Agent",
                "description": "Handles gaming metrics, game contracts, player activity, gaming performance, gaming collections",
                "keywords": ["game", "gaming", "player", "contract", "metrics", "activity", "performance", "collection"]
            },
            {
                "name": "NFT Price Estimation Agent", 
                "description": "Handles price predictions, price estimates, NFT valuations, collection pricing, token pricing",
                "keywords": ["price", "pricing", "estimate", "prediction", "valuation", "cost", "worth", "value", "metadata", "collections"]
            },
            {
                "name": "NFT Brand Agent",
                "description": "Handles brand NFTs, brand metrics, brand categories, specific brands like Starbucks, Nike, Adidas, etc.",
                "keywords": ["brand", "brands", "Starbucks", "Nike", "Adidas", "Coca-Cola", "McDonald's", "Gucci", "Louis Vuitton", "category", "categories"]
            },
            {
                "name": "NFT DeFi Agent",
                "description": "Handles DeFi pools, DEX protocols, pair addresses, Uniswap, Sushiswap, PancakeSwap, etc.",
                "keywords": ["defi", "dex", "pool", "pools", "protocol", "protocols", "pair", "address", "uniswap", "sushiswap", "pancakeswap", "curve", "balancer", "aave", "compound"]
            },
            {
                "name": "NFT Fungible Token Agent",
                "description": "Handles fungible tokens, ERC-20 tokens, historical prices, price estimates, token prices",
                "keywords": ["fungible", "token", "tokens", "erc-20", "erc20", "historical", "history", "price history", "token price", "usdc", "eth", "dai"]
            },
            {
                "name": "NFT Wallet Analytics Agent",
                "description": "Handles wallet analytics, wallet scores, wallet profiles, wallet performance, wallet ratings",
                "keywords": ["wallet", "analytics", "scores", "profile", "performance", "rating", "ratings", "portfolio", "trading", "metrics", "trends"]
            },
            {
                "name": "NFT Token Analytics Agent",
                "description": "Handles token metrics, token price predictions, DEX prices, token performance, token market data",
                "keywords": ["token metrics", "token price predictions", "DEX prices", "token performance", "token market data", "price forecasts", "volatility trends"]
            },
            {
                "name": "Portfolio Analysis Agent",
                "description": "Handles wallet portfolios, DeFi holdings, NFT holdings, ERC20 tokens, wallet labels, wallet scores, wallet metrics, comprehensive wallet analysis",
                "keywords": ["portfolio", "defi holdings", "nft holdings", "erc20 tokens", "wallet labels", "wallet scores", "wallet metrics", "comprehensive analysis", "wallet analysis", "holdings", "balance"]
            }
        ]
    }

# Example usage and FastAPI server startup
if __name__ == "__main__":
    import uvicorn
    
    # Initialize the orchestrator (API keys will be loaded from .env file)
    try:
        # Set verbose=True to see the orchestrator's thinking process
        orchestrator_instance = NFTOrchestrator(verbose=True)
        print("✅ NFT Orchestrator initialized successfully")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please make sure you have a .env file with the required API keys.")
        exit(1)
    
    print("🚀 Starting FastAPI server...")
    print("📖 API Documentation available at: http://localhost:8000/docs")
    print("🔍 Alternative docs at: http://localhost:8000/redoc")
    print("💬 Chat endpoint: POST http://localhost:8000/chat")
    print("🏥 Health check: GET http://localhost:8000/health")
    print("📋 Agents list: GET http://localhost:8000/agents")
    print("\n" + "="*60)
    
    # Start the FastAPI server
    uvicorn.run(
        "orchestrator:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 
