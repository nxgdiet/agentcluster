import requests
import json
import os
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import re

class NFTTokenAgent:
    def __init__(self, verbose: bool = True):
        """
        Initialize the NFT Token Analytics Agent with API keys from .env file
        
        Args:
            verbose (bool): Enable verbose logging to see agent's thinking process
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API keys from environment variables
        openai_api_key = os.getenv('OPENAI_API_KEY')
        unleash_api_key = os.getenv('UNLEASH_NFTS_API_KEY')
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        if not unleash_api_key:
            raise ValueError("UNLEASH_NFTS_API_KEY not found in .env file")
        
        self.client = OpenAI(api_key=openai_api_key)
        self.api_key = unleash_api_key
        self.verbose = verbose
        self.base_url = "https://api.unleashnfts.com/api/v2/token"
        
        # Supported blockchains for token analytics
        self.supported_blockchains = [
            "avalanche", "ethereum", "base", "berachain", "linea", "polygon", "unichain"
        ]
        
        # Supported time ranges
        self.supported_time_ranges = ["24h", "7d", "30d", "90d", "all"]
        
        # Define the tools for OpenAI function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_token_metrics",
                    "description": "Get key metrics and metadata for a specified token. Use this when users ask for token metrics, token performance, token metadata, token insights, or token market data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "blockchain": {
                                "type": "string",
                                "description": "Blockchain for the token",
                                "enum": ["avalanche", "ethereum", "base", "berachain", "linea", "polygon", "unichain"]
                            },
                            "token_address": {
                                "type": "string",
                                "description": "The token contract address"
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Offset for pagination",
                                "default": 0
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Limit for number of results",
                                "default": 30
                            }
                        },
                        "required": ["blockchain", "token_address"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_token_price_prediction",
                    "description": "Get token price prediction with key market indicators and volatility trends. Use this when users ask for price predictions, price forecasts, price estimates, or future price movements.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token_address": {
                                "type": "string",
                                "description": "The token contract address"
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Offset for pagination",
                                "default": 0
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Limit for number of results",
                                "default": 30
                            }
                        },
                        "required": ["token_address"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_token_dex_price",
                    "description": "Get the USD price of an ERC-20 token from decentralized exchanges (DEXs). Use this when users ask for DEX prices, real-time pricing, current token price, or market prices.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "blockchain": {
                                "type": "string",
                                "description": "Blockchain for the token",
                                "enum": ["avalanche", "ethereum", "base", "berachain", "linea", "polygon", "unichain"]
                            },
                            "token_address": {
                                "type": "string",
                                "description": "The token contract address"
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for price data",
                                "enum": ["24h", "7d", "30d", "90d", "all"],
                                "default": "all"
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Offset for pagination",
                                "default": 0
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Limit for number of results",
                                "default": 30
                            }
                        },
                        "required": ["blockchain", "token_address"]
                    }
                }
            }
        ]

    def get_token_metrics(
        self, 
        blockchain: str, 
        token_address: str,
        offset: int = 0,
        limit: int = 30
    ) -> Dict[str, Any]:
        """Get key metrics and metadata for a specified token"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_token_metrics")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - blockchain: {blockchain}")
            print(f"   - token_address: {token_address}")
            print(f"   - offset: {offset}")
            print(f"   - limit: {limit}")
        
        url = f"{self.base_url}/metrics"
        params = {
            "blockchain": blockchain,
            "token_address": token_address,
            "offset": offset,
            "limit": limit
        }
        headers = {
            "accept": "application/json",
            "x-api-key": self.api_key
        }
        
        if self.verbose:
            print(f"🌐 API REQUEST: GET {url}")
            print(f"📊 QUERY PARAMS: {params}")
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            result = response.json()
            
            if self.verbose:
                print(f"✅ API RESPONSE: Status {response.status_code}")
                print(f"📄 RESPONSE SIZE: {len(json.dumps(result))} characters")
                if isinstance(result, dict) and 'data' in result:
                    print(f"📈 DATA ITEMS: {len(result.get('data', []))} items returned")
            
            return result
        except requests.exceptions.RequestException as e:
            error_result = {"error": f"API request failed: {str(e)}"}
            if self.verbose:
                print(f"❌ API ERROR: {error_result}")
            return error_result

    def get_token_price_prediction(
        self, 
        token_address: str,
        offset: int = 0,
        limit: int = 30
    ) -> Dict[str, Any]:
        """Get token price prediction with key market indicators and volatility trends"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_token_price_prediction")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - token_address: {token_address}")
            print(f"   - offset: {offset}")
            print(f"   - limit: {limit}")
        
        url = f"{self.base_url}/price_prediction"
        params = {
            "token_address": token_address,
            "offset": offset,
            "limit": limit
        }
        headers = {
            "accept": "application/json",
            "x-api-key": self.api_key
        }
        
        if self.verbose:
            print(f"🌐 API REQUEST: GET {url}")
            print(f"📊 QUERY PARAMS: {params}")
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            result = response.json()
            
            if self.verbose:
                print(f"✅ API RESPONSE: Status {response.status_code}")
                print(f"📄 RESPONSE SIZE: {len(json.dumps(result))} characters")
                if isinstance(result, dict) and 'data' in result:
                    print(f"📈 DATA ITEMS: {len(result.get('data', []))} items returned")
            
            return result
        except requests.exceptions.RequestException as e:
            error_result = {"error": f"API request failed: {str(e)}"}
            if self.verbose:
                print(f"❌ API ERROR: {error_result}")
            return error_result

    def get_token_dex_price(
        self, 
        blockchain: str, 
        token_address: str,
        time_range: str = "all",
        offset: int = 0,
        limit: int = 30
    ) -> Dict[str, Any]:
        """Get the USD price of an ERC-20 token from decentralized exchanges (DEXs)"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_token_dex_price")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - blockchain: {blockchain}")
            print(f"   - token_address: {token_address}")
            print(f"   - time_range: {time_range}")
            print(f"   - offset: {offset}")
            print(f"   - limit: {limit}")
        
        url = f"{self.base_url}/dex_price"
        params = {
            "blockchain": blockchain,
            "token_address": token_address,
            "time_range": time_range,
            "offset": offset,
            "limit": limit
        }
        headers = {
            "accept": "application/json",
            "x-api-key": self.api_key
        }
        
        if self.verbose:
            print(f"🌐 API REQUEST: GET {url}")
            print(f"📊 QUERY PARAMS: {params}")
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            result = response.json()
            
            if self.verbose:
                print(f"✅ API RESPONSE: Status {response.status_code}")
                print(f"📄 RESPONSE SIZE: {len(json.dumps(result))} characters")
                if isinstance(result, dict) and 'data' in result:
                    print(f"📈 DATA ITEMS: {len(result.get('data', []))} items returned")
            
            return result
        except requests.exceptions.RequestException as e:
            error_result = {"error": f"API request failed: {str(e)}"}
            if self.verbose:
                print(f"❌ API ERROR: {error_result}")
            return error_result

    def execute_function_call(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the appropriate function based on the function call"""
        if self.verbose:
            print(f"\n🎯 EXECUTING FUNCTION: {function_name}")
            print(f"🔍 FUNCTION ARGUMENTS: {json.dumps(arguments, indent=2)}")
        
        if function_name == "get_token_metrics":
            return self.get_token_metrics(**arguments)
        elif function_name == "get_token_price_prediction":
            return self.get_token_price_prediction(**arguments)
        elif function_name == "get_token_dex_price":
            return self.get_token_dex_price(**arguments)
        else:
            error_result = {"error": f"Unknown function: {function_name}"}
            if self.verbose:
                print(f"❌ FUNCTION ERROR: {error_result}")
            return error_result

    def chat(self, user_message: str) -> str:
        """
        Process a natural language query and return relevant NFT token data
        
        Args:
            user_message (str): Natural language query from the user
            
        Returns:
            str: Formatted response with the requested data
        """
        if self.verbose:
            print(f"\n" + "="*60)
            print(f"🧠 AGENT THINKING PROCESS")
            print(f"="*60)
            print(f"💬 USER QUERY: {user_message}")
            print(f"🤖 Analyzing query and determining appropriate tools...")
        
        try:
            # Create the initial conversation with system prompt
            messages = [
                {
                    "role": "system",
                    "content": """You are an NFT Token Analytics Assistant. You help users get information about token metrics, price predictions, and DEX prices using three main tools:

1. get_token_metrics - Use this when users ask for token metrics, token performance, token metadata, token insights, or token market data. This provides key metrics and metadata for a specified token.

2. get_token_price_prediction - Use this when users ask for price predictions, price forecasts, price estimates, future price movements, or volatility trends. This provides token price prediction with key market indicators.

3. get_token_dex_price - Use this when users ask for DEX prices, real-time pricing, current token price, market prices, or decentralized exchange prices. This provides the USD price of an ERC-20 token from DEXs.

IMPORTANT: 
- If users ask for "metrics", "performance", "metadata", "insights", "market data" → use get_token_metrics
- If users ask for "prediction", "forecast", "estimate", "future", "volatility" → use get_token_price_prediction
- If users ask for "DEX price", "real-time", "current price", "market price", "decentralized" → use get_token_dex_price

Supported blockchains: avalanche, ethereum, base, berachain, linea, polygon, unichain
Supported time ranges: 24h, 7d, 30d, 90d, all

When users ask questions, determine which tool(s) to use and call them appropriately. Provide clear, helpful responses based on the data returned."""
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]

            if self.verbose:
                print(f"🔄 Making initial request to GPT-4o...")

            # Make the initial API call to GPT-4o
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            if self.verbose:
                print(f"📤 GPT-4o RESPONSE RECEIVED")
                if response.choices[0].message.tool_calls:
                    print(f"🛠️  GPT-4o wants to call {len(response.choices[0].message.tool_calls)} tool(s)")
                else:
                    print(f"💭 GPT-4o provided direct response (no tools needed)")

            # Check if the model wants to call a function
            if response.choices[0].message.tool_calls:
                # Add the assistant's response to messages
                messages.append(response.choices[0].message)
                
                # Process each tool call
                for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                    if self.verbose:
                        print(f"\n📞 TOOL CALL #{i+1}:")
                        print(f"🔧 Function: {tool_call.function.name}")
                        print(f"🆔 Call ID: {tool_call.id}")
                    
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Execute the function
                    function_result = self.execute_function_call(function_name, function_args)
                    
                    if self.verbose:
                        print(f"✨ TOOL EXECUTION COMPLETED")
                        print(f"🔄 Adding result to conversation context...")
                    
                    # Add the function result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(function_result)
                    })

                if self.verbose:
                    print(f"\n🔄 Sending results back to GPT-4o for final response...")

                # Get the final response from GPT-4o
                final_response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                
                final_content = final_response.choices[0].message.content
                
                if self.verbose:
                    print(f"✅ FINAL RESPONSE GENERATED")
                    print(f"📝 Response length: {len(final_content)} characters")
                    print(f"="*60)
                    print(f"🎯 AGENT FINAL RESPONSE:")
                    print(f"="*60)
                
                return final_content
            else:
                # No function call needed, return the direct response
                direct_response = response.choices[0].message.content
                
                if self.verbose:
                    print(f"✅ DIRECT RESPONSE (no tools needed)")
                    print(f"📝 Response length: {len(direct_response)} characters")
                    print(f"="*60)
                    print(f"🎯 AGENT FINAL RESPONSE:")
                    print(f"="*60)
                
                return direct_response

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            if self.verbose:
                print(f"❌ CRITICAL ERROR: {error_msg}")
            return error_msg

# Example usage
if __name__ == "__main__":
    # Initialize the agent (API keys will be loaded from .env file)
    try:
        # Set verbose=True to see the agent's thinking process
        agent = NFTTokenAgent(verbose=True)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure you have a .env file with the required API keys.")
        exit(1)
    
    # Example conversations for testing
    example_queries = [
        "Get token metrics for 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48 on ethereum",
        "Show me price prediction for token 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "Get DEX price for token 0xdAC17F958D2ee523a2206206994597C13D831ec7 on ethereum",
        "What are the metrics for USDC token on polygon?",
        "Show me price forecast for token 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
    ]
    
    print("NFT Token Analytics Agent - Ready to chat!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        response = agent.chat(user_input)
        print(f"Agent: {response}\n") 