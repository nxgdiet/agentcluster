import requests
import json
import os
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import re

class NFTFungibleAgent:
    def __init__(self, verbose: bool = True):
        """
        Initialize the NFT Fungible Token Agent with API keys from .env file
        
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
        self.base_url = "https://api.unleashnfts.com/api/v1/ft"
        
        # Chain IDs mapping
        self.chain_ids = {
            "ethereum": 1,
            "polygon": 137,
            "avalanche": 43114,
            "binance": 56,
            "solana": 101,
            "linea": 59144,
            "base": 8453,
            "arbitrum": 42161,
            "optimism": 10,
            "polygon_zkevm": 1101,
            "mantle": 5000,
            "scroll": 534352,
            "zksync": 324,
            "polygon_zkevm_testnet": 1442,
            "mantle_testnet": 5001,
            "scroll_testnet": 534353,
            "zksync_testnet": 280,
            "base_testnet": 84531,
            "arbitrum_testnet": 421613,
            "optimism_testnet": 11155420,
            "linea_testnet": 59140
        }
        
        # Supported currencies
        self.supported_currencies = ["usdc", "eth", "dai"]
        
        # Supported time ranges
        self.supported_time_ranges = ["24h", "7d", "30d", "90d"]
        
        # Define the tools for OpenAI function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_historical_price",
                    "description": "Get the day-wise historical price of fungible tokens. Use this when users ask for historical price data, price history, or past prices of tokens.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chain_id": {
                                "type": "integer",
                                "description": "Chain ID for the blockchain. Use ONLY the numeric chain ID: 1=ethereum, 137=polygon, 43114=avalanche, 56=binance, 101=solana, 59144=linea, 8453=base, 42161=arbitrum, 10=optimism, etc. DO NOT use chain names, only use the numeric ID.",
                                "enum": [1, 137, 43114, 56, 101, 59144, 8453, 42161, 10, 1101, 5000, 534352, 324, 1442, 5001, 534353, 280, 84531, 421613, 11155420, 59140]
                            },
                            "token_address": {
                                "type": "string",
                                "description": "The token contract address"
                            },
                            "currency": {
                                "type": "string",
                                "description": "Currency for price data",
                                "enum": ["usdc", "eth", "dai"],
                                "default": "usdc"
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for historical data",
                                "enum": ["24h", "7d", "30d", "90d"],
                                "default": "24h"
                            }
                        },
                        "required": ["chain_id", "token_address"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_price_estimate",
                    "description": "Get price estimation for ERC-20 tokens from Daily Model and Forecast Model. Use this when users ask for price predictions, estimates, or future price forecasts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chain_id": {
                                "type": "integer",
                                "description": "Chain ID for the blockchain. Use ONLY the numeric chain ID: 1=ethereum, 137=polygon, 43114=avalanche, 56=binance, 101=solana, 59144=linea, 8453=base, 42161=arbitrum, 10=optimism, etc. DO NOT use chain names, only use the numeric ID.",
                                "enum": [1, 137, 43114, 56, 101, 59144, 8453, 42161, 10, 1101, 5000, 534352, 324, 1442, 5001, 534353, 280, 84531, 421613, 11155420, 59140]
                            },
                            "token_address": {
                                "type": "string",
                                "description": "The token contract address"
                            }
                        },
                        "required": ["chain_id", "token_address"]
                    }
                }
            }
        ]

    def get_historical_price(
        self, 
        chain_id: int, 
        token_address: str, 
        currency: str = "usdc",
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Get the day-wise historical price of fungible tokens"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_historical_price")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - chain_id: {chain_id}")
            print(f"   - token_address: {token_address}")
            print(f"   - currency: {currency}")
            print(f"   - time_range: {time_range}")
        
        url = f"{self.base_url}/{chain_id}/{token_address}/price/historical"
        params = {
            "currency": currency,
            "time_range": time_range
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

    def get_price_estimate(
        self, 
        chain_id: int, 
        token_address: str
    ) -> Dict[str, Any]:
        """Get price estimation for ERC-20 tokens"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_price_estimate")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - chain_id: {chain_id}")
            print(f"   - token_address: {token_address}")
        
        url = f"{self.base_url}/{chain_id}/{token_address}/price-estimate"
        headers = {
            "accept": "application/json",
            "x-api-key": self.api_key
        }
        
        if self.verbose:
            print(f"🌐 API REQUEST: GET {url}")
            print(f"📊 QUERY PARAMS: {{}}")
        
        try:
            response = requests.get(url, headers=headers)
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
        
        if function_name == "get_historical_price":
            return self.get_historical_price(**arguments)
        elif function_name == "get_price_estimate":
            return self.get_price_estimate(**arguments)
        else:
            error_result = {"error": f"Unknown function: {function_name}"}
            if self.verbose:
                print(f"❌ FUNCTION ERROR: {error_result}")
            return error_result

    def chat(self, user_message: str) -> str:
        """
        Process a natural language query and return relevant NFT fungible token data
        
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
                    "content": """You are an NFT Fungible Token Metrics Assistant. You help users get information about fungible tokens using two main tools:

1. get_historical_price - Use this when users ask for historical price data, price history, past prices, or historical performance of tokens. This provides day-wise historical price data.

2. get_price_estimate - Use this when users ask for price predictions, estimates, future price forecasts, or price estimates for tokens. This provides price estimation from Daily Model and Forecast Model.

IMPORTANT: 
- If users ask for "historical", "past", "history", "previous", "last week", "last month" → use get_historical_price
- If users ask for "estimate", "prediction", "forecast", "future", "price estimate" → use get_price_estimate

CRITICAL: When users mention chain names, you MUST convert them to chain IDs:
- "ethereum" → use chain_id: 1
- "polygon" → use chain_id: 137  
- "avalanche" → use chain_id: 43114
- "binance" → use chain_id: 56
- "solana" → use chain_id: 101
- "linea" → use chain_id: 59144
- "base" → use chain_id: 8453
- "arbitrum" → use chain_id: 42161
- "optimism" → use chain_id: 10

NEVER use chain names in the API calls, ONLY use the numeric chain_id values.

Supported currencies: usdc (default), eth, dai
Supported time ranges: 24h (default), 7d, 30d, 90d

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
        agent = NFTFungibleAgent(verbose=True)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure you have a .env file with the required API keys.")
        exit(1)
    
    # Example conversations for testing
    example_queries = [
        "Get historical price for token 0x6B175474E89094C44Da98b954EedeAC495271d0F on ethereum",
        "Show me price history for USDC token on polygon for the last 7 days",
        "Get price estimate for token 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984 on ethereum",
        "What's the price prediction for UNI token on ethereum?",
        "Get historical price data for DAI token on avalanche in ETH currency for 30 days",
    ]
    
    print("NFT Fungible Token Agent - Ready to chat!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        response = agent.chat(user_input)
        print(f"Agent: {response}\n") 