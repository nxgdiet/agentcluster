import requests
import json
import os
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import re

class NFTWalletAgent:
    def __init__(self, verbose: bool = True):
        """
        Initialize the NFT Wallet Analytics Agent with API keys from .env file
        
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
        self.base_url = "https://api.unleashnfts.com/api/v2/nft/wallet"
        
        # Supported blockchains for wallet analytics
        self.supported_blockchains = [
            "avalanche", "base", "binance", "bitcoin", "ethereum", 
            "linea", "polygon", "root", "solana", "soneium", 
            "unichain", "unichain_sepolia"
        ]
        
        # Supported sort options
        self.supported_sort_by = ["volume", "portfolio_value", "transaction_count", "unique_collections"]
        self.supported_sort_order = ["asc", "desc"]
        self.supported_time_ranges = ["24h", "7d", "30d", "90d", "all"]
        
        # Define the tools for OpenAI function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_wallet_analytics",
                    "description": "Get detailed analytics on value and trends for a specific wallet. Use this when users ask for wallet analytics, wallet performance, trading activity, or wallet metrics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "wallet": {
                                "type": "string",
                                "description": "The wallet address to analyze"
                            },
                            "blockchain": {
                                "type": "string",
                                "description": "Blockchain for the wallet",
                                "enum": ["avalanche", "base", "binance", "bitcoin", "ethereum", "linea", "polygon", "root", "solana", "soneium", "unichain", "unichain_sepolia"]
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for analytics",
                                "enum": ["24h", "7d", "30d", "90d", "all"],
                                "default": "7d"
                            },
                            "sort_by": {
                                "type": "string",
                                "description": "Sort criteria for results",
                                "enum": ["volume", "portfolio_value", "transaction_count", "unique_collections"],
                                "default": "volume"
                            },
                            "sort_order": {
                                "type": "string",
                                "description": "Sort order (ascending or descending)",
                                "enum": ["asc", "desc"],
                                "default": "desc"
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
                        "required": ["wallet", "blockchain"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_wallet_scores",
                    "description": "Get detailed analytics on score values and trends for a specific wallet. Use this when users ask for wallet scores, wallet ratings, portfolio scores, or wallet performance scores.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "wallet": {
                                "type": "string",
                                "description": "The wallet address to analyze"
                            },
                            "blockchain": {
                                "type": "string",
                                "description": "Blockchain for the wallet",
                                "enum": ["avalanche", "base", "binance", "bitcoin", "ethereum", "linea", "polygon", "root", "solana", "soneium", "unichain", "unichain_sepolia"]
                            },
                            "sort_by": {
                                "type": "string",
                                "description": "Sort criteria for results",
                                "enum": ["portfolio_value", "volume", "transaction_count", "unique_collections"],
                                "default": "portfolio_value"
                            },
                            "sort_order": {
                                "type": "string",
                                "description": "Sort order (ascending or descending)",
                                "enum": ["asc", "desc"],
                                "default": "desc"
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for scores",
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
                        "required": ["wallet", "blockchain"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_wallet_profile",
                    "description": "Get comprehensive profiling information for a specific wallet. Use this when users ask for wallet profile, wallet details, NFT holdings, wallet insights, or wallet information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "wallet": {
                                "type": "string",
                                "description": "The wallet address to profile"
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
                        "required": ["wallet"]
                    }
                }
            }
        ]

    def get_wallet_analytics(
        self, 
        wallet: str, 
        blockchain: str,
        time_range: str = "7d",
        sort_by: str = "volume",
        sort_order: str = "desc",
        offset: int = 0,
        limit: int = 30
    ) -> Dict[str, Any]:
        """Get detailed analytics on value and trends for a specific wallet"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_wallet_analytics")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - wallet: {wallet}")
            print(f"   - blockchain: {blockchain}")
            print(f"   - time_range: {time_range}")
            print(f"   - sort_by: {sort_by}")
            print(f"   - sort_order: {sort_order}")
            print(f"   - offset: {offset}")
            print(f"   - limit: {limit}")
        
        url = f"{self.base_url}/analytics"
        params = {
            "wallet": wallet,
            "blockchain": blockchain,
            "time_range": time_range,
            "sort_by": sort_by,
            "sort_order": sort_order,
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

    def get_wallet_scores(
        self, 
        wallet: str, 
        blockchain: str,
        sort_by: str = "portfolio_value",
        sort_order: str = "desc",
        time_range: str = "all",
        offset: int = 0,
        limit: int = 30
    ) -> Dict[str, Any]:
        """Get detailed analytics on score values and trends for a specific wallet"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_wallet_scores")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - wallet: {wallet}")
            print(f"   - blockchain: {blockchain}")
            print(f"   - sort_by: {sort_by}")
            print(f"   - sort_order: {sort_order}")
            print(f"   - time_range: {time_range}")
            print(f"   - offset: {offset}")
            print(f"   - limit: {limit}")
        
        url = f"{self.base_url}/scores"
        params = {
            "wallet": wallet,
            "blockchain": blockchain,
            "sort_by": sort_by,
            "sort_order": sort_order,
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

    def get_wallet_profile(
        self, 
        wallet: str,
        offset: int = 0,
        limit: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive profiling information for a specific wallet"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_wallet_profile")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - wallet: {wallet}")
            print(f"   - offset: {offset}")
            print(f"   - limit: {limit}")
        
        url = f"{self.base_url}/profile"
        params = {
            "wallet": wallet,
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
        
        if function_name == "get_wallet_analytics":
            return self.get_wallet_analytics(**arguments)
        elif function_name == "get_wallet_scores":
            return self.get_wallet_scores(**arguments)
        elif function_name == "get_wallet_profile":
            return self.get_wallet_profile(**arguments)
        else:
            error_result = {"error": f"Unknown function: {function_name}"}
            if self.verbose:
                print(f"❌ FUNCTION ERROR: {error_result}")
            return error_result

    def chat(self, user_message: str) -> str:
        """
        Process a natural language query and return relevant NFT wallet data
        
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
                    "content": """You are an NFT Wallet Analytics Assistant. You help users get information about wallet analytics, scores, and profiles using three main tools:

1. get_wallet_analytics - Use this when users ask for wallet analytics, wallet performance, trading activity, wallet metrics, or wallet trends. This provides detailed analytics on value and trends.

2. get_wallet_scores - Use this when users ask for wallet scores, wallet ratings, portfolio scores, wallet performance scores, or wallet rankings. This provides detailed analytics on score values and trends.

3. get_wallet_profile - Use this when users ask for wallet profile, wallet details, NFT holdings, wallet insights, wallet information, or comprehensive wallet data. This provides comprehensive profiling information.

IMPORTANT: 
- If users ask for "analytics", "performance", "trading", "metrics", "trends" → use get_wallet_analytics
- If users ask for "scores", "ratings", "rankings", "portfolio scores" → use get_wallet_scores
- If users ask for "profile", "details", "holdings", "insights", "information" → use get_wallet_profile

Supported blockchains: avalanche, base, binance, bitcoin, ethereum, linea, polygon, root, solana, soneium, unichain, unichain_sepolia

Supported time ranges: 24h, 7d, 30d, 90d, all
Supported sort options: volume, portfolio_value, transaction_count, unique_collections
Supported sort orders: asc, desc

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
        agent = NFTWalletAgent(verbose=True)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure you have a .env file with the required API keys.")
        exit(1)
    
    # Example conversations for testing
    example_queries = [
        "Get wallet analytics for 0x2514844f312c02ae3c9d4feb40db4ec8830b6844 on ethereum",
        "Show me wallet scores for 0xfa99e2bE141dd93Ad016466f60EFAE04BC34D81F on polygon",
        "Get wallet profile for 0xfa99e2bE141dd93Ad016466f60EFAE04BC34D81F",
        "What are the trading analytics for wallet 0x1234567890abcdef on avalanche?",
        "Get portfolio scores for wallet 0xabcdef1234567890 on binance",
    ]
    
    print("NFT Wallet Analytics Agent - Ready to chat!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        response = agent.chat(user_input)
        print(f"Agent: {response}\n") 