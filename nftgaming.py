import requests
import json
import os
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import re

class NFTGamingAgent:
    def __init__(self, verbose: bool = True):
        """
        Initialize the NFT Gaming Agent with API keys from .env file
        
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
        self.base_url = "https://api.unleashnfts.com/api/v2/nft/gaming"
        
        # Supported blockchains
        self.supported_blockchains = [
            "avalanche", "base", "binance", "bitcoin", "berachain", 
            "ethereum", "linea", "polygon", "solana", "unichain"
        ]
        
        # Define the tools for OpenAI function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_game_contracts_info",
                    "description": "Get general overview of game contracts and metrics. Use ONLY for broad queries like 'show me game contracts' or 'list games'. DO NOT use for specific game names.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "blockchain": {
                                "type": "string",
                                "description": "Blockchain network to filter results (optional)",
                                "enum": ["avalanche", "base", "binance", "bitcoin", "berachain", "ethereum", "linea", "polygon", "solana", "unichain"]
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of results to return (default: 30)",
                                "default": 30
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Offset for pagination (default: 0)",
                                "default": 0
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_nft_gaming_metrics_by_contract",
                    "description": "Fetch NFT gaming metrics for a specific contract address",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "blockchain": {
                                "type": "string",
                                "description": "Blockchain network",
                                "enum": ["avalanche", "base", "binance", "bitcoin", "berachain", "ethereum", "linea", "polygon", "solana", "unichain"],
                                "default": "ethereum"
                            },
                            "contract_address": {
                                "type": "string",
                                "description": "Contract address to fetch metrics for"
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for metrics (e.g., 24h, 7d, 30d)",
                                "default": "24h"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 30
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Offset for pagination",
                                "default": 0
                            }
                        },
                        "required": ["contract_address"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_nft_gaming_metrics_by_game",
                    "description": "Fetch NFT gaming metrics for a specific game by name. Use this when users mention specific game names like 'yeti frens', 'axie infinity', etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "blockchain": {
                                "type": "string",
                                "description": "Blockchain network",
                                "enum": ["avalanche", "base", "binance", "bitcoin", "berachain", "ethereum", "linea", "polygon", "solana", "unichain"],
                                "default": "ethereum"
                            },
                            "game": {
                                "type": "string",
                                "description": "Name of the game to fetch metrics for"
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for metrics (e.g., 24h, 7d, 30d)",
                                "default": "24h"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 30
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Offset for pagination",
                                "default": 0
                            }
                        },
                        "required": ["game"]
                    }
                }
            }
        ]

    def get_game_contracts_info(self, limit: int = 30, offset: int = 0, blockchain: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information on game contracts"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_game_contracts_info")
            print(f"📥 INPUT PARAMETERS: limit={limit}, offset={offset}, blockchain={blockchain}")
        
        url = f"{self.base_url}/metrics"
        params = {
            "offset": offset,
            "limit": limit,
            "sort_by": "game",
            "sort_order": "desc"
        }
        
        # Add blockchain parameter if specified
        if blockchain:
            params["blockchain"] = blockchain
        
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

    def get_nft_gaming_metrics_by_contract(
        self, 
        contract_address: str, 
        blockchain: str = "ethereum", 
        time_range: str = "24h",
        limit: int = 30,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Fetch NFT gaming metrics by contract address"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_nft_gaming_metrics_by_contract")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - contract_address: {contract_address}")
            print(f"   - blockchain: {blockchain}")
            print(f"   - time_range: {time_range}")
            print(f"   - limit: {limit}")
            print(f"   - offset: {offset}")
        
        url = f"{self.base_url}/contract/metrics"
        params = {
            "blockchain": blockchain,
            "contract_address": contract_address,
            "time_range": time_range,
            "offset": offset,
            "limit": limit,
            "sort_by": "total_users",
            "sort_order": "desc"
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

    def get_nft_gaming_metrics_by_game(
        self, 
        game: str, 
        blockchain: str = "ethereum", 
        time_range: str = "24h",
        limit: int = 30,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Fetch NFT gaming metrics by game name"""
        # Convert game name to lowercase for API compatibility
        game_formatted = game.lower()
        
        if self.verbose:
            print(f"🔧 TOOL CALL: get_nft_gaming_metrics_by_game")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - game: {game} → formatted: {game_formatted}")
            print(f"   - blockchain: {blockchain}")
            print(f"   - time_range: {time_range}")
            print(f"   - limit: {limit}")
            print(f"   - offset: {offset}")
        
        url = f"{self.base_url}/collection/metrics"
        params = {
            "blockchain": blockchain,
            "game": game_formatted,  # Use the lowercase formatted name
            "time_range": time_range,
            "offset": offset,
            "limit": limit,
            "sort_by": "total_users",
            "sort_order": "desc"
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
        
        if function_name == "get_game_contracts_info":
            return self.get_game_contracts_info(**arguments)
        elif function_name == "get_nft_gaming_metrics_by_contract":
            return self.get_nft_gaming_metrics_by_contract(**arguments)
        elif function_name == "get_nft_gaming_metrics_by_game":
            return self.get_nft_gaming_metrics_by_game(**arguments)
        else:
            error_result = {"error": f"Unknown function: {function_name}"}
            if self.verbose:
                print(f"❌ FUNCTION ERROR: {error_result}")
            return error_result

    def chat(self, user_message: str) -> str:
        """
        Process a natural language query and return relevant NFT gaming data
        
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
                    "content": """You are an NFT Gaming Metrics Assistant. You help users get information about NFT gaming metrics using three main tools:

1. get_game_contracts_info - Use this ONLY when users ask for general information about game contracts, lists of games, or overview data. This is for broad queries like "show me game contracts" or "list games".

2. get_nft_gaming_metrics_by_contract - Use this when users provide a specific contract address (starts with 0x). This gets detailed metrics for a specific contract.

3. get_nft_gaming_metrics_by_game - Use this when users mention a specific game name (like "yeti frens", "axie infinity", "cryptokitties", etc.). This gets metrics for a specific game.

IMPORTANT: When users ask about a specific game by name, ALWAYS use get_nft_gaming_metrics_by_game, NOT get_game_contracts_info.

You can work with these blockchains: avalanche, base, binance, bitcoin, berachain, ethereum, linea, polygon, solana, unichain.

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
        agent = NFTGamingAgent(verbose=True)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure you have a .env file with the required API keys.")
        exit(1)
    
    # Example conversations
    example_queries = [
        "Show me information about game contracts",
        "Get metrics for contract address 0xf902312a2c54b6c3d3a5e13f31404b6026343d8b on ethereum",
        "What are the metrics for the game 'yeti frens' on ethereum?",
        "Show me metrics for Axie Infinity on polygon blockchain",
        "Get contract info but limit to 10 results",
    ]
    
    print("NFT Gaming Metrics Agent - Ready to chat!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        response = agent.chat(user_input)
        print(f"Agent: {response}\n")