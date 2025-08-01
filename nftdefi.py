import requests
import json
import os
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import re

class NFTDeFiAgent:
    def __init__(self, verbose: bool = True):
        """
        Initialize the NFT DeFi Agent with API keys from .env file
        
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
        self.base_url = "https://api.unleashnfts.com/api/v1/defi/pool"
        
        # Supported protocols
        self.supported_protocols = [
            "uniswap", "sushiswap", "pancakeswap", "curve", "balancer", 
            "aave", "compound", "yearn", "makerdao", "dydx", "1inch",
            "paraswap", "0x", "kyber", "bancor", "dodo", "perpetual"
        ]
        
        # Define the tools for OpenAI function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_dex_pool_metadata",
                    "description": "Get details of the DEX pool by passing the pair address. Use this when users provide a specific pair address.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pair_address": {
                                "type": "string",
                                "description": "The pair address of the DEX pool"
                            }
                        },
                        "required": ["pair_address"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_dex_pool_metrics",
                    "description": "Get the metric details of the DEX pool/position. Use this when users ask for metrics or performance data for a specific pair address.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pair_address": {
                                "type": "string",
                                "description": "The pair address of the DEX pool"
                            }
                        },
                        "required": ["pair_address"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_dex_pools_by_protocol",
                    "description": "Get all DEX positions details in the protocol. Use this when users ask about pools in a specific protocol like Uniswap, Sushiswap, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "protocol": {
                                "type": "string",
                                "description": "The DeFi protocol name",
                                "enum": [
                                    "uniswap", "sushiswap", "pancakeswap", "curve", "balancer", 
                                    "aave", "compound", "yearn", "makerdao", "dydx", "1inch",
                                    "paraswap", "0x", "kyber", "bancor", "dodo", "perpetual"
                                ]
                            }
                        },
                        "required": ["protocol"]
                    }
                }
            }
        ]

    def get_dex_pool_metadata(self, pair_address: str) -> Dict[str, Any]:
        """Get details of the DEX pool by passing the pair address"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_dex_pool_metadata")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - pair_address: {pair_address}")
        
        url = f"{self.base_url}/metadata"
        params = {
            "pair_address": pair_address
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

    def get_dex_pool_metrics(self, pair_address: str) -> Dict[str, Any]:
        """Get the metric details of the DEX pool/position"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_dex_pool_metrics")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - pair_address: {pair_address}")
        
        url = f"{self.base_url}/metrics"
        params = {
            "pair_address": pair_address
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

    def get_dex_pools_by_protocol(self, protocol: str) -> Dict[str, Any]:
        """Get all DEX positions details in the protocol"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_dex_pools_by_protocol")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - protocol: {protocol}")
        
        url = f"{self.base_url}"
        params = {
            "protocol": protocol
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
        
        if function_name == "get_dex_pool_metadata":
            return self.get_dex_pool_metadata(**arguments)
        elif function_name == "get_dex_pool_metrics":
            return self.get_dex_pool_metrics(**arguments)
        elif function_name == "get_dex_pools_by_protocol":
            return self.get_dex_pools_by_protocol(**arguments)
        else:
            error_result = {"error": f"Unknown function: {function_name}"}
            if self.verbose:
                print(f"❌ FUNCTION ERROR: {error_result}")
            return error_result

    def chat(self, user_message: str) -> str:
        """
        Process a natural language query and return relevant NFT DeFi data
        
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
                    "content": """You are an NFT DeFi Metrics Assistant. You help users get information about DeFi pools and protocols using three main tools:

1. get_dex_pool_metadata - Use this when users provide a specific pair address and want to know details about that DEX pool.

2. get_dex_pool_metrics - Use this when users provide a pair address and want to know performance metrics, trading volume, liquidity, or other metrics for that pool.

3. get_dex_pools_by_protocol - Use this when users ask about pools in a specific protocol (like Uniswap, Sushiswap, PancakeSwap, etc.) or want to see all pools in a protocol.

IMPORTANT: 
- If users provide a pair address and ask for "details" or "metadata" → use get_dex_pool_metadata
- If users provide a pair address and ask for "metrics", "performance", "volume", "liquidity" → use get_dex_pool_metrics
- If users mention a protocol name (Uniswap, Sushiswap, etc.) → use get_dex_pools_by_protocol

Supported protocols include: uniswap, sushiswap, pancakeswap, curve, balancer, aave, compound, yearn, makerdao, dydx, 1inch, paraswap, 0x, kyber, bancor, dodo, perpetual

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
        agent = NFTDeFiAgent(verbose=True)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure you have a .env file with the required API keys.")
        exit(1)
    
    # Example conversations for testing
    example_queries = [
        "Get metadata for pair address 0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc",
        "Show me metrics for pair 0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc",
        "What pools are available in Uniswap?",
        "Show me all Sushiswap pools",
        "Get performance metrics for pair 0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc",
    ]
    
    print("NFT DeFi Agent - Ready to chat!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        response = agent.chat(user_input)
        print(f"Agent: {response}\n") 