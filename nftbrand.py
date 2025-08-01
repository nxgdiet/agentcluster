import requests
import json
import os
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import re

class NFTBrandAgent:
    def __init__(self, verbose: bool = True):
        """
        Initialize the NFT Brand Agent with API keys from .env file
        
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
        self.base_url = "https://api.unleashnfts.com/api/v1/brand"
        
        # Supported brands
        self.supported_brands = [
            "Coachella", "Star Trek", "Kith Friends", "Grimace Digital", "Collectible", "Karafuru", 
            "Macys", "Ping Fong", "Puma", "Lamborghini", "Coca-Cola", 
            "TIMEPieces x Timbaland: The Beatclub Collection", "Hugo", "McDonalds", "StarBucks", 
            "Asics", "Louis vuitton", "Moncler", "Gucci", "Givenchy", "Reddit", "Clinique", 
            "Coach", "AP Photojournalism NFTs", "Hello kitty", "TommyHilfiger", "Budweiser", 
            "YSL Beauty Pride Block", "Liverpool Football club", "MG Motors", "Adidas", 
            "Burger King", "Nivea", "Nike", "Times Magazine", "AO ArtBall", "LimeWire", 
            "Chicago Bulls", "Prada", "Nickelodeon", "Rimova", "Reebok", "Burberry", 
            "Dolce and Gabbana", "Rolling Stone Magazine", "Adam Bomb Squad", "The Walking Dead", 
            "Hyundai", "Mercedes Benz", "BlockBar", "Pepsi", "Hublot", "Bugatti", "McLaren", 
            "Bud light", "Flyfish Club", "Porsche", "Lacoste", "Tiffany and co", "Tiger Beer", 
            "9dcc", "Netflix"
        ]
        
        # Supported categories
        self.supported_categories = [
            "Fashion", "Metaverse", "Social Media", "Sports", "Food & Beverage", "Cars", 
            "Sports Club", "Skincare & Cosmetics", "Restaurant & Hotel membership", "Books", 
            "Media & Entertainment", "Collectibles"
        ]
        
        # Chain IDs mapping
        self.chain_ids = {
            "ethereum": 1,
            "polygon": 137,
            "avalanche": 43114,
            "binance": 56,
            "solana": 101,
            "linea": 59144
        }
        
        # Define the tools for OpenAI function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_brand_details",
                    "description": "Get combined metrics for a specific brand NFT. Use this when users mention a specific brand name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "brand": {
                                "type": "string",
                                "description": "Name of the brand to fetch details for",
                                "enum": [
                                    "Coachella", "Star Trek", "Kith Friends", "Grimace Digital", "Collectible", "Karafuru", 
                                    "Macys", "Ping Fong", "Puma", "Lamborghini", "Coca-Cola", 
                                    "TIMEPieces x Timbaland: The Beatclub Collection", "Hugo", "McDonalds", "StarBucks", 
                                    "Asics", "Louis vuitton", "Moncler", "Gucci", "Givenchy", "Reddit", "Clinique", 
                                    "Coach", "AP Photojournalism NFTs", "Hello kitty", "TommyHilfiger", "Budweiser", 
                                    "YSL Beauty Pride Block", "Liverpool Football club", "MG Motors", "Adidas", 
                                    "Burger King", "Nivea", "Nike", "Times Magazine", "AO ArtBall", "LimeWire", 
                                    "Chicago Bulls", "Prada", "Nickelodeon", "Rimova", "Reebok", "Burberry", 
                                    "Dolce and Gabbana", "Rolling Stone Magazine", "Adam Bomb Squad", "The Walking Dead", 
                                    "Hyundai", "Mercedes Benz", "BlockBar", "Pepsi", "Hublot", "Bugatti", "McLaren", 
                                    "Bud light", "Flyfish Club", "Porsche", "Lacoste", "Tiffany and co", "Tiger Beer", 
                                    "9dcc", "Netflix"
                                ]
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for metrics (e.g., 24h, 7d, 30d)",
                                "default": "24h"
                            }
                        },
                        "required": ["brand"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_brand_metrics_by_contract",
                    "description": "Get combined metrics for brand collections by contract address. Use this when users provide a specific contract address for a brand.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chain_id": {
                                "type": "integer",
                                "description": "Chain ID for the blockchain",
                                "enum": [1, 137, 43114, 56, 101, 59144]
                            },
                            "contract_address": {
                                "type": "string",
                                "description": "Contract address of the brand collection"
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for metrics (e.g., 24h, 7d, 30d)",
                                "default": "24h"
                            }
                        },
                        "required": ["contract_address"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_brand_category_details",
                    "description": "Get brand category details for NFTs. Use this when users ask about brands in a specific category like 'Sports', 'Fashion', etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Category of brands to fetch",
                                "enum": [
                                    "Fashion", "Metaverse", "Social Media", "Sports", "Food & Beverage", "Cars", 
                                    "Sports Club", "Skincare & Cosmetics", "Restaurant & Hotel membership", "Books", 
                                    "Media & Entertainment", "Collectibles"
                                ]
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
                        "required": ["category"]
                    }
                }
            }
        ]

    def get_brand_details(self, brand: str, time_range: str = "24h") -> Dict[str, Any]:
        """Get combined metrics for a specific brand NFT"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_brand_details")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - brand: {brand}")
            print(f"   - time_range: {time_range}")
        
        url = f"{self.base_url}"
        params = {
            "brand": brand,
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

    def get_brand_metrics_by_contract(
        self, 
        contract_address: str, 
        chain_id: int = 1,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Get combined metrics for brand collections by contract address"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_brand_metrics_by_contract")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - contract_address: {contract_address}")
            print(f"   - chain_id: {chain_id}")
            print(f"   - time_range: {time_range}")
        
        url = f"{self.base_url}/{chain_id}/{contract_address}"
        params = {
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

    def get_brand_category_details(
        self, 
        category: str, 
        limit: int = 30,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get brand category details for NFTs"""
        if self.verbose:
            print(f"🔧 TOOL CALL: get_brand_category_details")
            print(f"📥 INPUT PARAMETERS:")
            print(f"   - category: {category}")
            print(f"   - limit: {limit}")
            print(f"   - offset: {offset}")
        
        url = f"{self.base_url}/category"
        params = {
            "category": category,
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
        
        if function_name == "get_brand_details":
            return self.get_brand_details(**arguments)
        elif function_name == "get_brand_metrics_by_contract":
            return self.get_brand_metrics_by_contract(**arguments)
        elif function_name == "get_brand_category_details":
            return self.get_brand_category_details(**arguments)
        else:
            error_result = {"error": f"Unknown function: {function_name}"}
            if self.verbose:
                print(f"❌ FUNCTION ERROR: {error_result}")
            return error_result

    def chat(self, user_message: str) -> str:
        """
        Process a natural language query and return relevant NFT brand data
        
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
                    "content": """You are an NFT Brand Metrics Assistant. You help users get information about NFT brand metrics using three main tools:

1. get_brand_details - Use this when users mention a specific brand name (like "Starbucks", "Nike", "Adidas", etc.). This gets metrics for a specific brand.

2. get_brand_metrics_by_contract - Use this when users provide a specific contract address for a brand collection. This gets metrics for a brand by contract address.

3. get_brand_category_details - Use this when users ask about brands in a specific category (like "Sports", "Fashion", "Food & Beverage", etc.). This gets all brands in a category.

IMPORTANT: 
- If users mention a specific brand name, use get_brand_details
- If users provide a contract address, use get_brand_metrics_by_contract
- If users ask about brands in a category, use get_brand_category_details

Supported brands include: Coachella, Star Trek, Kith Friends, Grimace Digital, Collectible, Karafuru, Macys, Ping Fong, Puma, Lamborghini, Coca-Cola, TIMEPieces x Timbaland: The Beatclub Collection, Hugo, McDonalds, StarBucks, Asics, Louis vuitton, Moncler, Gucci, Givenchy, Reddit, Clinique, Coach, AP Photojournalism NFTs, Hello kitty, TommyHilfiger, Budweiser, YSL Beauty Pride Block, Liverpool Football club, MG Motors, Adidas, Burger King, Nivea, Nike, Times Magazine, AO ArtBall, LimeWire, Chicago Bulls, Prada, Nickelodeon, Rimova, Reebok, Burberry, Dolce and Gabbana, Rolling Stone Magazine, Adam Bomb Squad, The Walking Dead, Hyundai, Mercedes Benz, BlockBar, Pepsi, Hublot, Bugatti, McLaren, Bud light, Flyfish Club, Porsche, Lacoste, Tiffany and co, Tiger Beer, 9dcc, Netflix

Supported categories include: Fashion, Metaverse, Social Media, Sports, Food & Beverage, Cars, Sports Club, Skincare & Cosmetics, Restaurant & Hotel membership, Books, Media & Entertainment, Collectibles

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
        agent = NFTBrandAgent(verbose=True)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure you have a .env file with the required API keys.")
        exit(1)
    
    # Example conversations for testing
    example_queries = [
        "Get brand details for Starbucks",
        "Show me Nike brand metrics",
        "What brands are in the Sports category?",
        "Get metrics for contract 0x73cceed2264de2b72931963dcda56ac5b1249735 on ethereum",
        "Show me Fashion brands",
    ]
    
    print("NFT Brand Agent - Ready to chat!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        response = agent.chat(user_input)
        print(f"Agent: {response}\n") 