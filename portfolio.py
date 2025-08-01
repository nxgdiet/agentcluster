import requests
import json
import os
from typing import Dict, Any, Optional, List
from openai import OpenAI
from dotenv import load_dotenv

class PortfolioAgent:
    def __init__(self, verbose: bool = True):
        """
        Initialize the Portfolio Analysis Agent
        
        Args:
            verbose (bool): Enable verbose logging to see agent's thinking process
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API keys from environment variables
        openai_api_key = os.getenv('OPENAI_API_KEY')
        unleash_api_key = os.getenv('UNLEASH_API_KEY', 'e3887325f0bc477f99f3c92383be5e74')
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.client = OpenAI(api_key=openai_api_key)
        self.unleash_api_key = unleash_api_key
        self.verbose = verbose
        
        # Define the tools for OpenAI function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_defi_portfolio",
                    "description": "Retrieve a comprehensive overview of a wallet's DeFi portfolio, including key metrics such as token holdings, blockchain details, and asset quantities.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "address": {
                                "type": "string",
                                "description": "The wallet address to analyze"
                            },
                            "blockchain": {
                                "type": "string",
                                "description": "The blockchain to analyze. Options: atleta_olympia, avalanche, base, berachain, binance, bitcoin, ethereum, linea, monad_testnet, polygon, polyhedra_testnet, root, solana, somnia_testnet, soneium, unichain, unichain_sepolia",
                                "enum": ["atleta_olympia", "avalanche", "base", "berachain", "binance", "bitcoin", "ethereum", "linea", "monad_testnet", "polygon", "polyhedra_testnet", "root", "solana", "somnia_testnet", "soneium", "unichain", "unichain_sepolia"]
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for the analysis. Options: 1d, 7d, 30d, 90d, 1y, all",
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
                        "required": ["address", "blockchain"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_nft_portfolio",
                    "description": "Retrieve a comprehensive overview of a wallet's NFT holdings, including key metrics such as collection details, contract type, token ID, and quantity owned.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "wallet": {
                                "type": "string",
                                "description": "The wallet address to analyze"
                            },
                            "blockchain": {
                                "type": "string",
                                "description": "The blockchain to analyze. Options: atleta_olympia, avalanche, base, berachain, binance, bitcoin, ethereum, linea, monad_testnet, polygon, polyhedra_testnet, root, solana, somnia_testnet, soneium, unichain, unichain_sepolia",
                                "enum": ["atleta_olympia", "avalanche", "base", "berachain", "binance", "bitcoin", "ethereum", "linea", "monad_testnet", "polygon", "polyhedra_testnet", "root", "solana", "somnia_testnet", "soneium", "unichain", "unichain_sepolia"]
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for the analysis. Options: 1d, 7d, 30d, 90d, 1y, all",
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
                    "name": "get_erc20_portfolio",
                    "description": "Retrieve a comprehensive overview of a wallet's ERC-20 token holdings, including key metrics such as token details, quantity owned, and blockchain information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "address": {
                                "type": "string",
                                "description": "The wallet address to analyze"
                            },
                            "blockchain": {
                                "type": "string",
                                "description": "The blockchain to analyze. Options: atleta_olympia, avalanche, base, berachain, binance, bitcoin, ethereum, linea, monad_testnet, polygon, polyhedra_testnet, root, solana, somnia_testnet, soneium, unichain, unichain_sepolia",
                                "enum": ["atleta_olympia", "avalanche", "base", "berachain", "binance", "bitcoin", "ethereum", "linea", "monad_testnet", "polygon", "polyhedra_testnet", "root", "solana", "somnia_testnet", "soneium", "unichain", "unichain_sepolia"]
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for the analysis. Options: 1d, 7d, 30d, 90d, 1y, all",
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
                        "required": ["address", "blockchain"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_wallet_label",
                    "description": "Retrieve a comprehensive overview of a wallet's classification and associated risk factors, including key metrics such as activity type, security status, and protocol involvement.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "address": {
                                "type": "string",
                                "description": "The wallet address to analyze"
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
                        "required": ["address"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_wallet_score",
                    "description": "Retrieve a comprehensive overview of a wallet's activity and risk assessment, including key metrics such as interaction patterns, classification, and risk scores.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "time_range": {
                                "type": "string",
                                "description": "Time range for the analysis. Options: 1d, 7d, 30d, 90d, 1y, all",
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
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_wallet_metrics",
                    "description": "Retrieve a comprehensive overview of a wallet's transactional activity, including key metrics such as transaction volume, inflow/outflow data, and wallet age.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "blockchain": {
                                "type": "string",
                                "description": "The blockchain to analyze. Options: ethereum, avalanche, linea, polygon",
                                "enum": ["ethereum", "avalanche", "linea", "polygon"]
                            },
                            "wallet": {
                                "type": "string",
                                "description": "The wallet address to analyze"
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for the analysis. Options: 1d, 7d, 30d, 90d, 1y, all",
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
                        "required": ["blockchain", "wallet"]
                    }
                }
            }
        ]

    def get_defi_portfolio(self, address: str, blockchain: str, time_range: str = "all", offset: int = 0, limit: int = 30) -> Dict[str, Any]:
        """Get DeFi portfolio for a wallet address"""
        if self.verbose:
            print(f"🔄 Getting DeFi portfolio for {address} on {blockchain}")
        
        url = f"https://api.unleashnfts.com/api/v2/wallet/balance/defi"
        
        headers = {
            "accept": "application/json",
            "x-api-key": self.unleash_api_key
        }
        
        params = {
            "address": address,
            "blockchain": blockchain,
            "time_range": time_range,
            "offset": offset,
            "limit": limit
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if self.verbose:
                print(f"✅ DeFi portfolio retrieved successfully")
                print(f"📊 Data points: {len(data.get('data', []))}")
                print(f"📄 API Response: {json.dumps(data, indent=2)}")
            
            return {
                "success": True,
                "data": data,
                "summary": f"Retrieved DeFi portfolio for {address} on {blockchain} with {len(data.get('data', []))} data points"
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to get DeFi portfolio: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

    def get_nft_portfolio(self, wallet: str, blockchain: str, time_range: str = "all", offset: int = 0, limit: int = 30) -> Dict[str, Any]:
        """Get NFT portfolio for a wallet address"""
        if self.verbose:
            print(f"🖼️ Getting NFT portfolio for {wallet} on {blockchain}")
        
        url = f"https://api.unleashnfts.com/api/v2/wallet/balance/nft"
        
        headers = {
            "accept": "application/json",
            "x-api-key": self.unleash_api_key
        }
        
        params = {
            "wallet": wallet,
            "blockchain": blockchain,
            "time_range": time_range,
            "offset": offset,
            "limit": limit
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if self.verbose:
                print(f"✅ NFT portfolio retrieved successfully")
                print(f"📊 Data points: {len(data.get('data', []))}")
                print(f"📄 API Response: {json.dumps(data, indent=2)}")
            
            return {
                "success": True,
                "data": data,
                "summary": f"Retrieved NFT portfolio for {wallet} on {blockchain} with {len(data.get('data', []))} data points"
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to get NFT portfolio: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

    def get_erc20_portfolio(self, address: str, blockchain: str, time_range: str = "all", offset: int = 0, limit: int = 30) -> Dict[str, Any]:
        """Get ERC20 portfolio for a wallet address"""
        if self.verbose:
            print(f"🪙 Getting ERC20 portfolio for {address} on {blockchain}")
        
        url = f"https://api.unleashnfts.com/api/v2/wallet/balance/token"
        
        headers = {
            "accept": "application/json",
            "x-api-key": self.unleash_api_key
        }
        
        params = {
            "address": address,
            "blockchain": blockchain,
            "time_range": time_range,
            "offset": offset,
            "limit": limit
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if self.verbose:
                print(f"✅ ERC20 portfolio retrieved successfully")
                print(f"📊 Data points: {len(data.get('data', []))}")
                print(f"📄 API Response: {json.dumps(data, indent=2)}")
            
            return {
                "success": True,
                "data": data,
                "summary": f"Retrieved ERC20 portfolio for {address} on {blockchain} with {len(data.get('data', []))} data points"
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to get ERC20 portfolio: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

    def get_wallet_label(self, address: str, offset: int = 0, limit: int = 30) -> Dict[str, Any]:
        """Get wallet label and classification"""
        if self.verbose:
            print(f"🏷️ Getting wallet label for {address}")
        
        url = f"https://api.unleashnfts.com/api/v2/wallet/label"
        
        headers = {
            "accept": "application/json",
            "x-api-key": self.unleash_api_key
        }
        
        params = {
            "address": address,
            "offset": offset,
            "limit": limit
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if self.verbose:
                print(f"✅ Wallet label retrieved successfully")
                print(f"📊 Data points: {len(data.get('data', []))}")
                print(f"📄 API Response: {json.dumps(data, indent=2)}")
            
            return {
                "success": True,
                "data": data,
                "summary": f"Retrieved wallet label for {address} with {len(data.get('data', []))} data points"
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to get wallet label: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

    def get_wallet_score(self, time_range: str = "all", offset: int = 0, limit: int = 30) -> Dict[str, Any]:
        """Get wallet score and risk assessment"""
        if self.verbose:
            print(f"📊 Getting wallet scores")
        
        url = f"https://api.unleashnfts.com/api/v2/wallet/score"
        
        headers = {
            "accept": "application/json",
            "x-api-key": self.unleash_api_key
        }
        
        params = {
            "time_range": time_range,
            "offset": offset,
            "limit": limit
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if self.verbose:
                print(f"✅ Wallet scores retrieved successfully")
                print(f"📊 Data points: {len(data.get('data', []))}")
                print(f"📄 API Response: {json.dumps(data, indent=2)}")
            
            return {
                "success": True,
                "data": data,
                "summary": f"Retrieved wallet scores with {len(data.get('data', []))} data points"
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to get wallet scores: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

    def get_wallet_metrics(self, blockchain: str, wallet: str, time_range: str = "all", offset: int = 0, limit: int = 30) -> Dict[str, Any]:
        """Get wallet metrics and transactional activity"""
        if self.verbose:
            print(f"📈 Getting wallet metrics for {wallet} on {blockchain}")
        
        url = f"https://api.unleashnfts.com/api/v2/wallet/metrics"
        
        headers = {
            "accept": "application/json",
            "x-api-key": self.unleash_api_key
        }
        
        params = {
            "blockchain": blockchain,
            "wallet": wallet,
            "time_range": time_range,
            "offset": offset,
            "limit": limit
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if self.verbose:
                print(f"✅ Wallet metrics retrieved successfully")
                print(f"📊 Data points: {len(data.get('data', []))}")
                print(f"📄 API Response: {json.dumps(data, indent=2)}")
            
            return {
                "success": True,
                "data": data,
                "summary": f"Retrieved wallet metrics for {wallet} on {blockchain} with {len(data.get('data', []))} data points"
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to get wallet metrics: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

    def execute_tool_call(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the appropriate tool function"""
        if self.verbose:
            print(f"\n🔧 EXECUTING TOOL: {function_name}")
            print(f"📋 ARGUMENTS: {json.dumps(arguments, indent=2)}")
        
        if function_name == "get_defi_portfolio":
            return self.get_defi_portfolio(**arguments)
        elif function_name == "get_nft_portfolio":
            return self.get_nft_portfolio(**arguments)
        elif function_name == "get_erc20_portfolio":
            return self.get_erc20_portfolio(**arguments)
        elif function_name == "get_wallet_label":
            return self.get_wallet_label(**arguments)
        elif function_name == "get_wallet_score":
            return self.get_wallet_score(**arguments)
        elif function_name == "get_wallet_metrics":
            return self.get_wallet_metrics(**arguments)
        else:
            error_result = {"error": f"Unknown function: {function_name}"}
            if self.verbose:
                print(f"❌ UNKNOWN FUNCTION: {function_name}")
            return error_result

    def chat(self, user_message: str) -> str:
        """
        Process a natural language query and execute appropriate portfolio analysis tools
        
        Args:
            user_message (str): Natural language query from the user
            
        Returns:
            str: Formatted response with the requested portfolio data
        """
        if self.verbose:
            print(f"\n" + "="*60)
            print(f"💼 PORTFOLIO AGENT THINKING PROCESS")
            print(f"="*60)
            print(f"💬 USER QUERY: {user_message}")
            print(f"🤖 Analyzing query and determining which portfolio tools to use...")
        
        try:
            # Create the initial conversation with system prompt
            messages = [
                {
                    "role": "system",
                    "content": """You are a Portfolio Analysis Agent that specializes in comprehensive wallet analysis. You have access to multiple tools to analyze different aspects of wallets and portfolios:

**Available Tools:**
1. **DeFi Portfolio** - Analyze a wallet's DeFi holdings, token balances, and blockchain details
2. **NFT Portfolio** - Analyze a wallet's NFT holdings, collection details, and token IDs
3. **ERC20 Portfolio** - Analyze ERC-20 token holdings and blockchain information
4. **Wallet Label** - Get wallet classification, risk factors, and activity type
5. **Wallet Score** - Get wallet activity assessment, interaction patterns, and risk scores
6. **Wallet Metrics** - Get transactional activity, volume, inflow/outflow data, and wallet age

**Tool Selection Guidelines:**
- If the query mentions "DeFi", "defi portfolio", "DeFi holdings" → use get_defi_portfolio
- If the query mentions "NFT", "NFT holdings", "NFT portfolio", "collections" → use get_nft_portfolio
- If the query mentions "ERC20", "tokens", "token holdings", "fungible tokens" → use get_erc20_portfolio
- If the query mentions "wallet label", "classification", "risk factors", "activity type" → use get_wallet_label
- If the query mentions "wallet score", "risk assessment", "interaction patterns" → use get_wallet_score
- If the query mentions "wallet metrics", "transactional activity", "volume", "inflow/outflow" → use get_wallet_metrics
- If the query mentions "portfolio", "comprehensive analysis", "wallet analysis" → use multiple tools as appropriate

**Required Parameters:**
- For DeFi Portfolio: address, blockchain (required)
- For NFT Portfolio: wallet, blockchain (required)
- For ERC20 Portfolio: address, blockchain (required)
- For Wallet Label: address (required)
- For Wallet Score: no required parameters
- For Wallet Metrics: blockchain, wallet (required)

**Blockchain Options:**
- DeFi/NFT/ERC20: atleta_olympia, avalanche, base, berachain, binance, bitcoin, ethereum, linea, monad_testnet, polygon, polyhedra_testnet, root, solana, somnia_testnet, soneium, unichain, unichain_sepolia
- Wallet Metrics: ethereum, avalanche, linea, polygon

**Time Range Options:** 1d, 7d, 30d, 90d, 1y, all (default: all)

Always extract wallet addresses and blockchain information from the user query. If no blockchain is specified, default to 'ethereum'."""
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]

            if self.verbose:
                print(f"🔄 Making tool selection decision with GPT-4o...")

            # Make the initial API call to GPT-4o for tool selection
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            if self.verbose:
                print(f"📤 GPT-4o TOOL SELECTION RESPONSE RECEIVED")
                if response.choices[0].message.tool_calls:
                    print(f"🛠️  GPT-4o wants to call {len(response.choices[0].message.tool_calls)} tool(s)")
                else:
                    print(f"💭 GPT-4o provided direct response (no tools needed)")

            # Check if the model wants to call tools
            if response.choices[0].message.tool_calls:
                # Add the assistant's response to messages
                messages.append(response.choices[0].message)
                
                # Process each tool call SEQUENTIALLY
                tool_results = []
                for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                    if self.verbose:
                        print(f"\n📞 TOOL CALL #{i+1}:")
                        print(f"🔧 Function: {tool_call.function.name}")
                        print(f"🆔 Call ID: {tool_call.id}")
                    
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if self.verbose:
                        print(f"⏳ EXECUTING TOOL #{i+1} - WAITING FOR COMPLETION...")
                    
                    # Execute the tool and WAIT for completion
                    tool_result = self.execute_tool_call(function_name, function_args)
                    tool_results.append(tool_result)
                    
                    if self.verbose:
                        print(f"✅ TOOL #{i+1} COMPLETED")
                        print(f"🔄 Adding result to conversation context...")
                    
                    # Add the tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result)
                    })
                    
                    # Wait a moment before processing next tool call (if any)
                    if i < len(response.choices[0].message.tool_calls) - 1:
                        if self.verbose:
                            print(f"⏳ MOVING TO NEXT TOOL CALL...")

                if self.verbose:
                    print(f"\n🔄 Sending results back to GPT-4o for final formatting...")

                # Get the final response from GPT-4o for formatting
                final_response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                
                final_content = final_response.choices[0].message.content
                
                if self.verbose:
                    print(f"\n💼 PORTFOLIO AGENT FINAL SUMMARY:")
                    print(f"📊 Tool Results: {len(tool_results)} tool(s) executed")
                    for i, result in enumerate(tool_results):
                        success = result.get("success", False)
                        summary = result.get("summary", "No summary available")
                        print(f"   {i+1}. {'✅' if success else '❌'} {summary}")
                
                if self.verbose:
                    print(f"✅ FINAL RESPONSE GENERATED")
                    print(f"📝 Response length: {len(final_content)} characters")
                    print(f"="*60)
                    print(f"💼 PORTFOLIO AGENT FINAL RESPONSE:")
                    print(f"="*60)
                
                return final_content
            else:
                # No tools needed, return the direct response
                direct_response = response.choices[0].message.content
                
                if self.verbose:
                    print(f"✅ DIRECT RESPONSE (no tools needed)")
                    print(f"📝 Response length: {len(direct_response)} characters")
                    print(f"="*60)
                    print(f"💼 PORTFOLIO AGENT FINAL RESPONSE:")
                    print(f"="*60)
                
                return direct_response

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            if self.verbose:
                print(f"❌ CRITICAL ERROR: {error_msg}")
            return error_msg

# Example usage
if __name__ == "__main__":
    # Initialize the portfolio agent (API keys will be loaded from .env file)
    try:
        # Set verbose=True to see the agent's thinking process
        portfolio_agent = PortfolioAgent(verbose=True)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure you have a .env file with the required API keys.")
        exit(1)
    
    # Example conversations for testing different portfolio analysis scenarios
    example_queries = [
        # DeFi Portfolio queries
        "Get DeFi portfolio for wallet 0xfa99e2bE141dd93Ad016466f60EFAE04BC34D81F on ethereum",
        "Show me DeFi holdings for address 0x1234567890abcdef on polygon",
        "Analyze DeFi portfolio for 0xabcdef1234567890 on avalanche",
        
        # NFT Portfolio queries
        "Get NFT portfolio for wallet 0xfa99e2bE141dd93Ad016466f60EFAE04BC34D81F on ethereum",
        "Show me NFT holdings for address 0x1234567890abcdef on polygon",
        "Analyze NFT portfolio for 0xabcdef1234567890 on solana",
        
        # ERC20 Portfolio queries
        "Get ERC20 portfolio for wallet 0xfa99e2bE141dd93Ad016466f60EFAE04BC34D81F on ethereum",
        "Show me token holdings for address 0x1234567890abcdef on polygon",
        "Analyze ERC20 portfolio for 0xabcdef1234567890 on avalanche",
        
        # Wallet Label queries
        "Get wallet label for 0xfa99e2bE141dd93Ad016466f60EFAE04BC34D81F",
        "Show me classification for wallet 0x1234567890abcdef",
        "Analyze risk factors for address 0xabcdef1234567890",
        
        # Wallet Score queries
        "Get wallet scores",
        "Show me risk assessment",
        "Analyze interaction patterns",
        
        # Wallet Metrics queries
        "Get wallet metrics for 0xfa99e2bE141dd93Ad016466f60EFAE04BC34D81F on ethereum",
        "Show me transactional activity for 0x1234567890abcdef on polygon",
        "Analyze volume data for 0xabcdef1234567890 on avalanche",
        
        # Comprehensive Portfolio queries
        "Analyze the complete portfolio for wallet 0xfa99e2bE141dd93Ad016466f60EFAE04BC34D81F on ethereum",
        "Get comprehensive wallet analysis for 0x1234567890abcdef",
        "Show me everything about wallet 0xabcdef1234567890 on polygon",
    ]
    
    print("Portfolio Analysis Agent - Ready to analyze wallets!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        response = portfolio_agent.chat(user_input)
        print(f"Portfolio Agent: {response}\n") 