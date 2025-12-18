# """
# unified_rag_system.py - Unified RAG System with Vector Search
# Combines customer and policy retrieval using semantic search over Cosmos DB
# """

# import os
# import json
# from azure.cosmos import CosmosClient
# from openai import AzureOpenAI
# from typing import List, Dict, Any, Optional, Tuple
# from datetime import datetime

# class UnifiedRAGSystem:
#     """
#     Unified RAG System that combines customer and policy retrieval
#     using vector search (semantic search) over Azure Cosmos DB
#     """
    
#     def __init__(self):
#         """Initialize all clients and connections"""
#         print("\n" + "=" * 80)
#         print("Initializing Unified RAG System...")
#         print("=" * 80)
        
#         # Initialize Azure OpenAI client for embeddings
#         print("\n📊 Connecting to Azure OpenAI...")
#         # self.llm = ChatOpenAI(
#         #         openai_api_key=OPENAI_API_KEY,
#         #         model_name=OPENAI_MODEL,
#         #         temperature=0.7
#         #     )
        
#         self.openai_client = AzureOpenAI(
#             azure_endpoint=AZURE_OPENAI_ENDPOINT,
#             api_key=AZURE_OPENAI_KEY,
#             api_version=AZURE_OPENAI_API_VERSION
#         )
#         print("✓ Azure OpenAI client initialized")
        
#         # Initialize Customer Database
#         print("\n👥 Connecting to Customer Database...")
#         self.customer_cosmos_client = CosmosClient(CUSTOMER_COSMOS_ENDPOINT, CUSTOMER_COSMOS_KEY)
#         self.customer_database = self.customer_cosmos_client.get_database_client(CUSTOMER_DATABASE_NAME)
#         self.customer_container = self.customer_database.get_container_client(CUSTOMER_CONTAINER_NAME)
#         print("✓ Customer database connected")
        
#         # Initialize Policy Database
#         print("\n📋 Connecting to Policy Database...")
#         self.policy_cosmos_client = CosmosClient(POLICY_COSMOS_ENDPOINT, POLICY_COSMOS_KEY)
#         self.policy_database = self.policy_cosmos_client.get_database_client(POLICY_DATABASE_NAME)
#         self.policy_container = self.policy_database.get_container_client(POLICY_CONTAINER_NAME)
#         print("✓ Policy database connected")
        
#         print("\n" + "=" * 80)
#         print("✓ Unified RAG System Ready!")
#         print("=" * 80 + "\n")
    
#     # ========================================================================
#     # EMBEDDING GENERATION
#     # ========================================================================
    
#     def generate_embedding(self, text: str) -> List[float]:
#         """
#         Generate embedding vector for any text query
        
#         Args:
#             text: Input text to embed
            
#         Returns:
#             List of floats representing the embedding vector
#         """



#         try:
#             response = self.openai_client.embeddings.create(
#                 input=text,
#                 model=AZURE_OPENAI_DEPLOYMENT
#             )
#             return response.data[0].embedding
#         except Exception as e:
#             print(f"❌ Error generating embedding: {e}")
#             raise
    
#     # ========================================================================
#     # CUSTOMER VECTOR SEARCH
#     # ========================================================================
    
#     def search_customers(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         """
#         Semantic search over customer data using vector embeddings
        
#         Args:
#             query: Natural language query (e.g., "high income engineers in California")
#             top_k: Number of results to return
            
#         Returns:
#             List of similar customers with similarity scores
#         """
#         try:
#             print(f"\n🔍 Searching customers for: '{query}'")
            
#             # Generate query embedding
#             query_embedding = self.generate_embedding(query)
            
#             # Vector search query
#             sql_query = """
#             SELECT TOP @top_k 
#                 c.customer_id, 
#                 c.first_name, 
#                 c.last_name, 
#                 c.email, 
#                 c.phone,
#                 c.occupation, 
#                 c.annual_income, 
#                 c.credit_score,
#                 c.marital_status,
#                 c.address,
#                 c.metadata,
#                 VectorDistance(c.embedding, @embedding) AS similarity_score
#             FROM c
#             ORDER BY VectorDistance(c.embedding, @embedding)
#             """
            
#             parameters = [
#                 {"name": "@top_k", "value": top_k},
#                 {"name": "@embedding", "value": query_embedding}
#             ]
            
#             results = list(self.customer_container.query_items(
#                 query=sql_query,
#                 parameters=parameters,
#                 enable_cross_partition_query=True
#             ))
            
#             print(f"✓ Found {len(results)} customers")
#             return results
            
#         except Exception as e:
#             print(f"❌ Error in customer vector search: {e}")
#             return []
    
#     # ========================================================================
#     # POLICY VECTOR SEARCH
#     # ========================================================================
    
#     def search_policies(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         """
#         Semantic search over policy data using vector embeddings
        
#         Args:
#             query: Natural language query (e.g., "active auto insurance policies")
#             top_k: Number of results to return
            
#         Returns:
#             List of similar policies with similarity scores
#         """
#         try:
#             print(f"\n🔍 Searching policies for: '{query}'")
            
#             # Generate query embedding
#             query_embedding = self.generate_embedding(query)
            
#             # Vector search query
#             sql_query = """
#             SELECT TOP @top_k 
#                 c.policy_id,
#                 c.customer_id,
#                 c.policy_number,
#                 c.policy_type,
#                 c.status,
#                 c.annual_premium,
#                 c.coverage_amount,
#                 c.deductible,
#                 c.payment_frequency,
#                 c.start_date,
#                 c.end_date,
#                 c.term_months,
#                 c.auto_renew,
#                 c.agent_id,
#                 c.metadata,
#                 VectorDistance(c.embedding, @embedding) AS similarity_score
#             FROM c
#             ORDER BY VectorDistance(c.embedding, @embedding)
#             """
            
#             parameters = [
#                 {"name": "@top_k", "value": top_k},
#                 {"name": "@embedding", "value": query_embedding}
#             ]
            
#             results = list(self.policy_container.query_items(
#                 query=sql_query,
#                 parameters=parameters,
#                 enable_cross_partition_query=True
#             ))
            
#             print(f"✓ Found {len(results)} policies")
#             return results
            
#         except Exception as e:
#             print(f"❌ Error in policy vector search: {e}")
#             return []
    
#     # ========================================================================
#     # UNIFIED SEARCH - Searches both customers and policies
#     # ========================================================================
    
#     def unified_search(self, query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
#         """
#         Perform semantic search across both customers and policies
        
#         Args:
#             query: Natural language query
#             top_k: Number of results per category
            
#         Returns:
#             Dictionary with 'customers' and 'policies' keys containing results
#         """
#         print(f"\n🔍 Unified Search: '{query}'")
#         print("-" * 80)
        
#         results = {
#             'customers': self.search_customers(query, top_k),
#             'policies': self.search_policies(query, top_k)
#         }
        
#         return results
    
#     # ========================================================================
#     # INTELLIGENT QUERY ROUTING
#     # ========================================================================
    
#     def intelligent_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
#         """
#         Intelligently route query to appropriate search based on keywords
        
#         Args:
#             query: Natural language query
#             top_k: Number of results to return
            
#         Returns:
#             Dictionary with search results and metadata
#         """
#         query_lower = query.lower()
        
#         # Customer-related keywords
#         customer_keywords = ['customer', 'client', 'person', 'income', 'occupation', 
#                            'credit', 'address', 'email', 'phone', 'engineer', 
#                            'manager', 'salary', 'earning']
        
#         # Policy-related keywords
#         policy_keywords = ['policy', 'insurance', 'premium', 'coverage', 'claim',
#                           'auto', 'home', 'life', 'health', 'business', 'deductible',
#                           'renew', 'active', 'cancelled', 'expired']
        
#         # Check for specific keywords
#         has_customer_keywords = any(keyword in query_lower for keyword in customer_keywords)
#         has_policy_keywords = any(keyword in query_lower for keyword in policy_keywords)
        
#         result = {
#             'query': query,
#             'search_type': None,
#             'customers': [],
#             'policies': []
#         }
        
#         # Route based on keywords
#         if has_customer_keywords and not has_policy_keywords:
#             result['search_type'] = 'customer'
#             result['customers'] = self.search_customers(query, top_k)
#         elif has_policy_keywords and not has_customer_keywords:
#             result['search_type'] = 'policy'
#             result['policies'] = self.search_policies(query, top_k)
#         else:
#             # Search both if ambiguous or contains both types
#             result['search_type'] = 'unified'
#             unified_results = self.unified_search(query, top_k)
#             result['customers'] = unified_results['customers']
#             result['policies'] = unified_results['policies']
        
#         return result
    
#     # ========================================================================
#     # CUSTOMER + POLICY COMBINED SEARCH
#     # ========================================================================
    
#     def get_customer_with_policies(self, customer_id: str) -> Dict[str, Any]:
#         """
#         Get complete customer information along with all their policies
        
#         Args:
#             customer_id: Customer ID to search for
            
#         Returns:
#             Dictionary containing customer info and their policies
#         """
#         try:
#             print(f"\n🔍 Retrieving customer and policies for: {customer_id}")
            
#             # Get customer details
#             customer_query = "SELECT * FROM c WHERE c.customer_id = @customer_id"
#             customer_params = [{"name": "@customer_id", "value": customer_id}]
            
#             customers = list(self.customer_container.query_items(
#                 query=customer_query,
#                 parameters=customer_params,
#                 enable_cross_partition_query=True
#             ))
            
#             if not customers:
#                 print(f"❌ Customer {customer_id} not found")
#                 return None
            
#             customer = customers[0]
#             if 'embedding' in customer:
#                 del customer['embedding']
            
#             # Get all policies for this customer
#             policy_query = "SELECT * FROM c WHERE c.customer_id = @customer_id"
#             policy_params = [{"name": "@customer_id", "value": customer_id}]
            
#             policies = list(self.policy_container.query_items(
#                 query=policy_query,
#                 parameters=policy_params,
#                 enable_cross_partition_query=True
#             ))
            
#             # Remove embeddings from policies
#             for policy in policies:
#                 if 'embedding' in policy:
#                     del policy['embedding']
            
#             result = {
#                 'customer': customer,
#                 'policies': policies,
#                 'policy_count': len(policies)
#             }
            
#             print(f"✓ Found customer with {len(policies)} policies")
#             return result
            
#         except Exception as e:
#             print(f"❌ Error retrieving customer with policies: {e}")
#             return None
    
#     # ========================================================================
#     # RAG QUERY - Complete RAG pipeline
#     # ========================================================================
    
#     def rag_query(self, query: str, max_results: int = 5) -> Dict[str, Any]:
#         """
#         Complete RAG pipeline: Retrieves relevant context using vector search
        
#         Args:
#             query: Natural language question
#             max_results: Maximum results to retrieve
            
#         Returns:
#             Dictionary with retrieved context and metadata
#         """
#         print("\n" + "=" * 80)
#         print("RAG QUERY PIPELINE")
#         print("=" * 80)
#         print(f"Query: {query}")
#         print("-" * 80)
        
#         # Step 1: Intelligent routing and retrieval
#         results = self.intelligent_search(query, max_results)
        
#         # Step 2: Format context
#         context = self._format_context(results)
        
#         # Step 3: Prepare RAG response
#         rag_response = {
#             'query': query,
#             'search_type': results['search_type'],
#             'context': context,
#             'customers_found': len(results['customers']),
#             'policies_found': len(results['policies']),
#             'customers': results['customers'],
#             'policies': results['policies'],
#             'timestamp': datetime.now().isoformat()
#         }
        
#         return rag_response
    
#     def _format_context(self, results: Dict[str, Any]) -> str:
#         """
#         Format retrieved results into context string for LLM consumption
        
#         Args:
#             results: Search results dictionary
            
#         Returns:
#             Formatted context string
#         """
#         context_parts = []
        
#         # Format customer context
#         if results['customers']:
#             context_parts.append("CUSTOMER INFORMATION:")
#             context_parts.append("-" * 40)
#             for idx, customer in enumerate(results['customers'], 1):
#                 context_parts.append(f"\n{idx}. {customer['first_name']} {customer['last_name']} ({customer['customer_id']})")
#                 context_parts.append(f"   Occupation: {customer['occupation']}")
#                 context_parts.append(f"   Income: ${customer['annual_income']:,}")
#                 context_parts.append(f"   Credit Score: {customer['credit_score']}")
#                 context_parts.append(f"   Email: {customer['email']}")
#                 context_parts.append(f"   Similarity: {customer['similarity_score']:.4f}")
        
#         # Format policy context
#         if results['policies']:
#             if context_parts:
#                 context_parts.append("\n")
#             context_parts.append("POLICY INFORMATION:")
#             context_parts.append("-" * 40)
#             for idx, policy in enumerate(results['policies'], 1):
#                 context_parts.append(f"\n{idx}. {policy['policy_type']} - {policy['policy_number']}")
#                 context_parts.append(f"   Customer: {policy['customer_id']}")
#                 context_parts.append(f"   Status: {policy['status']}")
#                 context_parts.append(f"   Premium: ${policy['annual_premium']:,.2f}")
#                 context_parts.append(f"   Coverage: ${policy['coverage_amount']:,}")
#                 context_parts.append(f"   Similarity: {policy['similarity_score']:.4f}")
        
#         return "\n".join(context_parts)
    
#     # ========================================================================
#     # DISPLAY METHODS
#     # ========================================================================
    
#     def display_results(self, results: Dict[str, Any]):
#         """Display search results in a formatted way"""
        
#         print("\n" + "=" * 80)
#         print("SEARCH RESULTS")
#         print("=" * 80)
#         print(f"Query: {results['query']}")
#         print(f"Search Type: {results['search_type'].upper()}")
#         print("-" * 80)
        
#         # Display customers
#         if results['customers']:
#             print(f"\n👥 CUSTOMERS FOUND: {len(results['customers'])}")
#             print("-" * 80)
#             for idx, customer in enumerate(results['customers'], 1):
#                 print(f"\n{idx}. {customer['first_name']} {customer['last_name']}")
#                 print(f"   ID: {customer['customer_id']}")
#                 print(f"   Occupation: {customer['occupation']}")
#                 print(f"   Income: ${customer['annual_income']:,}")
#                 print(f"   Credit Score: {customer['credit_score']}")
#                 print(f"   Location: {customer['address']['city']}, {customer['address']['state']}")
#                 print(f"   Similarity Score: {customer['similarity_score']:.4f}")
        
#         # Display policies
#         if results['policies']:
#             print(f"\n📋 POLICIES FOUND: {len(results['policies'])}")
#             print("-" * 80)
#             for idx, policy in enumerate(results['policies'], 1):
#                 print(f"\n{idx}. {policy['policy_type']} - {policy['policy_number']}")
#                 print(f"   Customer: {policy['customer_id']}")
#                 print(f"   Status: {policy['status']}")
#                 print(f"   Premium: ${policy['annual_premium']:,.2f}")
#                 print(f"   Coverage: ${policy['coverage_amount']:,}")
#                 print(f"   Payment: {policy['payment_frequency']}")
#                 print(f"   Similarity Score: {policy['similarity_score']:.4f}")
        
#         print("\n" + "=" * 80 + "\n")
    
#     def display_rag_response(self, rag_response: Dict[str, Any]):
#         """Display complete RAG response"""
        
#         print("\n" + "=" * 80)
#         print("RAG RESPONSE")
#         print("=" * 80)
#         print(f"Query: {rag_response['query']}")
#         print(f"Search Type: {rag_response['search_type'].upper()}")
#         print(f"Customers Found: {rag_response['customers_found']}")
#         print(f"Policies Found: {rag_response['policies_found']}")
#         print(f"Timestamp: {rag_response['timestamp']}")
#         print("-" * 80)
        
#         print("\nRETRIEVED CONTEXT:")
#         print("-" * 80)
#         print(rag_response['context'])
        
#         print("\n" + "=" * 80 + "\n")
    
#     # ========================================================================
#     # EXPORT METHODS
#     # ========================================================================
    
#     def export_results(self, results: Dict[str, Any], filename: str = None):
#         """
#         Export search results to JSON file
        
#         Args:
#             results: Search results dictionary
#             filename: Output filename (auto-generated if None)
#         """
#         if filename is None:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"rag_results_{timestamp}.json"
        
#         try:
#             with open(filename, 'w') as f:
#                 json.dump(results, f, indent=2)
#             print(f"✓ Results exported to {filename}")
#         except Exception as e:
#             print(f"❌ Error exporting results: {e}")


# # ============================================================================
# # INTERACTIVE MENU SYSTEM
# # ============================================================================

# def print_menu():
#     """Print interactive menu"""
#     print("\n" + "=" * 80)
#     print("UNIFIED RAG SYSTEM - MAIN MENU")
#     print("=" * 80)
#     print("1. Search Customers (Vector Search)")
#     print("2. Search Policies (Vector Search)")
#     print("3. Unified Search (Both Customers & Policies)")
#     print("4. Intelligent Search (Auto-route based on query)")
#     print("5. RAG Query (Complete RAG Pipeline)")
#     print("6. Get Customer with All Policies")
#     print("7. Exit")
#     print("-" * 80)


# def main():
#     """Main execution function"""
    
#     print("\n" + "=" * 80)
#     print("UNIFIED RAG SYSTEM WITH VECTOR SEARCH")
#     print("Semantic Search over Customer & Policy Data")
#     print("=" * 80)
    
#     try:
#         # Initialize RAG system
#         rag_system = UnifiedRAGSystem()
#     except Exception as e:
#         print(f"\n❌ Error initializing RAG system: {e}")
#         print("\nPlease check your configuration and credentials.")
#         return
    
#     # Interactive menu loop
#     while True:
#         print_menu()
#         choice = input("\nEnter your choice (1-7): ").strip()
        
#         if choice == "1":
#             # Customer vector search
#             query = input("\nEnter search query for customers: ").strip()
#             if not query:
#                 print("❌ Query cannot be empty")
#                 continue
            
#             try:
#                 top_k = int(input("Number of results (default 5): ").strip() or "5")
#             except ValueError:
#                 top_k = 5
            
#             results = rag_system.search_customers(query, top_k)
            
#             if results:
#                 print(f"\n✓ Found {len(results)} customers:\n")
#                 for idx, customer in enumerate(results, 1):
#                     print(f"{idx}. {customer['first_name']} {customer['last_name']} ({customer['customer_id']})")
#                     print(f"   Occupation: {customer['occupation']}")
#                     print(f"   Income: ${customer['annual_income']:,} | Credit: {customer['credit_score']}")
#                     print(f"   Similarity: {customer['similarity_score']:.4f}\n")
                
#                 # Export option
#                 export = input("Export results to JSON? (y/n): ").strip().lower()
#                 if export == 'y':
#                     rag_system.export_results({'customers': results}, 
#                                              f"customer_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
#             else:
#                 print("❌ No results found")
        
#         elif choice == "2":
#             # Policy vector search
#             query = input("\nEnter search query for policies: ").strip()
#             if not query:
#                 print("❌ Query cannot be empty")
#                 continue
            
#             try:
#                 top_k = int(input("Number of results (default 5): ").strip() or "5")
#             except ValueError:
#                 top_k = 5
            
#             results = rag_system.search_policies(query, top_k)
            
#             if results:
#                 print(f"\n✓ Found {len(results)} policies:\n")
#                 for idx, policy in enumerate(results, 1):
#                     print(f"{idx}. {policy['policy_type']} - {policy['policy_number']}")
#                     print(f"   Customer: {policy['customer_id']} | Status: {policy['status']}")
#                     print(f"   Premium: ${policy['annual_premium']:,.2f} | Coverage: ${policy['coverage_amount']:,}")
#                     print(f"   Similarity: {policy['similarity_score']:.4f}\n")
                
#                 # Export option
#                 export = input("Export results to JSON? (y/n): ").strip().lower()
#                 if export == 'y':
#                     rag_system.export_results({'policies': results}, 
#                                              f"policy_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
#             else:
#                 print("❌ No results found")
        
#         elif choice == "3":
#             # Unified search
#             query = input("\nEnter search query (searches both customers & policies): ").strip()
#             if not query:
#                 print("❌ Query cannot be empty")
#                 continue
            
#             try:
#                 top_k = int(input("Number of results per category (default 5): ").strip() or "5")
#             except ValueError:
#                 top_k = 5
            
#             results = rag_system.unified_search(query, top_k)
            
#             print(f"\n✓ Search completed")
#             print(f"   Customers found: {len(results['customers'])}")
#             print(f"   Policies found: {len(results['policies'])}")
            
#             rag_system.display_results({
#                 'query': query,
#                 'search_type': 'unified',
#                 'customers': results['customers'],
#                 'policies': results['policies']
#             })
            
#             # Export option
#             export = input("Export results to JSON? (y/n): ").strip().lower()
#             if export == 'y':
#                 rag_system.export_results({'query': query, **results})
        
#         elif choice == "4":
#             # Intelligent search
#             query = input("\nEnter your query (auto-routes to appropriate search): ").strip()
#             if not query:
#                 print("❌ Query cannot be empty")
#                 continue
            
#             try:
#                 top_k = int(input("Number of results (default 5): ").strip() or "5")
#             except ValueError:
#                 top_k = 5
            
#             results = rag_system.intelligent_search(query, top_k)
            
#             print(f"\n✓ Query analyzed and routed to: {results['search_type'].upper()} search")
#             rag_system.display_results(results)
            
#             # Export option
#             export = input("Export results to JSON? (y/n): ").strip().lower()
#             if export == 'y':
#                 rag_system.export_results(results)
        
#         elif choice == "5":
#             # Complete RAG query
#             query = input("\nEnter your RAG query: ").strip()
#             if not query:
#                 print("❌ Query cannot be empty")
#                 continue
            
#             try:
#                 top_k = int(input("Max results to retrieve (default 5): ").strip() or "5")
#             except ValueError:
#                 top_k = 5
            
#             rag_response = rag_system.rag_query(query, top_k)
#             rag_system.display_rag_response(rag_response)
            
#             # Export option
#             export = input("Export RAG response to JSON? (y/n): ").strip().lower()
#             if export == 'y':
#                 rag_system.export_results(rag_response)
        
#         elif choice == "6":
#             # Get customer with policies
#             customer_id = input("\nEnter Customer ID (e.g., CUST-0001): ").strip()
#             if not customer_id:
#                 print("❌ Customer ID cannot be empty")
#                 continue
            
#             result = rag_system.get_customer_with_policies(customer_id)
            
#             if result:
#                 customer = result['customer']
#                 policies = result['policies']
                
#                 print("\n" + "=" * 80)
#                 print("CUSTOMER PROFILE")
#                 print("=" * 80)
#                 print(f"Name: {customer['first_name']} {customer['last_name']}")
#                 print(f"ID: {customer['customer_id']}")
#                 print(f"Occupation: {customer['occupation']}")
#                 print(f"Income: ${customer['annual_income']:,}")
#                 print(f"Credit Score: {customer['credit_score']}")
#                 print(f"Email: {customer['email']}")
                
#                 print(f"\n📋 POLICIES ({len(policies)} total)")
#                 print("-" * 80)
#                 for idx, policy in enumerate(policies, 1):
#                     print(f"\n{idx}. {policy['policy_type']} - {policy['policy_number']}")
#                     print(f"   Status: {policy['status']}")
#                     print(f"   Premium: ${policy['annual_premium']:,.2f}")
#                     print(f"   Coverage: ${policy['coverage_amount']:,}")
                
#                 print("\n" + "=" * 80)
                
#                 # Export option
#                 export = input("\nExport to JSON? (y/n): ").strip().lower()
#                 if export == 'y':
#                     rag_system.export_results(result, 
#                                              f"{customer_id}_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
#             else:
#                 print(f"❌ Customer {customer_id} not found or error occurred")
        
#         elif choice == "7":
#             print("\n" + "=" * 80)
#             print("Thank you for using Unified RAG System!")
#             print("=" * 80 + "\n")
#             break
        
#         else:
#             print("\n❌ Invalid choice. Please select 1-7.")


# if __name__ == "__main__":
#     main()

"""
unified_rag_system.py - Unified RAG System with Vector Search
Combines customer and policy retrieval using semantic search over Cosmos DB
"""

import os
import json
from azure.cosmos import CosmosClient
from openai import AzureOpenAI
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()  
# ============================================================================


COSMOS_ret_ENDPOINT = os.getenv("COSMOS_ret_ENDPOINT")
COSMOS_ret_KEY = os.getenv("COSMOS_ret_KEY")
COSMOS_ret_DATABASE_NAME = os.getenv("COSMOS_ret_DATABASE_NAME")
COSMOS_ret_CONTAINER_NAME = os.getenv("COSMOS_ret_CONTAINER_NAME")

# Configuration
COSMOS_pol_ENDPOINT = os.getenv("COSMOS_pol_ENDPOINT")
COSMOS_pol_KEY = os.getenv("COSMOS_pol_KEY")
COSMOS_pol_DATABASE_NAME = os.getenv("COSMOS_pol_DATABASE_NAME")
COSMOS_pol_CONTAINER_NAME = os.getenv("COSMOS_pol_CONTAINER_NAME")


AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = "text-embedding-3-large"
AZURE_OPENAI_API_VERSION = "2024-02-01"


class UnifiedRAGSystem:
    """
    Unified RAG System that combines customer and policy retrieval
    using vector search (semantic search) over Azure Cosmos DB
    """
    
    def __init__(self):
        """Initialize all clients and connections"""
        print("\n" + "=" * 80)
        print("Initializing Unified RAG System...")
        print("=" * 80)
        
        # Initialize Azure OpenAI client for embeddings
        print("\n📊 Connecting to Azure OpenAI...")
        self.openai_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
        print("✓ Azure OpenAI client initialized")
        
        # Initialize Customer Database
        print("\n👥 Connecting to Customer Database...")
        self.customer_cosmos_client = CosmosClient(COSMOS_ret_ENDPOINT, COSMOS_ret_KEY)
        self.customer_database = self.customer_cosmos_client.get_database_client(COSMOS_ret_DATABASE_NAME)
        self.customer_container = self.customer_database.get_container_client(COSMOS_ret_CONTAINER_NAME)
        print("✓ Customer database connected")
        
        # Initialize Policy Database
        print("\n📋 Connecting to Policy Database...")
        self.policy_cosmos_client = CosmosClient(COSMOS_pol_ENDPOINT, COSMOS_pol_KEY)
        self.policy_database = self.policy_cosmos_client.get_database_client(COSMOS_pol_DATABASE_NAME)
        self.policy_container = self.policy_database.get_container_client(COSMOS_pol_CONTAINER_NAME)
        print("✓ Policy database connected")
        
        print("\n" + "=" * 80)
        print("✓ Unified RAG System Ready!")
        print("=" * 80 + "\n")
    
    # ========================================================================
    # EMBEDDING GENERATION
    # ========================================================================
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for any text query
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=AZURE_OPENAI_DEPLOYMENT
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            raise
    
    # ========================================================================
    # CUSTOMER VECTOR SEARCH
    # ========================================================================
    
    def search_customers(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search over customer data using vector embeddings
        
        Args:
            query: Natural language query (e.g., "high income engineers in California")
            top_k: Number of results to return
            
        Returns:
            List of similar customers with similarity scores
        """
        try:
            print(f"\n🔍 Searching customers for: '{query}'")
            
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Vector search query
            sql_query = """
            SELECT TOP @top_k 
                c.customer_id, 
                c.first_name, 
                c.last_name, 
                c.email, 
                c.phone,
                c.occupation, 
                c.annual_income, 
                c.credit_score,
                c.marital_status,
                c.address,
                c.metadata,
                VectorDistance(c.embedding, @embedding) AS similarity_score
            FROM c
            ORDER BY VectorDistance(c.embedding, @embedding)
            """
            
            parameters = [
                {"name": "@top_k", "value": top_k},
                {"name": "@embedding", "value": query_embedding}
            ]
            
            results = list(self.customer_container.query_items(
                query=sql_query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            print(f"✓ Found {len(results)} customers")
            return results
            
        except Exception as e:
            print(f"❌ Error in customer vector search: {e}")
            return []
    
    # ========================================================================
    # POLICY VECTOR SEARCH
    # ========================================================================
    
    def search_policies(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search over policy data using vector embeddings
        
        Args:
            query: Natural language query (e.g., "active auto insurance policies")
            top_k: Number of results to return
            
        Returns:
            List of similar policies with similarity scores
        """
        try:
            print(f"\n🔍 Searching policies for: '{query}'")
            
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Vector search query
            sql_query = """
            SELECT TOP @top_k 
                c.policy_id,
                c.customer_id,
                c.policy_number,
                c.policy_type,
                c.status,
                c.annual_premium,
                c.coverage_amount,
                c.deductible,
                c.payment_frequency,
                c.start_date,
                c.end_date,
                c.term_months,
                c.auto_renew,
                c.agent_id,
                c.metadata,
                VectorDistance(c.embedding, @embedding) AS similarity_score
            FROM c
            ORDER BY VectorDistance(c.embedding, @embedding)
            """
            
            parameters = [
                {"name": "@top_k", "value": top_k},
                {"name": "@embedding", "value": query_embedding}
            ]
            
            results = list(self.policy_container.query_items(
                query=sql_query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            print(f"✓ Found {len(results)} policies")
            return results
            
        except Exception as e:
            print(f"❌ Error in policy vector search: {e}")
            return []
    
    # ========================================================================
    # UNIFIED SEARCH - Searches both customers and policies
    # ========================================================================
    
    def unified_search(self, query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform semantic search across both customers and policies
        Returns COMBINED top_k results based on normalized similarity scores
        
        Args:
            query: Natural language query
            top_k: Total number of results to return (combined)
            
        Returns:
            Dictionary with 'customers' and 'policies' keys containing combined top_k results
        """
        print(f"\n🔍 Unified Search: '{query}'")
        print(f"   Retrieving combined top {top_k} results")
        print("-" * 80)
        
        # Get more candidates from each to ensure we have enough for selection
        candidate_multiplier = 3
        customer_results = self.search_customers(query, top_k * candidate_multiplier)
        policy_results = self.search_policies(query, top_k * candidate_multiplier)
        
        # Normalize scores within each category to make them comparable
        # For distance metrics, lower is better, so we normalize to 0-1 range
        
        if customer_results:
            customer_scores = [c.get('similarity_score', 0) for c in customer_results]
            min_customer = min(customer_scores)
            max_customer = max(customer_scores)
            score_range_customer = max_customer - min_customer if max_customer > min_customer else 1
            
            for customer in customer_results:
                original_score = customer.get('similarity_score', 0)
                # Normalize to 0-1, where 0 is best
                customer['normalized_score'] = (original_score - min_customer) / score_range_customer
                customer['result_type'] = 'customer'
        
        if policy_results:
            policy_scores = [p.get('similarity_score', 0) for p in policy_results]
            min_policy = min(policy_scores)
            max_policy = max(policy_scores)
            score_range_policy = max_policy - min_policy if max_policy > min_policy else 1
            
            for policy in policy_results:
                original_score = policy.get('similarity_score', 0)
                # Normalize to 0-1, where 0 is best
                policy['normalized_score'] = (original_score - min_policy) / score_range_policy
                policy['result_type'] = 'policy'
        
        # Combine all results
        all_results = customer_results + policy_results
        
        # Sort by normalized_score (lower is better)
        all_results.sort(key=lambda x: x.get('normalized_score', float('inf')))
        
        # Take top_k from combined results
        top_results = all_results[:top_k]
        
        # Separate back into customers and policies
        final_customers = [r for r in top_results if r['result_type'] == 'customer']
        final_policies = [r for r in top_results if r['result_type'] == 'policy']
        
        # Remove the temporary fields before returning
        for customer in final_customers:
            customer.pop('result_type', None)
            customer.pop('normalized_score', None)
        
        for policy in final_policies:
            policy.pop('result_type', None)
            policy.pop('normalized_score', None)
        
        print(f"✓ Combined results: {len(final_customers)} customers + {len(final_policies)} policies = {len(top_results)} total")
        
        results = {
            'customers': final_customers,
            'policies': final_policies
        }
        
        return results
    
    # ========================================================================
    # INTELLIGENT QUERY ROUTING
    # ========================================================================
    
    def intelligent_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Intelligently route query to appropriate search based on keywords
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            Dictionary with search results and metadata
        """
        query_lower = query.lower()
        
        # Customer-related keywords
        customer_keywords = ['customer', 'client', 'person', 'income', 'occupation', 
                           'credit', 'address', 'email', 'phone', 'engineer', 
                           'manager', 'salary', 'earning']
        
        # Policy-related keywords
        policy_keywords = ['policy', 'insurance', 'premium', 'coverage', 'claim',
                          'auto', 'home', 'life', 'health', 'business', 'deductible',
                          'renew', 'active', 'cancelled', 'expired']
        
        # Check for specific keywords
        has_customer_keywords = any(keyword in query_lower for keyword in customer_keywords)
        has_policy_keywords = any(keyword in query_lower for keyword in policy_keywords)
        
        result = {
            'query': query,
            'search_type': None,
            'customers': [],
            'policies': []
        }
        
        # Route based on keywords
        if has_customer_keywords and not has_policy_keywords:
            result['search_type'] = 'customer'
            result['customers'] = self.search_customers(query, top_k)
        elif has_policy_keywords and not has_customer_keywords:
            result['search_type'] = 'policy'
            result['policies'] = self.search_policies(query, top_k)
        else:
            # Search both if ambiguous or contains both types
            result['search_type'] = 'unified'
            unified_results = self.unified_search(query, top_k)
            result['customers'] = unified_results['customers']
            result['policies'] = unified_results['policies']
        
        return result
    
    # ========================================================================
    # CUSTOMER + POLICY COMBINED SEARCH
    # ========================================================================
    
    def get_customer_with_policies(self, customer_id: str) -> Dict[str, Any]:
        """
        Get complete customer information along with all their policies
        
        Args:
            customer_id: Customer ID to search for
            
        Returns:
            Dictionary containing customer info and their policies
        """
        try:
            print(f"\n🔍 Retrieving customer and policies for: {customer_id}")
            
            # Get customer details
            customer_query = "SELECT * FROM c WHERE c.customer_id = @customer_id"
            customer_params = [{"name": "@customer_id", "value": customer_id}]
            
            customers = list(self.customer_container.query_items(
                query=customer_query,
                parameters=customer_params,
                enable_cross_partition_query=True
            ))
            
            if not customers:
                print(f"❌ Customer {customer_id} not found")
                return None
            
            customer = customers[0]
            if 'embedding' in customer:
                del customer['embedding']
            
            # Get all policies for this customer
            policy_query = "SELECT * FROM c WHERE c.customer_id = @customer_id"
            policy_params = [{"name": "@customer_id", "value": customer_id}]
            
            policies = list(self.policy_container.query_items(
                query=policy_query,
                parameters=policy_params,
                enable_cross_partition_query=True
            ))
            
            # Remove embeddings from policies
            for policy in policies:
                if 'embedding' in policy:
                    del policy['embedding']
            
            result = {
                'customer': customer,
                'policies': policies,
                'policy_count': len(policies)
            }
            
            print(f"✓ Found customer with {len(policies)} policies")
            return result
            
        except Exception as e:
            print(f"❌ Error retrieving customer with policies: {e}")
            return None
    
    # ========================================================================
    # RAG QUERY - Complete RAG pipeline
    # ========================================================================
    
    def rag_query(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Complete RAG pipeline: Retrieves relevant context using vector search
        
        Args:
            query: Natural language question
            max_results: Maximum results to retrieve
            
        Returns:
            Dictionary with retrieved context and metadata
        """
        print("\n" + "=" * 80)
        print("RAG QUERY PIPELINE")
        print("=" * 80)
        print(f"Query: {query}")
        print("-" * 80)
        
        # Step 1: Intelligent routing and retrieval
        results = self.intelligent_search(query, max_results)
        
        # Step 2: Format context
        context = self._format_context(results)
        
        # Step 3: Prepare RAG response
        rag_response = {
            'query': query,
            'search_type': results['search_type'],
            'context': context,
            'customers_found': len(results['customers']),
            'policies_found': len(results['policies']),
            'customers': results['customers'],
            'policies': results['policies'],
            'timestamp': datetime.now().isoformat()
        }
        
        return rag_response
    
    def _format_context(self, results: Dict[str, Any]) -> str:
        """
        Format retrieved results into context string for LLM consumption
        
        Args:
            results: Search results dictionary
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Format customer context
        if results['customers']:
            context_parts.append("CUSTOMER INFORMATION:")
            context_parts.append("-" * 40)
            for idx, customer in enumerate(results['customers'], 1):
                context_parts.append(f"\n{idx}. {customer['first_name']} {customer['last_name']} ({customer['customer_id']})")
                context_parts.append(f"   Occupation: {customer['occupation']}")
                context_parts.append(f"   Income: ${customer['annual_income']:,}")
                context_parts.append(f"   Credit Score: {customer['credit_score']}")
                context_parts.append(f"   Email: {customer['email']}")
                context_parts.append(f"   Similarity: {customer['similarity_score']:.4f}")
        
        # Format policy context
        if results['policies']:
            if context_parts:
                context_parts.append("\n")
            context_parts.append("POLICY INFORMATION:")
            context_parts.append("-" * 40)
            for idx, policy in enumerate(results['policies'], 1):
                context_parts.append(f"\n{idx}. {policy['policy_type']} - {policy['policy_number']}")
                context_parts.append(f"   Customer: {policy['customer_id']}")
                context_parts.append(f"   Status: {policy['status']}")
                context_parts.append(f"   Premium: ${policy['annual_premium']:,.2f}")
                context_parts.append(f"   Coverage: ${policy['coverage_amount']:,}")
                context_parts.append(f"   Similarity: {policy['similarity_score']:.4f}")
        
        return "\n".join(context_parts)
    
    # ========================================================================
    # DISPLAY METHODS
    # ========================================================================
    
    def display_results(self, results: Dict[str, Any]):
        """Display search results in a formatted way"""
        
        print("\n" + "=" * 80)
        print("SEARCH RESULTS")
        print("=" * 80)
        print(f"Query: {results['query']}")
        print(f"Search Type: {results['search_type'].upper()}")
        print("-" * 80)
        
        # Display customers
        if results['customers']:
            print(f"\n👥 CUSTOMERS FOUND: {len(results['customers'])}")
            print("-" * 80)
            for idx, customer in enumerate(results['customers'], 1):
                print(f"\n{idx}. {customer['first_name']} {customer['last_name']}")
                print(f"   ID: {customer['customer_id']}")
                print(f"   Occupation: {customer['occupation']}")
                print(f"   Income: ${customer['annual_income']:,}")
                print(f"   Credit Score: {customer['credit_score']}")
                print(f"   Location: {customer['address']['city']}, {customer['address']['state']}")
                print(f"   Similarity Score: {customer['similarity_score']:.4f}")
        
        # Display policies
        if results['policies']:
            print(f"\n📋 POLICIES FOUND: {len(results['policies'])}")
            print("-" * 80)
            for idx, policy in enumerate(results['policies'], 1):
                print(f"\n{idx}. {policy['policy_type']} - {policy['policy_number']}")
                print(f"   Customer: {policy['customer_id']}")
                print(f"   Status: {policy['status']}")
                print(f"   Premium: ${policy['annual_premium']:,.2f}")
                print(f"   Coverage: ${policy['coverage_amount']:,}")
                print(f"   Payment: {policy['payment_frequency']}")
                print(f"   Similarity Score: {policy['similarity_score']:.4f}")
        
        print("\n" + "=" * 80 + "\n")
    
    def display_rag_response(self, rag_response: Dict[str, Any]):
        """Display complete RAG response"""
        
        print("\n" + "=" * 80)
        print("RAG RESPONSE")
        print("=" * 80)
        print(f"Query: {rag_response['query']}")
        print(f"Search Type: {rag_response['search_type'].upper()}")
        print(f"Customers Found: {rag_response['customers_found']}")
        print(f"Policies Found: {rag_response['policies_found']}")
        print(f"Timestamp: {rag_response['timestamp']}")
        print("-" * 80)
        
        print("\nRETRIEVED CONTEXT:")
        print("-" * 80)
        print(rag_response['context'])
        
        print("\n" + "=" * 80 + "\n")
    
    # ========================================================================
    # EXPORT METHODS
    # ========================================================================
    
    def export_results(self, results: Dict[str, Any], filename: str = None):
        """
        Export search results to JSON file
        
        Args:
            results: Search results dictionary
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Results exported to {filename}")
        except Exception as e:
            print(f"❌ Error exporting results: {e}")


# ============================================================================
# INTERACTIVE MENU SYSTEM
# ============================================================================

def print_menu():
    """Print interactive menu"""
    print("\n" + "=" * 80)
    print("UNIFIED RAG SYSTEM - MAIN MENU")
    print("=" * 80)
    print("1. Search Customers (Vector Search)")
    print("2. Search Policies (Vector Search)")
    print("3. Unified Search (Both Customers & Policies)")
    print("4. Intelligent Search (Auto-route based on query)")
    print("5. RAG Query (Complete RAG Pipeline)")
    print("6. Get Customer with All Policies")
    print("7. Exit")
    print("-" * 80)


def main():
    """Main execution function"""
    
    print("\n" + "=" * 80)
    print("UNIFIED RAG SYSTEM WITH VECTOR SEARCH")
    print("Semantic Search over Customer & Policy Data")
    print("=" * 80)
    
    try:
        # Initialize RAG system
        rag_system = UnifiedRAGSystem()
    except Exception as e:
        print(f"\n❌ Error initializing RAG system: {e}")
        print("\nPlease check your configuration and credentials.")
        return
    
    # Interactive menu loop
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            # Customer vector search
            query = input("\nEnter search query for customers: ").strip()
            if not query:
                print("❌ Query cannot be empty")
                continue
            
            try:
                top_k = int(input("Number of results (default 5): ").strip() or "5")
            except ValueError:
                top_k = 5
            
            results = rag_system.search_customers(query, top_k)
            
            if results:
                print(f"\n✓ Found {len(results)} customers:\n")
                for idx, customer in enumerate(results, 1):
                    print(f"{idx}. {customer['first_name']} {customer['last_name']} ({customer['customer_id']})")
                    print(f"   Occupation: {customer['occupation']}")
                    print(f"   Income: ${customer['annual_income']:,} | Credit: {customer['credit_score']}")
                    print(f"   Similarity: {customer['similarity_score']:.4f}\n")
                
                # Export option
                export = input("Export results to JSON? (y/n): ").strip().lower()
                if export == 'y':
                    rag_system.export_results({'customers': results}, 
                                             f"customer_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            else:
                print("❌ No results found")
        
        elif choice == "2":
            # Policy vector search
            query = input("\nEnter search query for policies: ").strip()
            if not query:
                print("❌ Query cannot be empty")
                continue
            
            try:
                top_k = int(input("Number of results (default 5): ").strip() or "5")
            except ValueError:
                top_k = 5
            
            results = rag_system.search_policies(query, top_k)
            
            if results:
                print(f"\n✓ Found {len(results)} policies:\n")
                for idx, policy in enumerate(results, 1):
                    print(f"{idx}. {policy['policy_type']} - {policy['policy_number']}")
                    print(f"   Customer: {policy['customer_id']} | Status: {policy['status']}")
                    print(f"   Premium: ${policy['annual_premium']:,.2f} | Coverage: ${policy['coverage_amount']:,}")
                    print(f"   Similarity: {policy['similarity_score']:.4f}\n")
                
                # Export option
                export = input("Export results to JSON? (y/n): ").strip().lower()
                if export == 'y':
                    rag_system.export_results({'policies': results}, 
                                             f"policy_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            else:
                print("❌ No results found")
        
        elif choice == "3":
            # Unified search
            query = input("\nEnter search query (searches both customers & policies): ").strip()
            if not query:
                print("❌ Query cannot be empty")
                continue
            
            try:
                top_k = int(input("Number of results per category (default 5): ").strip() or "5")
            except ValueError:
                top_k = 5
            
            results = rag_system.unified_search(query, top_k)
            
            print(f"\n✓ Search completed")
            print(f"   Customers found: {len(results['customers'])}")
            print(f"   Policies found: {len(results['policies'])}")
            
            rag_system.display_results({
                'query': query,
                'search_type': 'unified',
                'customers': results['customers'],
                'policies': results['policies']
            })
            
            # Export option
            export = input("Export results to JSON? (y/n): ").strip().lower()
            if export == 'y':
                rag_system.export_results({'query': query, **results})
        
        elif choice == "4":
            # Intelligent search
            query = input("\nEnter your query (auto-routes to appropriate search): ").strip()
            if not query:
                print("❌ Query cannot be empty")
                continue
            
            try:
                top_k = int(input("Number of results (default 5): ").strip() or "5")
            except ValueError:
                top_k = 5
            
            results = rag_system.intelligent_search(query, top_k)
            
            print(f"\n✓ Query analyzed and routed to: {results['search_type'].upper()} search")
            rag_system.display_results(results)
            
            # Export option
            export = input("Export results to JSON? (y/n): ").strip().lower()
            if export == 'y':
                rag_system.export_results(results)
        
        elif choice == "5":
            # Complete RAG query
            query = input("\nEnter your RAG query: ").strip()
            if not query:
                print("❌ Query cannot be empty")
                continue
            
            try:
                top_k = int(input("Max results to retrieve (default 5): ").strip() or "5")
            except ValueError:
                top_k = 5
            
            rag_response = rag_system.rag_query(query, top_k)
            rag_system.display_rag_response(rag_response)
            
            # Export option
            export = input("Export RAG response to JSON? (y/n): ").strip().lower()
            if export == 'y':
                rag_system.export_results(rag_response)
        
        elif choice == "6":
            # Get customer with policies
            customer_id = input("\nEnter Customer ID (e.g., CUST-0001): ").strip()
            if not customer_id:
                print("❌ Customer ID cannot be empty")
                continue
            
            result = rag_system.get_customer_with_policies(customer_id)
            
            if result:
                customer = result['customer']
                policies = result['policies']
                
                print("\n" + "=" * 80)
                print("CUSTOMER PROFILE")
                print("=" * 80)
                print(f"Name: {customer['first_name']} {customer['last_name']}")
                print(f"ID: {customer['customer_id']}")
                print(f"Occupation: {customer['occupation']}")
                print(f"Income: ${customer['annual_income']:,}")
                print(f"Credit Score: {customer['credit_score']}")
                print(f"Email: {customer['email']}")
                
                print(f"\n📋 POLICIES ({len(policies)} total)")
                print("-" * 80)
                for idx, policy in enumerate(policies, 1):
                    print(f"\n{idx}. {policy['policy_type']} - {policy['policy_number']}")
                    print(f"   Status: {policy['status']}")
                    print(f"   Premium: ${policy['annual_premium']:,.2f}")
                    print(f"   Coverage: ${policy['coverage_amount']:,}")
                
                print("\n" + "=" * 80)
                
                # Export option
                export = input("\nExport to JSON? (y/n): ").strip().lower()
                if export == 'y':
                    rag_system.export_results(result, 
                                             f"{customer_id}_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            else:
                print(f"❌ Customer {customer_id} not found or error occurred")
        
        elif choice == "7":
            print("\n" + "=" * 80)
            print("Thank you for using Unified RAG System!")
            print("=" * 80 + "\n")
            break
        
        else:
            print("\n❌ Invalid choice. Please select 1-7.")


if __name__ == "__main__":
    main()