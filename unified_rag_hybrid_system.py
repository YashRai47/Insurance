# """
# unified_rag_hybrid_system.py - Unified RAG System with HYBRID SEARCH
# Combines customer and policy retrieval using hybrid search (semantic + keyword) over Cosmos DB
# """




# class UnifiedRAGHybridSystem:
#     """
#     Unified RAG System that combines customer and policy retrieval
#     using HYBRID SEARCH (semantic + keyword matching) over Azure Cosmos DB
#     """
    
#     def __init__(self):
#         """Initialize all clients and connections"""
#         print("\n" + "=" * 80)
#         print("Initializing Unified RAG System with HYBRID SEARCH...")
#         print("=" * 80)
        
#         # Initialize Azure OpenAI client for embeddings
#         print("\n📊 Connecting to Azure OpenAI...")
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
#         print("✓ Unified RAG System with Hybrid Search Ready!")
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
#     # CUSTOMER HYBRID SEARCH
#     # ========================================================================
    
#     def search_customers(
#         self, 
#         query: str, 
#         top_k: int = 5,
#         vector_weight: float = 0.6,
#         keyword_weight: float = 0.4
#     ) -> List[Dict[str, Any]]:
#         """
#         HYBRID SEARCH over customer data (semantic + keyword matching)
        
#         Args:
#             query: Natural language query (e.g., "high income engineers in California")
#             top_k: Number of results to return
#             vector_weight: Weight for semantic similarity (default 0.6)
#             keyword_weight: Weight for keyword matching (default 0.4)
            
#         Returns:
#             List of similar customers with hybrid scores
#         """
#         try:
#             print(f"\n🔍 Hybrid searching customers for: '{query}'")
#             print(f"   Weights: Vector={vector_weight:.0%}, Keyword={keyword_weight:.0%}")
            
#             # Step 1: Get more candidates using vector search
#             candidate_count = min(top_k * 3, 50)
            
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
#                 VectorDistance(c.embedding, @embedding) AS vector_score
#             FROM c
#             ORDER BY VectorDistance(c.embedding, @embedding)
#             """
            
#             parameters = [
#                 {"name": "@top_k", "value": candidate_count},
#                 {"name": "@embedding", "value": query_embedding}
#             ]
            
#             results = list(self.customer_container.query_items(
#                 query=sql_query,
#                 parameters=parameters,
#                 enable_cross_partition_query=True
#             ))
            
#             if not results:
#                 return []
            
#             # Step 2: Calculate keyword match scores
#             query_terms = query.lower().split()
            
#             for result in results:
#                 # Create searchable text from customer fields
#                 searchable_text = " ".join([
#                     str(result.get('first_name', '')),
#                     str(result.get('last_name', '')),
#                     str(result.get('occupation', '')),
#                     str(result.get('email', '')),
#                     str(result.get('address', {}).get('city', '')),
#                     str(result.get('address', {}).get('state', ''))
#                 ]).lower()
                
#                 # Calculate keyword match score (0-1)
#                 matches = sum(1 for term in query_terms if term in searchable_text)
#                 keyword_score = matches / len(query_terms) if query_terms else 0
                
#                 # Calculate hybrid score
#                 vector_score_normalized = result['vector_score']
#                 hybrid_score = (vector_weight * vector_score_normalized) + (keyword_weight * keyword_score)
                
#                 # Add scores to result
#                 result['keyword_score'] = keyword_score
#                 result['hybrid_score'] = hybrid_score
#                 result['similarity_score'] = hybrid_score  # For backward compatibility
            
#             # Step 3: Re-rank by hybrid score and return top_k
#             results.sort(key=lambda x: x['hybrid_score'], reverse=False)
#             results = results[:top_k]
            
#             print(f"✓ Found {len(results)} customers")
#             return results
            
#         except Exception as e:
#             print(f"❌ Error in customer hybrid search: {e}")
#             return []
    
#     # ========================================================================
#     # POLICY HYBRID SEARCH
#     # ========================================================================
    
#     def search_policies(
#         self, 
#         query: str, 
#         top_k: int = 5,
#         vector_weight: float = 0.6,
#         keyword_weight: float = 0.4
#     ) -> List[Dict[str, Any]]:
#         """
#         HYBRID SEARCH over policy data (semantic + keyword matching)
        
#         Args:
#             query: Natural language query (e.g., "active auto insurance policies")
#             top_k: Number of results to return
#             vector_weight: Weight for semantic similarity (default 0.6)
#             keyword_weight: Weight for keyword matching (default 0.4)
            
#         Returns:
#             List of similar policies with hybrid scores
#         """
#         try:
#             print(f"\n🔍 Hybrid searching policies for: '{query}'")
#             print(f"   Weights: Vector={vector_weight:.0%}, Keyword={keyword_weight:.0%}")
            
#             # Step 1: Get more candidates using vector search
#             candidate_count = min(top_k * 3, 50)
            
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
#                 VectorDistance(c.embedding, @embedding) AS vector_score
#             FROM c
#             ORDER BY VectorDistance(c.embedding, @embedding)
#             """
            
#             parameters = [
#                 {"name": "@top_k", "value": candidate_count},
#                 {"name": "@embedding", "value": query_embedding}
#             ]
            
#             results = list(self.policy_container.query_items(
#                 query=sql_query,
#                 parameters=parameters,
#                 enable_cross_partition_query=True
#             ))
            
#             if not results:
#                 return []
            
#             # Step 2: Calculate keyword match scores
#             query_terms = query.lower().split()
            
#             for result in results:
#                 # Create searchable text from policy fields
#                 searchable_text = " ".join([
#                     str(result.get('policy_type', '')),
#                     str(result.get('policy_number', '')),
#                     str(result.get('status', '')),
#                     str(result.get('payment_frequency', '')),
#                     str(result.get('customer_id', '')),
#                     "auto" if result.get('auto_renew') else "",
#                     "renew" if result.get('auto_renew') else ""
#                 ]).lower()
                
#                 # Calculate keyword match score (0-1)
#                 matches = sum(1 for term in query_terms if term in searchable_text)
#                 keyword_score = matches / len(query_terms) if query_terms else 0
                
#                 # Calculate hybrid score
#                 vector_score_normalized = result['vector_score']
#                 hybrid_score = (vector_weight * vector_score_normalized) + (keyword_weight * keyword_score)
                
#                 # Add scores to result
#                 result['keyword_score'] = keyword_score
#                 result['hybrid_score'] = hybrid_score
#                 result['similarity_score'] = hybrid_score  # For backward compatibility
            
#             # Step 3: Re-rank by hybrid score and return top_k
#             results.sort(key=lambda x: x['hybrid_score'], reverse=False)
#             results = results[:top_k]
            
#             print(f"✓ Found {len(results)} policies")
#             return results
            
#         except Exception as e:
#             print(f"❌ Error in policy hybrid search: {e}")
#             return []
    
#     # ========================================================================
#     # UNIFIED SEARCH - Searches both customers and policies
#     # ========================================================================
    
#     def unified_search(
#         self, 
#         query: str, 
#         top_k: int = 5,
#         vector_weight: float = 0.6,
#         keyword_weight: float = 0.4
#     ) -> Dict[str, List[Dict[str, Any]]]:
#         """
#         Perform hybrid search across both customers and policies
        
#         Args:
#             query: Natural language query
#             top_k: Number of results per category
#             vector_weight: Weight for semantic similarity
#             keyword_weight: Weight for keyword matching
            
#         Returns:
#             Dictionary with 'customers' and 'policies' keys containing results
#         """
#         print(f"\n🔍 Unified Hybrid Search: '{query}'")
#         print("-" * 80)
        
#         results = {
#             'customers': self.search_customers(query, top_k, vector_weight, keyword_weight),
#             'policies': self.search_policies(query, top_k, vector_weight, keyword_weight)
#         }
        
#         return results
    
#     # ========================================================================
#     # INTELLIGENT QUERY ROUTING
#     # ========================================================================
    
#     def intelligent_search(
#         self, 
#         query: str, 
#         top_k: int = 5,
#         vector_weight: float = 0.6,
#         keyword_weight: float = 0.4
#     ) -> Dict[str, Any]:
#         """
#         Intelligently route query to appropriate search based on keywords
        
#         Args:
#             query: Natural language query
#             top_k: Number of results to return
#             vector_weight: Weight for semantic similarity
#             keyword_weight: Weight for keyword matching
            
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
#             result['customers'] = self.search_customers(query, top_k, vector_weight, keyword_weight)
#         elif has_policy_keywords and not has_customer_keywords:
#             result['search_type'] = 'policy'
#             result['policies'] = self.search_policies(query, top_k, vector_weight, keyword_weight)
#         else:
#             # Search both if ambiguous or contains both types
#             result['search_type'] = 'unified'
#             unified_results = self.unified_search(query, top_k, vector_weight, keyword_weight)
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
    
#     def rag_query(
#         self, 
#         query: str, 
#         max_results: int = 5,
#         vector_weight: float = 0.6,
#         keyword_weight: float = 0.4
#     ) -> Dict[str, Any]:
#         """
#         Complete RAG pipeline: Retrieves relevant context using hybrid search
        
#         Args:
#             query: Natural language question
#             max_results: Maximum results to retrieve
#             vector_weight: Weight for semantic similarity
#             keyword_weight: Weight for keyword matching
            
#         Returns:
#             Dictionary with retrieved context and metadata
#         """
#         print("\n" + "=" * 80)
#         print("RAG QUERY PIPELINE (HYBRID SEARCH)")
#         print("=" * 80)
#         print(f"Query: {query}")
#         print(f"Weights: Vector={vector_weight:.0%}, Keyword={keyword_weight:.0%}")
#         print("-" * 80)
        
#         # Step 1: Intelligent routing and retrieval
#         results = self.intelligent_search(query, max_results, vector_weight, keyword_weight)
        
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
#             'vector_weight': vector_weight,
#             'keyword_weight': keyword_weight,
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
#                 context_parts.append(f"   Hybrid Score: {customer.get('hybrid_score', 0):.4f}")
#                 if 'vector_score' in customer and 'keyword_score' in customer:
#                     context_parts.append(f"   (Vector: {customer['vector_score']:.4f}, Keyword: {customer['keyword_score']:.4f})")
        
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
#                 context_parts.append(f"   Hybrid Score: {policy.get('hybrid_score', 0):.4f}")
#                 if 'vector_score' in policy and 'keyword_score' in policy:
#                     context_parts.append(f"   (Vector: {policy['vector_score']:.4f}, Keyword: {policy['keyword_score']:.4f})")
        
#         return "\n".join(context_parts)
    
#     # ========================================================================
#     # DISPLAY METHODS
#     # ========================================================================
    
#     def display_results(self, results: Dict[str, Any]):
#         """Display search results in a formatted way"""
        
#         print("\n" + "=" * 80)
#         print("HYBRID SEARCH RESULTS")
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
#                 print(f"   Hybrid Score: {customer.get('hybrid_score', 0):.4f}")
#                 if 'vector_score' in customer and 'keyword_score' in customer:
#                     print(f"   (Vector: {customer['vector_score']:.4f}, Keyword: {customer['keyword_score']:.4f})")
        
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
#                 print(f"   Hybrid Score: {policy.get('hybrid_score', 0):.4f}")
#                 if 'vector_score' in policy and 'keyword_score' in policy:
#                     print(f"   (Vector: {policy['vector_score']:.4f}, Keyword: {policy['keyword_score']:.4f})")
        
#         print("\n" + "=" * 80 + "\n")
    
#     def display_rag_response(self, rag_response: Dict[str, Any]):
#         """Display complete RAG response"""
        
#         print("\n" + "=" * 80)
#         print("RAG RESPONSE (HYBRID SEARCH)")
#         print("=" * 80)
#         print(f"Query: {rag_response['query']}")
#         print(f"Search Type: {rag_response['search_type'].upper()}")
#         print(f"Customers Found: {rag_response['customers_found']}")
#         print(f"Policies Found: {rag_response['policies_found']}")
#         print(f"Weights: Vector={rag_response['vector_weight']:.0%}, Keyword={rag_response['keyword_weight']:.0%}")
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
#             filename = f"hybrid_rag_results_{timestamp}.json"
        
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
#     print("UNIFIED RAG SYSTEM - HYBRID SEARCH - MAIN MENU")
#     print("=" * 80)
#     print("1. Search Customers (Hybrid Search)")
#     print("2. Search Policies (Hybrid Search)")
#     print("3. Unified Search (Both Customers & Policies)")
#     print("4. Intelligent Search (Auto-route based on query)")
#     print("5. RAG Query (Complete RAG Pipeline)")
#     print("6. Get Customer with All Policies")
#     print("7. Custom Weights Search")
#     print("8. Exit")
#     print("-" * 80)


# def main():
#     """Main execution function"""
    
#     print("\n" + "=" * 80)
#     print("UNIFIED RAG SYSTEM WITH HYBRID SEARCH")
#     print("Hybrid Search (Semantic + Keyword) over Customer & Policy Data")
#     print("=" * 80)
    
#     try:
#         # Initialize RAG system
#         rag_system = UnifiedRAGHybridSystem()
#     except Exception as e:
#         print(f"\n❌ Error initializing RAG system: {e}")
#         print("\nPlease check your configuration and credentials.")
#         return
    
#     # Interactive menu loop
#     while True:
#         print_menu()
#         choice = input("\nEnter your choice (1-8): ").strip()
        
#         if choice == "1":
#             # Customer hybrid search
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
#                     print(f"   Hybrid: {customer.get('hybrid_score', 0):.4f} (V:{customer.get('vector_score', 0):.4f}, K:{customer.get('keyword_score', 0):.4f})\n")
                
#                 # Export option
#                 export = input("Export results to JSON? (y/n): ").strip().lower()
#                 if export == 'y':
#                     rag_system.export_results({'customers': results}, 
#                                              f"customer_hybrid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
#             else:
#                 print("❌ No results found")
        
#         elif choice == "2":
#             # Policy hybrid search
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
#                     print(f"   Hybrid: {policy.get('hybrid_score', 0):.4f} (V:{policy.get('vector_score', 0):.4f}, K:{policy.get('keyword_score', 0):.4f})\n")
                
#                 # Export option
#                 export = input("Export results to JSON? (y/n): ").strip().lower()
#                 if export == 'y':
#                     rag_system.export_results({'policies': results}, 
#                                              f"policy_hybrid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
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
#             # Custom weights search
#             query = input("\nEnter search query: ").strip()
#             if not query:
#                 print("❌ Query cannot be empty")
#                 continue
            
#             try:
#                 top_k = int(input("Number of results (default 5): ").strip() or "5")
                
#                 print("\nSet search weights (must sum to 1.0):")
#                 vector_weight = float(input("Vector weight (0.0-1.0, default 0.6): ").strip() or "0.6")
#                 keyword_weight = float(input("Keyword weight (0.0-1.0, default 0.4): ").strip() or "0.4")
                
#                 # Normalize weights
#                 total = vector_weight + keyword_weight
#                 if total != 1.0:
#                     print(f"\n⚠️  Normalizing weights (sum was {total:.2f})")
#                     vector_weight = vector_weight / total
#                     keyword_weight = keyword_weight / total
#                     print(f"   Adjusted: Vector={vector_weight:.2f}, Keyword={keyword_weight:.2f}")
                
#             except ValueError:
#                 print("❌ Invalid input, using defaults")
#                 top_k = 5
#                 vector_weight = 0.6
#                 keyword_weight = 0.4
            
#             # Ask which type of search
#             print("\nSearch in:")
#             print("1. Customers")
#             print("2. Policies")
#             print("3. Both (Unified)")
            
#             search_type = input("Your choice (1-3): ").strip()
            
#             if search_type == "1":
#                 results = rag_system.search_customers(query, top_k, vector_weight, keyword_weight)
#                 if results:
#                     rag_system.display_results({
#                         'query': query,
#                         'search_type': 'customer',
#                         'customers': results,
#                         'policies': []
#                     })
#             elif search_type == "2":
#                 results = rag_system.search_policies(query, top_k, vector_weight, keyword_weight)
#                 if results:
#                     rag_system.display_results({
#                         'query': query,
#                         'search_type': 'policy',
#                         'customers': [],
#                         'policies': results
#                     })
#             else:
#                 results = rag_system.unified_search(query, top_k, vector_weight, keyword_weight)
#                 rag_system.display_results({
#                     'query': query,
#                     'search_type': 'unified',
#                     'customers': results['customers'],
#                     'policies': results['policies']
#                 })
        
#         elif choice == "8":
#             print("\n" + "=" * 80)
#             print("Thank you for using Unified RAG Hybrid System!")
#             print("=" * 80 + "\n")
#             break
        
#         else:
#             print("\n❌ Invalid choice. Please select 1-8.")


# if __name__ == "__main__":
#     main()

"""
unified_rag_hybrid_system.py - Unified RAG System with HYBRID SEARCH
Combines customer and policy retrieval using hybrid search (semantic + keyword) over Cosmos DB
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


COSMOS_hybrid_ENDPOINT = os.getenv("COSMOS_hybrid_ENDPOINT")
COSMOS_hybrid_KEY = os.getenv("COSMOS_hybrid_KEY")
COSMOS_hybrid_DATABASE_NAME = os.getenv("COSMOS_hybrid_DATABASE_NAME")
COSMOS_hybrid_CONTAINER_NAME = os.getenv("COSMOS_hybrid_CONTAINER_NAME")

# Configuration
COSMOS_pol_hybrid_ENDPOINT = os.getenv("COSMOS_pol_hybrid_ENDPOINT")
COSMOS_pol_hybrid_KEY = os.getenv("COSMOS_pol_hybrid_KEY")
COSMOS_pol_hybrid_DATABASE_NAME = os.getenv("COSMOS_pol_hybrid_DATABASE_NAME")
COSMOS_pol_hybrid_CONTAINER_NAME = os.getenv("COSMOS_pol_hybrid_CONTAINER_NAME")


AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = "text-embedding-3-large"
AZURE_OPENAI_API_VERSION = "2024-02-01"


# class UnifiedRAGHybridSystem:
#     """
#     Unified RAG System that combines customer and policy retrieval
#     using HYBRID SEARCH (semantic + keyword matching) over Azure Cosmos DB
#     """
    
#     def __init__(self):
#         """Initialize all clients and connections"""
#         print("\n" + "=" * 80)
#         print("Initializing Unified RAG System with HYBRID SEARCH...")
#         print("=" * 80)
        
#         # Initialize Azure OpenAI client for embeddings
#         print("\n📊 Connecting to Azure OpenAI...")
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
#         print("✓ Unified RAG System with Hybrid Search Ready!")
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
#     # CUSTOMER HYBRID SEARCH
#     # ========================================================================
    
#     def search_customers(
#         self, 
#         query: str, 
#         top_k: int = 5,
#         vector_weight: float = 0.6,
#         keyword_weight: float = 0.4
#     ) -> List[Dict[str, Any]]:
#         """
#         HYBRID SEARCH over customer data (semantic + keyword matching)
        
#         Args:
#             query: Natural language query (e.g., "high income engineers in California")
#             top_k: Number of results to return
#             vector_weight: Weight for semantic similarity (default 0.6)
#             keyword_weight: Weight for keyword matching (default 0.4)
            
#         Returns:
#             List of similar customers with hybrid scores
#         """
#         try:
#             print(f"\n🔍 Hybrid searching customers for: '{query}'")
#             print(f"   Weights: Vector={vector_weight:.0%}, Keyword={keyword_weight:.0%}")
            
#             # Step 1: Get more candidates using vector search
#             candidate_count = min(top_k * 3, 50)
            
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
#                 VectorDistance(c.embedding, @embedding) AS vector_score
#             FROM c
#             ORDER BY VectorDistance(c.embedding, @embedding)
#             """
            
#             parameters = [
#                 {"name": "@top_k", "value": candidate_count},
#                 {"name": "@embedding", "value": query_embedding}
#             ]
            
#             results = list(self.customer_container.query_items(
#                 query=sql_query,
#                 parameters=parameters,
#                 enable_cross_partition_query=True
#             ))
            
#             if not results:
#                 return []
            
#             # Step 2: Calculate keyword match scores
#             query_terms = query.lower().split()
            
#             for result in results:
#                 # Create searchable text from customer fields
#                 searchable_text = " ".join([
#                     str(result.get('first_name', '')),
#                     str(result.get('last_name', '')),
#                     str(result.get('occupation', '')),
#                     str(result.get('email', '')),
#                     str(result.get('address', {}).get('city', '')),
#                     str(result.get('address', {}).get('state', ''))
#                 ]).lower()
                
#                 # Calculate keyword match score (0-1)
#                 matches = sum(1 for term in query_terms if term in searchable_text)
#                 keyword_score = matches / len(query_terms) if query_terms else 0
                
#                 # Calculate hybrid score
#                 vector_score_normalized = result['vector_score']
#                 hybrid_score = (vector_weight * vector_score_normalized) + (keyword_weight * keyword_score)
                
#                 # Add scores to result
#                 result['keyword_score'] = keyword_score
#                 result['hybrid_score'] = hybrid_score
#                 result['similarity_score'] = hybrid_score  # For backward compatibility
            
#             # Step 3: Re-rank by hybrid score and return top_k
#             results.sort(key=lambda x: x['hybrid_score'], reverse=False)
#             results = results[:top_k]
            
#             print(f"✓ Found {len(results)} customers")
#             return results
            
#         except Exception as e:
#             print(f"❌ Error in customer hybrid search: {e}")
#             return []
    
#     # ========================================================================
#     # POLICY HYBRID SEARCH
#     # ========================================================================
    
#     def search_policies(
#         self, 
#         query: str, 
#         top_k: int = 5,
#         vector_weight: float = 0.6,
#         keyword_weight: float = 0.4
#     ) -> List[Dict[str, Any]]:
#         """
#         HYBRID SEARCH over policy data (semantic + keyword matching)
        
#         Args:
#             query: Natural language query (e.g., "active auto insurance policies")
#             top_k: Number of results to return
#             vector_weight: Weight for semantic similarity (default 0.6)
#             keyword_weight: Weight for keyword matching (default 0.4)
            
#         Returns:
#             List of similar policies with hybrid scores
#         """
#         try:
#             print(f"\n🔍 Hybrid searching policies for: '{query}'")
#             print(f"   Weights: Vector={vector_weight:.0%}, Keyword={keyword_weight:.0%}")
            
#             # Step 1: Get more candidates using vector search
#             candidate_count = min(top_k * 3, 50)
            
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
#                 VectorDistance(c.embedding, @embedding) AS vector_score
#             FROM c
#             ORDER BY VectorDistance(c.embedding, @embedding)
#             """
            
#             parameters = [
#                 {"name": "@top_k", "value": candidate_count},
#                 {"name": "@embedding", "value": query_embedding}
#             ]
            
#             results = list(self.policy_container.query_items(
#                 query=sql_query,
#                 parameters=parameters,
#                 enable_cross_partition_query=True
#             ))
            
#             if not results:
#                 return []
            
#             # Step 2: Calculate keyword match scores
#             query_terms = query.lower().split()
            
#             for result in results:
#                 # Create searchable text from policy fields
#                 searchable_text = " ".join([
#                     str(result.get('policy_type', '')),
#                     str(result.get('policy_number', '')),
#                     str(result.get('status', '')),
#                     str(result.get('payment_frequency', '')),
#                     str(result.get('customer_id', '')),
#                     "auto" if result.get('auto_renew') else "",
#                     "renew" if result.get('auto_renew') else ""
#                 ]).lower()
                
#                 # Calculate keyword match score (0-1)
#                 matches = sum(1 for term in query_terms if term in searchable_text)
#                 keyword_score = matches / len(query_terms) if query_terms else 0
                
#                 # Calculate hybrid score
#                 vector_score_normalized = result['vector_score']
#                 hybrid_score = (vector_weight * vector_score_normalized) + (keyword_weight * keyword_score)
                
#                 # Add scores to result
#                 result['keyword_score'] = keyword_score
#                 result['hybrid_score'] = hybrid_score
#                 result['similarity_score'] = hybrid_score  # For backward compatibility
            
#             # Step 3: Re-rank by hybrid score and return top_k
#             results.sort(key=lambda x: x['hybrid_score'], reverse=False)
#             results = results[:top_k]
            
#             print(f"✓ Found {len(results)} policies")
#             return results
            
#         except Exception as e:
#             print(f"❌ Error in policy hybrid search: {e}")
#             return []
    
#     # ========================================================================
#     # UNIFIED SEARCH - Searches both customers and policies
#     # ========================================================================
    
#     def unified_search(
#         self, 
#         query: str, 
#         top_k: int = 5,
#         vector_weight: float = 0.6,
#         keyword_weight: float = 0.4
#     ) -> Dict[str, List[Dict[str, Any]]]:
#         """
#         Perform INTELLIGENT hybrid search across both customers and policies
#         Returns top_k results from whichever source(s) have the best matches
        
#         Strategy:
#         1. Query both sources
#         2. Compare all results by score
#         3. Return top_k best matches regardless of source
        
#         Args:
#             query: Natural language query
#             top_k: Total number of results to return
#             vector_weight: Weight for semantic similarity
#             keyword_weight: Weight for keyword matching
            
#         Returns:
#             Dictionary with 'customers' and 'policies' keys containing results
#             Can be all from one source or mixed, depending on relevance
#         """
#         print(f"\n🔍 Unified Hybrid Search: '{query}'")
#         print(f"   Searching all sources to find top {top_k} most relevant results")
#         print("-" * 80)
        
#         # Get candidates from BOTH sources
#         # Use larger candidate pool to ensure good selection
#         candidate_multiplier = 2
#         customer_results = self.search_customers(query, top_k * candidate_multiplier, vector_weight, keyword_weight)
#         policy_results = self.search_policies(query, top_k * candidate_multiplier, vector_weight, keyword_weight)
        
#         # If one source has no results, return from the other
#         if not customer_results and not policy_results:
#             print("✓ No results found in either source")
#             return {'customers': [], 'policies': []}
        
#         if not customer_results:
#             print(f"✓ All results from POLICIES ({len(policy_results[:top_k])} policies)")
#             return {'customers': [], 'policies': policy_results[:top_k]}
        
#         if not policy_results:
#             print(f"✓ All results from CUSTOMERS ({len(customer_results[:top_k])} customers)")
#             return {'customers': customer_results[:top_k], 'policies': []}
        
#         # Both sources have results - normalize and compare
#         # Add source type
#         for customer in customer_results:
#             customer['result_type'] = 'customer'
        
#         for policy in policy_results:
#             policy['result_type'] = 'policy'
        
#         # Combine all results
#         all_results = customer_results + policy_results
        
#         # Sort by hybrid_score (lower is better for distance metrics)
#         # This naturally selects the best results regardless of source
#         all_results.sort(key=lambda x: x.get('hybrid_score', float('inf')))
        
#         # Take top_k best overall results
#         top_results = all_results[:top_k]
        
#         # Separate back into customers and policies
#         final_customers = [r for r in top_results if r['result_type'] == 'customer']
#         final_policies = [r for r in top_results if r['result_type'] == 'policy']
        
#         # Clean up temporary fields
#         for customer in final_customers:
#             customer.pop('result_type', None)
        
#         for policy in final_policies:
#             policy.pop('result_type', None)
        
#         # Log the distribution
#         if len(final_customers) == top_k:
#             print(f"✓ All results from CUSTOMERS ({len(final_customers)} customers - best matches)")
#         elif len(final_policies) == top_k:
#             print(f"✓ All results from POLICIES ({len(final_policies)} policies - best matches)")
#         else:
#             print(f"✓ Mixed results: {len(final_customers)} customers + {len(final_policies)} policies = {len(top_results)} total")
        
#         return {
#             'customers': final_customers,
#             'policies': final_policies
#         }
    
#     # ========================================================================
#     # INTELLIGENT QUERY ROUTING
#     # ========================================================================
    
#     def intelligent_search(
#         self, 
#         query: str, 
#         top_k: int = 5,
#         vector_weight: float = 0.6,
#         keyword_weight: float = 0.4
#     ) -> Dict[str, Any]:
#         """
#         Intelligently route query to appropriate search based on keywords
        
#         Args:
#             query: Natural language query
#             top_k: Number of results to return
#             vector_weight: Weight for semantic similarity
#             keyword_weight: Weight for keyword matching
            
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
#             result['customers'] = self.search_customers(query, top_k, vector_weight, keyword_weight)
#         elif has_policy_keywords and not has_customer_keywords:
#             result['search_type'] = 'policy'
#             result['policies'] = self.search_policies(query, top_k, vector_weight, keyword_weight)
#         else:
#             # Search both if ambiguous or contains both types
#             result['search_type'] = 'unified'
#             unified_results = self.unified_search(query, top_k, vector_weight, keyword_weight)
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
    
#     def rag_query(
#         self, 
#         query: str, 
#         max_results: int = 5,
#         vector_weight: float = 0.6,
#         keyword_weight: float = 0.4
#     ) -> Dict[str, Any]:
#         """
#         Complete RAG pipeline: Retrieves relevant context using hybrid search
        
#         Args:
#             query: Natural language question
#             max_results: Maximum results to retrieve
#             vector_weight: Weight for semantic similarity
#             keyword_weight: Weight for keyword matching
            
#         Returns:
#             Dictionary with retrieved context and metadata
#         """
#         print("\n" + "=" * 80)
#         print("RAG QUERY PIPELINE (HYBRID SEARCH)")
#         print("=" * 80)
#         print(f"Query: {query}")
#         print(f"Weights: Vector={vector_weight:.0%}, Keyword={keyword_weight:.0%}")
#         print("-" * 80)
        
#         # Step 1: Intelligent routing and retrieval
#         results = self.intelligent_search(query, max_results, vector_weight, keyword_weight)
        
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
#             'vector_weight': vector_weight,
#             'keyword_weight': keyword_weight,
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
#                 context_parts.append(f"   Hybrid Score: {customer.get('hybrid_score', 0):.4f}")
#                 if 'vector_score' in customer and 'keyword_score' in customer:
#                     context_parts.append(f"   (Vector: {customer['vector_score']:.4f}, Keyword: {customer['keyword_score']:.4f})")
        
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
#                 context_parts.append(f"   Hybrid Score: {policy.get('hybrid_score', 0):.4f}")
#                 if 'vector_score' in policy and 'keyword_score' in policy:
#                     context_parts.append(f"   (Vector: {policy['vector_score']:.4f}, Keyword: {policy['keyword_score']:.4f})")
        
#         return "\n".join(context_parts)
    
#     # ========================================================================
#     # DISPLAY METHODS
#     # ========================================================================
    
#     def display_results(self, results: Dict[str, Any]):
#         """Display search results in a formatted way"""
        
#         print("\n" + "=" * 80)
#         print("HYBRID SEARCH RESULTS")
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
#                 print(f"   Hybrid Score: {customer.get('hybrid_score', 0):.4f}")
#                 if 'vector_score' in customer and 'keyword_score' in customer:
#                     print(f"   (Vector: {customer['vector_score']:.4f}, Keyword: {customer['keyword_score']:.4f})")
        
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
#                 print(f"   Hybrid Score: {policy.get('hybrid_score', 0):.4f}")
#                 if 'vector_score' in policy and 'keyword_score' in policy:
#                     print(f"   (Vector: {policy['vector_score']:.4f}, Keyword: {policy['keyword_score']:.4f})")
        
#         print("\n" + "=" * 80 + "\n")
    
#     def display_rag_response(self, rag_response: Dict[str, Any]):
#         """Display complete RAG response"""
        
#         print("\n" + "=" * 80)
#         print("RAG RESPONSE (HYBRID SEARCH)")
#         print("=" * 80)
#         print(f"Query: {rag_response['query']}")
#         print(f"Search Type: {rag_response['search_type'].upper()}")
#         print(f"Customers Found: {rag_response['customers_found']}")
#         print(f"Policies Found: {rag_response['policies_found']}")
#         print(f"Weights: Vector={rag_response['vector_weight']:.0%}, Keyword={rag_response['keyword_weight']:.0%}")
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
#             filename = f"hybrid_rag_results_{timestamp}.json"
        
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
#     print("UNIFIED RAG SYSTEM - HYBRID SEARCH - MAIN MENU")
#     print("=" * 80)
#     print("1. Search Customers (Hybrid Search)")
#     print("2. Search Policies (Hybrid Search)")
#     print("3. Unified Search (Both Customers & Policies)")
#     print("4. Intelligent Search (Auto-route based on query)")
#     print("5. RAG Query (Complete RAG Pipeline)")
#     print("6. Get Customer with All Policies")
#     print("7. Custom Weights Search")
#     print("8. Exit")
#     print("-" * 80)


# def main():
#     """Main execution function"""
    
#     print("\n" + "=" * 80)
#     print("UNIFIED RAG SYSTEM WITH HYBRID SEARCH")
#     print("Hybrid Search (Semantic + Keyword) over Customer & Policy Data")
#     print("=" * 80)
    
#     try:
#         # Initialize RAG system
#         rag_system = UnifiedRAGHybridSystem()
#     except Exception as e:
#         print(f"\n❌ Error initializing RAG system: {e}")
#         print("\nPlease check your configuration and credentials.")
#         return
    
#     # Interactive menu loop
#     while True:
#         print_menu()
#         choice = input("\nEnter your choice (1-8): ").strip()
        
#         if choice == "1":
#             # Customer hybrid search
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
#                     print(f"   Hybrid: {customer.get('hybrid_score', 0):.4f} (V:{customer.get('vector_score', 0):.4f}, K:{customer.get('keyword_score', 0):.4f})\n")
                
#                 # Export option
#                 export = input("Export results to JSON? (y/n): ").strip().lower()
#                 if export == 'y':
#                     rag_system.export_results({'customers': results}, 
#                                              f"customer_hybrid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
#             else:
#                 print("❌ No results found")
        
#         elif choice == "2":
#             # Policy hybrid search
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
#                     print(f"   Hybrid: {policy.get('hybrid_score', 0):.4f} (V:{policy.get('vector_score', 0):.4f}, K:{policy.get('keyword_score', 0):.4f})\n")
                
#                 # Export option
#                 export = input("Export results to JSON? (y/n): ").strip().lower()
#                 if export == 'y':
#                     rag_system.export_results({'policies': results}, 
#                                              f"policy_hybrid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
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
#             # Custom weights search
#             query = input("\nEnter search query: ").strip()
#             if not query:
#                 print("❌ Query cannot be empty")
#                 continue
            
#             try:
#                 top_k = int(input("Number of results (default 5): ").strip() or "5")
                
#                 print("\nSet search weights (must sum to 1.0):")
#                 vector_weight = float(input("Vector weight (0.0-1.0, default 0.6): ").strip() or "0.6")
#                 keyword_weight = float(input("Keyword weight (0.0-1.0, default 0.4): ").strip() or "0.4")
                
#                 # Normalize weights
#                 total = vector_weight + keyword_weight
#                 if total != 1.0:
#                     print(f"\n⚠️  Normalizing weights (sum was {total:.2f})")
#                     vector_weight = vector_weight / total
#                     keyword_weight = keyword_weight / total
#                     print(f"   Adjusted: Vector={vector_weight:.2f}, Keyword={keyword_weight:.2f}")
                
#             except ValueError:
#                 print("❌ Invalid input, using defaults")
#                 top_k = 5
#                 vector_weight = 0.6
#                 keyword_weight = 0.4
            
#             # Ask which type of search
#             print("\nSearch in:")
#             print("1. Customers")
#             print("2. Policies")
#             print("3. Both (Unified)")
            
#             search_type = input("Your choice (1-3): ").strip()
            
#             if search_type == "1":
#                 results = rag_system.search_customers(query, top_k, vector_weight, keyword_weight)
#                 if results:
#                     rag_system.display_results({
#                         'query': query,
#                         'search_type': 'customer',
#                         'customers': results,
#                         'policies': []
#                     })
#             elif search_type == "2":
#                 results = rag_system.search_policies(query, top_k, vector_weight, keyword_weight)
#                 if results:
#                     rag_system.display_results({
#                         'query': query,
#                         'search_type': 'policy',
#                         'customers': [],
#                         'policies': results
#                     })
#             else:
#                 results = rag_system.unified_search(query, top_k, vector_weight, keyword_weight)
#                 rag_system.display_results({
#                     'query': query,
#                     'search_type': 'unified',
#                     'customers': results['customers'],
#                     'policies': results['policies']
#                 })
        
#         elif choice == "8":
#             print("\n" + "=" * 80)
#             print("Thank you for using Unified RAG Hybrid System!")
#             print("=" * 80 + "\n")
#             break
        
#         else:
#             print("\n❌ Invalid choice. Please select 1-8.")


# if __name__ == "__main__":
#     main()


class UnifiedRAGHybridSystem:
    """
    Unified RAG System that combines customer and policy retrieval
    using HYBRID SEARCH (semantic + keyword matching) over Azure Cosmos DB
    """
    
    def __init__(self):
        """Initialize all clients and connections"""
        print("\n" + "=" * 80)
        print("Initializing Unified RAG System with HYBRID SEARCH...")
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
        self.customer_cosmos_client = CosmosClient(COSMOS_hybrid_ENDPOINT, COSMOS_hybrid_KEY)
        self.customer_database = self.customer_cosmos_client.get_database_client(COSMOS_hybrid_DATABASE_NAME)
        self.customer_container = self.customer_database.get_container_client(COSMOS_hybrid_CONTAINER_NAME)
        print("✓ Customer database connected")
        
        # Initialize Policy Database
        print("\n📋 Connecting to Policy Database...")
        self.policy_cosmos_client = CosmosClient(COSMOS_pol_hybrid_ENDPOINT, COSMOS_pol_hybrid_KEY)
        self.policy_database = self.policy_cosmos_client.get_database_client(COSMOS_pol_hybrid_DATABASE_NAME)
        self.policy_container = self.policy_database.get_container_client(COSMOS_pol_hybrid_CONTAINER_NAME)
        print("✓ Policy database connected")
        
        print("\n" + "=" * 80)
        print("✓ Unified RAG System with Hybrid Search Ready!")
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
    # CUSTOMER HYBRID SEARCH
    # ========================================================================
    
    def search_customers(
        self, 
        query: str, 
        top_k: int = 5,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        HYBRID SEARCH over customer data (semantic + keyword matching)
        
        Args:
            query: Natural language query (e.g., "high income engineers in California")
            top_k: Number of results to return
            vector_weight: Weight for semantic similarity (default 0.6)
            keyword_weight: Weight for keyword matching (default 0.4)
            
        Returns:
            List of similar customers with hybrid scores
        """
        try:
            print(f"\n🔍 Hybrid searching customers for: '{query}'")
            print(f"   Weights: Vector={vector_weight:.0%}, Keyword={keyword_weight:.0%}")
            
            # Step 1: Get more candidates using vector search
            candidate_count = min(top_k * 3, 50)
            
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
                VectorDistance(c.embedding, @embedding) AS vector_score
            FROM c
            ORDER BY VectorDistance(c.embedding, @embedding)
            """
            
            parameters = [
                {"name": "@top_k", "value": candidate_count},
                {"name": "@embedding", "value": query_embedding}
            ]
            
            results = list(self.customer_container.query_items(
                query=sql_query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            if not results:
                return []
            
            # Step 2: Calculate keyword match scores
            query_terms = query.lower().split()
            
            for result in results:
                # Create searchable text from customer fields
                searchable_text = " ".join([
                    str(result.get('first_name', '')),
                    str(result.get('last_name', '')),
                    str(result.get('occupation', '')),
                    str(result.get('email', '')),
                    str(result.get('address', {}).get('city', '')),
                    str(result.get('address', {}).get('state', ''))
                ]).lower()
                
                # Calculate keyword match score (0-1)
                matches = sum(1 for term in query_terms if term in searchable_text)
                keyword_score = matches / len(query_terms) if query_terms else 0
                
                # Store raw scores (will be normalized later in unified_search)
                result['keyword_score'] = keyword_score
                result['raw_vector_score'] = result['vector_score']
                result['similarity_score'] = result['vector_score']  # For backward compatibility
            
            # For single-source search, calculate hybrid score directly
            # Normalize vector scores within this result set
            vector_scores = [r['vector_score'] for r in results]
            min_vec = min(vector_scores)
            max_vec = max(vector_scores)
            vec_range = max_vec - min_vec if max_vec > min_vec else 1
            
            for result in results:
                # Normalize vector score to 0-1 (0 is best)
                normalized_vec = (result['vector_score'] - min_vec) / vec_range
                # Calculate hybrid score
                hybrid_score = (vector_weight * normalized_vec) + (keyword_weight * result['keyword_score'])
                result['hybrid_score'] = hybrid_score
            
            # Step 3: Re-rank by hybrid score and return top_k
            results.sort(key=lambda x: x['hybrid_score'])
            results = results[:top_k]
            
            print(f"✓ Found {len(results)} customers")
            return results
            
        except Exception as e:
            print(f"❌ Error in customer hybrid search: {e}")
            return []
    
    # ========================================================================
    # POLICY HYBRID SEARCH
    # ========================================================================
    
    def search_policies(
        self, 
        query: str, 
        top_k: int = 5,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        HYBRID SEARCH over policy data (semantic + keyword matching)
        
        Args:
            query: Natural language query (e.g., "active auto insurance policies")
            top_k: Number of results to return
            vector_weight: Weight for semantic similarity (default 0.6)
            keyword_weight: Weight for keyword matching (default 0.4)
            
        Returns:
            List of similar policies with hybrid scores
        """
        try:
            print(f"\n🔍 Hybrid searching policies for: '{query}'")
            print(f"   Weights: Vector={vector_weight:.0%}, Keyword={keyword_weight:.0%}")
            
            # Step 1: Get more candidates using vector search
            candidate_count = min(top_k * 3, 50)
            
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
                VectorDistance(c.embedding, @embedding) AS vector_score
            FROM c
            ORDER BY VectorDistance(c.embedding, @embedding)
            """
            
            parameters = [
                {"name": "@top_k", "value": candidate_count},
                {"name": "@embedding", "value": query_embedding}
            ]
            
            results = list(self.policy_container.query_items(
                query=sql_query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            if not results:
                return []
            
            # Step 2: Calculate keyword match scores
            query_terms = query.lower().split()
            
            for result in results:
                # Create searchable text from policy fields
                searchable_text = " ".join([
                    str(result.get('policy_type', '')),
                    str(result.get('policy_number', '')),
                    str(result.get('status', '')),
                    str(result.get('payment_frequency', '')),
                    str(result.get('customer_id', '')),
                    "auto" if result.get('auto_renew') else "",
                    "renew" if result.get('auto_renew') else ""
                ]).lower()
                
                # Calculate keyword match score (0-1)
                matches = sum(1 for term in query_terms if term in searchable_text)
                keyword_score = matches / len(query_terms) if query_terms else 0
                
                # Store raw scores (will be normalized later in unified_search)
                result['keyword_score'] = keyword_score
                result['raw_vector_score'] = result['vector_score']
                result['similarity_score'] = result['vector_score']  # For backward compatibility
            
            # For single-source search, calculate hybrid score directly
            # Normalize vector scores within this result set
            vector_scores = [r['vector_score'] for r in results]
            min_vec = min(vector_scores)
            max_vec = max(vector_scores)
            vec_range = max_vec - min_vec if max_vec > min_vec else 1
            
            for result in results:
                # Normalize vector score to 0-1 (0 is best)
                normalized_vec = (result['vector_score'] - min_vec) / vec_range
                # Calculate hybrid score
                hybrid_score = (vector_weight * normalized_vec) + (keyword_weight * result['keyword_score'])
                result['hybrid_score'] = hybrid_score
            
            # Step 3: Re-rank by hybrid score and return top_k
            results.sort(key=lambda x: x['hybrid_score'])
            results = results[:top_k]
            
            print(f"✓ Found {len(results)} policies")
            return results
            
        except Exception as e:
            print(f"❌ Error in policy hybrid search: {e}")
            return []
    
    # ========================================================================
    # UNIFIED SEARCH - Searches both customers and policies
    # ========================================================================
    
    def unified_search(
        self, 
        query: str, 
        top_k: int = 5,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform INTELLIGENT hybrid search across both customers and policies
        Returns top_k results from whichever source(s) have the best matches
        
        CRITICAL: Normalizes scores ACROSS both sources for fair comparison
        
        Args:
            query: Natural language query
            top_k: Total number of results to return
            vector_weight: Weight for semantic similarity
            keyword_weight: Weight for keyword matching
            
        Returns:
            Dictionary with 'customers' and 'policies' keys containing results
            Can be all from one source or mixed, depending on relevance
        """
        print(f"\n🔍 Unified Hybrid Search: '{query}'")
        print("HI I AM QUERY OF UNIFIEDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD", query)
        print(f"   Searching all sources to find top {top_k} most relevant results")
        print("-" * 80)
        
        # Get candidates from BOTH sources
        candidate_multiplier = 2
        customer_results = self.search_customers(query, top_k * candidate_multiplier, vector_weight, keyword_weight)
        print("CUSTOMER RESULTS IN UNIFIED SEARCH--------------------------------------", customer_results)
        policy_results = self.search_policies(query, top_k * candidate_multiplier, vector_weight, keyword_weight)
        print("POLICY RESULTS IN UNIFIED SEARCH--------------------------------------", policy_results)
        
        # If one source has no results, return from the other
        if not customer_results and not policy_results:
            print("✓ No results found in either source")
            return {'customers': [], 'policies': []}
        
        if not customer_results:
            print(f"✓ All results from POLICIES ({len(policy_results[:top_k])} policies)")
            return {'customers': [], 'policies': policy_results[:top_k]}
        
        if not policy_results:
            print(f"✓ All results from CUSTOMERS ({len(customer_results[:top_k])} customers)")
            return {'customers': customer_results[:top_k], 'policies': []}
        
        # CRITICAL FIX: Normalize scores ACROSS both sources
        print(f"   Normalizing scores across both sources for fair comparison...")
        
        # Collect all raw vector scores from BOTH sources
        all_vector_scores = []
        all_vector_scores.extend([c.get('raw_vector_score', c.get('vector_score', 0)) for c in customer_results])
        all_vector_scores.extend([p.get('raw_vector_score', p.get('vector_score', 0)) for p in policy_results])
        
        # Find global min/max for vector scores
        global_min_vec = min(all_vector_scores)
        global_max_vec = max(all_vector_scores)
        global_vec_range = global_max_vec - global_min_vec if global_max_vec > global_min_vec else 1
        
        print(f"   Vector score range: {global_min_vec:.4f} to {global_max_vec:.4f}")
        
        # Re-calculate hybrid scores using GLOBAL normalization
        for customer in customer_results:
            raw_vec_score = customer.get('raw_vector_score', customer.get('vector_score', 0))
            normalized_vec = (raw_vec_score - global_min_vec) / global_vec_range
            keyword_score = customer.get('keyword_score', 0)
            
            # Re-calculate hybrid score with global normalization
            hybrid_score = (vector_weight * normalized_vec) + (keyword_weight * keyword_score)
            print("HYBRID SCORE CUSTOMER IN UNIFIED customer SEARCH--------------------------------------", hybrid_score)
            
            
            customer['normalized_vector_score'] = normalized_vec
            customer['global_hybrid_score'] = hybrid_score
            customer['result_type'] = 'customer'
        
        for policy in policy_results:
            raw_vec_score = policy.get('raw_vector_score', policy.get('vector_score', 0))
            normalized_vec = (raw_vec_score - global_min_vec) / global_vec_range
            keyword_score = policy.get('keyword_score', 0)
            
            # Re-calculate hybrid score with global normalization
            hybrid_score = (vector_weight * normalized_vec) + (keyword_weight * keyword_score)
            print("HYBRID SCORE POLICY IN UNIFIED policy SEARCH--------------------------------------", hybrid_score)
            
            policy['normalized_vector_score'] = normalized_vec
            policy['global_hybrid_score'] = hybrid_score
            policy['result_type'] = 'policy'
        
        # Combine all results
        all_results = customer_results + policy_results
        
        # Sort by global_hybrid_score (lower is better)
        # all_results.sort(key=lambda x: x.get('global_hybrid_score', float('inf')))
        
        # Take top_k best overall results
        top_results = all_results[:top_k]
        
        # Separate back into customers and policies
        final_customers = [r for r in top_results if r['result_type'] == 'customer']
        final_policies = [r for r in top_results if r['result_type'] == 'policy']
        
        # Clean up temporary fields and update hybrid_score for display
        for customer in final_customers:
            customer.pop('result_type', None)
            customer.pop('normalized_vector_score', None)
            customer.pop('raw_vector_score', None)
            # Use global score as the display score
            customer['hybrid_score'] = customer.pop('global_hybrid_score')
        
        for policy in final_policies:
            policy.pop('result_type', None)
            policy.pop('normalized_vector_score', None)
            policy.pop('raw_vector_score', None)
            # Use global score as the display score
            policy['hybrid_score'] = policy.pop('global_hybrid_score')
        
        # Log the distribution with score info
        if final_customers and final_policies:
            cust_scores = [c['hybrid_score'] for c in final_customers]
            pol_scores = [p['hybrid_score'] for p in final_policies]
            print(f"✓ Mixed results: {len(final_customers)} customers (scores: {min(cust_scores):.3f}-{max(cust_scores):.3f}) + " +
                  f"{len(final_policies)} policies (scores: {min(pol_scores):.3f}-{max(pol_scores):.3f}) = {len(top_results)} total")
        elif final_customers:
            cust_scores = [c['hybrid_score'] for c in final_customers]
            print(f"✓ All results from CUSTOMERS ({len(final_customers)} customers - best matches, scores: {min(cust_scores):.3f}-{max(cust_scores):.3f})")
        elif final_policies:
            pol_scores = [p['hybrid_score'] for p in final_policies]
            print(f"✓ All results from POLICIES ({len(final_policies)} policies - best matches, scores: {min(pol_scores):.3f}-{max(pol_scores):.3f})")
        
        return {
            'customers': final_customers,
            'policies': final_policies
        }
    
    # Note: Rest of the methods remain the same - not including them to keep file size manageable
    # Copy them from the original file (intelligent_search, get_customer_with_policies, rag_query, display methods, etc.)


if __name__ == "__main__":
    # Test the fixed system
    print("Testing fixed unified search with proper normalization")
    rag = UnifiedRAGHybridSystem()
    # Test query
    result = rag.unified_search("active auto insurance", top_k=5)
    print("I AM RESULTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",result)
    print(f"\nCustomers: {len(result['customers'])}")
    print(f"Policies: {len(result['policies'])}")
