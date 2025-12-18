"""
example_usage_hybrid.py - Example usage of the Unified RAG Hybrid System

This script demonstrates various ways to use the hybrid RAG system programmatically
Hybrid Search = Semantic (Vector) Search + Keyword Matching
"""

from unified_rag_hybrid_system import UnifiedRAGHybridSystem
from datetime import datetime


def example_customer_hybrid_search():
    """Example: Search for customers using hybrid search"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Customer Hybrid Search")
    print("=" * 80)
    
    rag = UnifiedRAGHybridSystem()
    
    # Search for high-income engineers
    query = "high income software engineers in California"
    results = rag.search_customers(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"Results found: {len(results)}\n")
    
    for i, customer in enumerate(results, 1):
        print(f"{i}. {customer['first_name']} {customer['last_name']}")
        print(f"   Occupation: {customer['occupation']}")
        print(f"   Income: ${customer['annual_income']:,}")
        print(f"   Hybrid Score: {customer.get('hybrid_score', 0):.4f}")
        if 'vector_score' in customer and 'keyword_score' in customer:
            print(f"   (Vector: {customer['vector_score']:.4f}, Keyword: {customer['keyword_score']:.4f})")
        print()


def example_policy_hybrid_search():
    """Example: Search for policies using hybrid search"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Policy Hybrid Search")
    print("=" * 80)
    
    rag = UnifiedRAGHybridSystem()
    
    # Search for active auto insurance
    query = "active auto insurance with high coverage"
    results = rag.search_policies(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"Results found: {len(results)}\n")
    
    for i, policy in enumerate(results, 1):
        print(f"{i}. {policy['policy_type']} - {policy['policy_number']}")
        print(f"   Status: {policy['status']}")
        print(f"   Premium: ${policy['annual_premium']:,.2f}")
        print(f"   Coverage: ${policy['coverage_amount']:,}")
        print(f"   Hybrid Score: {policy.get('hybrid_score', 0):.4f}")
        if 'vector_score' in policy and 'keyword_score' in policy:
            print(f"   (Vector: {policy['vector_score']:.4f}, Keyword: {policy['keyword_score']:.4f})")
        print()


def example_unified_hybrid_search():
    """Example: Hybrid search across both customers and policies"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Unified Hybrid Search (Customers + Policies)")
    print("=" * 80)
    
    rag = UnifiedRAGHybridSystem()
    
    # Search across both databases
    query = "premium customers with life insurance"
    results = rag.unified_search(query, top_k=2)
    
    print(f"\nQuery: {query}")
    print(f"Customers found: {len(results['customers'])}")
    print(f"Policies found: {len(results['policies'])}\n")
    
    # Display customers
    if results['customers']:
        print("Top Customers:")
        for i, customer in enumerate(results['customers'], 1):
            print(f"  {i}. {customer['first_name']} {customer['last_name']} - {customer['occupation']}")
            print(f"     Hybrid: {customer.get('hybrid_score', 0):.4f}")
    
    # Display policies
    if results['policies']:
        print("\nTop Policies:")
        for i, policy in enumerate(results['policies'], 1):
            print(f"  {i}. {policy['policy_type']} - ${policy['annual_premium']:,.2f}")
            print(f"     Hybrid: {policy.get('hybrid_score', 0):.4f}")


def example_intelligent_hybrid_search():
    """Example: Intelligent search with auto-routing and hybrid search"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Intelligent Hybrid Search (Auto-routing)")
    print("=" * 80)
    
    rag = UnifiedRAGHybridSystem()
    
    # Test different query types
    queries = [
        "engineers with good credit scores",  # Should route to customers
        "expired insurance policies",          # Should route to policies
        "high income customers with insurance" # Should search both
    ]
    
    for query in queries:
        results = rag.intelligent_search(query, top_k=2)
        print(f"\nQuery: {query}")
        print(f"Routed to: {results['search_type'].upper()} search")
        print(f"Customers: {len(results['customers'])}, Policies: {len(results['policies'])}")


def example_hybrid_rag_query():
    """Example: Complete RAG query with hybrid search"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Complete Hybrid RAG Query")
    print("=" * 80)
    
    rag = UnifiedRAGHybridSystem()
    
    # Execute RAG query
    query = "find wealthy customers interested in life insurance"
    rag_response = rag.rag_query(query, max_results=3)
    
    # Display RAG response
    print(f"\nQuery: {rag_response['query']}")
    print(f"Search Type: {rag_response['search_type']}")
    print(f"Weights: Vector={rag_response['vector_weight']:.0%}, Keyword={rag_response['keyword_weight']:.0%}")
    print(f"Retrieved: {rag_response['customers_found']} customers, {rag_response['policies_found']} policies")
    print(f"\nFormatted Context:")
    print("-" * 80)
    print(rag_response['context'])
    
    # This context can now be sent to an LLM for answer generation


def example_custom_weights():
    """Example: Custom weight configuration for hybrid search"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Custom Weights Hybrid Search")
    print("=" * 80)
    
    rag = UnifiedRAGHybridSystem()
    
    query = "software engineer"
    
    # Test 1: More semantic (good for conceptual queries)
    print("\n--- Test 1: More Semantic (80% Vector, 20% Keyword) ---")
    results1 = rag.search_customers(query, top_k=3, vector_weight=0.8, keyword_weight=0.2)
    print(f"Found {len(results1)} results")
    for i, customer in enumerate(results1, 1):
        print(f"{i}. {customer['first_name']} {customer['last_name']} - {customer['occupation']}")
        print(f"   Hybrid: {customer.get('hybrid_score', 0):.4f}")
    
    # Test 2: More keyword-based (good for specific terms)
    print("\n--- Test 2: More Keyword (20% Vector, 80% Keyword) ---")
    results2 = rag.search_customers(query, top_k=3, vector_weight=0.2, keyword_weight=0.8)
    print(f"Found {len(results2)} results")
    for i, customer in enumerate(results2, 1):
        print(f"{i}. {customer['first_name']} {customer['last_name']} - {customer['occupation']}")
        print(f"   Hybrid: {customer.get('hybrid_score', 0):.4f}")
    
    # Test 3: Balanced (default)
    print("\n--- Test 3: Balanced (60% Vector, 40% Keyword) ---")
    results3 = rag.search_customers(query, top_k=3, vector_weight=0.6, keyword_weight=0.4)
    print(f"Found {len(results3)} results")
    for i, customer in enumerate(results3, 1):
        print(f"{i}. {customer['first_name']} {customer['last_name']} - {customer['occupation']}")
        print(f"   Hybrid: {customer.get('hybrid_score', 0):.4f}")


def example_customer_with_policies():
    """Example: Get customer with all their policies"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Customer Profile with Policies")
    print("=" * 80)
    
    rag = UnifiedRAGHybridSystem()
    
    # Get customer with all policies
    customer_id = "CUST-0001"  # Replace with actual customer ID
    result = rag.get_customer_with_policies(customer_id)
    
    if result:
        customer = result['customer']
        policies = result['policies']
        
        print(f"\nCustomer: {customer['first_name']} {customer['last_name']}")
        print(f"Income: ${customer['annual_income']:,}")
        print(f"Total Policies: {result['policy_count']}\n")
        
        print("Policy Summary:")
        for i, policy in enumerate(policies, 1):
            print(f"  {i}. {policy['policy_type']} - {policy['status']}")
            print(f"     Premium: ${policy['annual_premium']:,.2f}")
    else:
        print(f"Customer {customer_id} not found")


def example_export_hybrid_results():
    """Example: Export hybrid search results to JSON"""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Export Hybrid Results")
    print("=" * 80)
    
    rag = UnifiedRAGHybridSystem()
    
    # Perform search
    query = "high value customers"
    results = rag.search_customers(query, top_k=5)
    
    # Export to JSON
    filename = f"customer_hybrid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    rag.export_results({'query': query, 'customers': results}, filename)
    print(f"\nResults exported to: {filename}")


def example_batch_hybrid_queries():
    """Example: Process multiple queries efficiently with hybrid search"""
    print("\n" + "=" * 80)
    print("EXAMPLE 9: Batch Hybrid Query Processing")
    print("=" * 80)
    
    rag = UnifiedRAGHybridSystem()
    
    # Multiple queries
    queries = [
        "software engineers",
        "healthcare professionals",
        "business owners"
    ]
    
    print("\nProcessing batch queries with hybrid search...\n")
    
    all_results = {}
    for query in queries:
        results = rag.search_customers(query, top_k=2)
        all_results[query] = results
        print(f"'{query}': {len(results)} results")
        if results:
            print(f"  Top result: {results[0]['first_name']} {results[0]['last_name']}")
            print(f"  Hybrid Score: {results[0].get('hybrid_score', 0):.4f}")
    
    # Export all results
    filename = f"batch_hybrid_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    rag.export_results(all_results, filename)
    print(f"\nBatch results exported to: {filename}")


def example_filtered_hybrid_search():
    """Example: Combine hybrid search with post-filtering"""
    print("\n" + "=" * 80)
    print("EXAMPLE 10: Hybrid Search + Filtering")
    print("=" * 80)
    
    rag = UnifiedRAGHybridSystem()
    
    # Search customers
    query = "experienced professionals"
    results = rag.search_customers(query, top_k=10)
    
    # Filter results by income (post-retrieval filtering)
    high_income = [r for r in results if r['annual_income'] > 100000]
    
    print(f"\nQuery: {query}")
    print(f"Total results: {len(results)}")
    print(f"High income (>100k): {len(high_income)}\n")
    
    for i, customer in enumerate(high_income[:3], 1):
        print(f"{i}. {customer['first_name']} {customer['last_name']}")
        print(f"   Income: ${customer['annual_income']:,}")
        print(f"   Hybrid Score: {customer.get('hybrid_score', 0):.4f}")
        if 'vector_score' in customer and 'keyword_score' in customer:
            print(f"   (Vector: {customer['vector_score']:.4f}, Keyword: {customer['keyword_score']:.4f})")
        print()


def example_hybrid_score_analysis():
    """Example: Analyze hybrid scores and component scores"""
    print("\n" + "=" * 80)
    print("EXAMPLE 11: Hybrid Score Analysis")
    print("=" * 80)
    
    rag = UnifiedRAGHybridSystem()
    
    # Search with hybrid
    query = "engineer in technology"
    results = rag.search_customers(query, top_k=5)
    
    print(f"\nQuery: '{query}'")
    print(f"Results: {len(results)}\n")
    
    print(f"{'Rank':<6} {'Name':<25} {'Occupation':<25} {'Hybrid':<8} {'Vector':<8} {'Keyword':<8}")
    print("-" * 95)
    
    for i, customer in enumerate(results, 1):
        name = f"{customer['first_name']} {customer['last_name']}"[:24]
        occ = customer['occupation'][:24]
        hybrid = customer.get('hybrid_score', 0)
        vector = customer.get('vector_score', 0)
        keyword = customer.get('keyword_score', 0)
        
        print(f"{i:<6} {name:<25} {occ:<25} {hybrid:<8.4f} {vector:<8.4f} {keyword:<8.4f}")
    
    print("-" * 95)
    print("\nScore Interpretation:")
    print("  • Hybrid Score: Combined weighted score (default: 0.6 × Vector + 0.4 × Keyword)")
    print("  • Vector Score: Semantic similarity (meaning-based)")
    print("  • Keyword Score: Exact keyword matching")


def example_policy_type_comparison():
    """Example: Compare different policy types using hybrid search"""
    print("\n" + "=" * 80)
    print("EXAMPLE 12: Policy Type Comparison")
    print("=" * 80)
    
    rag = UnifiedRAGHybridSystem()
    
    policy_types = ["auto insurance", "life insurance", "health insurance"]
    
    print("\nComparing policy types with hybrid search:\n")
    
    for policy_type in policy_types:
        results = rag.search_policies(policy_type, top_k=3, vector_weight=0.3, keyword_weight=0.7)
        print(f"--- {policy_type.upper()} ---")
        print(f"Found: {len(results)} policies")
        if results:
            for i, policy in enumerate(results, 1):
                print(f"  {i}. {policy['policy_type']} - {policy['status']}")
                print(f"     Premium: ${policy['annual_premium']:,.2f}")
                print(f"     Hybrid: {policy.get('hybrid_score', 0):.4f} (K: {policy.get('keyword_score', 0):.4f})")
        print()


def example_semantic_vs_keyword():
    """Example: Compare semantic-heavy vs keyword-heavy search"""
    print("\n" + "=" * 80)
    print("EXAMPLE 13: Semantic vs Keyword Comparison")
    print("=" * 80)
    
    rag = UnifiedRAGHybridSystem()
    
    # Query with specific term
    query = "software engineer"
    
    print(f"\nQuery: '{query}'\n")
    
    # Semantic-heavy search (90% vector, 10% keyword)
    print("--- SEMANTIC-HEAVY (90% Vector, 10% Keyword) ---")
    semantic_results = rag.search_customers(query, top_k=3, vector_weight=0.9, keyword_weight=0.1)
    for i, customer in enumerate(semantic_results, 1):
        print(f"{i}. {customer['first_name']} {customer['last_name']} - {customer['occupation']}")
        print(f"   Hybrid: {customer.get('hybrid_score', 0):.4f}")
    
    print("\n--- KEYWORD-HEAVY (10% Vector, 90% Keyword) ---")
    keyword_results = rag.search_customers(query, top_k=3, vector_weight=0.1, keyword_weight=0.9)
    for i, customer in enumerate(keyword_results, 1):
        print(f"{i}. {customer['first_name']} {customer['last_name']} - {customer['occupation']}")
        print(f"   Hybrid: {customer.get('hybrid_score', 0):.4f}")
    
    print("\n--- BALANCED (60% Vector, 40% Keyword) ---")
    balanced_results = rag.search_customers(query, top_k=3, vector_weight=0.6, keyword_weight=0.4)
    for i, customer in enumerate(balanced_results, 1):
        print(f"{i}. {customer['first_name']} {customer['last_name']} - {customer['occupation']}")
        print(f"   Hybrid: {customer.get('hybrid_score', 0):.4f}")


def example_conceptual_query():
    """Example: Conceptual query that benefits from semantic search"""
    print("\n" + "=" * 80)
    print("EXAMPLE 14: Conceptual Query (More Semantic)")
    print("=" * 80)
    
    rag = UnifiedRAGHybridSystem()
    
    # Conceptual query - semantic search should help
    query = "wealthy professionals in healthcare"
    
    # Use more semantic weight for conceptual queries
    results = rag.search_customers(query, top_k=5, vector_weight=0.8, keyword_weight=0.2)
    
    print(f"\nQuery: '{query}'")
    print("Strategy: More semantic (80% vector) for conceptual understanding\n")
    
    for i, customer in enumerate(results, 1):
        print(f"{i}. {customer['first_name']} {customer['last_name']}")
        print(f"   Occupation: {customer['occupation']}")
        print(f"   Income: ${customer['annual_income']:,}")
        print(f"   Hybrid: {customer.get('hybrid_score', 0):.4f} (V:{customer.get('vector_score', 0):.4f}, K:{customer.get('keyword_score', 0):.4f})")
        print()


def example_exact_term_query():
    """Example: Exact term query that benefits from keyword matching"""
    print("\n" + "=" * 80)
    print("EXAMPLE 15: Exact Term Query (More Keyword)")
    print("=" * 80)
    
    rag = UnifiedRAGHybridSystem()
    
    # Exact term query - keyword matching should help
    query = "active AUTO policy"
    
    # Use more keyword weight for exact terms
    results = rag.search_policies(query, top_k=5, vector_weight=0.3, keyword_weight=0.7)
    
    print(f"\nQuery: '{query}'")
    print("Strategy: More keyword (70% keyword) for exact term matching\n")
    
    for i, policy in enumerate(results, 1):
        print(f"{i}. {policy['policy_type']} - {policy['policy_number']}")
        print(f"   Status: {policy['status']}")
        print(f"   Hybrid: {policy.get('hybrid_score', 0):.4f} (V:{policy.get('vector_score', 0):.4f}, K:{policy.get('keyword_score', 0):.4f})")
        print()


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("UNIFIED RAG HYBRID SYSTEM - USAGE EXAMPLES")
    print("=" * 80)
    
    examples = [
        ("Customer Hybrid Search", example_customer_hybrid_search),
        ("Policy Hybrid Search", example_policy_hybrid_search),
        ("Unified Hybrid Search", example_unified_hybrid_search),
        ("Intelligent Hybrid Search", example_intelligent_hybrid_search),
        ("Hybrid RAG Query", example_hybrid_rag_query),
        ("Custom Weights", example_custom_weights),
        ("Customer with Policies", example_customer_with_policies),
        ("Export Hybrid Results", example_export_hybrid_results),
        ("Batch Hybrid Queries", example_batch_hybrid_queries),
        ("Filtered Hybrid Search", example_filtered_hybrid_search),
        ("Hybrid Score Analysis", example_hybrid_score_analysis),
        ("Policy Type Comparison", example_policy_type_comparison),
        ("Semantic vs Keyword", example_semantic_vs_keyword),
        ("Conceptual Query", example_conceptual_query),
        ("Exact Term Query", example_exact_term_query)
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    print("0. Run all examples")
    print("q. Quit")
    
    while True:
        choice = input("\nSelect example to run (0-15, q to quit): ").strip().lower()
        
        if choice == 'q':
            print("\nGoodbye!")
            break
        
        try:
            choice_num = int(choice)
            if choice_num == 0:
                # Run all examples
                for name, example_func in examples:
                    try:
                        example_func()
                        input("\nPress Enter to continue...")
                    except Exception as e:
                        print(f"\n❌ Error in {name}: {e}")
                        input("\nPress Enter to continue...")
                break
            elif 1 <= choice_num <= len(examples):
                # Run selected example
                name, example_func = examples[choice_num - 1]
                try:
                    example_func()
                except Exception as e:
                    print(f"\n❌ Error: {e}")
            else:
                print("Invalid choice. Please select 0-15.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")


if __name__ == "__main__":
    main()
