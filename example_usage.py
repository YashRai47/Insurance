"""
example_usage.py - Example usage of the Unified RAG System

This script demonstrates various ways to use the RAG system programmatically
"""

from unified_rag_system import UnifiedRAGSystem


def example_customer_search():
    """Example: Search for customers using vector search"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Customer Vector Search")
    print("=" * 80)
    
    rag = UnifiedRAGSystem()
    
    # Search for high-income engineers
    query = "high income software engineers in California"
    results = rag.search_customers(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"Results found: {len(results)}\n")
    
    for i, customer in enumerate(results, 1):
        print(f"{i}. {customer['first_name']} {customer['last_name']}")
        print(f"   Occupation: {customer['occupation']}")
        print(f"   Income: ${customer['annual_income']:,}")
        print(f"   Similarity: {customer['similarity_score']:.4f}\n")


def example_policy_search():
    """Example: Search for policies using vector search"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Policy Vector Search")
    print("=" * 80)
    
    rag = UnifiedRAGSystem()
    
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
        print(f"   Similarity: {policy['similarity_score']:.4f}\n")


def example_unified_search():
    """Example: Search both customers and policies"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Unified Search (Customers + Policies)")
    print("=" * 80)
    
    rag = UnifiedRAGSystem()
    
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
    
    # Display policies
    if results['policies']:
        print("\nTop Policies:")
        for i, policy in enumerate(results['policies'], 1):
            print(f"  {i}. {policy['policy_type']} - ${policy['annual_premium']:,.2f}")


def example_intelligent_search():
    """Example: Intelligent search with auto-routing"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Intelligent Search (Auto-routing)")
    print("=" * 80)
    
    rag = UnifiedRAGSystem()
    
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


def example_rag_query():
    """Example: Complete RAG query with formatted context"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Complete RAG Query")
    print("=" * 80)
    
    rag = UnifiedRAGSystem()
    
    # Execute RAG query
    query = "find wealthy customers interested in life insurance"
    rag_response = rag.rag_query(query, max_results=3)
    
    # Display RAG response
    print(f"\nQuery: {rag_response['query']}")
    print(f"Search Type: {rag_response['search_type']}")
    print(f"Retrieved: {rag_response['customers_found']} customers, {rag_response['policies_found']} policies")
    print(f"\nFormatted Context:")
    print("-" * 80)
    print(rag_response['context'])
    
    # This context can now be sent to an LLM for answer generation


def example_customer_with_policies():
    """Example: Get customer with all their policies"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Customer Profile with Policies")
    print("=" * 80)
    
    rag = UnifiedRAGSystem()
    
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


def example_export_results():
    """Example: Export search results to JSON"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Export Results")
    print("=" * 80)
    
    rag = UnifiedRAGSystem()
    
    # Perform search
    query = "high value customers"
    results = rag.search_customers(query, top_k=5)
    
    # Export to JSON
    filename = "customer_search_results.json"
    rag.export_results({'query': query, 'customers': results}, filename)
    print(f"\nResults exported to: {filename}")


def example_batch_queries():
    """Example: Process multiple queries efficiently"""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Batch Query Processing")
    print("=" * 80)
    
    rag = UnifiedRAGSystem()
    
    # Multiple queries
    queries = [
        "software engineers",
        "healthcare professionals",
        "business owners"
    ]
    
    print("\nProcessing batch queries...\n")
    
    all_results = {}
    for query in queries:
        results = rag.search_customers(query, top_k=2)
        all_results[query] = results
        print(f"'{query}': {len(results)} results")
    
    # Export all results
    rag.export_results(all_results, "batch_results.json")
    print("\nBatch results exported to: batch_results.json")


def example_filtered_search():
    """Example: Combine vector search with filters"""
    print("\n" + "=" * 80)
    print("EXAMPLE 9: Vector Search + Filtering")
    print("=" * 80)
    
    rag = UnifiedRAGSystem()
    
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
        print(f"   Similarity: {customer['similarity_score']:.4f}\n")


def example_similarity_threshold():
    """Example: Filter by similarity threshold"""
    print("\n" + "=" * 80)
    print("EXAMPLE 10: Similarity Threshold Filtering")
    print("=" * 80)
    
    rag = UnifiedRAGSystem()
    
    # Search with high similarity threshold
    query = "insurance agents"
    results = rag.search_customers(query, top_k=10)
    
    # Filter by similarity score
    similarity_threshold = 0.80
    high_similarity = [r for r in results if r['similarity_score'] >= similarity_threshold]
    
    print(f"\nQuery: {query}")
    print(f"Total results: {len(results)}")
    print(f"High similarity (>={similarity_threshold}): {len(high_similarity)}\n")
    
    for i, customer in enumerate(high_similarity, 1):
        print(f"{i}. {customer['first_name']} {customer['last_name']}")
        print(f"   Occupation: {customer['occupation']}")
        print(f"   Similarity: {customer['similarity_score']:.4f}\n")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("UNIFIED RAG SYSTEM - USAGE EXAMPLES")
    print("=" * 80)
    
    examples = [
        ("Customer Search", example_customer_search),
        ("Policy Search", example_policy_search),
        ("Unified Search", example_unified_search),
        ("Intelligent Search", example_intelligent_search),
        ("RAG Query", example_rag_query),
        ("Customer with Policies", example_customer_with_policies),
        ("Export Results", example_export_results),
        ("Batch Queries", example_batch_queries),
        ("Filtered Search", example_filtered_search),
        ("Similarity Threshold", example_similarity_threshold)
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    print("0. Run all examples")
    print("q. Quit")
    
    while True:
        choice = input("\nSelect example to run (0-10, q to quit): ").strip().lower()
        
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
                print("Invalid choice. Please select 0-10.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")


if __name__ == "__main__":
    main()
