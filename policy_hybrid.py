"""
policy_hybrid.py - Retrieve policy data from Azure Cosmos DB using HYBRID SEARCH
Combines vector search (semantic) + keyword matching for optimal results
"""

import os
import json
from azure.cosmos import CosmosClient
from openai import AzureOpenAI
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import os

load_dotenv()  
# Configuration
COSMOS_pol_hybrid_ENDPOINT = os.getenv("COSMOS_pol_hybrid_ENDPOINT")
COSMOS_pol_hybrid_KEY = os.getenv("COSMOS_pol_hybrid_KEY")
COSMOS_pol_hybrid_DATABASE_NAME = os.getenv("COSMOS_pol_hybrid_DATABASE_NAME")
COSMOS_pol_hybrid_CONTAINER_NAME = os.getenv("COSMOS_pol_hybrid_CONTAINER_NAME")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = "text-embedding-3-large"
AZURE_OPENAI_API_VERSION = "2024-02-01"

class PolicyRetriever:
    def __init__(self):
        """Initialize Cosmos DB and Azure OpenAI clients"""
        print("Initializing policy retrieval client with HYBRID SEARCH...")

        # Initialize Cosmos DB COSMOS_pol_hybrid_ENDPOINT
        self.cosmos_client = CosmosClient(COSMOS_pol_hybrid_ENDPOINT, COSMOS_pol_hybrid_KEY)
        self.database = self.cosmos_client.get_database_client(COSMOS_pol_hybrid_DATABASE_NAME)
        self.container = self.database.get_container_client(COSMOS_pol_hybrid_CONTAINER_NAME)

        # Initialize Azure OpenAI client for vector search
        self.openai_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )

        print("✓ Client initialized successfully")
        print("✓ Hybrid Search enabled (Vector + Keyword)\n")

    def get_policies_by_customer_id(self, customer_id: str) -> List[Dict[str, Any]]:
        """Retrieve all policies for a specific customer by customer_id"""

        try:
            query = "SELECT * FROM c WHERE c.customer_id = @customer_id"
            parameters = [{"name": "@customer_id", "value": customer_id}]

            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))

            # Remove embedding from response (too large and not needed for display)
            for item in items:
                if 'embedding' in item:
                    del item['embedding']

            return items

        except Exception as e:
            print(f"Error retrieving policies: {e}")
            return []

    def get_policy_by_id(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete policy details by policy_id"""

        try:
            query = "SELECT * FROM c WHERE c.policy_id = @policy_id"
            parameters = [{"name": "@policy_id", "value": policy_id}]

            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))

            if items:
                policy = items[0]
                # Remove embedding from response
                if 'embedding' in policy:
                    del policy['embedding']
                return policy
            else:
                return None

        except Exception as e:
            print(f"Error retrieving policy: {e}")
            return None

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for search query"""

        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=AZURE_OPENAI_DEPLOYMENT
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    def hybrid_search(
        self, 
        query_text: str, 
        top_k: int = 5,
        vector_weight: float = 0.6,
        fulltext_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        HYBRID SEARCH: Combines vector search (semantic) + keyword matching
        
        Implementation: Uses post-processing approach for maximum compatibility
        1. Performs vector search to get candidates
        2. Calculates keyword match scores
        3. Combines scores with weights
        
        Args:
            query_text: Search query
            top_k: Number of results to return
            vector_weight: Weight for vector similarity (0-1). Default 0.6
            fulltext_weight: Weight for keyword matching (0-1). Default 0.4
            
        Returns:
            List of policies with combined hybrid scores
        """

        try:
            print(f"🔍 Running HYBRID SEARCH (Vector: {vector_weight:.0%}, Keyword: {fulltext_weight:.0%})")
            
            # Step 1: Get more candidates using vector search
            # We fetch 3x the requested results to have a good pool for re-ranking
            candidate_count = min(top_k * 3, 50)
            
            # Generate embedding for query
            query_embedding = self.generate_embedding(query_text)

            # Vector search query
            query = """
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

            results = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))

            if not results:
                return []
            
            # Step 2: Calculate keyword match scores for each result
            query_terms = query_text.lower().split()
            
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
                
                # Normalize vector score to 0-1 range (vector_score is already 0-1)
                vector_score_normalized = result['vector_score']
                
                # Calculate hybrid score
                hybrid_score = (vector_weight * vector_score_normalized) + (fulltext_weight * keyword_score)
                
                # Add scores to result
                result['fulltext_score'] = keyword_score
                result['hybrid_score'] = hybrid_score
            
            # Step 3: Re-rank by hybrid score and return top_k
            results.sort(key=lambda x: x['hybrid_score'], reverse=False)  # Lower is better for distance
            results = results[:top_k]
            
            print(f"✓ Found {len(results)} results using hybrid search")
            return results

        except Exception as e:
            print(f"❌ Error in hybrid search: {e}")
            print(f"   Attempting fallback to vector-only search...")
            # Fallback to vector search if hybrid fails
            return self.vector_search_fallback(query_text, top_k)

    def vector_search_fallback(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Fallback to pure vector search if hybrid search fails
        """
        
        try:
            print(f"⚠️  Using vector-only search as fallback")
            
            # Generate embedding for query
            query_embedding = self.generate_embedding(query_text)

            # Perform vector search
            query = """
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
                VectorDistance(c.embedding, @embedding) AS vector_score,
                VectorDistance(c.embedding, @embedding) AS hybrid_score
            FROM c
            ORDER BY VectorDistance(c.embedding, @embedding)
            """

            parameters = [
                {"name": "@top_k", "value": top_k},
                {"name": "@embedding", "value": query_embedding}
            ]

            results = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))

            return results

        except Exception as e:
            print(f"❌ Error in vector search fallback: {e}")
            return []

    # Alias for backward compatibility
    def vector_search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Backward compatibility wrapper - now uses hybrid search by default
        """
        return self.hybrid_search(query_text, top_k)

    def search_by_criteria(
            self,
            policy_type: Optional[str] = None,
            status: Optional[str] = None,
            min_premium: Optional[float] = None,
            max_premium: Optional[float] = None,
            min_coverage: Optional[int] = None,
            payment_frequency: Optional[str] = None,
            auto_renew: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Search policies by various criteria"""

        try:
            conditions = []
            parameters = []

            if policy_type:
                conditions.append("c.policy_type = @policy_type")
                parameters.append({"name": "@policy_type", "value": policy_type})

            if status:
                conditions.append("c.status = @status")
                parameters.append({"name": "@status", "value": status})

            if min_premium is not None:
                conditions.append("c.annual_premium >= @min_premium")
                parameters.append({"name": "@min_premium", "value": min_premium})

            if max_premium is not None:
                conditions.append("c.annual_premium <= @max_premium")
                parameters.append({"name": "@max_premium", "value": max_premium})

            if min_coverage is not None:
                conditions.append("c.coverage_amount >= @min_coverage")
                parameters.append({"name": "@min_coverage", "value": min_coverage})

            if payment_frequency:
                conditions.append("c.payment_frequency = @payment_frequency")
                parameters.append({"name": "@payment_frequency", "value": payment_frequency})

            if auto_renew is not None:
                conditions.append("c.auto_renew = @auto_renew")
                parameters.append({"name": "@auto_renew", "value": auto_renew})

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"SELECT * FROM c WHERE {where_clause}"

            results = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))

            # Remove embeddings from results
            for result in results:
                if 'embedding' in result:
                    del result['embedding']

            return results

        except Exception as e:
            print(f"Error searching by criteria: {e}")
            return []

    def hybrid_search_with_filters(
        self,
        query_text: str,
        top_k: int = 5,
        vector_weight: float = 0.6,
        fulltext_weight: float = 0.4,
        policy_type: Optional[str] = None,
        status: Optional[str] = None,
        min_premium: Optional[float] = None,
        max_premium: Optional[float] = None,
        min_coverage: Optional[int] = None,
        payment_frequency: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        ADVANCED: Hybrid search combined with filters
        Uses post-processing approach for maximum compatibility
        
        Args:
            query_text: Search query
            top_k: Number of results
            vector_weight: Weight for vector similarity
            fulltext_weight: Weight for keyword matching
            policy_type: Policy type filter
            status: Status filter
            min_premium: Minimum premium filter
            max_premium: Maximum premium filter
            min_coverage: Minimum coverage filter
            payment_frequency: Payment frequency filter
            
        Returns:
            Filtered hybrid search results
        """
        
        try:
            print(f"🔍 Running HYBRID SEARCH with FILTERS")
            
            # Build filter conditions for the SQL query
            conditions = ["1=1"]  # Always true base condition
            parameters = []
            
            # Generate embedding
            query_embedding = self.generate_embedding(query_text)
            parameters.append({"name": "@embedding", "value": query_embedding})
            
            # Get more candidates for filtering
            candidate_count = min(top_k * 5, 100)
            parameters.append({"name": "@top_k", "value": candidate_count})
            
            # Add filters to SQL query
            if policy_type:
                conditions.append("c.policy_type = @policy_type")
                parameters.append({"name": "@policy_type", "value": policy_type})
            
            if status:
                conditions.append("c.status = @status")
                parameters.append({"name": "@status", "value": status})
            
            if min_premium is not None:
                conditions.append("c.annual_premium >= @min_premium")
                parameters.append({"name": "@min_premium", "value": min_premium})
            
            if max_premium is not None:
                conditions.append("c.annual_premium <= @max_premium")
                parameters.append({"name": "@max_premium", "value": max_premium})
            
            if min_coverage is not None:
                conditions.append("c.coverage_amount >= @min_coverage")
                parameters.append({"name": "@min_coverage", "value": min_coverage})
            
            if payment_frequency:
                conditions.append("c.payment_frequency = @payment_frequency")
                parameters.append({"name": "@payment_frequency", "value": payment_frequency})
            
            where_clause = " AND ".join(conditions)
            
            # Vector search query with filters
            query = f"""
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
            WHERE {where_clause}
            ORDER BY VectorDistance(c.embedding, @embedding)
            """
            
            results = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            if not results:
                print(f"✓ No results found matching the filters")
                return []
            
            # Post-process: Add keyword matching scores
            query_terms = query_text.lower().split()
            
            for result in results:
                # Create searchable text
                searchable_text = " ".join([
                    str(result.get('policy_type', '')),
                    str(result.get('policy_number', '')),
                    str(result.get('status', '')),
                    str(result.get('payment_frequency', '')),
                    str(result.get('customer_id', ''))
                ]).lower()
                
                # Calculate keyword match score
                matches = sum(1 for term in query_terms if term in searchable_text)
                keyword_score = matches / len(query_terms) if query_terms else 0
                
                # Calculate hybrid score
                vector_score_normalized = result['vector_score']
                hybrid_score = (vector_weight * vector_score_normalized) + (fulltext_weight * keyword_score)
                
                # Add scores
                result['fulltext_score'] = keyword_score
                result['hybrid_score'] = hybrid_score
            
            # Re-rank by hybrid score
            results.sort(key=lambda x: x['hybrid_score'], reverse=False)
            results = results[:top_k]
            
            print(f"✓ Found {len(results)} filtered hybrid search results")
            return results
            
        except Exception as e:
            print(f"❌ Error in hybrid search with filters: {e}")
            print(f"   Falling back to criteria search...")
            # Fallback to regular criteria search
            return self.search_by_criteria(
                policy_type=policy_type,
                status=status,
                min_premium=min_premium,
                max_premium=max_premium,
                min_coverage=min_coverage,
                payment_frequency=payment_frequency
            )

    def get_policy_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about policies in the database"""

        try:
            stats = {}

            # Total count
            count_query = "SELECT VALUE COUNT(1) FROM c"
            stats['total_policies'] = list(self.container.query_items(
                query=count_query,
                enable_cross_partition_query=True
            ))[0]

            # Average premium
            avg_premium_query = "SELECT VALUE AVG(c.annual_premium) FROM c"
            stats['average_premium'] = round(list(self.container.query_items(
                query=avg_premium_query,
                enable_cross_partition_query=True
            ))[0], 2)

            # Average coverage
            avg_coverage_query = "SELECT VALUE AVG(c.coverage_amount) FROM c"
            stats['average_coverage'] = round(list(self.container.query_items(
                query=avg_coverage_query,
                enable_cross_partition_query=True
            ))[0], 2)

            # Policy type distribution
            type_query = """
            SELECT c.policy_type AS type, COUNT(1) AS count 
            FROM c 
            GROUP BY c.policy_type
            """
            stats['type_distribution'] = list(self.container.query_items(
                query=type_query,
                enable_cross_partition_query=True
            ))

            # Status distribution
            status_query = """
            SELECT c.status AS status, COUNT(1) AS count 
            FROM c 
            GROUP BY c.status
            """
            stats['status_distribution'] = list(self.container.query_items(
                query=status_query,
                enable_cross_partition_query=True
            ))

            return stats

        except Exception as e:
            print(f"Error fetching statistics: {e}")
            return {}

    def print_policies_summary(self, policies: List[Dict[str, Any]]):
        """Print summary of multiple policies"""

        if not policies:
            print("No policies to display")
            return

        print("\n" + "=" * 80)
        print(f"POLICIES SUMMARY ({len(policies)} policies)")
        print("=" * 80 + "\n")

        for i, policy in enumerate(policies, 1):
            print(f"{i}. {policy.get('policy_type')} - {policy.get('policy_number')}")
            print(f"   Status: {policy.get('status')}")
            print(f"   Premium: ${policy.get('annual_premium', 0):,.2f} | Coverage: ${policy.get('coverage_amount', 0):,}")
            print(f"   Payment: {policy.get('payment_frequency')} | Auto Renew: {policy.get('auto_renew')}")
            print()

        print("=" * 80 + "\n")

    def print_policy_details(self, policy: Dict[str, Any]):
        """Print detailed policy information"""

        if not policy:
            print("No policy data to display")
            return

        print("\n" + "=" * 80)
        print(f"POLICY DETAILS: {policy.get('policy_id')}")
        print("=" * 80)

        # Basic Info
        print("\n📋 POLICY INFORMATION")
        print("-" * 80)
        print(f"Policy Number:  {policy.get('policy_number')}")
        print(f"Policy Type:    {policy.get('policy_type')}")
        print(f"Status:         {policy.get('status')}")
        print(f"Customer ID:    {policy.get('customer_id')}")

        # Financial Info
        print("\n💰 FINANCIAL DETAILS")
        print("-" * 80)
        print(f"Annual Premium:   ${policy.get('annual_premium', 0):,.2f}")
        print(f"Coverage Amount:  ${policy.get('coverage_amount', 0):,}")
        print(f"Deductible:       ${policy.get('deductible', 0):,}")
        print(f"Payment Frequency: {policy.get('payment_frequency')}")

        # Term Info
        print("\n📅 TERM DETAILS")
        print("-" * 80)
        print(f"Start Date:  {policy.get('start_date')}")
        print(f"End Date:    {policy.get('end_date')}")
        print(f"Term:        {policy.get('term_months')} months")
        print(f"Auto Renew:  {policy.get('auto_renew')}")

        # Additional Info
        print("\n⚙️  ADDITIONAL INFORMATION")
        print("-" * 80)
        print(f"Agent ID:         {policy.get('agent_id')}")
        if 'metadata' in policy and policy['metadata']:
            print(f"Discount Applied: {policy['metadata'].get('discount_applied', 0):.2%}")
            print(f"Risk Category:    {policy['metadata'].get('risk_category', 'N/A')}")

        print("=" * 80 + "\n")


def main():
    """Main execution function with interactive menu"""

    print("=" * 80)
    print("Policy Data Retrieval System - HYBRID SEARCH")
    print("=" * 80)

    try:
        retriever = PolicyRetriever()
    except Exception as e:
        print(f"\n❌ Error initializing retriever: {e}")
        return

    # Interactive menu
    while True:
        print("\n" + "=" * 80)
        print("SELECT AN OPTION:")
        print("=" * 80)
        print("1. Retrieve policies by Customer ID")
        print("2. Retrieve policy by Policy ID")
        print("3. Hybrid search (semantic + keyword)")
        print("4. Hybrid search with custom weights")
        print("5. Hybrid search with filters")
        print("6. Search by criteria (filters only)")
        print("7. View database statistics")
        print("8. Exit")
        print("-" * 80)

        choice = input("\nEnter your choice (1-8): ").strip()

        if choice == "1":
            # Retrieve by customer ID
            customer_id = input("\nEnter Customer ID (e.g., CUST-xxxx): ").strip()

            if not customer_id:
                print("❌ Customer ID cannot be empty")
                continue

            print(f"\nSearching for policies for customer: {customer_id}...")
            policies = retriever.get_policies_by_customer_id(customer_id)

            if policies:
                retriever.print_policies_summary(policies)

                # Option to view detailed policy
                view_detail = input("View detailed policy? Enter policy number (or press Enter to skip): ").strip()
                if view_detail:
                    # Find policy by number
                    selected_policy = next((p for p in policies if view_detail in p.get('policy_number', '')), None)
                    if selected_policy:
                        retriever.print_policy_details(selected_policy)
                    else:
                        print(f"❌ Policy '{view_detail}' not found in results")

                # Option to save to file
                save = input("\nSave all policies to file? (y/n): ").strip().lower()
                if save == 'y':
                    filename = f"{customer_id}_policies.json"
                    with open(filename, 'w') as f:
                        json.dump(policies, f, indent=2)
                    print(f"✓ Saved to {filename}")
            else:
                print(f"\n❌ No policies found for customer '{customer_id}'")

        elif choice == "2":
            # Retrieve by policy ID
            policy_id = input("\nEnter Policy ID (e.g., POL-xxxx): ").strip()

            if not policy_id:
                print("❌ Policy ID cannot be empty")
                continue

            print(f"\nSearching for policy: {policy_id}...")
            policy = retriever.get_policy_by_id(policy_id)

            if policy:
                retriever.print_policy_details(policy)

                # Option to save to file
                save = input("Save to file? (y/n): ").strip().lower()
                if save == 'y':
                    filename = f"{policy_id}.json"
                    with open(filename, 'w') as f:
                        json.dump(policy, f, indent=2)
                    print(f"✓ Saved to {filename}")
            else:
                print(f"\n❌ Policy '{policy_id}' not found in database")

        elif choice == "3":
            # Hybrid search with default weights
            query = input("\nEnter search query (e.g., 'active auto insurance'): ").strip()

            if not query:
                print("❌ Query cannot be empty")
                continue

            try:
                top_k = int(input("Number of results (default 5): ").strip() or "5")
            except ValueError:
                top_k = 5

            print(f"\nSearching for: '{query}'...")
            results = retriever.hybrid_search(query, top_k)

            if results:
                print(f"\n✓ Found {len(results)} similar policies:\n")
                for i, policy in enumerate(results, 1):
                    print(f"{i}. {policy['policy_type']} - {policy['policy_number']}")
                    print(f"   Customer: {policy['customer_id']} | Status: {policy['status']}")
                    print(f"   Premium: ${policy['annual_premium']:,.2f} | Coverage: ${policy['coverage_amount']:,}")
                    if 'hybrid_score' in policy:
                        print(f"   Hybrid Score: {policy['hybrid_score']:.4f}")
                        if 'vector_score' in policy and 'fulltext_score' in policy:
                            print(f"   (Vector: {policy['vector_score']:.4f}, Keyword: {policy['fulltext_score']:.4f})")
                    print()
            else:
                print("❌ No results found")

        elif choice == "4":
            # Hybrid search with custom weights
            query = input("\nEnter search query: ").strip()

            if not query:
                print("❌ Query cannot be empty")
                continue

            try:
                top_k = int(input("Number of results (default 5): ").strip() or "5")
                
                print("\nSet search weights (must sum to 1.0):")
                vector_weight = float(input("Vector search weight (0.0-1.0, default 0.6): ").strip() or "0.6")
                fulltext_weight = float(input("Keyword search weight (0.0-1.0, default 0.4): ").strip() or "0.4")
                
                # Normalize weights
                total = vector_weight + fulltext_weight
                if total != 1.0:
                    print(f"\n⚠️  Normalizing weights (sum was {total:.2f})")
                    vector_weight = vector_weight / total
                    fulltext_weight = fulltext_weight / total
                    print(f"   Adjusted: Vector={vector_weight:.2f}, Keyword={fulltext_weight:.2f}")
                
            except ValueError:
                print("❌ Invalid input, using defaults")
                top_k = 5
                vector_weight = 0.6
                fulltext_weight = 0.4

            print(f"\nSearching for: '{query}'...")
            results = retriever.hybrid_search(query, top_k, vector_weight, fulltext_weight)

            if results:
                print(f"\n✓ Found {len(results)} results:\n")
                for i, policy in enumerate(results, 1):
                    print(f"{i}. {policy['policy_type']} - {policy['policy_number']}")
                    print(f"   Status: {policy['status']}")
                    print(f"   Premium: ${policy['annual_premium']:,.2f} | Coverage: ${policy['coverage_amount']:,}")
                    if 'hybrid_score' in policy:
                        print(f"   Hybrid Score: {policy['hybrid_score']:.4f}")
                    print()
            else:
                print("❌ No results found")

        elif choice == "5":
            # Hybrid search with filters
            query = input("\nEnter search query: ").strip()

            if not query:
                print("❌ Query cannot be empty")
                continue

            print("\nEnter filters (press Enter to skip):")
            
            try:
                top_k = int(input("Number of results (default 5): ").strip() or "5")
                
                policy_type = input("Policy type (AUTO/HOME/LIFE/HEALTH/BUSINESS): ").strip().upper()
                policy_type = policy_type if policy_type else None

                status = input("Status (ACTIVE/CANCELLED/EXPIRED/PENDING): ").strip().upper()
                status = status if status else None

                min_premium = input("Minimum premium: ").strip()
                min_premium = float(min_premium) if min_premium else None

                max_premium = input("Maximum premium: ").strip()
                max_premium = float(max_premium) if max_premium else None

                min_coverage = input("Minimum coverage: ").strip()
                min_coverage = int(min_coverage) if min_coverage else None

                payment_freq = input("Payment frequency (MONTHLY/QUARTERLY/SEMI_ANNUAL/ANNUAL): ").strip().upper()
                payment_freq = payment_freq if payment_freq else None

                print(f"\nSearching for: '{query}' with filters...")
                results = retriever.hybrid_search_with_filters(
                    query_text=query,
                    top_k=top_k,
                    policy_type=policy_type,
                    status=status,
                    min_premium=min_premium,
                    max_premium=max_premium,
                    min_coverage=min_coverage,
                    payment_frequency=payment_freq
                )

                if results:
                    print(f"\n✓ Found {len(results)} filtered results:\n")
                    for i, policy in enumerate(results, 1):
                        print(f"{i}. {policy['policy_type']} - {policy['policy_number']}")
                        print(f"   Status: {policy['status']}")
                        print(f"   Premium: ${policy['annual_premium']:,.2f} | Coverage: ${policy['coverage_amount']:,}")
                        if 'hybrid_score' in policy:
                            print(f"   Hybrid Score: {policy['hybrid_score']:.4f}")
                        print()
                else:
                    print("❌ No results match the criteria")

            except ValueError:
                print("❌ Invalid input for numeric fields")

        elif choice == "6":
            # Search by criteria only
            print("\nEnter search criteria (press Enter to skip):")

            policy_type = input("Policy type (AUTO/HOME/LIFE/HEALTH/BUSINESS): ").strip().upper()
            policy_type = policy_type if policy_type else None

            status = input("Status (ACTIVE/CANCELLED/EXPIRED/PENDING): ").strip().upper()
            status = status if status else None

            try:
                min_premium = input("Minimum premium: ").strip()
                min_premium = float(min_premium) if min_premium else None

                max_premium = input("Maximum premium: ").strip()
                max_premium = float(max_premium) if max_premium else None

                min_coverage = input("Minimum coverage amount: ").strip()
                min_coverage = int(min_coverage) if min_coverage else None

                payment_freq = input("Payment frequency (MONTHLY/QUARTERLY/SEMI_ANNUAL/ANNUAL): ").strip().upper()
                payment_freq = payment_freq if payment_freq else None

                auto_renew_input = input("Auto renew (true/false): ").strip().lower()
                auto_renew = None
                if auto_renew_input == 'true':
                    auto_renew = True
                elif auto_renew_input == 'false':
                    auto_renew = False

                print("\nSearching...")
                results = retriever.search_by_criteria(
                    policy_type=policy_type,
                    status=status,
                    min_premium=min_premium,
                    max_premium=max_premium,
                    min_coverage=min_coverage,
                    payment_frequency=payment_freq,
                    auto_renew=auto_renew
                )

                if results:
                    print(f"\n✓ Found {len(results)} policies matching criteria:\n")
                    for i, policy in enumerate(results[:10], 1):
                        print(f"{i}. {policy['policy_type']} - {policy['policy_number']} ({policy['status']})")
                        print(f"   Customer: {policy['customer_id']}")
                        print(f"   Premium: ${policy['annual_premium']:,.2f} | Coverage: ${policy['coverage_amount']:,}")
                        print()

                    if len(results) > 10:
                        print(f"... and {len(results) - 10} more results")
                else:
                    print("❌ No policies match the criteria")

            except ValueError:
                print("❌ Invalid input for numeric fields")

        elif choice == "7":
            # Statistics
            print("\nFetching database statistics...")
            stats = retriever.get_policy_statistics()

            if stats:
                print("\n" + "=" * 80)
                print("DATABASE STATISTICS")
                print("=" * 80)
                print(f"\nTotal Policies: {stats.get('total_policies', 0):,}")
                print(f"Average Premium: ${stats.get('average_premium', 0):,.2f}")
                print(f"Average Coverage: ${stats.get('average_coverage', 0):,.2f}")

                print("\nPolicy Type Distribution:")
                for ptype in stats.get('type_distribution', []):
                    print(f"  {ptype['type']}: {ptype['count']} policies")

                print("\nStatus Distribution:")
                for status in stats.get('status_distribution', []):
                    print(f"  {status['status']}: {status['count']} policies")

                print("=" * 80)

        elif choice == "8":
            print("\n👋 Goodbye!")
            break

        else:
            print("\n❌ Invalid choice. Please select 1-8.")


if __name__ == "__main__":
    main()