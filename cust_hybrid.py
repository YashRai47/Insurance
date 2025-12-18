"""
retrieval.py - Retrieve customer data from Azure Cosmos DB using HYBRID SEARCH
Combines vector search (semantic) + full-text search (keyword) for optimal results
"""

import os
import json
from azure.cosmos import CosmosClient
from openai import AzureOpenAI
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
import os

load_dotenv()  


COSMOS_hybrid_ENDPOINT = os.getenv("COSMOS_hybrid_ENDPOINT")
COSMOS_hybrid_KEY = os.getenv("COSMOS_hybrid_KEY")
COSMOS_hybrid_DATABASE_NAME = os.getenv("COSMOS_hybrid_DATABASE_NAME")
COSMOS_hybrid_CONTAINER_NAME = os.getenv("COSMOS_hybrid_CONTAINER_NAME")


AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = "text-embedding-3-large"
AZURE_OPENAI_API_VERSION = "2024-02-01"

class CustomerRetriever:
    def __init__(self):
        """Initialize Cosmos DB and Azure OpenAI clients"""
        print("Initializing retrieval client with HYBRID SEARCH...")

        # Initialize Cosmos DB client
        self.cosmos_client = CosmosClient(COSMOS_hybrid_ENDPOINT, COSMOS_hybrid_KEY)
        self.database = self.cosmos_client.get_database_client(COSMOS_hybrid_DATABASE_NAME)
        self.container = self.database.get_container_client(COSMOS_hybrid_CONTAINER_NAME)

        # Initialize Azure OpenAI client for vector search
        self.openai_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )

        print("✓ Client initialized successfully")
        print("✓ Hybrid Search enabled (Vector + Full-Text)\n")

    def get_customer_by_id(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete customer details by customer_id"""

        try:
            query = "SELECT * FROM c WHERE c.customer_id = @customer_id"
            parameters = [{"name": "@customer_id", "value": customer_id}]

            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))

            if items:
                customer = items[0]
                # Remove embedding from response (too large and not needed for display)
                if 'embedding' in customer:
                    del customer['embedding']
                return customer
            else:
                return None

        except Exception as e:
            print(f"Error retrieving customer: {e}")
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
            List of customers with combined hybrid scores
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
        Fallback to pure vector search if hybrid search is not supported
        This is the original vector search method
        """
        
        try:
            print(f"⚠️  Using vector-only search as fallback")
            
            # Generate embedding for query
            query_embedding = self.generate_embedding(query_text)

            # Perform vector search
            query = """
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
            min_income: Optional[int] = None,
            max_income: Optional[int] = None,
            min_credit_score: Optional[int] = None,
            customer_segment: Optional[str] = None,
            state: Optional[str] = None,
            occupation: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search customers by various criteria"""

        try:
            conditions = []
            parameters = []

            if min_income is not None:
                conditions.append("c.annual_income >= @min_income")
                parameters.append({"name": "@min_income", "value": min_income})

            if max_income is not None:
                conditions.append("c.annual_income <= @max_income")
                parameters.append({"name": "@max_income", "value": max_income})

            if min_credit_score is not None:
                conditions.append("c.credit_score >= @min_credit_score")
                parameters.append({"name": "@min_credit_score", "value": min_credit_score})

            if customer_segment:
                conditions.append("c.metadata.customer_segment = @segment")
                parameters.append({"name": "@segment", "value": customer_segment})

            if state:
                conditions.append("c.address.state = @state")
                parameters.append({"name": "@state", "value": state})

            if occupation:
                conditions.append("CONTAINS(c.occupation, @occupation)")
                parameters.append({"name": "@occupation", "value": occupation})

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
        min_income: Optional[int] = None,
        max_income: Optional[int] = None,
        min_credit_score: Optional[int] = None,
        customer_segment: Optional[str] = None,
        state: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        ADVANCED: Hybrid search combined with filters
        Uses post-processing approach for maximum compatibility
        
        Args:
            query_text: Search query
            top_k: Number of results
            vector_weight: Weight for vector similarity
            fulltext_weight: Weight for keyword matching
            min_income: Minimum income filter
            max_income: Maximum income filter
            min_credit_score: Minimum credit score filter
            customer_segment: Customer segment filter
            state: State filter
            
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
            if min_income is not None:
                conditions.append("c.annual_income >= @min_income")
                parameters.append({"name": "@min_income", "value": min_income})
            
            if max_income is not None:
                conditions.append("c.annual_income <= @max_income")
                parameters.append({"name": "@max_income", "value": max_income})
            
            if min_credit_score is not None:
                conditions.append("c.credit_score >= @min_credit_score")
                parameters.append({"name": "@min_credit_score", "value": min_credit_score})
            
            if customer_segment:
                conditions.append("c.metadata.customer_segment = @segment")
                parameters.append({"name": "@segment", "value": customer_segment})
            
            if state:
                conditions.append("c.address.state = @state")
                parameters.append({"name": "@state", "value": state})
            
            where_clause = " AND ".join(conditions)
            
            # Vector search query with filters
            query = f"""
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
                    str(result.get('first_name', '')),
                    str(result.get('last_name', '')),
                    str(result.get('occupation', '')),
                    str(result.get('email', '')),
                    str(result.get('address', {}).get('city', '')),
                    str(result.get('address', {}).get('state', ''))
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
                min_income=min_income,
                max_income=max_income,
                min_credit_score=min_credit_score,
                customer_segment=customer_segment,
                state=state
            )

    def get_customer_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about customers in the database"""

        try:
            stats = {}

            # Total count
            count_query = "SELECT VALUE COUNT(1) FROM c"
            stats['total_customers'] = list(self.container.query_items(
                query=count_query,
                enable_cross_partition_query=True
            ))[0]

            # Average income
            avg_income_query = "SELECT VALUE AVG(c.annual_income) FROM c"
            stats['average_income'] = round(list(self.container.query_items(
                query=avg_income_query,
                enable_cross_partition_query=True
            ))[0], 2)

            # Average credit score
            avg_credit_query = "SELECT VALUE AVG(c.credit_score) FROM c"
            stats['average_credit_score'] = round(list(self.container.query_items(
                query=avg_credit_query,
                enable_cross_partition_query=True
            ))[0], 2)

            # Segment distribution
            segment_query = """
            SELECT c.metadata.customer_segment AS segment, COUNT(1) AS count 
            FROM c 
            GROUP BY c.metadata.customer_segment
            """
            stats['segment_distribution'] = list(self.container.query_items(
                query=segment_query,
                enable_cross_partition_query=True
            ))

            return stats

        except Exception as e:
            print(f"Error fetching statistics: {e}")
            return {}

    def print_customer_details(self, customer: Dict[str, Any]):
        """Pretty print customer details"""

        if not customer:
            print("No customer data to display")
            return

        print("\n" + "=" * 80)
        print(f"CUSTOMER DETAILS: {customer.get('customer_id')}")
        print("=" * 80)

        # Basic Info
        print("\n📋 BASIC INFORMATION")
        print("-" * 80)
        print(f"Name:           {customer.get('first_name')} {customer.get('last_name')}")
        print(f"Email:          {customer.get('email')}")
        print(f"Phone:          {customer.get('phone')}")
        print(f"Date of Birth:  {customer.get('date_of_birth')}")
        print(f"SSN:            {customer.get('ssn')}")
        print(f"Marital Status: {customer.get('marital_status')}")

        # Address
        if 'address' in customer:
            addr = customer['address']
            print("\n📍 ADDRESS")
            print("-" * 80)
            print(f"Street:   {addr.get('street')}")
            print(f"City:     {addr.get('city')}")
            print(f"State:    {addr.get('state')}")
            print(f"Zip Code: {addr.get('zip_code')}")
            print(f"Country:  {addr.get('country')}")

        # Professional Info
        print("\n💼 PROFESSIONAL INFORMATION")
        print("-" * 80)
        print(f"Occupation:    {customer.get('occupation')}")
        print(f"Annual Income: ${customer.get('annual_income'):,}")
        print(f"Credit Score:  {customer.get('credit_score')}")

        # Metadata
        if 'metadata' in customer:
            meta = customer['metadata']
            print("\n⚙️  METADATA")
            print("-" * 80)
            print(f"Customer Segment:  {meta.get('customer_segment')}")
            print(f"Marketing Opt-in:  {meta.get('marketing_opt_in')}")
            print(f"Paperless Billing: {meta.get('paperless_billing')}")
            print(f"Preferred Contact: {meta.get('preferred_contact')}")

        # Timestamps
        print("\n📅 TIMESTAMPS")
        print("-" * 80)
        print(f"Created Date: {customer.get('created_date')}")
        print(f"Created Time: {customer.get('created_timestamp')}")

        print("=" * 80 + "\n")


def main():
    """Main execution function with interactive menu"""

    print("=" * 80)
    print("Customer Data Retrieval System - HYBRID SEARCH")
    print("=" * 80)

    try:
        retriever = CustomerRetriever()
    except Exception as e:
        print(f"\n❌ Error initializing retriever: {e}")
        return

    # Interactive menu
    while True:
        print("\n" + "=" * 80)
        print("SELECT AN OPTION:")
        print("=" * 80)
        print("1. Retrieve customer by ID")
        print("2. Hybrid search (semantic + keyword)")
        print("3. Hybrid search with custom weights")
        print("4. Hybrid search with filters")
        print("5. Search by criteria (filters only)")
        print("6. View database statistics")
        print("7. Exit")
        print("-" * 80)

        choice = input("\nEnter your choice (1-7): ").strip()

        if choice == "1":
            # Retrieve by customer ID
            customer_id = input("\nEnter Customer ID (e.g., CUST-xxxx): ").strip()

            if not customer_id:
                print("❌ Customer ID cannot be empty")
                continue

            print(f"\nSearching for customer: {customer_id}...")
            customer = retriever.get_customer_by_id(customer_id)

            if customer:
                retriever.print_customer_details(customer)

                # Option to save to file
                save = input("Save to file? (y/n): ").strip().lower()
                if save == 'y':
                    filename = f"{customer_id}.json"
                    with open(filename, 'w') as f:
                        json.dump(customer, f, indent=2)
                    print(f"✓ Saved to {filename}")
            else:
                print(f"\n❌ Customer '{customer_id}' not found in database")

        elif choice == "2":
            # Hybrid search with default weights
            query = input("\nEnter search query (e.g., 'high income engineer'): ").strip()

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
                print(f"\n✓ Found {len(results)} similar customers:\n")
                for i, customer in enumerate(results, 1):
                    print(f"{i}. {customer['first_name']} {customer['last_name']}")
                    print(f"   ID: {customer['customer_id']}")
                    print(f"   Occupation: {customer['occupation']}")
                    print(f"   Income: ${customer['annual_income']:,}")
                    if 'hybrid_score' in customer:
                        print(f"   Hybrid Score: {customer['hybrid_score']:.4f}")
                        if 'vector_score' in customer and 'fulltext_score' in customer:
                            print(f"   (Vector: {customer['vector_score']:.4f}, Text: {customer['fulltext_score']:.4f})")
                    print()
            else:
                print("❌ No results found")

        elif choice == "3":
            # Hybrid search with custom weights
            query = input("\nEnter search query: ").strip()

            if not query:
                print("❌ Query cannot be empty")
                continue

            try:
                top_k = int(input("Number of results (default 5): ").strip() or "5")
                
                print("\nSet search weights (must sum to 1.0):")
                vector_weight = float(input("Vector search weight (0.0-1.0, default 0.6): ").strip() or "0.6")
                fulltext_weight = float(input("Full-text search weight (0.0-1.0, default 0.4): ").strip() or "0.4")
                
                # Normalize weights
                total = vector_weight + fulltext_weight
                if total != 1.0:
                    print(f"\n⚠️  Normalizing weights (sum was {total:.2f})")
                    vector_weight = vector_weight / total
                    fulltext_weight = fulltext_weight / total
                    print(f"   Adjusted: Vector={vector_weight:.2f}, Text={fulltext_weight:.2f}")
                
            except ValueError:
                print("❌ Invalid input, using defaults")
                top_k = 5
                vector_weight = 0.6
                fulltext_weight = 0.4

            print(f"\nSearching for: '{query}'...")
            results = retriever.hybrid_search(query, top_k, vector_weight, fulltext_weight)

            if results:
                print(f"\n✓ Found {len(results)} results:\n")
                for i, customer in enumerate(results, 1):
                    print(f"{i}. {customer['first_name']} {customer['last_name']}")
                    print(f"   Occupation: {customer['occupation']}")
                    print(f"   Income: ${customer['annual_income']:,}")
                    if 'hybrid_score' in customer:
                        print(f"   Hybrid Score: {customer['hybrid_score']:.4f}")
                    print()
            else:
                print("❌ No results found")

        elif choice == "4":
            # Hybrid search with filters
            query = input("\nEnter search query: ").strip()

            if not query:
                print("❌ Query cannot be empty")
                continue

            print("\nEnter filters (press Enter to skip):")
            
            try:
                top_k = int(input("Number of results (default 5): ").strip() or "5")
                
                min_income = input("Minimum income: ").strip()
                min_income = int(min_income) if min_income else None

                max_income = input("Maximum income: ").strip()
                max_income = int(max_income) if max_income else None

                min_credit = input("Minimum credit score: ").strip()
                min_credit = int(min_credit) if min_credit else None

                segment = input("Customer segment (BRONZE/SILVER/GOLD/PLATINUM): ").strip().upper()
                segment = segment if segment else None

                state = input("State (e.g., CA, NY): ").strip().upper()
                state = state if state else None

                print(f"\nSearching for: '{query}' with filters...")
                results = retriever.hybrid_search_with_filters(
                    query_text=query,
                    top_k=top_k,
                    min_income=min_income,
                    max_income=max_income,
                    min_credit_score=min_credit,
                    customer_segment=segment,
                    state=state
                )

                if results:
                    print(f"\n✓ Found {len(results)} filtered results:\n")
                    for i, customer in enumerate(results, 1):
                        print(f"{i}. {customer['first_name']} {customer['last_name']}")
                        print(f"   Occupation: {customer['occupation']}")
                        print(f"   Income: ${customer['annual_income']:,} | Credit: {customer['credit_score']}")
                        if 'hybrid_score' in customer:
                            print(f"   Hybrid Score: {customer['hybrid_score']:.4f}")
                        print()
                else:
                    print("❌ No results match the criteria")

            except ValueError:
                print("❌ Invalid input for numeric fields")

        elif choice == "5":
            # Search by criteria only
            print("\nEnter search criteria (press Enter to skip):")

            try:
                min_income = input("Minimum income: ").strip()
                min_income = int(min_income) if min_income else None

                max_income = input("Maximum income: ").strip()
                max_income = int(max_income) if max_income else None

                min_credit = input("Minimum credit score: ").strip()
                min_credit = int(min_credit) if min_credit else None

                segment = input("Customer segment (BRONZE/SILVER/GOLD/PLATINUM): ").strip().upper()
                segment = segment if segment else None

                state = input("State (e.g., CA, NY): ").strip().upper()
                state = state if state else None

                occupation = input("Occupation keyword: ").strip()
                occupation = occupation if occupation else None

                print("\nSearching...")
                results = retriever.search_by_criteria(
                    min_income=min_income,
                    max_income=max_income,
                    min_credit_score=min_credit,
                    customer_segment=segment,
                    state=state,
                    occupation=occupation
                )

                if results:
                    print(f"\n✓ Found {len(results)} customers matching criteria:\n")
                    for i, customer in enumerate(results[:10], 1):
                        print(f"{i}. {customer['first_name']} {customer['last_name']} ({customer['customer_id']})")
                        print(f"   Occupation: {customer['occupation']}")
                        print(f"   Income: ${customer['annual_income']:,} | Credit: {customer['credit_score']}")
                        print()

                    if len(results) > 10:
                        print(f"... and {len(results) - 10} more results")
                else:
                    print("❌ No customers match the criteria")

            except ValueError:
                print("❌ Invalid input for numeric fields")

        elif choice == "6":
            # Statistics
            print("\nFetching database statistics...")
            stats = retriever.get_customer_statistics()

            if stats:
                print("\n" + "=" * 80)
                print("DATABASE STATISTICS")
                print("=" * 80)
                print(f"\nTotal Customers: {stats.get('total_customers', 0):,}")
                print(f"Average Income: ${stats.get('average_income', 0):,.2f}")
                print(f"Average Credit Score: {stats.get('average_credit_score', 0):.2f}")

                print("\nCustomer Segment Distribution:")
                for segment in stats.get('segment_distribution', []):
                    print(f"  {segment['segment']}: {segment['count']} customers")
                print("=" * 80)

        elif choice == "7":
            print("\n👋 Goodbye!")
            break

        else:
            print("\n❌ Invalid choice. Please select 1-7.")


if __name__ == "__main__":
    main()
