"""
retrieval.py - Retrieve customer data from Azure Cosmos DB
"""

import os
import json
from azure.cosmos import CosmosClient
from openai import AzureOpenAI
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
import os

load_dotenv()  

#
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = "text-embedding-3-large"
AZURE_OPENAI_API_VERSION = "2024-02-01"

COSMOS_ret_ENDPOINT = os.getenv("COSMOS_ret_ENDPOINT")
COSMOS_ret_KEY = os.getenv("COSMOS_ret_KEY")
COSMOS_ret_DATABASE_NAME = os.getenv("COSMOS_ret_DATABASE_NAME")
COSMOS_ret_CONTAINER_NAME = os.getenv("COSMOS_ret_CONTAINER_NAME")

class CustomerRetriever:
    def __init__(self):
        """Initialize Cosmos DB and Azure OpenAI clients"""
        print("Initializing retrieval client...")

        # Initialize Cosmos DB client
        self.cosmos_client = CosmosClient(COSMOS_ret_ENDPOINT, COSMOS_ret_KEY)
        self.database = self.cosmos_client.get_database_client(COSMOS_ret_DATABASE_NAME)
        self.container = self.database.get_container_client(COSMOS_ret_CONTAINER_NAME)

        # Initialize Azure OpenAI client for vector search
        self.openai_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )

        print("✓ Client initialized successfully\n")

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

    def vector_search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar customers using vector search"""

        try:
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
                VectorDistance(c.embedding, @embedding) AS similarity_score
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
            print(f"Error in vector search: {e}")
            return []

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
                SELECT c.metadata.customer_segment as segment, COUNT(1) as count
                FROM c
                GROUP BY c.metadata.customer_segment
            """
            stats['segment_distribution'] = list(self.container.query_items(
                query=segment_query,
                enable_cross_partition_query=True
            ))

            return stats

        except Exception as e:
            print(f"Error getting statistics: {e}")
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
    print("Customer Data Retrieval System")
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
        print("2. Vector search (semantic search)")
        print("3. Search by criteria (filters)")
        print("4. View database statistics")
        print("5. Exit")
        print("-" * 80)

        choice = input("\nEnter your choice (1-5): ").strip()

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
            # Vector search
            query = input("\nEnter search query (e.g., 'high income engineer'): ").strip()

            if not query:
                print("❌ Query cannot be empty")
                continue

            try:
                top_k = int(input("Number of results (default 5): ").strip() or "5")
            except ValueError:
                top_k = 5

            print(f"\nSearching for: '{query}'...")
            results = retriever.vector_search(query, top_k)

            if results:
                print(f"\n✓ Found {len(results)} similar customers:\n")
                for i, customer in enumerate(results, 1):
                    print(f"{i}. {customer['first_name']} {customer['last_name']}")
                    print(f"   ID: {customer['customer_id']}")
                    print(f"   Occupation: {customer['occupation']}")
                    print(f"   Income: ${customer['annual_income']:,}")
                    print(f"   Similarity Score: {customer['similarity_score']:.4f}")
                    print()
            else:
                print("❌ No results found")

        elif choice == "3":
            # Search by criteria
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

        elif choice == "4":
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

        elif choice == "5":
            print("\n👋 Goodbye!")
            break

        else:
            print("\n❌ Invalid choice. Please select 1-5.")


if __name__ == "__main__":
    main()