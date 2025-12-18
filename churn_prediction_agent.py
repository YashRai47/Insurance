

"""
churn_prediction_agent.py - Interactive LangGraph Agent for Insurance Churn Prediction
"""

import os
import json
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from datetime import datetime
from dateutil import parser as date_parser
import operator

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
# Import our retrieval modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Note: Assuming the retrieval files are in the same directory
# from retrieval import CustomerRetriever
# from policy_retrieval import PolicyRetriever

# For standalone testing, we'll create simplified versions
# In production, you would uncomment the imports above

# ============================================================================
# CONFIGURATION - Choose one option below
# ============================================================================

# OPTION 1: Use OpenAI directly (gpt-4o-mini, gpt-4, etc.)
USE_OPENAI = True  # Set to True to use OpenAI, False to use Azure OpenAI

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-4o-mini"  # or "gpt-4", "gpt-3.5-turbo", etc.

USE_MOCK_DATA = False


class AgentState(TypedDict):
    """State for the churn prediction agent"""
    messages: Annotated[List, operator.add]
    customer_id: Optional[str]
    customer_data: Optional[Dict[str, Any]]
    policy_data: Optional[List[Dict[str, Any]]]
    churn_analysis: Optional[Dict[str, Any]]
    stage: str  # Stages: greeting, collect_id, retrieve_data, analyze, present_results


class ChurnPredictionAgent:
    """Interactive insurance agent for churn prediction using LangGraph"""

    def __init__(self):
        """Initialize the agent with Azure OpenAI and retrievers"""
        print("🚀 Initializing Churn Prediction Agent...")
        
        # Initialize LLM based on configuration
        if USE_OPENAI:
            print("📡 Using OpenAI API...")
            self.llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                model_name=OPENAI_MODEL,
                temperature=0.7
            )
        else:

            print("📡 Using Azure OpenAI...")
            # self.llm = AzureChatOpenAI(
            #     azure_endpoint=AZURE_OPENAI_ENDPOINT,
            #     api_key=AZURE_OPENAI_KEY,
            #     api_version=AZURE_OPENAI_API_VERSION,
            #     deployment_name=AZURE_OPENAI_DEPLOYMENT,
            #     temperature=0.7
            # )
        
        # Initialize retrievers based on configuration
        if USE_MOCK_DATA:
            print("⚠️  Using mock data for testing")
            self.customer_retriever = None
            self.policy_retriever = None
        else:
            try:
                from cust_ret import CustomerRetriever
                from policy_retrieval import PolicyRetriever
                
                print("📊 Initializing database retrievers...")
                self.customer_retriever = CustomerRetriever()
                self.policy_retriever = PolicyRetriever()
                print("✓ Database retrievers initialized")
            except ImportError as e:
                print(f"⚠️  Warning: Could not import retrievers: {e}")
                print("⚠️  Falling back to mock data")
                self.customer_retriever = None
                self.policy_retriever = None
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize retrievers: {e}")
                print("⚠️  Falling back to mock data")
                self.customer_retriever = None
                self.policy_retriever = None
        
        # Build the graph
        self.graph = self._build_graph()
        
        print("✓ Agent initialized successfully!\n")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("greeting", self.greeting_node)
        workflow.add_node("collect_customer_id", self.collect_customer_id_node)
        workflow.add_node("retrieve_data", self.retrieve_data_node)
        workflow.add_node("analyze_churn", self.analyze_churn_node)
        workflow.add_node("present_results", self.present_results_node)
        workflow.add_node("handle_followup", self.handle_followup_node)
        
        # Set entry point
        workflow.set_entry_point("greeting")
        
        # Add edges
        workflow.add_edge("greeting", "collect_customer_id")
        workflow.add_edge("collect_customer_id", "retrieve_data")
        workflow.add_edge("retrieve_data", "analyze_churn")
        workflow.add_edge("analyze_churn", "present_results")
        workflow.add_edge("present_results", "handle_followup")
        
        # Conditional edge from handle_followup
        workflow.add_conditional_edges(
            "handle_followup",
            self.should_continue,
            {
                "collect_customer_id": "collect_customer_id",
                "end": END
            }
        )
        
        return workflow.compile()

    def greeting_node(self, state: AgentState) -> AgentState:
        """Initial greeting to the user"""
        
        greeting_message = """
Hello! 👋 I'm your AI Insurance Advisor specializing in customer retention analysis.

I help identify customers who might be at risk of leaving so we can take proactive steps 
to keep them satisfied and ensure they have the best coverage for their needs.

To get started, I'll need to analyze a customer's profile and their policy history. 
Let's begin!
        """
        
        state["messages"].append(AIMessage(content=greeting_message.strip()))
        state["stage"] = "collect_id"
        
        return state

    def collect_customer_id_node(self, state: AgentState) -> AgentState:
        """Collect customer ID from user"""
        
        if state.get("customer_id"):
            # Customer ID already collected, skip
            return state
        
        prompt = "\n📋 Please provide the Customer ID you'd like me to analyze (e.g., CUST-xxxx-xxxx-xxxx):"
        state["messages"].append(AIMessage(content=prompt))
        
        # In interactive mode, this will wait for user input
        # The input will be processed externally and added to state
        state["stage"] = "awaiting_id"
        
        return state

    def retrieve_data_node(self, state: AgentState) -> AgentState:
        """Retrieve customer and policy data"""
        
        customer_id = state.get("customer_id")
        
        if not customer_id:
            state["messages"].append(AIMessage(
                content="❌ I couldn't find a valid Customer ID. Please try again."
            ))
            return state
        
        state["messages"].append(AIMessage(
            content=f"\n🔍 Retrieving data for Customer ID: {customer_id}...\n"
        ))
        
        customer_data = None
        policy_data = []
        
        try:
            # Use real retrievers if available, otherwise mock data
            if self.customer_retriever is not None and self.policy_retriever is not None:
                # Try to retrieve customer data
                try:
                    customer_data = self.customer_retriever.get_customer_by_id(customer_id)
                except Exception as e:
                    print(f"Warning: Error retrieving customer data: {str(e)}")
                    customer_data = None
                
                # Try to retrieve policy data regardless of customer data
                try:
                    policy_data = self.policy_retriever.get_policies_by_customer_id(customer_id)
                    if policy_data is None:
                        policy_data = []
                except Exception as e:
                    print(f"Warning: Error retrieving policy data: {str(e)}")
                    policy_data = []
                
                # Check if we have any data at all
                if not customer_data and not policy_data:
                    state["messages"].append(AIMessage(
                        content=f"❌ No data found for Customer ID '{customer_id}'.\n\nThis could mean:\n• The customer doesn't exist in the customer database\n• The customer has no policies\n• There may be a connection issue\n\nPlease verify the Customer ID and try again."
                    ))
                    state["customer_id"] = None
                    state["stage"] = "collect_id"
                    return state
                
                # We have at least some data, proceed
                if not customer_data:
                    state["messages"].append(AIMessage(
                        content="⚠️  Note: Customer profile not found in database, but policy data is available.\nProceeding with policy analysis only.\n"
                    ))
                    # Create minimal customer data from customer_id
                    customer_data = {
                        'customer_id': customer_id,
                        'first_name': 'Unknown',
                        'last_name': 'Customer',
                        'email': 'N/A',
                        'phone': 'N/A',
                        'date_of_birth': None,
                        'occupation': 'Not specified',
                        'annual_income': 0,
                        'credit_score': 0,
                        'metadata': {}
                    }
                
                if not policy_data:
                    state["messages"].append(AIMessage(
                        content="⚠️  Note: No policies found for this customer.\nProceeding with customer profile analysis only.\n"
                    ))
                
                state["customer_data"] = customer_data
                state["policy_data"] = policy_data
                
            else:
                # Mock data for testing
                customer_data = self._get_mock_customer_data(customer_id)
                policy_data = self._get_mock_policy_data(customer_id)
                
                state["customer_data"] = customer_data
                state["policy_data"] = policy_data
            
            policy_count = len(state['policy_data']) if state['policy_data'] else 0
            state["messages"].append(AIMessage(
                content=f"✓ Retrieved data: Customer profile and {policy_count} policies.\n"
            ))
            
            state["stage"] = "analyze"
            
        except Exception as e:
            state["messages"].append(AIMessage(
                content=f"❌ Unexpected error retrieving data: {str(e)}\nPlease check your configuration and try again."
            ))
            state["stage"] = "collect_id"
        
        return state

    def analyze_churn_node(self, state: AgentState) -> AgentState:
        """Analyze churn risk using LLM"""
        
        state["messages"].append(AIMessage(content="🧠 Analyzing churn risk factors...\n"))
        
        customer_data = state.get("customer_data", {})
        policy_data = state.get("policy_data", [])
        
        # Ensure customer_data is not None
        if customer_data is None:
            customer_data = {}
        
        # Ensure policy_data is not None
        if policy_data is None:
            policy_data = []
        
        # Calculate churn indicators
        analysis = self._calculate_churn_factors(customer_data, policy_data)
        
        # Check if we have sufficient data for analysis
        has_customer_profile = customer_data.get('first_name') != 'Unknown'
        has_policies = len(policy_data) > 0
        
        if not has_policies:
            # No policies - high churn risk by default
            analysis['llm_analysis'] = """
⚠️  LIMITED DATA ANALYSIS

**Churn Risk: CRITICAL (No Active Policies)**

This customer currently has no policies in our system. This represents either:
1. A new potential customer who hasn't purchased yet
2. A former customer who has cancelled all policies (100% churn)
3. Data synchronization issues

**Immediate Actions Required:**
1. Verify if this is an active customer or prospect
2. Review historical policy data if this is a former customer
3. If former customer: Conduct win-back campaign
4. If prospect: Initiate sales outreach
5. Check for data quality issues

**Recommendation:** Manual review required before proceeding with retention efforts.
"""
            state["churn_analysis"] = analysis
            state["stage"] = "present"
            return state
        
        # Use LLM to generate detailed analysis
        system_prompt = """You are an expert insurance analyst specializing in customer retention. 
        Your task is to analyze customer data and predict churn risk with actionable insights.
        Be professional, empathetic, and provide specific recommendations.
        
        If customer profile data is limited, focus your analysis on policy behavior patterns."""
        
        # Safe access to customer data with defaults
        first_name = customer_data.get('first_name', 'Unknown')
        last_name = customer_data.get('last_name', 'Customer')
        age = self._calculate_age(customer_data.get('date_of_birth', ''))
        occupation = customer_data.get('occupation', 'Not specified')
        annual_income = customer_data.get('annual_income', 0)
        credit_score = customer_data.get('credit_score', 0)
        metadata = customer_data.get('metadata', {})
        if metadata is None:
            metadata = {}
        customer_segment = metadata.get('customer_segment', 'Not classified')
        
        # Build analysis context
        if has_customer_profile:
            customer_context = f"""
CUSTOMER PROFILE:
- Name: {first_name} {last_name}
- Age: {age} years
- Occupation: {occupation}
- Annual Income: ${annual_income:,}
- Credit Score: {credit_score}
- Customer Segment: {customer_segment}
"""
        else:
            customer_context = f"""
CUSTOMER PROFILE:
⚠️  Limited customer profile data available. Analysis based primarily on policy behavior.
- Customer ID: {customer_data.get('customer_id', 'Unknown')}
"""
        
        analysis_prompt = f"""
Based on the following data, provide a comprehensive churn risk analysis:

{customer_context}

POLICY SUMMARY:
- Total Policies: {analysis['total_policies']}
- Active Policies: {analysis['active_policies']}
- Cancelled Policies: {analysis['cancelled_policies']}
- Expired Policies: {analysis['expired_policies']}
- Total Annual Premium: ${analysis['total_premium']:,.2f}
- Average Policy Duration: {analysis['avg_policy_duration']:.1f} months

CHURN INDICATORS:
- Cancellation Rate: {analysis['cancellation_rate']:.1%}
- Recent Cancellations: {analysis['recent_cancellations']}
- Payment Issues: {analysis['payment_issues']}
- Policy Diversity: {analysis['policy_diversity']} different policy types
- Customer Tenure: {analysis['customer_tenure_months']:.1f} months

POLICY DETAILS:
"""
        
        # Add policy type breakdown
        if policy_data:
            policy_types = {}
            for policy in policy_data:
                ptype = policy.get('policy_type', 'Unknown')
                status = policy.get('status', 'Unknown')
                if ptype not in policy_types:
                    policy_types[ptype] = {'active': 0, 'cancelled': 0, 'expired': 0}
                policy_types[ptype][status.lower()] = policy_types[ptype].get(status.lower(), 0) + 1
            
            analysis_prompt += "\n"
            for ptype, counts in policy_types.items():
                analysis_prompt += f"- {ptype}: {counts.get('active', 0)} active, {counts.get('cancelled', 0)} cancelled, {counts.get('expired', 0)} expired\n"
        
        analysis_prompt += f"""

Please provide:
1. Overall Churn Risk Score (Low/Medium/High/Critical) with confidence level
2. Key risk factors identified from policy behavior
3. Protective factors (reasons they might stay)
4. Specific actionable recommendations to reduce churn risk
5. Suggested next steps for customer outreach

{"Note: Customer profile data is limited, so focus on policy behavior patterns and observable trends." if not has_customer_profile else ""}

Format your response in a professional yet conversational tone, as if speaking to an insurance manager.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = self.llm.invoke(messages)
            analysis['llm_analysis'] = response.content
            
        except Exception as e:
            analysis['llm_analysis'] = f"Error generating detailed analysis: {str(e)}\n\nPlease check your Azure OpenAI configuration.\n\nMake sure you're using a CHAT model (gpt-4 or gpt-35-turbo), not an embedding model."
        
        state["churn_analysis"] = analysis
        state["stage"] = "present"
        
        return state

    def present_results_node(self, state: AgentState) -> AgentState:
        """Present churn analysis results to user"""
        
        analysis = state.get("churn_analysis", {})
        customer_data = state.get("customer_data", {})
        
        # Ensure data is not None
        if analysis is None:
            analysis = {}
        if customer_data is None:
            customer_data = {}
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(analysis)
        risk_level = self._get_risk_level(risk_score)
        risk_emoji = "🟢" if risk_level == "LOW" else "🟡" if risk_level == "MEDIUM" else "🔴"
        
        # Safe access to customer data
        first_name = customer_data.get('first_name', 'Unknown')
        last_name = customer_data.get('last_name', 'Customer')
        customer_id = state.get('customer_id', 'Unknown')
        
        result_message = f"""
{'='*80}
{risk_emoji} CHURN RISK ANALYSIS REPORT {risk_emoji}
{'='*80}

Customer: {first_name} {last_name}
Customer ID: {customer_id}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
📊 RISK ASSESSMENT
{'='*80}

Overall Churn Risk: {risk_level} ({risk_score:.1f}/100)

Key Metrics:
• Total Policies: {analysis.get('total_policies', 0)}
• Active Policies: {analysis.get('active_policies', 0)}
• Cancellation Rate: {analysis.get('cancellation_rate', 0):.1%}
• Customer Tenure: {analysis.get('customer_tenure_months', 0):.1f} months
• Total Annual Premium: ${analysis.get('total_premium', 0):,.2f}

{'='*80}
🔍 DETAILED ANALYSIS
{'='*80}

{analysis.get('llm_analysis', 'Analysis not available')}

{'='*80}
        """
        
        state["messages"].append(AIMessage(content=result_message.strip()))
        state["stage"] = "followup"
        
        return state

    def handle_followup_node(self, state: AgentState) -> AgentState:
        """Handle follow-up questions or new analysis"""
        
        followup_message = """
Would you like to:
1. Analyze another customer
2. Export this report
3. End session

Please let me know how I can help you further.
        """
        
        state["messages"].append(AIMessage(content=followup_message.strip()))
        state["stage"] = "awaiting_followup"
        
        return state

    def should_continue(self, state: AgentState) -> str:
        """Determine if we should continue or end"""
        
        # This will be controlled by user input
        # Return "collect_customer_id" to analyze another customer
        # Return "end" to finish
        
        last_message = state["messages"][-1] if state["messages"] else None
        
        if isinstance(last_message, HumanMessage):
            content = last_message.content.lower()
            if "another" in content or "analyze" in content or "1" in content:
                # Reset state for new analysis
                state["customer_id"] = None
                state["customer_data"] = None
                state["policy_data"] = None
                state["churn_analysis"] = None
                return "collect_customer_id"
        
        return "end"

    # Helper Methods
    
    def _calculate_churn_factors(self, customer_data: Dict, policy_data: List[Dict]) -> Dict[str, Any]:
        """Calculate various churn risk factors"""
        
        # Handle None or empty policy data
        if policy_data is None:
            policy_data = []
        
        total_policies = len(policy_data)
        active_policies = sum(1 for p in policy_data if p.get('status') == 'ACTIVE')
        cancelled_policies = sum(1 for p in policy_data if p.get('status') == 'CANCELLED')
        expired_policies = sum(1 for p in policy_data if p.get('status') == 'EXPIRED')
        
        cancellation_rate = cancelled_policies / total_policies if total_policies > 0 else 0
        
        # Calculate total premium
        total_premium = sum(p.get('annual_premium', 0) for p in policy_data if p.get('status') == 'ACTIVE')
        
        # Calculate average policy duration
        durations = []
        for policy in policy_data:
            if policy.get('start_date') and policy.get('end_date'):
                try:
                    start = date_parser.parse(str(policy['start_date']))
                    end = date_parser.parse(str(policy['end_date']))
                    duration_months = (end - start).days / 30.44
                    durations.append(duration_months)
                except:
                    pass
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Recent cancellations (last 12 months)
        recent_cancellations = 0
        current_date = datetime.now()
        for policy in policy_data:
            if policy.get('status') == 'CANCELLED' and policy.get('end_date'):
                try:
                    end_date = date_parser.parse(str(policy['end_date']))
                    if (current_date - end_date).days < 365:
                        recent_cancellations += 1
                except:
                    pass
        
        # Policy diversity (different types)
        policy_types = set(p.get('policy_type') for p in policy_data if p.get('policy_type'))
        policy_diversity = len(policy_types)
        
        # Customer tenure
        customer_tenure_months = 0
        if policy_data:
            earliest_start = None
            for policy in policy_data:
                if policy.get('start_date'):
                    try:
                        start = date_parser.parse(str(policy['start_date']))
                        if earliest_start is None or start < earliest_start:
                            earliest_start = start
                    except:
                        pass
            
            if earliest_start:
                customer_tenure_months = (current_date - earliest_start).days / 30.44
        
        # Payment issues (auto_renew = false might indicate issues)
        payment_issues = sum(1 for p in policy_data if not p.get('auto_renew', True))
        
        return {
            'total_policies': total_policies,
            'active_policies': active_policies,
            'cancelled_policies': cancelled_policies,
            'expired_policies': expired_policies,
            'cancellation_rate': cancellation_rate,
            'total_premium': total_premium,
            'avg_policy_duration': avg_duration,
            'recent_cancellations': recent_cancellations,
            'policy_diversity': policy_diversity,
            'customer_tenure_months': customer_tenure_months,
            'payment_issues': payment_issues
        }
    
    def _calculate_risk_score(self, analysis: Dict[str, Any]) -> float:
        """
        Production-grade churn risk scoring
        Score range: 0 (no risk) → 100 (maximum churn risk)
        """

        # Defensive defaults
        cancellation_rate = float(analysis.get("cancellation_rate", 0.0))
        recent_cancellations = int(analysis.get("recent_cancellations", 0))
        active_policies = int(analysis.get("active_policies", 0))
        policy_diversity = int(analysis.get("policy_diversity", 0))
        tenure = float(analysis.get("customer_tenure_months", 0.0))
        payment_issues = int(analysis.get("payment_issues", 0))
        total_premium = float(analysis.get("total_premium", 0.0))

        score = 0.0

        # ------------------------------------------------------------------
        # 1. Cancellation behavior (strongest churn predictor)
        # ------------------------------------------------------------------
        score += min(cancellation_rate * 40, 40)   # max 40 points

        # Recency multiplier
        score += min(recent_cancellations * 8, 20)  # max 20 points

        # ------------------------------------------------------------------
        # 2. Policy portfolio strength (protective factor)
        # ------------------------------------------------------------------
        if active_policies <= 0:
            score += 25
        elif active_policies == 1:
            score += 15
        elif active_policies == 2:
            score += 8
        else:
            score -= 5  # strong retention signal

        # Diversity reduces churn
        if policy_diversity <= 1:
            score += 10
        elif policy_diversity == 2:
            score += 5
        else:
            score -= 5

        # ------------------------------------------------------------------
        # 3. Tenure decay (non-linear)
        # ------------------------------------------------------------------
        if tenure <= 6:
            score += 20
        elif tenure <= 12:
            score += 15
        elif tenure <= 24:
            score += 8
        elif tenure <= 36:
            score += 4
        else:
            score -= 6

        # ------------------------------------------------------------------
        # 4. Payment friction
        # ------------------------------------------------------------------
        score += min(payment_issues * 6, 12)

        # ------------------------------------------------------------------
        # 5. Premium sensitivity (low premium = low commitment)
        # ------------------------------------------------------------------
        if total_premium <= 500:
            score += 8
        elif total_premium <= 1500:
            score += 4
        else:
            score -= 4

        # ------------------------------------------------------------------
        # 6. Risk interaction amplification
        # ------------------------------------------------------------------
        if cancellation_rate > 0.3 and tenure < 12:
            score += 10

        if payment_issues > 0 and recent_cancellations > 0:
            score += 8

        # ------------------------------------------------------------------
        # 7. Final normalization
        # ------------------------------------------------------------------
        score = max(0.0, min(score, 100.0))
        return round(score, 1)


    # def _calculate_risk_score(self, analysis: Dict[str, Any]) -> float:
    #     """Calculate overall risk score (0-100, higher = more risk)"""
        
    #     score = 0.0
        
    #     # Cancellation rate (0-30 points)
    #     score += analysis['cancellation_rate'] * 60
        
    #     # Recent cancellations (0-20 points)
    #     score += min(analysis['recent_cancellations'] * 10, 20)
        
    #     # Active policies (fewer = higher risk, 0-15 points)
    #     if analysis['active_policies'] == 0:
    #         score += 15
    #     elif analysis['active_policies'] == 1:
    #         score += 10
    #     elif analysis['active_policies'] == 2:
    #         score += 5
        
    #     # Policy diversity (less diversity = higher risk, 0-10 points)
    #     if analysis['policy_diversity'] <= 1:
    #         score += 10
    #     elif analysis['policy_diversity'] == 2:
    #         score += 5
        
    #     # Customer tenure (newer customers = higher risk, 0-15 points)
    #     if analysis['customer_tenure_months'] < 12:
    #         score += 15
    #     elif analysis['customer_tenure_months'] < 24:
    #         score += 10
    #     elif analysis['customer_tenure_months'] < 36:
    #         score += 5
        
    #     # Payment issues (0-10 points)
    #     score += min(analysis['payment_issues'] * 5, 10)
        
    #     return min(score, 100)


    def _get_risk_level(self, score: float) -> str:
        if score < 25:
            return "LOW"
        elif score < 50:
            return "MEDIUM"
        elif score < 75:
            return "HIGH"
        else:
            return "CRITICAL"

    # def _get_risk_level(self, score: float) -> str:
    #     """Convert risk score to level"""
    #     if score < 30:
    #         return "LOW"
    #     elif score < 60:
    #         return "MEDIUM"
    #     else:
    #         return "HIGH"

    def _calculate_age(self, dob: str) -> int:
        """Calculate age from date of birth"""
        if not dob:
            return 0
        try:
            birth_date = date_parser.parse(str(dob))
            today = datetime.now()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            return age
        except Exception as e:
            print(f"Warning: Could not parse date of birth: {dob}")
            return 0

    # Mock data methods (for testing without database)
    
    def _get_mock_customer_data(self, customer_id: str) -> Dict[str, Any]:
        """Mock customer data for testing"""
        return {
            'customer_id': customer_id,
            'first_name': 'John',
            'last_name': 'Doe',
            'email': 'john.doe@example.com',
            'phone': '555-0123',
            'date_of_birth': '1985-05-15',
            'occupation': 'Software Engineer',
            'annual_income': 95000,
            'credit_score': 720,
            'metadata': {
                'customer_segment': 'GOLD',
                'marketing_opt_in': True,
                'preferred_contact': 'email'
            }
        }

    def _get_mock_policy_data(self, customer_id: str) -> List[Dict[str, Any]]:
        """Mock policy data for testing"""
        return [
            {
                'policy_id': 'POL-001',
                'policy_number': 'PN-1234567890',
                'customer_id': customer_id,
                'policy_type': 'AUTO',
                'status': 'ACTIVE',
                'annual_premium': 1200.00,
                'coverage_amount': 100000,
                'start_date': '2022-01-15',
                'end_date': '2025-01-15',
                'auto_renew': True
            },
            {
                'policy_id': 'POL-002',
                'policy_number': 'PN-9876543210',
                'customer_id': customer_id,
                'policy_type': 'HOME',
                'status': 'ACTIVE',
                'annual_premium': 1800.00,
                'coverage_amount': 300000,
                'start_date': '2021-06-01',
                'end_date': '2024-06-01',
                'auto_renew': True
            },
            {
                'policy_id': 'POL-003',
                'policy_number': 'PN-5555555555',
                'customer_id': customer_id,
                'policy_type': 'LIFE',
                'status': 'CANCELLED',
                'annual_premium': 600.00,
                'coverage_amount': 250000,
                'start_date': '2020-03-10',
                'end_date': '2023-08-15',
                'auto_renew': False
            }
        ]

    def run_interactive(self):
        """Run the agent in interactive mode"""
        
        print("\n" + "="*80)
        print("🏢 INSURANCE CHURN PREDICTION AGENT")
        print("="*80 + "\n")
        
        while True:
            # Initialize state for new session
            state = {
                "messages": [],
                "customer_id": None,
                "customer_data": None,
                "policy_data": None,
                "churn_analysis": None,
                "stage": "greeting"
            }
            
            # Greeting
            state = self.greeting_node(state)
            for msg in state["messages"]:
                if isinstance(msg, AIMessage):
                    print(msg.content)
                    print()
            
            # Collect Customer ID
            state["messages"] = []
            state = self.collect_customer_id_node(state)
            for msg in state["messages"]:
                if isinstance(msg, AIMessage):
                    print(msg.content)
            
            customer_id = input("\nYour input: ").strip()
            
            if not customer_id:
                print("❌ Customer ID cannot be empty. Please try again.\n")
                continue
            
            state["customer_id"] = customer_id
            state["messages"] = []
            
            # Retrieve Data
            state = self.retrieve_data_node(state)
            for msg in state["messages"]:
                if isinstance(msg, AIMessage):
                    print(msg.content)
            
            # Check if retrieval was successful
            if not state.get("customer_data"):
                continue
            
            # Analyze Churn
            state["messages"] = []
            state = self.analyze_churn_node(state)
            for msg in state["messages"]:
                if isinstance(msg, AIMessage):
                    print(msg.content)
            
            # Present Results
            state["messages"] = []
            state = self.present_results_node(state)
            for msg in state["messages"]:
                if isinstance(msg, AIMessage):
                    print(msg.content)
            
            # Follow-up
            state["messages"] = []
            state = self.handle_followup_node(state)
            for msg in state["messages"]:
                if isinstance(msg, AIMessage):
                    print(msg.content)
            
            user_choice = input("\nYour choice: ").strip()
            
            if "1" in user_choice or "another" in user_choice.lower():
                print("\n" + "="*80)
                print("Starting new analysis...")
                print("="*80 + "\n")
                continue
            elif "2" in user_choice or "export" in user_choice.lower():
                self._export_report(state)
                print("\n✓ Report exported successfully!")
                
                continue_prompt = input("\nWould you like to analyze another customer? (yes/no): ").strip().lower()
                if continue_prompt == "yes" or continue_prompt == "y":
                    print("\n" + "="*80)
                    print("Starting new analysis...")
                    print("="*80 + "\n")
                    continue
                else:
                    break
            else:
                break
        
        print("\n" + "="*80)
        print("Thank you for using the Churn Prediction Agent! Goodbye! 👋")
        print("="*80 + "\n")

    def _export_report(self, state: AgentState):
        """Export analysis report to JSON file"""
        
        customer_id = state.get("customer_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"churn_report_{customer_id}_{timestamp}.json"
        
        report = {
            "customer_id": customer_id,
            "analysis_date": datetime.now().isoformat(),
            "customer_data": state.get("customer_data"),
            "policy_summary": {
                "total_policies": len(state.get("policy_data", [])),
                "policies": state.get("policy_data")
            },
            "churn_analysis": state.get("churn_analysis")
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved to: {filename}")


def main():
    """Main entry point"""
    
    try:
        agent = ChurnPredictionAgent()
        agent.run_interactive()
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user. Goodbye! 👋")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease check your configuration and try again.")


if __name__ == "__main__":
    main()