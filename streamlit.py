# """
# streamlit_churn_app.py - Streamlit UI for Insurance Churn Prediction Agent
# Updated with ChatAgent Tab for Semantic and Hybrid Search
# """

# import streamlit as st
# import json
# from datetime import datetime
# from typing import Dict, Any, List
# import sys
# import os

# # Import the agent
# from churn_prediction_agent import ChurnPredictionAgent

# # Import RAG systems
# from unified_rag_system import UnifiedRAGSystem
# from unified_rag_hybrid_system import UnifiedRAGHybridSystem

# # Page configuration
# st.set_page_config(
#     page_title="Insurance Churn Prediction",
#     page_icon="🏢",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#         text-align: center;
#         padding: 1rem 0;
#         margin-bottom: 2rem;
#     }
    
#     .metric-card {
#         background-color: #f0f2f6;
#         padding: 1.5rem;
#         border-radius: 10px;
#         border-left: 5px solid #1f77b4;
#         margin: 1rem 0;
#     }
    
#     .risk-low {
#         border-left-color: #28a745 !important;
#         background-color: #d4edda;
#     }
    
#     .risk-medium {
#         border-left-color: #ffc107 !important;
#         background-color: #fff3cd;
#     }
    
#     .risk-high {
#         border-left-color: #dc3545 !important;
#         background-color: #f8d7da;
#     }
    
#     .risk-critical {
#         border-left-color: #dc3545 !important;
#         background-color: #f5c6cb;
#     }
    
#     .policy-card {
#         background-color: #ffffff;
#         padding: 1rem;
#         border-radius: 8px;
#         border: 1px solid #e0e0e0;
#         margin: 0.5rem 0;
#     }
    
#     .stButton>button {
#         width: 100%;
#         background-color: #1f77b4;
#         color: white;
#         font-weight: bold;
#         padding: 0.5rem 1rem;
#         border-radius: 5px;
#         border: none;
#     }
    
#     .stButton>button:hover {
#         background-color: #155a8a;
#     }
    
#     .info-box {
#         background-color: #e7f3ff;
#         padding: 1rem;
#         border-radius: 8px;
#         border-left: 4px solid #2196F3;
#         margin: 1rem 0;
#     }
    
#     .warning-box {
#         background-color: #fff3cd;
#         padding: 1rem;
#         border-radius: 8px;
#         border-left: 4px solid #ffc107;
#         margin: 1rem 0;
#     }
    
#     .chat-result-card {
#         background-color: #f8f9fa;
#         padding: 1.5rem;
#         border-radius: 10px;
#         border: 1px solid #dee2e6;
#         margin: 1rem 0;
#     }
    
#     .search-mode-badge {
#         display: inline-block;
#         padding: 0.5rem 1rem;
#         border-radius: 20px;
#         font-weight: bold;
#         margin: 0.5rem;
#     }
    
#     .semantic-badge {
#         background-color: #d1ecf1;
#         color: #0c5460;
#     }
    
#     .hybrid-badge {
#         background-color: #d4edda;
#         color: #155724;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'agent' not in st.session_state:
#     st.session_state.agent = None
# if 'analysis_history' not in st.session_state:
#     st.session_state.analysis_history = []
# if 'current_analysis' not in st.session_state:
#     st.session_state.current_analysis = None
# if 'rag_semantic' not in st.session_state:
#     st.session_state.rag_semantic = None
# if 'rag_hybrid' not in st.session_state:
#     st.session_state.rag_hybrid = None
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# def initialize_agent():
#     """Initialize the churn prediction agent"""
#     try:
#         with st.spinner("🚀 Initializing Churn Prediction Agent..."):
#             agent = ChurnPredictionAgent()
#             st.session_state.agent = agent
#             return True
#     except Exception as e:
#         st.error(f"❌ Error initializing agent: {str(e)}")
#         return False

# def initialize_rag_systems():
#     """Initialize RAG systems for ChatAgent"""
#     try:
#         with st.spinner("🔄 Initializing RAG Systems..."):
#             if st.session_state.rag_semantic is None:
#                 st.session_state.rag_semantic = UnifiedRAGSystem()
#             if st.session_state.rag_hybrid is None:
#                 st.session_state.rag_hybrid = UnifiedRAGHybridSystem()
#             return True
#     except Exception as e:
#         st.error(f"❌ Error initializing RAG systems: {str(e)}")
#         return False

# def get_risk_color(risk_level: str) -> str:
#     """Get color code for risk level"""
#     colors = {
#         "LOW": "#28a745",
#         "MEDIUM": "#ffc107",
#         "HIGH": "#dc3545",
#         "CRITICAL": "#dc3545"
#     }
#     return colors.get(risk_level.upper(), "#6c757d")

# def get_risk_emoji(risk_level: str) -> str:
#     """Get emoji for risk level"""
#     emojis = {
#         "LOW": "🟢",
#         "MEDIUM": "🟡",
#         "HIGH": "🔴",
#         "CRITICAL": "🔴"
#     }
#     return emojis.get(risk_level.upper(), "⚪")

# def analyze_customer(customer_id: str) -> Dict[str, Any]:
#     """Analyze a customer and return results"""
#     agent = st.session_state.agent
    
#     # Create initial state
#     state = {
#         "messages": [],
#         "customer_id": customer_id,
#         "customer_data": None,
#         "policy_data": None,
#         "churn_analysis": None,
#         "stage": "retrieve"
#     }
    
#     # Retrieve data
#     state = agent.retrieve_data_node(state)
    
#     # Check if retrieval was successful
#     if not state.get("customer_data") and not state.get("policy_data"):
#         return {
#             "success": False,
#             "error": "No data found for this customer ID"
#         }
    
#     # Analyze
#     state = agent.analyze_churn_node(state)
    
#     # Calculate risk score
#     analysis = state.get("churn_analysis", {})
#     risk_score = agent._calculate_risk_score(analysis)
#     risk_level = agent._get_risk_level(risk_score)
    
#     return {
#         "success": True,
#         "customer_id": customer_id,
#         "customer_data": state.get("customer_data"),
#         "policy_data": state.get("policy_data"),
#         "analysis": analysis,
#         "risk_score": risk_score,
#         "risk_level": risk_level,
#         "timestamp": datetime.now().isoformat()
#     }

# def display_customer_profile(customer_data: Dict[str, Any]):
#     """Display customer profile information"""
#     st.markdown("### 👤 Customer Profile")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("**Name:**")
#         st.write(f"{customer_data.get('first_name', 'N/A')} {customer_data.get('last_name', 'N/A')}")
        
#         st.markdown("**Email:**")
#         st.write(customer_data.get('email', 'N/A'))
    
#     with col2:
#         st.markdown("**Occupation:**")
#         st.write(customer_data.get('occupation', 'N/A'))
        
#         st.markdown("**Annual Income:**")
#         income = customer_data.get('annual_income', 0)
#         if income > 0:
#             st.write(f"${income:,}")
#         else:
#             st.write("N/A")
    
#     with col3:
#         st.markdown("**Credit Score:**")
#         credit = customer_data.get('credit_score', 0)
#         if credit > 0:
#             st.write(credit)
#         else:
#             st.write("N/A")
        
#         st.markdown("**Customer Segment:**")
#         metadata = customer_data.get('metadata', {})
#         if metadata:
#             st.write(metadata.get('customer_segment', 'N/A'))
#         else:
#             st.write("N/A")

# def display_policy_summary(policy_data: List[Dict[str, Any]]):
#     """Display policy summary"""
#     st.markdown("### 📋 Policy Portfolio")
    
#     if not policy_data:
#         st.info("No policies found for this customer.")
#         return
    
#     # Summary metrics
#     active_policies = sum(1 for p in policy_data if p.get('status') == 'ACTIVE')
#     cancelled_policies = sum(1 for p in policy_data if p.get('status') == 'CANCELLED')
#     expired_policies = sum(1 for p in policy_data if p.get('status') == 'EXPIRED')
#     total_premium = sum(p.get('annual_premium', 0) for p in policy_data if p.get('status') == 'ACTIVE')
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("Total Policies", len(policy_data))
#     with col2:
#         st.metric("Active", active_policies, delta=None, delta_color="normal")
#     with col3:
#         st.metric("Cancelled", cancelled_policies, delta=None, delta_color="inverse")
#     with col4:
#         st.metric("Total Premium", f"${total_premium:,.2f}")
    
#     # Policy details
#     st.markdown("#### Policy Details")
    
#     for idx, policy in enumerate(policy_data, 1):
#         status = policy.get('status', 'Unknown')
#         status_color = {
#             'ACTIVE': '🟢',
#             'CANCELLED': '🔴',
#             'EXPIRED': '🟠',
#             'PENDING': '🟡'
#         }.get(status, '⚪')
        
#         with st.expander(f"{status_color} Policy #{idx}: {policy.get('policy_type', 'N/A')} - {policy.get('policy_number', 'N/A')}"):
#             pcol1, pcol2 = st.columns(2)
            
#             with pcol1:
#                 st.markdown(f"**Status:** {status}")
#                 st.markdown(f"**Policy Type:** {policy.get('policy_type', 'N/A')}")
#                 st.markdown(f"**Annual Premium:** ${policy.get('annual_premium', 0):,.2f}")
#                 st.markdown(f"**Coverage Amount:** ${policy.get('coverage_amount', 0):,}")
            
#             with pcol2:
#                 st.markdown(f"**Start Date:** {policy.get('start_date', 'N/A')}")
#                 st.markdown(f"**End Date:** {policy.get('end_date', 'N/A')}")
#                 st.markdown(f"**Auto Renew:** {policy.get('auto_renew', 'N/A')}")
#                 st.markdown(f"**Payment Frequency:** {policy.get('payment_frequency', 'N/A')}")

# def display_risk_assessment(risk_score: float, risk_level: str, analysis: Dict[str, Any]):
#     """Display risk assessment"""
#     st.markdown("### 📊 Risk Assessment")
    
#     # Risk level card
#     risk_emoji = get_risk_emoji(risk_level)
#     risk_color = get_risk_color(risk_level)
    
#     risk_class = f"risk-{risk_level.lower()}"
    
#     st.markdown(f"""
#     <div class="metric-card {risk_class}">
#         <h2 style="margin: 0; color: {risk_color};">
#             {risk_emoji} Overall Churn Risk: {risk_level}
#         </h2>
#         <p style="font-size: 1.5rem; margin: 0.5rem 0;">
#             Risk Score: {risk_score:.1f}/100
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Key metrics
#     st.markdown("#### Key Metrics")
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric(
#             "Cancellation Rate",
#             f"{analysis.get('cancellation_rate', 0):.1%}",
#             help="Percentage of policies that have been cancelled"
#         )
    
#     with col2:
#         st.metric(
#             "Customer Tenure",
#             f"{analysis.get('customer_tenure_months', 0):.1f} mo",
#             help="How long the customer has been with us"
#         )
    
#     with col3:
#         st.metric(
#             "Policy Diversity",
#             analysis.get('policy_diversity', 0),
#             help="Number of different policy types"
#         )
    
#     with col4:
#         st.metric(
#             "Recent Cancellations",
#             analysis.get('recent_cancellations', 0),
#             help="Cancellations in the last 12 months"
#         )

# def display_ai_analysis(llm_analysis: str):
#     """Display AI-generated analysis"""
#     st.markdown("### 🤖 AI-Powered Analysis")
    
#     st.markdown(f"""
#     <div class="info-box">
#         {llm_analysis}
#     </div>
#     """, unsafe_allow_html=True)

# def export_report(analysis_result: Dict[str, Any]):
#     """Export analysis report to JSON"""
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     customer_id = analysis_result.get('customer_id', 'unknown')
#     filename = f"churn_report_{customer_id}_{timestamp}.json"
    
#     report = {
#         "customer_id": customer_id,
#         "analysis_date": analysis_result.get('timestamp'),
#         "customer_data": analysis_result.get('customer_data'),
#         "policy_summary": {
#             "total_policies": len(analysis_result.get('policy_data', [])),
#             "policies": analysis_result.get('policy_data')
#         },
#         "churn_analysis": analysis_result.get('analysis'),
#         "risk_assessment": {
#             "risk_level": analysis_result.get('risk_level'),
#             "risk_score": analysis_result.get('risk_score')
#         }
#     }
    
#     return json.dumps(report, indent=2), filename

# # ============================================================================
# # CHAT AGENT FUNCTIONS
# # ============================================================================

# def display_chat_result_customer(customer: Dict[str, Any], idx: int):
#     """Display a single customer result"""
#     st.markdown(f"""
#     <div class="chat-result-card">
#         <h4>👤 Customer {idx}: {customer['first_name']} {customer['last_name']}</h4>
#         <p><strong>ID:</strong> {customer['customer_id']}</p>
#         <p><strong>Occupation:</strong> {customer['occupation']}</p>
#         <p><strong>Income:</strong> ${customer['annual_income']:,}</p>
#         <p><strong>Credit Score:</strong> {customer['credit_score']}</p>
#         <p><strong>Location:</strong> {customer['address']['city']}, {customer['address']['state']}</p>
#         <p><strong>Similarity Score:</strong> {customer.get('similarity_score', customer.get('hybrid_score', 0)):.4f}</p>
#     </div>
#     """, unsafe_allow_html=True)

# def display_chat_result_policy(policy: Dict[str, Any], idx: int):
#     """Display a single policy result"""
#     st.markdown(f"""
#     <div class="chat-result-card">
#         <h4>📋 Policy {idx}: {policy['policy_type']}</h4>
#         <p><strong>Policy Number:</strong> {policy['policy_number']}</p>
#         <p><strong>Customer ID:</strong> {policy['customer_id']}</p>
#         <p><strong>Status:</strong> {policy['status']}</p>
#         <p><strong>Premium:</strong> ${policy['annual_premium']:,.2f}/year</p>
#         <p><strong>Coverage:</strong> ${policy['coverage_amount']:,}</p>
#         <p><strong>Payment Frequency:</strong> {policy['payment_frequency']}</p>
#         <p><strong>Similarity Score:</strong> {policy.get('similarity_score', policy.get('hybrid_score', 0)):.4f}</p>
#     </div>
#     """, unsafe_allow_html=True)

# def perform_chat_search(query: str, search_mode: str, top_k: int = 5):
#     """Perform search using selected RAG system"""
#     try:
#         if search_mode == "Semantic Search":
#             rag_system = st.session_state.rag_semantic
#             results = rag_system.unified_search(query, top_k)
#         else:  # Hybrid Search
#             rag_system = st.session_state.rag_hybrid
#             results = rag_system.unified_search(query, top_k)
        
#         return {
#             'success': True,
#             'customers': results['customers'],
#             'policies': results['policies'],
#             'query': query,
#             'search_mode': search_mode,
#             'timestamp': datetime.now().isoformat()
#         }
#     except Exception as e:
#         return {
#             'success': False,
#             'error': str(e)
#         }

# # Main App
# def main():
#     # Header
#     st.markdown('<div class="main-header">🏢 Insurance - Customer Churn Prediction System</div>', unsafe_allow_html=True)
    
#     # Sidebar
#     with st.sidebar:
#         st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Insurance+AI", use_container_width=True)
        
#         st.markdown("## 📊 Dashboard")
        
#         # Initialize agent button
#         if st.session_state.agent is None:
#             if st.button("🚀 Initialize Agent", use_container_width=True):
#                 initialize_agent()
#                 st.success("✅ Azure Cosmos DB connection Successful")
#         else:
#             st.success("✅ Agent Ready")
        
#         st.markdown("---")
        
#         # Analysis history
#         st.markdown("### 📜 Analysis History")
#         if st.session_state.analysis_history:
#             for idx, analysis in enumerate(reversed(st.session_state.analysis_history[-5:]), 1):
#                 risk_emoji = get_risk_emoji(analysis.get('risk_level', 'UNKNOWN'))
#                 customer_id = analysis.get('customer_id', 'Unknown')[:20]
#                 if st.button(f"{risk_emoji} {customer_id}...", key=f"history_{idx}"):
#                     st.session_state.current_analysis = analysis
#         else:
#             st.info("No analysis history yet")
        
#         st.markdown("---")
        
#         # About
#         with st.expander("ℹ️ About"):
#             st.markdown("""
#             **Churn Prediction Agent**
            
#             This AI-powered system analyzes customer and policy data to predict churn risk and provide actionable retention recommendations.
            
#             **Features:**
#             - Real-time churn risk assessment
#             - AI-generated insights
#             - Policy portfolio analysis
#             - Export reports
#             - RAG-powered ChatAgent
#             """)
    
#     # Main content tabs
#     tab_analysis, tab_chat = st.tabs(["🔍 Customer Analysis", "💬 ChatAgent"])
    
#     # ========================================================================
#     # CUSTOMER ANALYSIS TAB (Original functionality)
#     # ========================================================================
#     with tab_analysis:
#         if st.session_state.agent is None:
#             st.info("👈 Please initialize the agent using the sidebar button to get started.")
#         else:
#             # Input section
#             st.markdown("## 🔍 Customer Analysis")
            
#             col1, col2 = st.columns([3, 1])
            
#             with col1:
#                 customer_id = st.text_input(
#                     "Enter Customer ID",
#                     placeholder="e.g., CUST-eadf24a1-f497-4351-be57-fb1ee8e34042",
#                     help="Enter the customer ID you want to analyze"
#                 )
            
#             with col2:
#                 st.markdown("<br>", unsafe_allow_html=True)
#                 analyze_button = st.button("🔬 Analyze Customer", type="primary", use_container_width=True)
            
#             # Analyze customer
#             if analyze_button and customer_id:
#                 with st.spinner("🧠 Analyzing customer data..."):
#                     result = analyze_customer(customer_id)
                    
#                     if result.get('success'):
#                         st.session_state.current_analysis = result
#                         st.session_state.analysis_history.append(result)
#                         st.success("✅ Analysis complete!")
#                     else:
#                         st.error(f"❌ {result.get('error', 'Analysis failed')}")
            
#             # Display current analysis
#             if st.session_state.current_analysis:
#                 result = st.session_state.current_analysis
                
#                 st.markdown("---")
                
#                 # Customer ID header
#                 st.markdown(f"## Analysis for Customer: `{result.get('customer_id')}`")
                
#                 # Tabs for different sections
#                 tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Risk Assessment", "👤 Customer Profile", "📋 Policies", "🤖 AI Analysis", "Communication Agent"])
                
#                 with tab1:
#                     display_risk_assessment(
#                         result.get('risk_score', 0),
#                         result.get('risk_level', 'UNKNOWN'),
#                         result.get('analysis', {})
#                     )
                
#                 with tab2:
#                     customer_data = result.get('customer_data')
#                     if customer_data:
#                         display_customer_profile(customer_data)
#                     else:
#                         st.warning("⚠️ Customer profile data not available")
                
#                 with tab3:
#                     policy_data = result.get('policy_data', [])
#                     display_policy_summary(policy_data)
                
#                 with tab4:
#                     analysis = result.get('analysis', {})
#                     llm_analysis = analysis.get('llm_analysis', 'No analysis available')
#                     display_ai_analysis(llm_analysis)
                
#                 with tab5:
#                     if st.button("Generate Email"):
#                         st.markdown(
#                             """
#                 **Dear Customer,**

#                 I hope this message finds you well. As your insurance advisor, I wanted to personally reach out regarding the renewal of your policy for the coming year. Over the last term, your policy has been quietly protecting you against uncertainties that often come without warning, and renewing it ensures that continuity of protection remains uninterrupted when it matters most.

#                 To make this decision easier, I am pleased to offer you a **special loyalty discount** available exclusively for valued customers like you. Renewing now not only secures your benefits at a reduced premium but also locks in your coverage before any future rate changes.

#                 I would strongly recommend renewing at the earliest so you can continue with complete peace of mind.

#                 **Warm regards,**  
#                 Insurance Advisor
#                             """
#                         )
                
#                 # Export button
#                 st.markdown("---")
#                 col1, col2, col3 = st.columns([1, 1, 2])
                
#                 with col1:
#                     report_json, filename = export_report(result)
#                     st.download_button(
#                         label="📥 Download Report (JSON)",
#                         data=report_json,
#                         file_name=filename,
#                         mime="application/json",
#                         use_container_width=True
#                     )
                
#                 with col2:
#                     if st.button("🔄 New Analysis", use_container_width=True):
#                         st.session_state.current_analysis = None
#                         st.rerun()
    
#     # ========================================================================
#     # CHAT AGENT TAB (New functionality)
#     # ========================================================================
#     with tab_chat:
#         st.markdown("## 💬 ChatAgent - RAG-Powered Search")
#         st.markdown("Search across customers and policies using advanced RAG techniques")
        
#         # Initialize RAG systems if not already done
#         if st.session_state.rag_semantic is None or st.session_state.rag_hybrid is None:
#             col1, col2, col3 = st.columns([1, 2, 1])
#             with col1:
#                 if st.button("🔄 Initialize RAG Systems", type="primary", use_container_width=True):
#                     if initialize_rag_systems():
#                         st.success("✅ RAG Systems initialized successfully!")
#                         st.rerun()
#         else:
#             # Search mode selection
#             st.markdown("### 🎯 Select Search Mode")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.markdown("""
#                 <div class="info-box">
#                     <h4>🔵 Semantic Search</h4>
#                     <p>Pure vector-based semantic search using embeddings. Best for conceptual queries and finding similar meanings.</p>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             with col2:
#                 st.markdown("""
#                 <div class="info-box" style="background-color: #d4edda; border-left-color: #28a745;">
#                     <h4>🟢 Hybrid Search</h4>
#                     <p>Combines semantic search with keyword matching. Best for queries with specific terms or mixed requirements.</p>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             search_mode = st.radio(
#                 "Choose your search mode:",
#                 ["Semantic Search", "Hybrid Search"],
#                 horizontal=True,
#                 help="Semantic Search uses pure vector similarity, while Hybrid Search combines vectors with keyword matching"
#             )
            
#             # Display badge for selected mode
#             badge_class = "semantic-badge" if search_mode == "Semantic Search" else "hybrid-badge"
#             st.markdown(f"""
#             <div class="search-mode-badge {badge_class}">
#                 Selected: {search_mode}
#             </div>
#             """, unsafe_allow_html=True)
            
#             st.markdown("---")
            
#             # Search interface
#             st.markdown("### 🔍 Search Query")
            
#             col1, col2 = st.columns([4, 1])
            
#             with col1:
#                 query = st.text_input(
#                     "Enter your search query",
#                     placeholder="e.g., 'high income engineers with auto insurance' or 'customers with cancelled policies'",
#                     help="Enter a natural language query to search across customers and policies"
#                 )
            
#             with col2:
#                 top_k = st.number_input("Results", min_value=1, max_value=20, value=5, help="Number of results per category")
            
#             search_button = st.button("🔎 Search", type="primary", use_container_width=True)
            
#             # Perform search
#             if search_button and query:
#                 with st.spinner(f"🔍 Searching using {search_mode}..."):
#                     search_result = perform_chat_search(query, search_mode, top_k)
                    
#                     if search_result.get('success'):
#                         # Add to chat history
#                         st.session_state.chat_history.append(search_result)
                        
#                         st.success(f"✅ Search completed using {search_mode}!")
                        
#                         # Display results
#                         st.markdown("---")
#                         st.markdown("### 📊 Search Results")
                        
#                         customers = search_result.get('customers', [])
#                         policies = search_result.get('policies', [])
                        
#                         # Summary metrics
#                         col1, col2, col3 = st.columns(3)
#                         with col1:
#                             st.metric("Customers Found", len(customers))
#                         with col2:
#                             st.metric("Policies Found", len(policies))
#                         with col3:
#                             st.metric("Search Mode", search_mode.split()[0])
                        
#                         # Display customers
#                         if customers:
#                             st.markdown("#### 👥 Customers")
#                             for idx, customer in enumerate(customers, 1):
#                                 display_chat_result_customer(customer, idx)
#                         else:
#                             st.info("No customers found matching your query")
                        
#                         # Display policies
#                         if policies:
#                             st.markdown("#### 📋 Policies")
#                             for idx, policy in enumerate(policies, 1):
#                                 display_chat_result_policy(policy, idx)
#                         else:
#                             st.info("No policies found matching your query")
                        
#                         # Export option
#                         st.markdown("---")
#                         export_data = json.dumps(search_result, indent=2, default=str)
#                         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                         filename = f"search_results_{search_mode.replace(' ', '_').lower()}_{timestamp}.json"
                        
#                         st.download_button(
#                             label="📥 Download Search Results",
#                             data=export_data,
#                             file_name=filename,
#                             mime="application/json",
#                             use_container_width=True
#                         )
#                     else:
#                         st.error(f"❌ Search failed: {search_result.get('error')}")


# if __name__ == "__main__":
#     main()


"""
streamlit_churn_app.py - Streamlit UI for Insurance Churn Prediction Agent
Updated with ChatAgent Tab for Semantic and Hybrid Search
"""

import streamlit as st
import json
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Import the agent
from churn_prediction_agent import ChurnPredictionAgent

# Import RAG systems
from unified_rag_system import UnifiedRAGSystem
from unified_rag_hybrid_system import UnifiedRAGHybridSystem

# Page configuration
st.set_page_config(
    page_title="Insurance Churn Prediction",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .risk-low {
        border-left-color: #28a745 !important;
        background-color: #d4edda;
    }
    
    .risk-medium {
        border-left-color: #ffc107 !important;
        background-color: #fff3cd;
    }
    
    .risk-high {
        border-left-color: #dc3545 !important;
        background-color: #f8d7da;
    }
    
    .risk-critical {
        border-left-color: #dc3545 !important;
        background-color: #f5c6cb;
    }
    
    .policy-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #155a8a;
    }
    
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .chat-result-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    
    .search-mode-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
    }
    
    .semantic-badge {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    
    .hybrid-badge {
        background-color: #d4edda;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'rag_semantic' not in st.session_state:
    st.session_state.rag_semantic = None
if 'rag_hybrid' not in st.session_state:
    st.session_state.rag_hybrid = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_agent():
    """Initialize the churn prediction agent"""
    try:
        with st.spinner("🚀 Initializing Churn Prediction Agent..."):
            agent = ChurnPredictionAgent()
            st.session_state.agent = agent
            return True
    except Exception as e:
        st.error(f"❌ Error initializing agent: {str(e)}")
        return False

def initialize_rag_systems():
    """Initialize RAG systems for ChatAgent"""
    try:
        with st.spinner("🔄 Initializing RAG Systems..."):
            if st.session_state.rag_semantic is None:
                st.session_state.rag_semantic = UnifiedRAGSystem()
            if st.session_state.rag_hybrid is None:
                st.session_state.rag_hybrid = UnifiedRAGHybridSystem()
            return True
    except Exception as e:
        st.error(f"❌ Error initializing RAG systems: {str(e)}")
        return False

def get_risk_color(risk_level: str) -> str:
    """Get color code for risk level"""
    colors = {
        "LOW": "#28a745",
        "MEDIUM": "#ffc107",
        "HIGH": "#dc3545",
        "CRITICAL": "#dc3545"
    }
    return colors.get(risk_level.upper(), "#6c757d")

def get_risk_emoji(risk_level: str) -> str:
    """Get emoji for risk level"""
    emojis = {
        "LOW": "🟢",
        "MEDIUM": "🟡",
        "HIGH": "🔴",
        "CRITICAL": "🔴"
    }
    return emojis.get(risk_level.upper(), "⚪")

def analyze_customer(customer_id: str) -> Dict[str, Any]:
    """Analyze a customer and return results"""
    agent = st.session_state.agent
    
    # Create initial state
    state = {
        "messages": [],
        "customer_id": customer_id,
        "customer_data": None,
        "policy_data": None,
        "churn_analysis": None,
        "stage": "retrieve"
    }
    
    # Retrieve data
    state = agent.retrieve_data_node(state)
    
    # Check if retrieval was successful
    if not state.get("customer_data") and not state.get("policy_data"):
        return {
            "success": False,
            "error": "No data found for this customer ID"
        }
    
    # Analyze
    state = agent.analyze_churn_node(state)
    
    # Calculate risk score
    analysis = state.get("churn_analysis", {})
    risk_score = agent._calculate_risk_score(analysis)
    risk_level = agent._get_risk_level(risk_score)
    
    return {
        "success": True,
        "customer_id": customer_id,
        "customer_data": state.get("customer_data"),
        "policy_data": state.get("policy_data"),
        "analysis": analysis,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "timestamp": datetime.now().isoformat()
    }

def display_customer_profile(customer_data: Dict[str, Any]):
    """Display customer profile information"""
    st.markdown("### 👤 Customer Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Name:**")
        st.write(f"{customer_data.get('first_name', 'N/A')} {customer_data.get('last_name', 'N/A')}")
        
        st.markdown("**Email:**")
        st.write(customer_data.get('email', 'N/A'))
    
    with col2:
        st.markdown("**Occupation:**")
        st.write(customer_data.get('occupation', 'N/A'))
        
        st.markdown("**Annual Income:**")
        income = customer_data.get('annual_income', 0)
        if income > 0:
            st.write(f"${income:,}")
        else:
            st.write("N/A")
    
    with col3:
        st.markdown("**Credit Score:**")
        credit = customer_data.get('credit_score', 0)
        if credit > 0:
            st.write(credit)
        else:
            st.write("N/A")
        
        st.markdown("**Customer Segment:**")
        metadata = customer_data.get('metadata', {})
        if metadata:
            st.write(metadata.get('customer_segment', 'N/A'))
        else:
            st.write("N/A")

def display_policy_summary(policy_data: List[Dict[str, Any]]):
    """Display policy summary"""
    st.markdown("### 📋 Policy Portfolio")
    
    if not policy_data:
        st.info("No policies found for this customer.")
        return
    
    # Summary metrics
    active_policies = sum(1 for p in policy_data if p.get('status') == 'ACTIVE')
    cancelled_policies = sum(1 for p in policy_data if p.get('status') == 'CANCELLED')
    expired_policies = sum(1 for p in policy_data if p.get('status') == 'EXPIRED')
    total_premium = sum(p.get('annual_premium', 0) for p in policy_data if p.get('status') == 'ACTIVE')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Policies", len(policy_data))
    with col2:
        st.metric("Active", active_policies, delta=None, delta_color="normal")
    with col3:
        st.metric("Cancelled", cancelled_policies, delta=None, delta_color="inverse")
    with col4:
        st.metric("Total Premium", f"${total_premium:,.2f}")
    
    # Policy details
    st.markdown("#### Policy Details")
    
    for idx, policy in enumerate(policy_data, 1):
        status = policy.get('status', 'Unknown')
        status_color = {
            'ACTIVE': '🟢',
            'CANCELLED': '🔴',
            'EXPIRED': '🟠',
            'PENDING': '🟡'
        }.get(status, '⚪')
        
        with st.expander(f"{status_color} Policy #{idx}: {policy.get('policy_type', 'N/A')} - {policy.get('policy_number', 'N/A')}"):
            pcol1, pcol2 = st.columns(2)
            
            with pcol1:
                st.markdown(f"**Status:** {status}")
                st.markdown(f"**Policy Type:** {policy.get('policy_type', 'N/A')}")
                st.markdown(f"**Annual Premium:** ${policy.get('annual_premium', 0):,.2f}")
                st.markdown(f"**Coverage Amount:** ${policy.get('coverage_amount', 0):,}")
            
            with pcol2:
                st.markdown(f"**Start Date:** {policy.get('start_date', 'N/A')}")
                st.markdown(f"**End Date:** {policy.get('end_date', 'N/A')}")
                st.markdown(f"**Auto Renew:** {policy.get('auto_renew', 'N/A')}")
                st.markdown(f"**Payment Frequency:** {policy.get('payment_frequency', 'N/A')}")

def display_risk_assessment(risk_score: float, risk_level: str, analysis: Dict[str, Any]):
    """Display risk assessment"""
    st.markdown("### 📊 Risk Assessment")
    
    # Risk level card
    risk_emoji = get_risk_emoji(risk_level)
    risk_color = get_risk_color(risk_level)
    
    risk_class = f"risk-{risk_level.lower()}"
    
    st.markdown(f"""
    <div class="metric-card {risk_class}">
        <h2 style="margin: 0; color: {risk_color};">
            {risk_emoji} Overall Churn Risk: {risk_level}
        </h2>
        <p style="font-size: 1.5rem; margin: 0.5rem 0;">
            Risk Score: {risk_score:.1f}/100
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    st.markdown("#### Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Cancellation Rate",
            f"{analysis.get('cancellation_rate', 0):.1%}",
            help="Percentage of policies that have been cancelled"
        )
    
    with col2:
        st.metric(
            "Customer Tenure",
            f"{analysis.get('customer_tenure_months', 0):.1f} mo",
            help="How long the customer has been with us"
        )
    
    with col3:
        st.metric(
            "Policy Diversity",
            analysis.get('policy_diversity', 0),
            help="Number of different policy types"
        )
    
    with col4:
        st.metric(
            "Recent Cancellations",
            analysis.get('recent_cancellations', 0),
            help="Cancellations in the last 12 months"
        )

def display_ai_analysis(llm_analysis: str):
    """Display AI-generated analysis"""
    st.markdown("### 🤖 AI-Powered Analysis")
    
    st.markdown(f"""
    <div class="info-box">
        {llm_analysis}
    </div>
    """, unsafe_allow_html=True)

def export_report(analysis_result: Dict[str, Any]):
    """Export analysis report to JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    customer_id = analysis_result.get('customer_id', 'unknown')
    filename = f"churn_report_{customer_id}_{timestamp}.json"
    
    report = {
        "customer_id": customer_id,
        "analysis_date": analysis_result.get('timestamp'),
        "customer_data": analysis_result.get('customer_data'),
        "policy_summary": {
            "total_policies": len(analysis_result.get('policy_data', [])),
            "policies": analysis_result.get('policy_data')
        },
        "churn_analysis": analysis_result.get('analysis'),
        "risk_assessment": {
            "risk_level": analysis_result.get('risk_level'),
            "risk_score": analysis_result.get('risk_score')
        }
    }
    
    return json.dumps(report, indent=2), filename

# ============================================================================
# CHAT AGENT FUNCTIONS
# ============================================================================

def display_chat_result_customer(customer: Dict[str, Any], idx: int):
    """Display a single customer result"""
    st.markdown(f"""
    <div class="chat-result-card">
        <h4>👤 Customer {idx}: {customer['first_name']} {customer['last_name']}</h4>
        <p><strong>ID:</strong> {customer['customer_id']}</p>
        <p><strong>Occupation:</strong> {customer['occupation']}</p>
        <p><strong>Income:</strong> ${customer['annual_income']:,}</p>
        <p><strong>Credit Score:</strong> {customer['credit_score']}</p>
        <p><strong>Location:</strong> {customer['address']['city']}, {customer['address']['state']}</p>
        <p><strong>Similarity Score:</strong> {customer.get('similarity_score', customer.get('hybrid_score', 0)):.4f}</p>
    </div>
    """, unsafe_allow_html=True)

def display_chat_result_policy(policy: Dict[str, Any], idx: int):
    """Display a single policy result"""
    st.markdown(f"""
    <div class="chat-result-card">
        <h4>📋 Policy {idx}: {policy['policy_type']}</h4>
        <p><strong>Policy Number:</strong> {policy['policy_number']}</p>
        <p><strong>Customer ID:</strong> {policy['customer_id']}</p>
        <p><strong>Status:</strong> {policy['status']}</p>
        <p><strong>Premium:</strong> ${policy['annual_premium']:,.2f}/year</p>
        <p><strong>Coverage:</strong> ${policy['coverage_amount']:,}</p>
        <p><strong>Payment Frequency:</strong> {policy['payment_frequency']}</p>
        <p><strong>Similarity Score:</strong> {policy.get('similarity_score', policy.get('hybrid_score', 0)):.4f}</p>
    </div>
    """, unsafe_allow_html=True)

def perform_chat_search(query: str, search_mode: str, top_k: int = 5):
    """Perform search using selected RAG system"""
    try:
        if search_mode == "Semantic Search":
            rag_system = st.session_state.rag_semantic
            results = rag_system.unified_search(query, top_k)
        else:  # Hybrid Search
            rag_system = st.session_state.rag_hybrid
            results = rag_system.unified_search(query, top_k)
        
        return {
            'success': True,
            'customers': results['customers'],
            'policies': results['policies'],
            'query': query,
            'search_mode': search_mode,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Main App
def main():
    # Header
    st.markdown('<div class="main-header">🏢 Insurance - Customer Churn Prediction System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Insurance+AI", use_container_width=True)
        
        st.markdown("## 📊 Dashboard")
        
        # Initialize agent button
        if st.session_state.agent is None:
            if st.button("🚀 Initialize Agent", use_container_width=True):
                initialize_agent()
                st.success("✅ Azure Cosmos DB connection Successful")
        else:
            st.success("✅ Agent Ready")
        
        st.markdown("---")
        
        # Analysis history
        st.markdown("### 📜 Analysis History")
        if st.session_state.analysis_history:
            for idx, analysis in enumerate(reversed(st.session_state.analysis_history[-5:]), 1):
                risk_emoji = get_risk_emoji(analysis.get('risk_level', 'UNKNOWN'))
                customer_id = analysis.get('customer_id', 'Unknown')[:20]
                if st.button(f"{risk_emoji} {customer_id}...", key=f"history_{idx}"):
                    st.session_state.current_analysis = analysis
        else:
            st.info("No analysis history yet")
        
        st.markdown("---")
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Churn Prediction Agent**
            
            This AI-powered system analyzes customer and policy data to predict churn risk and provide actionable retention recommendations.
            
            **Features:**
            - Real-time churn risk assessment
            - AI-generated insights
            - Policy portfolio analysis
            - Export reports
            - RAG-powered ChatAgent
            """)
    
    # Main content tabs
    tab_analysis, tab_chat = st.tabs(["🔍 Customer Analysis", "💬 ChatAgent"])
    
    # ========================================================================
    # CUSTOMER ANALYSIS TAB (Original functionality)
    # ========================================================================
    with tab_analysis:
        if st.session_state.agent is None:
            st.info("👈 Please initialize the agent using the sidebar button to get started.")
        else:
            # Input section
            st.markdown("## 🔍 Customer Analysis")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                customer_id = st.text_input(
                    "Enter Customer ID",
                    placeholder="e.g., CUST-eadf24a1-f497-4351-be57-fb1ee8e34042",
                    help="Enter the customer ID you want to analyze"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                analyze_button = st.button("🔬 Analyze Customer", type="primary", use_container_width=True)
            
            # Analyze customer
            if analyze_button and customer_id:
                with st.spinner("🧠 Analyzing customer data..."):
                    result = analyze_customer(customer_id)
                    
                    if result.get('success'):
                        st.session_state.current_analysis = result
                        st.session_state.analysis_history.append(result)
                        st.success("✅ Analysis complete!")
                    else:
                        st.error(f"❌ {result.get('error', 'Analysis failed')}")
            
            # Display current analysis
            if st.session_state.current_analysis:
                result = st.session_state.current_analysis
                
                st.markdown("---")
                
                # Customer ID header
                st.markdown(f"## Analysis for Customer: `{result.get('customer_id')}`")
                
                # Tabs for different sections
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Risk Assessment", "👤 Customer Profile", "📋 Policies", "🤖 AI Analysis", "Communication Agent"])
                
                with tab1:
                    display_risk_assessment(
                        result.get('risk_score', 0),
                        result.get('risk_level', 'UNKNOWN'),
                        result.get('analysis', {})
                    )
                
                with tab2:
                    customer_data = result.get('customer_data')
                    if customer_data:
                        display_customer_profile(customer_data)
                    else:
                        st.warning("⚠️ Customer profile data not available")
                
                with tab3:
                    policy_data = result.get('policy_data', [])
                    display_policy_summary(policy_data)
                
                with tab4:
                    analysis = result.get('analysis', {})
                    llm_analysis = analysis.get('llm_analysis', 'No analysis available')
                    display_ai_analysis(llm_analysis)
                
                with tab5:
                    if st.button("Generate Email"):
                        st.markdown(
                            """
                **Dear Customer,**

                I hope this message finds you well. As your insurance advisor, I wanted to personally reach out regarding the renewal of your policy for the coming year. Over the last term, your policy has been quietly protecting you against uncertainties that often come without warning, and renewing it ensures that continuity of protection remains uninterrupted when it matters most.

                To make this decision easier, I am pleased to offer you a **special loyalty discount** available exclusively for valued customers like you. Renewing now not only secures your benefits at a reduced premium but also locks in your coverage before any future rate changes.

                I would strongly recommend renewing at the earliest so you can continue with complete peace of mind.

                **Warm regards,**  
                Insurance Advisor
                            """
                        )
                
                # Export button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    report_json, filename = export_report(result)
                    st.download_button(
                        label="📥 Download Report (JSON)",
                        data=report_json,
                        file_name=filename,
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col2:
                    if st.button("🔄 New Analysis", use_container_width=True):
                        st.session_state.current_analysis = None
                        st.rerun()
    
    # ========================================================================
    # CHAT AGENT TAB (New functionality)
    # ========================================================================
    with tab_chat:
        st.markdown("## 💬 ChatAgent - RAG-Powered Search")
        st.markdown("Search across customers and policies using advanced RAG techniques")
        
        # Initialize RAG systems if not already done
        if st.session_state.rag_semantic is None or st.session_state.rag_hybrid is None:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("🔄 Initialize RAG Systems", type="primary", use_container_width=True):
                    if initialize_rag_systems():
                        st.success("✅ RAG Systems initialized successfully!")
                        st.rerun()
        else:
            # Search mode selection
            st.markdown("### 🎯 Select Search Mode")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="info-box">
                    <h4>🔵 Semantic Search</h4>
                    <p>Pure vector-based semantic search using embeddings. Best for conceptual queries and finding similar meanings.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="info-box" style="background-color: #d4edda; border-left-color: #28a745;">
                    <h4>🟢 Hybrid Search</h4>
                    <p>Combines semantic search with keyword matching. Best for queries with specific terms or mixed requirements.</p>
                </div>
                """, unsafe_allow_html=True)
            
            search_mode = st.radio(
                "Choose your search mode:",
                ["Semantic Search", "Hybrid Search"],
                horizontal=True,
                help="Semantic Search uses pure vector similarity, while Hybrid Search combines vectors with keyword matching"
            )
            
            # Display badge for selected mode
            badge_class = "semantic-badge" if search_mode == "Semantic Search" else "hybrid-badge"
            st.markdown(f"""
            <div class="search-mode-badge {badge_class}">
                Selected: {search_mode}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Search interface
            st.markdown("### 🔍 Search Query")
            
            col1, col2 = st.columns([4, 1])
            
            with col1:
                query = st.text_input(
                    "Enter your search query",
                    placeholder="e.g., 'high income engineers with auto insurance' or 'customers with cancelled policies'",
                    help="Enter a natural language query to search across customers and policies"
                )
            
            with col2:
                top_k = st.number_input("Results", min_value=1, max_value=20, value=5, help="Number of results per category")
            
            search_button = st.button("🔎 Search", type="primary", use_container_width=True)
            
            # Perform search
            if search_button and query:
                with st.spinner(f"🔍 Searching using {search_mode}..."):
                    search_result = perform_chat_search(query, search_mode, top_k)
                    
                    if search_result.get('success'):
                        # Add to chat history
                        st.session_state.chat_history.append(search_result)
                        
                        st.success(f"✅ Search completed using {search_mode}!")
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### 📊 Search Results")
                        
                        customers = search_result.get('customers', [])
                        policies = search_result.get('policies', [])
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Customers Found", len(customers))
                        with col2:
                            st.metric("Policies Found", len(policies))
                        with col3:
                            st.metric("Search Mode", search_mode.split()[0])
                        
                        # Display customers
                        if customers:
                            st.markdown("#### 👥 Customers")
                            for idx, customer in enumerate(customers, 1):
                                display_chat_result_customer(customer, idx)
                        else:
                            st.info("No customers found matching your query")
                        
                        # Display policies
                        if policies:
                            st.markdown("#### 📋 Policies")
                            for idx, policy in enumerate(policies, 1):
                                display_chat_result_policy(policy, idx)
                        else:
                            st.info("No policies found matching your query")
                        
                        # Export option
                        st.markdown("---")
                        export_data = json.dumps(search_result, indent=2, default=str)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"search_results_{search_mode.replace(' ', '_').lower()}_{timestamp}.json"
                        
                        st.download_button(
                            label="📥 Download Search Results",
                            data=export_data,
                            file_name=filename,
                            mime="application/json",
                            use_container_width=True
                        )
                    else:
                        st.error(f"❌ Search failed: {search_result.get('error')}")


if __name__ == "__main__":
    main()