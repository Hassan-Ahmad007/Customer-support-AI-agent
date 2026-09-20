import streamlit as st
from crewai import Agent, LLM

from tools import company_knowledge_search, order_lookup
from escalation import escalate_to_human


api_key = st.secrets["GROQ_API_KEY"]


llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0.2
)


customer_support_agent = Agent(
    role="Customer Support Specialist",

    goal=(
        "Help customers by answering company questions, "
        "checking orders, remembering conversation context, "
        "and escalating unresolved issues to human support."
    ),

    backstory=(
        "You are a customer support specialist. "
        "Use the company knowledge tool for company information, "
        "policies, products, returns, refunds, shipping, and troubleshooting. "
        "Use the order lookup tool for order information. "
        "Never invent information. "
        "If the customer asks for a human, escalate the request. "
        "If you cannot confidently solve the issue using available tools, "
        "escalate it. "
        "When escalating, create a short issue summary and provide the ticket ID. "
        "Use previous conversation messages to understand follow up questions."
    ),

    tools=[
        company_knowledge_search,
        order_lookup,
        escalate_to_human
    ],

    llm=llm,

    verbose=True,
    use_system_prompt=False,

    allow_delegation=False
)
