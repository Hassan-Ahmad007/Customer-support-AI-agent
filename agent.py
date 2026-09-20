import os
import streamlit as st

from crewai import Agent, LLM

from tools import company_knowledge_search, order_lookup
from escalation import escalate_to_human


api_key = st.secrets["GROQ_API_KEY"]


llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=api_key
)


customer_support_agent = Agent(
    role="Customer Support Specialist",

    goal=(
        "Help customers by answering company related questions, "
        "checking order information, remembering the conversation context, "
        "and escalating unresolved issues to a human when necessary."
    ),

    backstory=(
        "You are a customer support specialist for the company. "
        "You have access to three tools. "
        "The first searches the company's internal knowledge. "
        "The second checks customer order information. "
        "The third escalates issues to human support. "

        "Use the company knowledge tool for questions about products, "
        "policies, returns, refunds, shipping, troubleshooting, and other "
        "company information. "

        "Use the order lookup tool when the customer asks about an order. "

        "Never invent company policies, product information, or order details. "

        "If the customer asks to speak to a human, escalate the request. "

        "If you cannot confidently solve the customer's problem using the "
        "available information, escalate it to human support. "

        "When escalating, create a short and useful summary of the issue "
        "and provide the customer with the generated ticket ID. "

        "Use the conversation history to understand follow up questions."
    ),

    tools=[
        company_knowledge_search,
        order_lookup,
        escalate_to_human
    ],

    llm=llm,

    verbose=True,

    allow_delegation=False
)
