import os
import json
import streamlit as st

from crewai import Crew, Task

from agent import customer_support_agent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Support AI",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

if not os.getenv("GROQ_API_KEY"):

    st.error(
        "GROQ_API_KEY is not configured."
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Customer Support AI")

    st.subheader("Agent Tools")

    st.write("✓ Company Knowledge")
    st.write("✓ Order Database")
    st.write("✓ Human Escalation")

    st.divider()

    st.caption(
        "Single Agent Customer Support System"
    )


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Customer Support AI")

st.write(
    "Ask questions about company policies, products, "
    "orders, shipping, or request human support."
)


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================

user_message = st.chat_input(
    "How can I help you?"
)


if user_message:

    # --------------------------------------------------------
    # Show user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    with st.chat_message("user"):

        st.markdown(user_message)


    # --------------------------------------------------------
    # Build conversation history
    # --------------------------------------------------------

    conversation_history = ""

    for message in st.session_state.messages:

        role = message["role"].upper()

        conversation_history += (
            f"{role}: "
            f"{message['content']}\n"
        )


    # --------------------------------------------------------
    # Create task
    # --------------------------------------------------------

    task = Task(

        description=f"""
        Respond to the customer's latest message.

        Customer's latest message:
        {user_message}

        Previous conversation:
        {conversation_history}

        Use the available tools whenever necessary.

        Important:

        • Use company_knowledge_search for company information.
        • Use order_lookup for order related questions.
        • Use escalate_to_human when the customer asks for
          human support or the issue cannot be confidently
          resolved.
        • Do not invent information.
        • Remember information from the previous conversation.
        • Give the customer a clear and concise response.
        """,

        expected_output="""
        A helpful customer support response.
        If escalation was required, clearly state that the
        issue has been escalated and provide the ticket ID.
        """,

        agent=customer_support_agent
    )


    # --------------------------------------------------------
    # Create Crew
    # --------------------------------------------------------

    crew = Crew(

        agents=[
            customer_support_agent
        ],

        tasks=[
            task
        ],

        verbose=True
    )


    # --------------------------------------------------------
    # Run Agent
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Customer Support Agent is working..."
        ):

            try:

                result = crew.kickoff()

                response = str(result)

            except Exception as e:

                response = (
                    "I encountered a problem while processing "
                    "your request. Please try again."
                )

                st.error(
                    f"Error: {str(e)}"
                )


        st.markdown(response)


    # --------------------------------------------------------
    # Save response to conversation
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
