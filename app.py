import streamlit as st
from swarm import Swarm, Agent
from datetime import datetime
from dotenv import load_dotenv
import requests
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
MODEL = "gpt-4o"

# Initialize Swarm client
client = Swarm()

# Perplexity API setup
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

def search_web(query):
    logger.info(f"Starting 2024 research for topic: {query}")
    
    # Add 2024 to the query explicitly
    enhanced_query = f"{query} 2024 latest developments"
    
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": """You are a professional research assistant focused on 2024 developments. 
                Prioritize information from 2024, and if not available, only use very recent sources. 
                Ensure all insights and trends are current and relevant to 2024."""
            },
            {
                "role": "user",
                "content": enhanced_query
            }
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "return_citations": True,
        "search_domain_filter": [],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "day",  # Changed from "month" to "day" for more recent results
        "stream": False
    }
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        logger.info("Gathering information from reliable sources")
        response = requests.post(PERPLEXITY_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        logger.info("Successfully gathered information")
        
        content = result['choices'][0]['message']['content']
        citations = result.get('citations', [])
        
        formatted_response = f"{content}\n\nSources:\n"
        for citation in citations:
            formatted_response += f"- {citation['title']}: {citation['url']}\n"
            
        logger.info("Research compilation complete")
        return formatted_response
        
    except Exception as e:
        logger.error(f"Research gathering error: {str(e)}")
        return f"Error during research: {str(e)}"

# Research Specialist for initial information gathering
research_specialist = Agent(
    name="Research Specialist",
    instructions="""You are a professional research specialist focused on finding compelling 2024 stories and insights.
    When researching, strictly prioritize:
    - Information and developments from 2024
    - Latest industry trends and breaking news
    - Most recent expert opinions and forecasts
    - Up-to-date statistics and data from 2024
    - Current developments and emerging trends
    
    Research Guidelines:
    - Always verify the date of information
    - Prioritize sources from 2024
    - If using older sources, explicitly note when the information is from
    - Focus on forward-looking insights and predictions for 2024
    - Highlight what's new or different in 2024 compared to previous years
    
    Avoid:
    - Outdated information from previous years
    - Generic or timeless content
    - Historical perspectives unless directly relevant to current trends
    
    The goal is to provide the most current and relevant insights for a LinkedIn post that demonstrates 
    knowledge of the latest developments.""",
    functions=[search_web],
    model=MODEL
)

# Content Strategist for narrative development
content_strategist = Agent(
    name="Content Strategist",
    instructions="""You are an expert content strategist who transforms current research into compelling narratives.
    Your role is to:
    1. Prioritize 2024-specific insights and developments
    2. Identify emerging trends and their potential impact
    3. Connect current events to future implications
    4. Highlight what's new and different in 2024
    5. Focus on forward-looking perspectives
    6. Include specific 2024 statistics, launches, or developments
    
    Remember: 
    - The audience wants to know what's happening NOW
    - Always emphasize the timeliness of information
    - Make connections between current events and future trends
    - Position insights in the context of 2024's unique challenges and opportunities
    
    The goal is to make readers feel informed about the very latest developments in their field.""",
    model=MODEL
)

# LinkedIn Content Writer for final post creation
linkedin_writer = Agent(
    name="LinkedIn Writer",
    instructions="""You are me - a tech professional writing a LinkedIn post about current trends and developments. 
    Write in a conversational, authentic voice that demonstrates awareness of the latest industry movements.

    Key guidelines:
    - Start with a 2024-specific hook or insight
    - Reference current events or recent developments
    - Use phrases like "Just this week..." or "In recent developments..."
    - Include specific dates or timeframes when mentioning developments
    - End with a forward-looking question or thought about where things are heading
    - Keep everything grounded in current context
    
    Style requirements:
    - Write how people actually talk
    - No corporate jargon or buzzwords
    - No emojis or excessive punctuation
    - Keep paragraphs short and scannable
    - Aim for 4-6 short paragraphs maximum
    
    Remember: The post should feel like it's coming from someone who's actively following and 
    engaging with the latest developments in their field. Make it clear this is current, 
    relevant information, not evergreen content.""",
    model=MODEL
)

def run_workflow(query):
    logger.info(f"Starting content creation workflow for: {query}")
    
    # Initial research phase
    logger.info("Starting initial research phase")
    research_response = client.run(
        agent=research_specialist,
        messages=[{"role": "user", "content": f"Research this topic thoroughly: {query}"}],
    )
    logger.info("Initial research completed")
    
    raw_research = research_response.messages[-1]["content"]
    logger.info("Research data collected")

    # Content strategy phase
    logger.info("Starting content strategy phase")
    strategy_response = client.run(
        agent=content_strategist,
        messages=[{"role": "user", "content": raw_research }],
    )
    logger.info("Content strategy completed")

    structured_content = strategy_response.messages[-1]["content"]
    
    # LinkedIn post creation phase
    logger.info("Starting LinkedIn post creation")
    return client.run(
        agent=linkedin_writer,
        messages=[{"role": "user", "content": structured_content }],
        stream=True
    )

def main():
    st.set_page_config(page_title="LinkedIn Content Generator", page_icon="📝")
    st.title("LinkedIn Content Generator")
    st.subheader("Transform any topic into an engaging LinkedIn post")

    if 'topic' not in st.session_state:
        st.session_state.topic = ""
    if 'post' not in st.session_state:
        st.session_state.post = ""

    col1, col2 = st.columns([3, 1])

    with col1:
        topic = st.text_input("Enter your topic:", value=st.session_state.topic, 
                            placeholder="e.g., AI in healthcare, remote work trends, sustainability...")

    with col2:
        if st.button("Clear"):
            st.session_state.topic = ""
            st.session_state.post = ""
            st.rerun()

    if st.button("Generate LinkedIn Post") and topic:
        with st.spinner("Crafting your LinkedIn post..."):
            logger.info("Starting post generation")
            streaming_response = run_workflow(topic)
            st.session_state.topic = topic
            
            message_placeholder = st.empty()
            full_response = ""
            
            for chunk in streaming_response:
                if isinstance(chunk, dict) and 'delim' in chunk:
                    logger.info(f"Processing chunk: {chunk['delim']}")
                    continue
                    
                if isinstance(chunk, dict) and 'content' in chunk:
                    content = chunk.get('content', '')
                    if content:
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")
            
            logger.info("Post generation completed")
            message_placeholder.markdown(full_response)
            st.session_state.post = full_response


if __name__ == "__main__":
    logger.info("Starting LinkedIn Content Generator")
    main()