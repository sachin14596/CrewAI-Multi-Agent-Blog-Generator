from crewai import  Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
import streamlit as st
import time
from litellm import RateLimitError

from dotenv import load_dotenv
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="Content Researcher & Writer", page_icon="", layout="wide")

# Title and description
st.title("Content Researcher & Writer, Powered by CrewAI")
st.markdown("Generate blog posts about any topic using AI agents")

with st.sidebar:
    st.header("Content Settings")

    # Make the text input take up more space
    topic = st.text_area(
        "Enter Your Topipc",
        height=100,
        placeholder="Enter the Topic"
    )

    # Add more sidebar controls if needed
    st.markdown("### LLM Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

    # Add some spacing
    st.markdown("---")

    # Make the generate button moreprominent in the sidebar
    generate_button = st.button("Generate Content", type="primary", use_container_width=True)

    # Add some helpful information
    with st.expander("How to use"):
        st.markdown("""
                    1. Enter your desired content topic
                    2. Play with the temperature
                    3. Click 'Generate Content' to start
                    4. Wait for the AI to generate your article
                    5. Download the result as a markdown file
                    """)

def generate_content(topic):
    llm = LLM(model="gpt-3.5-turbo-0125") 
    

    search_tool=SerperDevTool(n_results=10)

    # Agent 1: Senior Research Analyst
    senior_research_analyst = Agent(
    role = "Senior Research Analyst",
    goal = f"Research, Analyze and Synthesize comprehensive information on {topic} from web sources",
    backstory = "You're an expert research analyst with advanced web research skills"
                "You excel at finding, analyzing and synthesizing information from"
                "across the internet using search tools. You're skilled at"
                "distinguishing reliable sources from unreliable ones,"
                "fact-checking, cross-referencing information, and"
                "identifying key patterns and insights. You provide"
                "well-organized research briefs with proper citations"
                "and source verification. Your analysis includes both"
                "raw data and interpreted insights, making complex"
                "information accessible and actionable.",
    allow_delegation = False,
    verbose = False,
    tools = [search_tool],
    llm = llm
    )

    # Agent 2: Content Writer
    content_writer = Agent(
    role = "Content Writer",
    goal = "Transform research findings into engaging blog posts while maintaining accuracy",
    backstory = "You're a skilled content writer specialized in creating"
                "engaging, accessible content from technical research."
                "You work closely with the Senior Research Analyst and excel at maintaining the perfect"
                "balance between informative and citations from the research"
                "are properly incorporated. You have a talent for making"
                "complex topics approachable without oversimplifying them",
    allow_delegation = False,
    verbose = False,
    llm = llm
    )

    # Task 1: Research

    research_tasks = Task(
        description = ("""
                        1. Conduct comprehensive research on {topic} including:
                            - Recent development and news
                            - Key industry trends and innovations
                            - Expert opinions and analyses
                            - Statistical data and market insights
                        2. Evaluate source credibility and fact-checking all information
                        3. Organize findings into a structured research brief
                        4. Include all relevant citations and sources
                        """
                    ),
        expected_output =  """
                            A detailed research report containing:
                            - Executive summary of key findings
                            - Comprehensive analysis of current trends and developments
                            - List of verified facts and statistics
                            - All citations and links to original sources
                            - Clear categorization of main themes and patterns
                            Please format with clear sections and bullet points for easy references.
                            """,
        agent = senior_research_analyst
    )


    # Task 2: Content Writing
    writing_task = Task(
        description = ("""
                    Using the research brief provided, create an engaging blog post that:
                    1. Transform technical information into accessible content
                    2. Maintains all factual accuracy and citations from the research
                    3. Includes:
                            - Attention-grabbing introduction
                            - Well-structured body sections with clear headings
                            - Compelling conclusion
                    4. Preserves all sources citations in [Source: URL] format
                    5. Includes a references section at the end
                    """),
        expected_output = """
                            A polished blog post in markdown format that:
                            - Engages readers while maintaining accuracy
                            - Contains properly structured sections
                            - Includes inline citations hyperlinked to the original source url
                            - Presents information in an accessible yet informative way
                            - Follows proper markdown formatting, use H1 for the title and H3 for the sub-sections
                            """,
        agent = content_writer
    )

    crew = Crew(
        agents = [senior_research_analyst, content_writer],
        tasks = [research_tasks,writing_task],
        verbose = False
    )

    return crew.kickoff(inputs={"topic": topic})

def safe_generate_content(topic, retries=5, wait_time=25):
    for i in range(retries):
        try:
            return generate_content(topic)
        except RateLimitError as e:
            st.warning(f"Rate limit hit. Retrying in {wait_time} seconds... ({i+1}/{retries})")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded due to rate limiting.")


# Main content area
if generate_button:
    with st.spinner('Generating Content... This may take a moment'):
        try:
            result = safe_generate_content(topic)

            if result and hasattr(result, 'raw') and result.raw:
                st.markdown("### Generated Content")
                st.markdown(result.raw)

                st.download_button(
                    label="Download Content",
                    data=result.raw,
                    file_name=f"{topic.lower().replace(' ', '_')}_article.md",
                    mime="text/markdown"
                )
            else:
                st.error("LLM returned an empty or invalid response. Please try again.")

        except Exception as e:
            st.error(f"An Error Occurred: {str(e)}")

