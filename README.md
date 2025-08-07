# Agentic IA for movie recommendations with Python and Strands Agents

Context 1: I like to go to the cinema. I normally go to the cinema on Saturday afternoons, at the first showing. In the city where I live there are three cinemas and all belong to the same company called Sade. I normally check the cinema schedules on their website, SadeCines.com, to see what's playing. Also, I track the movies I see on Letterboxd. There I have my diary and also a list with the movies I see in the cinema. I rate the movies when I finish watching them. My first impression. I do that not to share with others, only to have a personal record of what I like and dislike.

Context 2: I'm on holidays and I like to code also, so I decided to build an AI agent that helps me decide what movie to watch on Saturday afternoons. This project is an example of over-engineering, I know, but I've done it as an exercise using Strands Agents, a framework for building multi-tool LLM agents that I'm using these days.

The aim of the project is to create an AI agent that can access the internet to check the cinema schedules, my Letterboxd profile, and then recommend me a movie to watch on Saturday afternoons. Normally the LLMs are good at reasoning, but they don't have access to the internet. Also, they are not good at doing mathematical operations, but with agents we can use tools to do that. So I decided to build an agent that can access the internet (to check the cinema schedules, my Letterboxd profile and IMDb/Metacritic's scores) and create the needed code to do the mathematical operations needed.

Strands Agents (it is similar to LangChain) allows us to build multi-tool LLM agents. In this example I'm using the pre-built tools provided by the framework, like:
- **calculator**: for performing mathematical operations
- **think**: for reasoning and decision-making
- **current_time**: to get the current date and time
- **file_write**: to write the recommendations to a file
- **batch**: to execute multiple tools in parallel
- **code_interpreter**: to execute Python code dynamically (sandboxed in an AWS environment)
- **browser**: to scrape the cinema schedules from SadeCines.com and my Letterboxd profile (also sandboxed in an AWS environment)

Code interpreter is a powerful tool that allows us to execute Python code dynamically, which is useful for performing mathematical operations and data processing. For me it is the key to push the agents to the next level. LLMs can generate python code very well. They can generate code to build a Pandas dataframe, to filter the data, to calculate the average rating, etc. But they can also generate code that can be harmful, like deleting files, or accessing sensitive data. So we need to be careful with the code we execute. This issue is especially important when we are using prompts from users (in a chat, for example). Strands Agents provides a tool called **python-repl** that allows us to execute Python code locally within our environment. If you rely on your prompts it can be an option (I've sent a pull request to Strands Agents to make it a bit more safe). But in this project I'm using the **code_interpreter** tool, which is a sandboxed environment provided by AWS. This allows us to execute Python code safely without the risk of executing harmful code in your host environment.

In this project we need to scrape webpages to retrieve information from internet. Strands Agents provides us a built-in tool, called **use_browser**, to use a headless browser locally to access the Internet. In this project, I'm using the **browser** tool, which is also a sandboxed environment provided by AWS Bedrock. This allows us to scrape webs (using Playwright) and interact with web pages without the risk of executing harmful code in your host environment.

With this information, to build the agent is pretty straightforward. The idea of agents is not to code everything from scratch, but to provide to the agent the needed tools to solve the problem, and let the agent figure out how to use them using the prompts. When we work with LLM we have two kinds of prompts: the system prompt and the user prompt. The system prompt is used to define the agent's behavior, while the user prompt is used to provide the input data.

In this project I'm using those prompts:
```python
from settings import BASE_DIR

SYSTEM_PROMPT = f"""
You are an expert movie recommendation assistant to help me decide what to watch.

You have access to the following URLs and available movie analyses:
- https://sadecines.com/ With the movie schedules in my city's cinemas.
    Sadecines has a checkbox to filter the day of the week, so you can select Saturday.
- https://letterboxd.com/gonzalo123/films/diary/ Movies I have watched and rated.
- https://letterboxd.com/gonzalo123/list/cine-2025/detail/ Movies I have already seen in theaters in 2025.

You must take into account the user's preferences:
- Avoid movies in the "children" and "family" genres.
- I don't really like intimate or drama movies, except for rare exceptions.
- I like entertaining movies, action, science fiction, adventure, and comedies.

Take into account when making recommendations:
- The ratings of the movies on IMDb and Metacritic.
- But mainly consider my personal preferences,
    which can be seen in the list of movies I have watched and rated on Letterboxd.
"""

QUESTION = f"""
Analyze the movies showing this Saturday in the first session.

Present only those you recommend, excluding those not relevant according to my preferences,
and order them from best to worst according to your criteria.

Show the result in a table with the following columns:
- Title
- Genre
- IMDb Rating
- Metacritic Rating
- Summary
- Start Time
- End Time

Save the final report in a file named YYYYMMDD.md, following this structure:
{BASE_DIR}/
    └ reports/
        └ YYYYMMDD.md       # Movie analysis of the day, format `YYYYMMDD`
"""
```

And the code of the agent is very simple (I'm using AWS Bedrock to run the agent)

```python
import logging

from botocore.config import Config
from strands import Agent
from strands.models import BedrockModel
from strands_tools import calculator, current_time, think, file_write, batch
from strands_tools.browser import AgentCoreBrowser
from strands_tools.code_interpreter import AgentCoreCodeInterpreter

from promts import SYSTEM_PROMPT, QUESTION
from settings import AWS_REGION, MODEL_TEMPERATURE, MODEL, LLM_READ_TIMEOUT, LLM_CONNECT_TIMEOUT, LLM_MAX_ATTEMPTS

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level="INFO",
    datefmt="%d/%m/%Y %X",
)

logger = logging.getLogger(__name__)

agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    model=BedrockModel(
        model_id=MODEL,
        temperature=MODEL_TEMPERATURE,
        boto_client_config=Config(
            read_timeout=LLM_READ_TIMEOUT,
            connect_timeout=LLM_CONNECT_TIMEOUT,
            retries={'max_attempts': LLM_MAX_ATTEMPTS}
        )
    ),
    tools=[
        calculator, think, current_time, file_write, batch,
        AgentCoreCodeInterpreter(region=AWS_REGION).code_interpreter,
        AgentCoreBrowser(region=AWS_REGION).browser]
)

result = agent(QUESTION)
logger.info(f"Total tokens: {result.metrics.accumulated_usage['totalTokens']}")
logger.info(f"Execution time: {sum(result.metrics.cycle_durations):.2f} seconds")
logger.info(f"Tools used: {list(result.metrics.tool_metrics.keys())}")
```

The lines of code never is a goal (we only need to write readable and maintainable code), but in this example we have more code in the prompts than in the code itself. Maybe it's the sigh of our times.

And that's all. I must say again that this project is just an example. It is an over-engineering example. Scaling this project would be very expensive. Working a little bit in a custom scraper in addition to custom python code, can do the same to solve this specific problem without the usage, and paid, the IA (cheap for a single user usage, but expensive when scaled). I think it is a good example to show how Agents and the power of the code interpreter and the browser tools in a few lines of code. And remember, I'm on holidays and I like to code (don't blame me for that).
