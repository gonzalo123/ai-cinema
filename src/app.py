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
