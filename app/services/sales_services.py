import pandas as pd

from langchain_experimental.agents import create_pandas_dataframe_agent

from app.core.llm_factory import LLMFactory
from app.prompts.templates import SALES_AGENT_PREFIX


def analyze_sales_data(
    file_path: str,
    question: str,
    provider: str = "openai"
):

    # CSV / Excel
    if file_path.endswith(".csv"):

        df = pd.read_csv(file_path)

    else:

        df = pd.read_excel(file_path)

    llm = LLMFactory.get_llm(
        provider=provider,
        temperature=0.1
    )

    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        allow_dangerous_code=True,
        prefix=SALES_AGENT_PREFIX,
        agent_executor_kwargs={
            "handle_parsing_errors": True
        }
    )

    response = agent.invoke({
        "input": question
    })

    return response["output"]