import json
import re
from datetime import datetime

from config import config
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.forecasting_agent.prompts.prompts import ForecastingPrompts
from src.forecasting_agent.tools.tools import (
    think,
    analyze,
    financial_data_extractor,
    qualitative_analysis,
)

class ForecastingAgent:
    def __init__(self):
        self.forecasting_model = config.forecasting_model
        
        self.forecasting_llm = ChatGroq(
            model=self.forecasting_model,
            temperature=0.1,
            max_retries=2
        )
        # self.forecasting_llm = ChatOpenAI(
        #     model = config.forecasting_model,
        #     temperature=0.1,
        #     reasoning_effort='high'
        # )
        self.forecasting_prompts = ForecastingPrompts()
        self.tools = [
            financial_data_extractor,
            qualitative_analysis,
            think,
            analyze,
        ]

    async def build_forecasting_prompt(self, query):

        output_format = """{
"financial_metrics_extracted": {
    "sales": " quarterly sales/revenue",
    "net_profit": "quaterly net profit",
    "operating_profit": "quarterly operating profit",
},
"qualitative_analysis": {
    "management_sentiment": "",
    "recurring_themes": "Key themes mentioned across multiple conferences",
    "forward_looking_statements": "Specific forward-looking statements made by management about future performance, guidance, or strategic direction",
},
"forecast": {
    "revenue_outlook": "Projected revenue outlook for upcoming quarters",
    "profitability_outlook": "Expected profitability trends and margin projections",
    "key_growth_drivers": "Main factors expected to drive growth",
    "risks": "Key risks and challenges identified",
    "opportunities": "Potential opportunities and growth areas"
}
}"""

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        system_prompt = self.forecasting_prompts.system_prompt.format(
            query=query,
            output_format=output_format,
            current_time=current_time
        )

        system_prompt = system_prompt.replace('{', '{{').replace('}', '}}')

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        return prompt

    async def extract_json_from_text(self, text):
        """Extract JSON from agent's text output."""
        if not text:
            return None
        
        cleaned_text = text.strip()
        
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[len("```json"):].strip()
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[len("```"):].strip()
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-len("```")].strip()
        
        try:
            return json.loads(cleaned_text)
        except Exception:
            pass
        
        # Try to extract key fields using regex patterns
        patterns = {
            'financial_metrics_extracted': r'"financial_metrics_extracted"\s*:\s*(\{[^}]*\})',
            'qualitative_analysis': r'"qualitative_analysis"\s*:\s*(\{[^}]*\})',
            'forecast': r'"forecast"\s*:\s*(\{[^}]*\})',
            'revenue_outlook': r'"revenue_outlook"\s*:\s*"([^"]*?)"',
            'profitability_outlook': r'"profitability_outlook"\s*:\s*"([^"]*?)"',
            'key_growth_drivers': r'"key_growth_drivers"\s*:\s*"([^"]*?)"',
            'risks': r'"risks"\s*:\s*"([^"]*?)"',
            'opportunities': r'"opportunities"\s*:\s*"([^"]*?)"',
            'management_sentiment': r'"management_sentiment"\s*:\s*"([^"]*?)"',
            'recurring_themes': r'"recurring_themes"\s*:\s*"([^"]*?)"',
            'forward_looking_statements': r'"forward_looking_statements"\s*:\s*"([^"]*?)"'
        }
        
        result = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, cleaned_text, re.DOTALL)
            if match:
                result[key] = match.group(1).strip()
        
        return result if result else None

    async def forecasting_call(self, query, session_id=None):
        try:
            forecasting_prompt = await self.build_forecasting_prompt(query)

            agent = create_tool_calling_agent(
                self.forecasting_llm, self.tools, forecasting_prompt
            )
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                stream_runnable=False,
                max_iterations=25,
                return_intermediate_steps=True
            )

            with get_openai_callback() as cb:
                forecast_resp = await agent_executor.ainvoke({
                    "input": query
                })
                usage_info = {
                    'model': self.forecasting_model,
                    'usage': {
                        'prompt_tokens': cb.prompt_tokens,
                        'prompt_tokens_details_cached_tokens': cb.prompt_tokens_cached,
                        'completion_tokens': cb.completion_tokens,
                        'total_tokens': cb.total_tokens,
                    }
                }
                print(f"Token usage: {usage_info}")

            forecast_response = await self.extract_json_from_text(forecast_resp['output'])

            if forecast_response:
                print(f"{session_id or 'N/A'}: Successfully generated forecast")
                return {
                    'status_code': 200,
                    'forecast_data': forecast_response
                }
            else:
                print(
                    f"WARNING: {session_id or 'N/A'}: Failed to extract forecast response from agent output"
                )
                return {
                    'status_code': 500,
                    'status_messages': 'Failed to extract forecast response from agent output'
                }

        except Exception as e:
            print(
                f"ERROR: {session_id or 'N/A'}: There was an error while forecasting: {e}"
            )
            return {
                'status_code': 500,
                'status_messages': f'There was an error while forecasting: {e}'
            }
