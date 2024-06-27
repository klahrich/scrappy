from typing import List
import marvin
from marvin.beta.assistants import Assistant
from marvin.beta.assistants import Thread
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

FOUNDER_SYSTEM_PROMPT = """
You are an entrepreneur and innovator looking for a novel idea.
Your ultimate goal is to find ideas that are desirable (do people have this problem and want a solution for it),
feasible (it's possible to build this solution) and viable (there's a business model that makes sense).
You want to review existing solutions in the market and make sure you have found a gap,
i.e. a problem that is not well solved by current solutions.
Finally, you want to be able to point to the target segment for this problem and your solution.
"""

marvin.settings.openai.api_key = os.environ["OPENAI_API_KEY"]
PERPLEXITY_API_KEY = os["PERPLEXITY_API_KEY"]


class MarvinAssistant:

    openai_client = OpenAI()

    perplexity_client = OpenAI(api_key=PERPLEXITY_API_KEY,
                            base_url="https://api.perplexity.ai")

    def __init__(self):
        self.thread = Thread()
        self.ai = Assistant(name="Founder", 
                            instructions=FOUNDER_SYSTEM_PROMPT)

    def ask(self, messages:str|List, target=None, instructions=None, as_list=False):
            if isinstance(messages, list):
                for prompt in messages:
                    self.thread.add(prompt)
            else:
                self.thread.add(messages)
            self.thread.run(self.ai)
            messages = self.thread.get_messages()
            last_message = messages[-1]
            last_message_txt = '\n'.join([t.text.value for t in last_message.content])

            if target is None:
                return last_message_txt
            else:
                if as_list:
                    return marvin.extract(last_message_txt, 
                                            target=target, 
                                            instructions=instructions) 
                else:
                    return marvin.cast(last_message_txt, 
                                        target=target, 
                                        instructions=instructions)

    def ask_perplexity(self, messages:str|List, model="llama-3-sonar-small-32k-chat", target=None, instructions=None, as_list=False):

        if isinstance(messages, str):
            prompt = [{
                "role": "user",
                "content": messages
            }]
        else:
            prompt = messages

        response = self.perplexity_client.chat.completions.create(
            model=model,
            messages=prompt,
        )

        self.thread.add(f"""
        Please add the following conversation to your knowledge base.
                        
        {prompt}

        {response.choices[0].message.content}
        """)

        if target is None:
            return response.choices[0].message.content
        
        else:
            if as_list:
                return marvin.extract(response.choices[0].message.content,
                                      target=target,
                                      instructions=instructions)
            else:
                return marvin.cast(response.choices[0].message.content,
                                      target=target,
                                      instructions=instructions)