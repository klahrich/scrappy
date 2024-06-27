from typing import List, Tuple
from pydantic import BaseModel
import os
#from marvin.beta.assistants import Assistant
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

#marvin.settings.openai.chat.completions.model = 'gpt-4o'

#ai = Assistant(name="Researcher", instructions="You are an expert research analyst.")

print('start')

import marvin

print('imported fucking marvin')

class Question(BaseModel):
    question: str
    reasoning: str

class AnalysisPlan(BaseModel):
    questions: List[Question]

class Step(BaseModel):
    question: Question
    answer: str

class NextStep(BaseModel):
    step: Step
    follow_up: str

topic = "Payment API providers for businesses in Canada"

prompt = f"""
You are an expert research analyst.
Given the following topic, generate a list of initial questions to start your research on the web.
Topic: {topic}
"""

print('sending prompt')

completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are an expert research assistant."},
    {"role": "user", "content": prompt}
  ]
)

r = completion.choices[0].message

print(r)

analysis_plan = marvin.extract(r, target=AnalysisPlan)