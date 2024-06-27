from pydantic import BaseModel, Field
from typing import List
import marvin
from marvin.beta.assistants import Assistant
from marvin.beta.assistants import Thread
import os
from openai import OpenAI
import logging
from dotenv import load_dotenv

load_dotenv()

marvin.settings.openai.api_key = os.environ["OPENAI_API_KEY"]
PERPLEXITY_API_KEY = os["PERPLEXITY_API_KEY"]

client = OpenAI()

perplexity_client = OpenAI(api_key=PERPLEXITY_API_KEY,
                            base_url="https://api.perplexity.ai")

logger = logging.getLogger('scrappy.founder')


class Question(BaseModel):
    text: str
    reasoning: str

class QA(BaseModel):
    question: Question
    answer: str

class PitchReview(BaseModel):
    score: int
    investable: bool = Field(..., description="Would you invest in this idea? True or False.")
    summary: str = Field(..., description='High-level feedback summary')
    follow_up_questions: List[Question] | None


TOPIC = "Payment API providers for businesses in Canada. Emphasis on Canada."

FOUNDER_SYSTEM_PROMPT = """
You are an entrepreneur looking to do market research on a given topic.
Your ultimate goal is to find ideas that are desirable (do people have this problem and want a solution for it),
feasible (it's possible to build this solution) and viable (there's a business model that makes sense).
You also want to review existing solutions in the market and you want to make sure you have found a gap,
i.e. a problem that is not well solved by current solutions.
Finally, you want to be able to point to the target segment for this problem/solution.
"""

FOUNDER_USER_PROMPT = """
Given the following topic, generate a list of crutial initial questions to start your research. 
Limit yourself to a maximum of 5 questions.
As a founder, choose the most important questions to ask about this topic, in order to decide whether
there's a worthy business opportunity here.
For each question, also explain your reasoning for including it.
Topic: {}
"""

INVESTOR_SYSTEM_PROMPT = """
You are an investor reviewing an idea pitched to you by an entrepreneur.
The entrepreneur must convince you that they have studied the desirability, feasibility and viability of the idea.
Please review their pitch and decide whether they have proven to yo uthat their idea fills a gap
and is differentiated enough from the existing competition.
"""

INVESTOR_USER_PROMPT = """
Here is the pitch from the entrepreneur.
Are you convinced? Would you invest money in this idea?
If so, just answer Yes.
If not, list the questions would like to ask this entrepreneur to complement their pitch.

The pitch:

{}
"""

GOAL_QUESTION_1 = """
What is the market need?

Explanation: To understand a market need is to understand a (currently) unfilled desire or potential. 
New ventures generally begin with a hypothesis that this potential is real, 
but the first order of business is always to validate this assumption, 
and usually to revise the hypothesis on the basis of facts and evidence.
"""

GOAL_QUESTION_2 = """
What is a feasible solution?

Explanation: The entrepreneur will try to understand what will be required to offer a compelling solution 
that will actually spur the change in consumption required (solve the market need).
 This discipline will minimize that chance of creating a product concept that misses the mark or 
 fails to take into account critical dimensions of the need or possible inhibitors to adoption.
"""

GOAL_QUESTION_3 = """
What is the competition

Explanation: Competition is the universe of alternatives that the potential customer has. 
Competition must be considered from the customers perspective. 
What are the full set of alternatives that the customer has.
"""


class AIAgent:

    def __init__(self):
        self.thread = Thread()

    def ask_perplexity(self, messages:str|List, model="llama-3-sonar-small-32k-online"):
        if isinstance(messages, str):
            prompt = [{
                "role": "user",
                "content": messages
            }]
        else:
            prompt = messages

        response = perplexity_client.chat.completions.create(
            model=model,
            messages=prompt,
        )
        return response.choices[0].message.content

    def ask_assistant(self, messages:str|List, target=None, instructions=None, as_list=False):
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


class Founder(AIAgent):

    def __init__(self, topic):
        super().__init__()
        self.goal_questions = [GOAL_QUESTION_1, GOAL_QUESTION_2, GOAL_QUESTION_3]
        self.docket = []
        self.ai = Assistant(name="Founder", 
                            instructions=FOUNDER_SYSTEM_PROMPT)
        self.topic = topic
        self.user_prompt = FOUNDER_USER_PROMPT.format(topic)
        self.questions_answers:List[QA] = []

    def set_goal_questions(self, questions):
        self.goal_questions = questions

    def get_max_iter(self):
        pass

    def start(self) -> List[QA]:
        logger.info('Generating initial questions...')
        questions = self.ask_assistant(self.user_prompt,
                                target=Question,
                                as_list=True)

        for idx,q in enumerate(questions):
            logger.info(f'Answering question {idx+1}')

            a = self.ask_assistant(['Please give your answer to the following question:\n',
                            'Question: ' + q.text, 'Reasoning: ' + q.reasoning])

            self.questions_answers.append(QA(question=q, answer=a))

    def pitch(self):
        proposal = self.ask_assistant(
            ['Given the questions and answers obtained so far, propose an idea in this space:',
            self.topic,
            'Support your pitch with a succint explanation of why your idea is desirable, viable, and feasible.']
        )
        return proposal

    def improve_pitch(self, pitch_review:PitchReview):
        improvement_prompt = """
        You've received the following feedback on your pitch from an experienced investor.
        The feedback comes in the form of a series of questions that the investor wants you to answer,
        in order to satisfy the criteria of their investing framework.

        Please do your best to answer these questions. Here they are:
        """
        self.ask_assistant(improvement_prompt)

        for idx,q in enumerate(pitch_review.follow_up_questions):
            logger.info(f'Answering question {idx+1}')

            a = self.ask_assistant(['Question:\n' + q.text, 
                              'Reasoning:\n' + q.reasoning])

            self.questions_answers.append(QA(question=q, answer=a))

    def get_goal_questions(self):
        return self.goal_questions

    def get_docket(self) -> List[QA]:
        return self.questions_answers


class Investor(AIAgent):

    def __init__(self):
        super().__init__()
        self.ai = Assistant(name="Founder", 
                            instructions=INVESTOR_SYSTEM_PROMPT)

    def review(self, pitch):
        user_prompt = INVESTOR_USER_PROMPT.format(pitch)
        review = self.ask_assistant(user_prompt, target=PitchReview, as_list=False,
                            instructions=""" 
                            1) Please score this pitch on a scale of 1 to 5 along the three axes of 
                            desirability, viability, and feasibility, with respect of the important questions
                            that an investor would want to have answer to, before investing in this idea; 
                            1 being the worst score, 5 the best score.
                            A score of 1 means that none of the important questions have been properly answered.
                            A score of 2 means there are many important questions still unanswered.
                            A score of 3 means there are a few important questions still unanswered.
                            A score of 4 means there are a few minor questions that are unanswered,
                            but they could be answered later.
                            A score of 5 means that all questions have been answered.
                            
                            2) Would you invest in this idea?
                            
                            3) If no, what additional questions would you need the entrepreneur to answer
                            for you to be comfortable to invest?

                            4) Finally, please provide a high-level summary of your feedback to the entrepreneur.

                            """)
        return review