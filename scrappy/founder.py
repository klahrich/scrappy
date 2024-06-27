from scrappy.marvin_assistant import MarvinAssistant
from typing import List
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger('scrappy.founder')


FOUNDER_SYSTEM_PROMPT = """
You are an entrepreneur looking to solve a specific problem for a specific target market.
You are working closely with an Innovator who will give you a list of potential Jobs To Be Done (JTBDs).
Your job consists of going through through these JTBDs, identifying customer painpoints and potential solutions.
You will have access to the web to perform your research.
Your ultimate goal is to find one idea that is desirable (do people have this problem and want a solution for it),
feasible (it's possible to build a solution) and viable (there's a business model that makes sense).
Importantly, you must be able to tell how your solution differs from the competition.
"""

FOUNDER_USER_PROMPT_1 = """
Here's a JTBD:

{jtbd}

As a startup founder, answer the following:

- what is the problem/painpoint to be solved
- propose a solution to this problem
"""

FOUNDER_USER_PROMPT_2 = """
    What solutions exist for the following problem?

    {problem}

    Only list unique and specific solution providers. 
    Please do not repeat the same provider or solution multiple times.

    For each solution:
    - name the provider
    - give a max 2-sentence summary of the solution
    - list a few limitations of this solution, if there are any
"""

FOUNDER_USER_PROMPT_3 = """
    Given the problem:

    {problem}

    Given the existing solutions:

    {existing_solutions}

    And given your proposed solution:

    {proposed_solution}

    Is the proposed solution (as stated above) already differentiated from the existing solutions (Yes/No)?
    'Differentiated' means it has 1, 2 or 3 key features that make it a 10x 'better' solution versus the existing solutions.
    'Better' in this context can mean different things: cost, efficiency, coverage, speed, etc.

    Note that if you answer Yes, you should be able to explain how the proposed solution is different from any of the above existing solutions.

    If Yes, list the top 3 key dimensions along which the proposed solution is differentiated. Keep your answer short and snappy.
    If No, your job is one.
"""


class ProblemStatement(BaseModel):
    problem:str = Field(..., description="The customer problem or painpoint to be solved.")
    solution:str = Field(..., description="The proposed solution to the problem.")

class Solution(BaseModel):
    provider: str
    summary: str = Field(..., description="A one or two sentence summary description of the solution.")
    limitations: List[str] = Field(..., description="Maximum 3 limitations of the solution.")

class SolutionComparison(BaseModel):
    is_differentiated: bool
    explanation: List[str]|None = Field(..., description="Key ways in which the proposed solution is differentiated, if any.")

class Founder(MarvinAssistant):

    def __init__(self):
        super().__init__()
        self.ask(FOUNDER_SYSTEM_PROMPT)
        

    def problem_statement(self, jtbd:str) -> ProblemStatement:

        r = self.ask(FOUNDER_USER_PROMPT_1.format(jtbd=jtbd), target=ProblemStatement, as_list=False)
        return r
    
    def search_solutions(self, problem:ProblemStatement) -> List[Solution]:
        s = self.ask_perplexity(FOUNDER_USER_PROMPT_2.format(problem=problem.problem), target=Solution, as_list=True)
        return s
    
    def compare_solution(self, problem:ProblemStatement, existing_solutions:List[Solution]) -> SolutionComparison:

        s = self.ask(FOUNDER_USER_PROMPT_3.format(problem=problem.problem,
                                                proposed_solution=problem.solution,
                                                existing_solutions=[es.model_dump() for es in existing_solutions]),
                    target=SolutionComparison,
                    as_list=False
        )
        return s




    
    
