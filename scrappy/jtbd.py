from pydantic import BaseModel, Field
from typing import List, Literal, ClassVar
from enum import Enum

CORE_GROWTH_INSTR = """
Core growth is the act of meeting unmet outcome expectations associated with a job that 
customers want to achieve.
This is the easiest way to innovate for most companies because it entails
perfecting the current paradigm.
For example, customers
want to pour juice into a cup with greater ease (desired outcome expectation)
and without the risk of spilling (undesired outcome expectation).
So the juice bottle is redesigned to have an indentation for easy gripping.
"""

RELATED_JOB_GROWTH_INSTR = """
This is the next easiest way to innovate and entails
bundling solutions that achieve the outcome expectations of more than one main or related JTBD.
Starbucks is an example of a solution that addresses
many jobs, such as drink caffeinated beverages, drink healthy alternative
beverages, carry on business conversations, surf the Internet, or study and
read books in a relaxing environment.
The key is to focus on adjacency: I want coffee, but I also want to read
a book and get on the Internet, or socialize with my friends. Or, I want a car
to rent so I can get from here to there, but I also want easy directions so I
also get a GPS in the car.
"""

NEW_JOB_GROWTH_INSTR = """
This is the product of evolving technology and change,
and it’s more difficult to achieve than core or related job growth. It entails
expanding the solution space to accomplish different JTBDs. Candle companies
that existed for decades, for instance, had to look for new applications
after the advent of the lightbulb. So they made products that were appealing
to those who wanted to decorate their homes or to create a romantic
environment for dinner. The JTBD was no longer to illuminate.
"""

DISRUPTIVE_GROWTH_INSTR = """
This focuses on what the literature and innovation
experts call nonconsumption.
Certain solutions are available to certain
classes of people, but not all or more people.
There are four drivers of nonconsumption: price, time, skill, and access
to the technology or solution.
This is the most difficult growth strategy to enact because it entails
cannibalizing what you and others in your industry do. Examples
of disruptive growth are home pregnancy tests, online stock trading, and
self-administered medical monitoring and treatment devices.
"""


FOCUS_MARKET_INSTR = '\n'.join(
    [CORE_GROWTH_INSTR, RELATED_JOB_GROWTH_INSTR, NEW_JOB_GROWTH_INSTR, DISRUPTIVE_GROWTH_INSTR]
)


class FocusMarket(BaseModel):

    description:ClassVar[str] = """
    The focus market for the job to be done (JTBD).
    Core and disruptive growth strategies are focused on existing JTBDs, while
    related and new job growth strategies are focused on new JTBDs. Also, core
    and related growth strategies are about serving existing customers, while new
    and disruptive growth strategies are about adding and serving new customers.
    """

    type: Literal['Core growth', 'Related job growth', 'New job growth', 'Disruptive growth']
    reasoning:str


class OutcomeExpectation(BaseModel):

    expectation_statement:str = Field(..., description = """
    Outcome expectation statement associated with a JTBD.
    """)

    outcome_type: Literal['Customer desired outcome',
                           'Customer undesired outcome',
                           'Provider desired outcome',
                           'Provider undesired outcome']


class JTBD(BaseModel):
    job_statement:str
    focus_market:FocusMarket = Field(..., description=FocusMarket.description)
    outcome_expectations:List[OutcomeExpectation]




