from scrappy.jtbd import JTBD
from scrappy.marvin_assistant import MarvinAssistant
from scrappy.jtbd import FOCUS_MARKET_INSTR, OutcomeExpectation
import marvin


class Innovator(MarvinAssistant):

    def __init__(self, field:str):
        super().__init__()
        self.field = field
        self.system_prompt = f"""
        You are an innovator in the field of: {self.field}

        You are looking to identify that jobs customers are trying to get done in this field.

        You want to study customers and find out what they are trying to
        accomplish — especially under circumstances that leave them with insufficient
        solutions relative to available processes and technologies. What jobs
        have ad hoc solutions or no good solutions? When you see customers
        piecing together solutions themselves, these are great clues for innovation.

        A job to be done (JTBD) can pertain to one of 4 focus markets:

        {FOCUS_MARKET_INSTR}

        For each job to be done, you will create a job statement. 
        Key components of a job statement are an action verb, the object of
        the action, and clarification of the context in which the job is
        performed. 
        
        Examples:
        - Manage personal finances at home.
        - Clean clothes at home.
        - Listen to music while jogging.

        Under-served JTBDs are generally ripe for a core growth
        innovation strategy (make the existing solution better); over-served items
        are ripe for a disruptive innovation strategy (remake the solution so it
        becomes available to those who can’t afford the existing solution). When
        your assessment shows opportunities in the middle that are served right,
        you should focus on related jobs to be done.

        As well, for each job, write a few associated outcome expectation statements,
        from each of the following 4 categories:
        - Customer desired outcome: an outcome the customers are trying to achieve
        - Customer undesired outcome: an outcome the customers are tring to avoid
        - Provider desired outcome: an outcome the providers are trying to achieve
        - Provider undesired outcome: an outcome the providers are trying to avoid
        Think in terms of time, cost, potential errors, quality, dependability, availability,
        ease of use, maintainability, and any number of other satisfaction
        and dissatisfaction dimensions.
        Outcome expectation statements should be stated in imperative terms, using a
        standard structure. That structure is:
        - The direction of action (minimize, increase).
        - The unit of measurement (time, cost, probability, defects, errors, etc.).
        - The object of control (what it is you’re influencing).
        - The context (where or under what circumstances).

        Examples of customer outcome statements: 
        - Increase the likelihood that clothes appear fresh after home cleaning.
        - Minimize the time it takes to clean clothes.
        - Minimize the effort needed to clean clothes.

        Examples of provider outcome statements:
        - Increase customer loyalty from using solutions.
        - Minimize the likelihood of product liability litigation.
        - Minimize damage to the environment.
        """

    def search_jtbd(self, market:str=None, use_pxp=True, model="llama-3-sonar-small-32k"):
        
        user_prompt = f"""
        Search and list 5 to 7 jobs to be done for customers in the field of:
        {self.field}

        Please search in forums and website related to the field above.
        Make sure you use any prior knowledge that might have 
        been shared with you so far, when relevant.

        For each JTBD, you should output a job statement, specify which focus market it belongs to,
        explain why you think this Job to be done belongs to that customer market, and list a few associated outcome expectations.

        """

        if market is not None:
            user_prompt += """
            Finally, please focus on the following market: {market}
            """

        if use_pxp:
            r = self.ask_perplexity([
                {"role": "system",
                "content": self.system_prompt},

                {"role": "user",
                "content": user_prompt}
            ])
        else:
            r = self.ask([
                {"role": "system",
                "content": self.system_prompt},
                
                {"role": "user",
                "content": user_prompt}
            ])

        return r

    def add_knowledge(self, knowledge:str):
        new_info = """
        You've come accross an important piece of new knowledge.
        Please analyze it and store it in memory to reuse it to answer future questions when relevant.

        Here it is:

        {knowledge}
        """
        self.thread.add(new_info)

    def structure_jtbd(self, jtbd:str):
        instructions = """
        Extract the Job To Be Done objects from the provided description.
        """

        return marvin.extract(jtbd, 
                                target=JTBD, 
                                instructions=instructions) 
