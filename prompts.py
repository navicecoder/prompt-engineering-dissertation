def zero_shot(question):
    #question = "who landed on moon first?"
    prompt = '''Solve the problem. 
            Give your output as Final answer: <number> 
            Do not put commas, dollar signs, LaTeX, or any other formatting in the final answer.''' +question
    return prompt

def few_shot(question):
    prompt = f"""
        Help me solve these. Have a look at these examples. and give output as them.

        Example 1
        Question:
        Tom has 3 apples. He buys 5 more apples. How many apples does he have?
        Answer:
        Tom starts with 3 apples.
        He buys 5 more.
        3 + 5 = 8.
        Final Answer: 8

        Example 2
        Question:
        A box contains 12 pencils. Sarah gives 4 pencils to her friend. How many pencils are left?
        Answer:
        Sarah starts with 12 pencils.
        She gives away 4.
        12 - 4 = 8.
        Final Answer: 8

        Now solve the following problem.
        Question: {question}
        Answer:
        At the end, write: Final Answer: <number>"""
    return prompt


def cot(question):
    prompt = f"""
    Let's think step by step and solve this problem.
    Question: {question}
    Give your answer at the end as:
    Final Answer: <number>
    Don't include commas, dollar signs, LaTeX, or unnecessary symbols in the final answer.
    """
    return prompt

def role_prompt(question):
    prompt = f"""
    You are a maths teacher. Solve the following math problem carefully.
    Question: {question}

    Give your answer at the end as: Final Answer: <number>
    Do not include commas, dollar signs, LaTeX, or any unnecessary symbol in the final answer.
    """
    return prompt