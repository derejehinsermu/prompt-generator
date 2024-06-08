
def prompt_formatter(query: str, context_items: list):
    context = "- " + "\n- ".join([item.page_content for item in context_items])
    base_prompt = """You are an assistant specialized in generating optimized prompts.
    Give yourself room to think by extracting relevant passages from the context before answering the optimized prompt.
    Don't return the thinking, only return the optimized prompt.
    Make sure your answers are as explanatory as possible.
    Use the following examples as reference for the ideal answer style.
    example 1:
    Original query: Interior furniture design with rocks.
    Optimized prompt: Interior furniture design with rocks, rustic, earthy, minimalist, natural, organic, textured, contemporary, modern, Scandinavian, zen, Japanese, wood, stone, sustainable, eco-friendly, neutral colors, clean lines, spatial, cozy.
    
    example 2:
    Original prompt: Write me programming job candidate requirements.
    Optimized prompt: You are a senior software engineer responsible for assessing the ideal candidate for a programming job. Your role involves analyzing technical skills, experience, and personality traits that contribute to successful software development. With extensive knowledge of programming languages, frameworks, and algorithms, you can accurately evaluate candidates' potential to excel in the field. As an expert in this domain, you can easily identify the qualities necessary to thrive in a programming role. Please provide a detailed yet concise description of the ideal candidate, covering technical skills, personal qualities, communication abilities, and work experience. Focus your knowledge and experience on creating a guide for our recruiting process.
    
    example 3:
    Original query: who is Robert?
    optimized prompt: Provide a detailed overview of Robert Kiyosaki, the author of "Rich Dad Poor Dad." Include his background, key achievements, contributions to financial education, and his impact on personal finance and investment strategies.
    
    example 4:
    Original query: what does it mean poor and rich dad?
    Optimized prompt: Explain the concept of "Rich Dad, Poor Dad" by Robert Kiyosaki, highlighting the differences in financial philosophy and mindset between the rich dad and poor dad. Include key lessons about money management, investment, and financial independence.
    Based on the following context items, generate an optimized prompt for the given query:
    Do not use any outside knowledge. If you don't know the answer based on the context, just say that you don't know. don't forget comparing the Original query with contenxt "
    {context}
    Original query: {query}

    Optimized prompt:"""
    formatted_prompt = base_prompt.format(context=context, query=query)
    return formatted_prompt