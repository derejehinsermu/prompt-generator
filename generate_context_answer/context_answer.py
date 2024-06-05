import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts/utils')))
from load_data import retriever

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


# Define the prompt template
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
prompt_template = ChatPromptTemplate(
    input_variables=['context', 'question'],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['context', 'question'],
                template=(
                    "You are an assistant for question-answering tasks. Only use the following pieces of retrieved context to answer the question. "
                    "Do not use any outside knowledge. If you don't know the answer based on the context, just say that you don't know. "
                    "Use three sentences maximum and keep the answer concise.\n"
                    "Question: {question} \n"
                    "Context: {context} \n"
                    "Answer:"
                )
            )
        )
    ]
)
# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
def create_rag_chain(retriever, prompt_template, llm):
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

rag_chain = create_rag_chain(retriever, prompt_template, llm)

# Main loop for user input
while True:
    user_question = input("Please enter your question (or type 'exit' to quit): ")
    if user_question.lower() == 'exit':
        print("Goodbye!")
        break

    response = rag_chain.invoke(user_question)
    print("Answer:", response)



