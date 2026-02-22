import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

load_dotenv()

class DebateState(TypedDict):
    topic : str
    messages : Annotated[list, add_messages]
    max_iterations : int
    iteration : int
    winner : str
    loser : str

class Verdict(BaseModel):
    winner : str = Field(description="read the conversation and announce the winner.")
    loser : str = Field(description="read the conversation and announce the loser.")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def proponent_agent(state: DebateState):
    history = "\n\n".join([f"{msg.name}: {msg.content}" for msg in state["messages"]])
    
    system_prompt = SystemMessage(content=f"You are the 'Proponent' in a debate about: '{state['topic']}'. Argue in favor of the topic. Base your arguments on logic. Keep your response to a single, concise paragraph.")

    if not state["messages"]:
        prompt = HumanMessage(content="You are starting the debate. Please make your opening argument.")
    else:
        prompt = HumanMessage(content=f"Here is the debate history so far:\n{history}\n\nNow, it is your turn to speak. Provide your counter-argument.")
        
    response = llm.invoke([system_prompt, prompt])

    return {"messages": [AIMessage(content=response.content, name="Proponent")]}

def opponent_agent(state: DebateState):
    history = "\n\n".join([f"{msg.name}: {msg.content}" for msg in state["messages"]])
    
    system_prompt = SystemMessage(content=f"You are the 'Opponent' in a debate about: '{state['topic']}'. Argue against the topic and counter the Proponent's points. Keep your response to a single, concise paragraph.")
    
    prompt = HumanMessage(content=f"Here is the debate history so far:\n{history}\n\nNow, it is your turn to speak. Provide your counter-argument against the Proponent.")
    
    response = llm.invoke([system_prompt, prompt])

    current_iteration = state.get('iteration', 0) + 1
    
    return {
        "messages": [AIMessage(content=response.content, name="Opponent")],
        "iteration": current_iteration
    }

def judge_agent(state: DebateState):
    history = "\n\n".join([f"{msg.name}: {msg.content}" for msg in state["messages"]])
    prompt = f"You are a Debate judge, two agents are in a debate, one is proponent and another is opponent. You read their conversation and you need to announce a winner and a loser. Here is the conversation:\n\n{history}"

    structured_llm = llm.with_structured_output(Verdict)
    result = structured_llm.invoke(prompt)

    return {"winner": result.winner, "loser": result.loser}

def should_continue(state: DebateState):
    if state['iteration'] >= state['max_iterations']:
        return "judge"
    else:
        return "proponent"

workflow = StateGraph(DebateState)

workflow.add_node("judge", judge_agent)
workflow.add_node("proponent", proponent_agent)
workflow.add_node("opponent", opponent_agent)

workflow.add_edge(START,"proponent")
workflow.add_edge("proponent","opponent")
workflow.add_conditional_edges("opponent", should_continue)
workflow.add_edge("judge", END)

app = workflow.compile()

if __name__ == "__main__":
    topic = input("Enter Debate Topic : ")

    final_state = app.invoke({
        "topic": topic,
        "messages": [],
        "iteration": 0,
        "max_iterations": 2 
    })

    for msg in final_state["messages"]:
        print(f"\n{msg.name.upper()}:\n{msg.content}")
        
    print("\n" + "="*40)
    print("JUDGE VERDICT")
    print(f"The winner is : {final_state['winner']}")
    print(f"The loser is : {final_state['loser']}")