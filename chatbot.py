import os
from flask import Flask, request, render_template, session, jsonify
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


app = Flask(__name__)
app.secret_key = os.urandom(24)

load_dotenv('.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

tools = [TavilySearchResults(max_results=1)]


# Only certain models support this
chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. You may not need to use tools for every query - the user may just want to chat!",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)



agent = create_openai_tools_agent(chat, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
demo_ephemeral_chat_history_for_chain = ChatMessageHistory()
conversational_agent_executor = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="input",
    output_messages_key="output",
    history_messages_key="chat_history",
)
@app.route('/')
def index():
    return render_template('chat.html')  # Assuming you have a chat.html template for the chat UI

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    session_id = session.get('session_id', 'default_session')
    
    
    if 'chat_history' not in session:
        session['chat_history'] = []
        
    # Update chat history with the user's message
    session['chat_history'].append(('human', user_message))
    
    # Prepare the input for the conversational agent
    input_dict = {"input": user_message}
    config_dict = {"configurable": {"session_id": session_id}}
    
    # Generate a response using the conversational agent executor
    response = conversational_agent_executor.invoke(input_dict, config_dict)
    response_message = response.get('output', 'Sorry, I couldn\'t process your message.')
    
    # Update chat history with the agent's response
    session['chat_history'].append(('system', response_message))
    
    # Send the response back to the user
    return jsonify({"message": response_message})

if __name__ == '__main__':
    app.run(debug=True)
