from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor, tool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import json
from typing import Optional
from datetime import datetime
from langchain.agents import tool
import pandas as pd

# Load .env file
load_dotenv()

# Access the API key from the environment file
groq_api_key = os.getenv('GROQ_API_KEY')

# Set custom API key and base URL for OpenAI
os.environ["OPENAI_API_KEY"] = groq_api_key
os.environ["OPENAI_API_BASE"] = os.getenv('GROQ_BASE_URL')


# Create a small dataset
data = {
    'Product': ['Laptop', 'Smartphone', 'Tablet', 'Desktop'],
    'Sales': [150, 200, 100, 90]
}
df = pd.DataFrame(data)


def extract_dataframe(json_str: str):
    """
    Extracts a Pandas DataFrame and column names from a JSON-like string.

    Parameters:
        json_str (str): A string representation of a JSON object containing:
            - "data": A dictionary where keys are column names and values are lists of data.
            - "sales_column": The name of the sales column.
            - "product_column": The name of the product column.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame created from the "data" dictionary.
            - str: The name of the sales column.
            - str: The name of the product column.

    Example:
        json_str = "{'data': {'Product': ['Laptop', 'Phone'], 'Sales': [100, 200]}, 
                     'sales_column': 'Sales', 'product_column': 'Product'}"
        df, sales_col, product_col = extract_dataframe(json_str)
        
        print(df)
        #   Product  Sales
        # 0  Laptop    100
        # 1   Phone    200
        
        print(sales_col)  # Output: Sales
        print(product_col)  # Output: Product
    """
    # Convert string to dictionary
    data_dict = json.loads(json_str.replace("'", '"'))  # Replace single quotes with double quotes
    
    # Extract DataFrame
    df = pd.DataFrame(data_dict["data"])
    
    # Extract column names
    sales_column = data_dict["sales_column"]
    product_column = data_dict["product_column"]
    
    return df, sales_column, product_column

@tool
def analyze_sales_data(
    json_str: str
) -> str:
    """
    Analyzes sales data to identify the top-selling product.
    
    Args: A json string with the following components:
        data: Input sales data, can be either:
              - Pandas DataFrame
              - Dictionary convertible to DataFrame
        sales_column: Name of column containing sales numbers (default: 'Sales')
        product_column: Name of column containing product names (default: 'Product')
        
    Returns:
        str: Formatted analysis result or error message
        
    Examples:
        >>> df = pd.DataFrame({'Product': ['A', 'B'], 'Sales': [100, 200]})
        >>> analyze_sales_data(df)
        'Top-selling product is B with 200 units sold.'
        
        >>> analyze_sales_data({'Product': [1,2], 'Revenue': [300,400]}, 
        ...                    sales_column='Revenue', product_column='Product')
        'Top-selling product is 2 with 400 units sold.'
        
    Notes:
        - Requires pandas installation
        - Handles empty datasets and missing columns
        - Automatically converts dictionaries to DataFrames
    """
    try:
        df, sales_column, product_column = extract_dataframe(json_str)
        # Convert dict input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        # Validate DataFrame structure
        if df.empty:
            return "Error: Empty dataset provided"
            
        for col in [sales_column, product_column]:
            if col not in df.columns:
                return f"Error: Missing required column - {col}"
                
        # Find maximum sales entry
        max_row = df.loc[df[sales_column].idxmax()]
        
        return (
            f"Top-selling product is {max_row[product_column]} "
            f"with {max_row[sales_column]:,} units sold."
        )
        
    except KeyError as e:
        return f"Data validation error: {str(e)}"
    except ValueError as e:
        return f"Analysis error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

@tool
def get_current_time(timezone: Optional[str] = None) -> str:
    """
    Get the current date and time in either local timezone or specified timezone.
    
    Args:
        timezone (Optional[str]): IANA timezone name (e.g., 'America/New_York', 'UTC'). 
                                  Defaults to local time.
                                  
    Returns:
        str: Formatted datetime string with timezone information
        
    Examples:
        get_current_time() -> "2024-02-20 15:30:45 (Local Time)"
        get_current_time("Europe/London") -> "2024-02-20 20:30:45 GMT"
    """
    try:
        now = datetime.now()
        if timezone:
            try:
                import pytz
                tz = pytz.timezone(timezone)
                now = datetime.now(tz)
                tz_info = now.strftime('%Z')
            except pytz.UnknownTimeZoneError:
                return f"Invalid timezone: {timezone}"
        else:
            tz_info = "Local Time"
            
        return now.strftime(f"%Y-%m-%d %H:%M:%S ({tz_info})")
        
    except Exception as e:
        return f"Error getting time: {str(e)}"



# Initialize the OpenAI LLM
llm = ChatOpenAI(
    #model_name="llama3-70b-8192",  # Specify model name if needed
    model_name="qwen-2.5-32b",
    openai_api_key=groq_api_key,
    openai_api_base=os.environ["OPENAI_API_BASE"]
)

# Define the tools list
tools = [analyze_sales_data,get_current_time]
prohibited_keywords = ["hack", "exploit", "illegal"]
safe_response = "The requested information cannot be provided as it violates our safety guidelines."

# Create the tool-calling agent and executor
instruction = f"""
You are a team of AI agents. One of you is a data analyst, and the other is a business expert.
- The data analyst provides insights from the sales data.
- The business expert generates strategies based on the data.
- The dataframe {df} contains the sales data
- If your response or the user input contains words in the {prohibited_keywords} list, repond with the words {safe_response}
"""
base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt = base_prompt.partial(instructions=instruction)

# Initialize memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create an agent
agent = create_react_agent(
    prompt=base_prompt.partial(instructions=instruction),
    llm=llm,
    tools=tools,
)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True, memory=memory)

# Run the agent in a loop to allow the user to ask many questions
while True:
    try:
        # Get user input
        user_input = input("\nEnter your question (type 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Run the query with the necessary variables for the prompt template
        response = agent_executor.invoke({"chat_history": [], "input": user_input})
        
        # Print the response
        print(f"\nAgent Response: {response['output']}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        break
    except Exception as e:
        print(f"\nError: {str(e)}")