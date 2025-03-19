# AI Sales Data Analysis Agent

## Overview
This repository contains a **simple AI agent** that can analyze a dataset and provide insights about the data. The AI agent leverages **LangChain, OpenAI's API, and Pandas** to process and interpret sales data.

## Features
- Uses **LangChain** for AI-driven analysis.
- Extracts and processes sales data from a dataset.
- Provides **business insights** and strategies based on the data.
- Supports interactive queries in the terminal.
- Maintains **chat history** for better conversation flow.

## Installation and Setup
This project uses **Pipenv** for dependency management. Follow these steps to set up the environment:

### 1. Clone the repository
```sh
git clone https://github.com/waltermaina/sample-AI-agent.git
cd sample-AI-agent
```

### 2. Install Pipenv (if not installed)
```sh
pip install pipenv
```

### 3. Create and activate a virtual environment
```sh
pipenv install
pipenv shell
```

### 4. Install required dependencies
```sh
pipenv install langchain-core langchain-community langchain-openai python-dotenv pandas
```

### 5. Set up environment variables
Create a **.env** file in the project directory and add your OpenAI API key:
```
GROQ_API_KEY=your_openai_api_key
GROQ_BASE_URL=your_openai_api_base_url
```

## Running the AI Agent
Once the setup is complete, run the AI agent using:
```sh
python main.py
```

## Sample Output
Enter your question (type 'exit' to quit): Analyze the sales data and suggest business strategies

> Entering new AgentExecutor chain...
Agent Response: The analysis of the sales data indicates that Smartphones are the top-selling product, with 200 units sold. Here are some business strategies you could consider:

1. **Increase Marketing Focus on Smartphones**
2. **Stock Management**
3. **Cross-Selling and Upselling**
4. **Customer Satisfaction and Loyalty**
5. **Competitor Analysis and Pricing**

By implementing these strategies, you can capitalize on the successful sales of smartphones and potentially boost revenue and market share.    

> Finished chain.

Agent Response: The analysis of the sales data indicates that Smartphones are the top-selling product, with 200 units sold. Here are some business strategies you could consider:

1. **Increase Marketing Focus on Smartphones**: Allocate more marketing resources to campaigns highlighting the benefits of smartphones. This may involve digital marketing efforts, social media ads, and perhaps even a special promotional event or sale to enhance brand visibility.      

2. **Stock Management**: Ensure that the stock inventory for smartphones is well-managed to avoid shortages during peak sales periods. Consider working closely with suppliers to ensure a steady supply chain.

3. **Cross-Selling and Upselling**: With smartphones being the best-selling item, consider promoting complementary products such as phone cases, screen protectors, or accessories that go well with smartphones.

4. **Customer Satisfaction and Loyalty**: Enhance the buying experience for smartphones by offering a free consultation, installation service, 
or a loyalty program that rewards customers with discounts on their next purchase, encouraging repeat business.

5. **Competitor Analysis and Pricing**: Conduct a market analysis to assess the competitive landscape for smartphones, ensuring that pricing and services offered are competitive and attract potential customers.

By implementing these strategies, you can capitalize on the successful sales of smartphones and potentially boost revenue and market share.


## Contributing
Feel free to fork the repository and submit pull requests with improvements!

## License
This project is licensed under the MIT License.


