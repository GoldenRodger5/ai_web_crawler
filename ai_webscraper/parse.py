from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
import time
from openai import OpenAIError

# Initialize OpenAI model with hardcoded API key
model = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key="{openai_api_key}")

template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)

def parse_with_openai(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    parsed_results = []

    for i, chunk in enumerate(dom_chunks, start=1):
        retries = 3  # Retry up to 3 times on failure
        while retries > 0:
            try:
                response = chain.invoke({"dom_content": chunk, "parse_description": parse_description})
                print(f"Parsed batch {i} of {len(dom_chunks)}")
                
                # Ensure the response is properly extracted as a string
                if isinstance(response, AIMessage):
                    parsed_results.append(response.content.strip() if response.content else "")
                elif isinstance(response, dict) and "content" in response:
                    parsed_results.append(response["content"].strip() if response["content"] else "")
                elif isinstance(response, list):
                    parsed_results.append(" ".join(str(r.content).strip() for r in response if hasattr(r, 'content') and r.content))
                else:
                    parsed_results.append(str(response).strip())
                break  # Exit retry loop if successful
            except OpenAIError as e:
                retries -= 1
                wait_time = (4 - retries) * 5  # Exponential backoff (5s, 10s, 15s)
                print(f"OpenAI API error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        else:
            print(f"Skipping batch {i} due to repeated OpenAI API errors.")
            parsed_results.append("")  # Append empty result if all retries fail
    
    return "/n".join(parsed_results)