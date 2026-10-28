# Integration of Mathematical Calculations with a Chat Completion System Using LLM Function-Calling

**Name:** Tamilarasan R

**Register Number:** 212225040459

## AIM

To design and implement a Python function for calculating the volume of a cylinder and integrate it with a chat completion system using the function-calling feature of a Large Language Model (LLM).

## PROBLEM STATEMENT

To develop a Python function that calculates the volume of a cylinder using its radius and height. Integrate the function with an LLM-based chat completion system so that the LLM can call the function, obtain the calculated result, and generate the final answer.

## DESIGN STEPS

### STEP 1:

Import the required libraries, load the API key from the `.env` file, and initialize the Groq chat model using LangChain.

### STEP 2:

Define a Python function named `cylinder_volume()` that accepts the radius and height as inputs and calculates the volume using the mathematical formula. Bind the function to the LLM using `bind_tools()`.

### STEP 3:

Send a request to the LLM, execute the requested function using the provided arguments, and return the calculated result to the LLM. Display the function details and the final answer.

## PROGRAM

```python
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage

model = init_chat_model("groq:openai/gpt-oss-120b")

@tool
def cylinder_volume(radius: float, height: float) -> float:
    """Calculate the volume of a cylinder using its radius and height."""
    return 3.14159 * radius * radius * height

tools = {"cylinder_volume": cylinder_volume}
model_with_tools = model.bind_tools([cylinder_volume])

messages = [
    ("human", "Calculate the volume of a cylinder with radius 3 and height 8 using the cylinder_volume tool.")
]

response = model_with_tools.invoke(messages)
messages.append(response)

for tool_call in response.tool_calls:
    result = tools[tool_call["name"]].invoke(tool_call["args"])

    messages.append(
        ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
    )

    print("Function:", tool_call["name"])
    print("Arguments:", tool_call["args"])
    print("Calculated Volume:", result)

final_response = model_with_tools.invoke(messages)

print("\nFinal Answer:")
print(final_response.content)
```

## OUTPUT

```text
Function: cylinder_volume
Arguments: {'radius': 3, 'height': 8}
Calculated Volume: 226.19448

Final Answer:
The volume of the cylinder is approximately 226.19 cubic units.
```

## RESULT

Thus, a Python function was implemented to calculate the volume of a cylinder and integrated with an LLM-based chat completion system using function calling. The LLM requested the function, Python executed the calculation, and the result was returned to the LLM to generate the final answer.
