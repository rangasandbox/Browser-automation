import os
import json
import google.generativeai as genai
import logging


# Define the schema for page parsing
page_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "scenario": genai.protos.Schema(type=genai.protos.Type.STRING),
        "descriptionIDs": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(type=genai.protos.Type.STRING),
        ),
        "formDetails": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "type": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "label": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "className": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "id": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "opener-data-id": genai.protos.Schema(
                        type=genai.protos.Type.STRING
                    ),
                },
            ),
        ),
        "nextAction": genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "buttonText": genai.protos.Schema(type=genai.protos.Type.STRING),
                "type": genai.protos.Schema(type=genai.protos.Type.STRING),
                "url": genai.protos.Schema(type=genai.protos.Type.STRING),
                "className": genai.protos.Schema(type=genai.protos.Type.STRING),
                "id": genai.protos.Schema(type=genai.protos.Type.STRING),
            },
        ),
        "errors": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "type": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "message": genai.protos.Schema(type=genai.protos.Type.STRING),
                },
            ),
        ),
        "specialRequirements": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "requirement": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "action": genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "buttonText": genai.protos.Schema(
                                type=genai.protos.Type.STRING
                            ),
                            "className": genai.protos.Schema(
                                type=genai.protos.Type.STRING
                            ),
                        },
                    ),
                },
            ),
        ),
    },
)

# Define the function declaration
parse_page = genai.protos.FunctionDeclaration(
    name="parse_page",
    description="Parses the HTML structure and content of a page and returns structured information about it.",
    parameters=page_schema,
)


# Function to generate the task prompt
# def generate_page_parse_prompt(page):
#     return f"""
# Analyze the following HTML structure and content for a {page} page. Parse the HTML data, understand the page's purpose, and generate a structured output detailing the next actionable steps for a job application process.


# Please provide the structured output according to the parse_page function specification.
# """


def generate_page_parse_prompt(page):
    return f"""
Analyze the following HTML structure and content for a {page} page. Parse the HTML data, understand the page's purpose, and generate a structured output detailing the next actionable steps for a job application process and Extract and detail all relevant form fields required for job applications. This includes text areas, input fields, file uploaders, checkboxes, radio buttons, dropdowns, and select options. Ensure that all form labels are clearly identified..

Here is some information required for the structured output:
- descriptionIDs - List of id that describe the content of the page.
- id - ID is necessary for the form fields.

Please avoid suggesting any login options during the parsing process. Ensure that the structured output follows the specifications outlined in the parse_page function.
"""


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def convert_to_dict(obj):
    if isinstance(obj, list):
        return [convert_to_dict(item) for item in obj]
    elif hasattr(obj, "items"):
        return {key: convert_to_dict(value) for key, value in obj.items()}
    else:
        return obj


def process_ai_response(html_content):
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            tools=[parse_page],
        )
        generated_prompt = generate_page_parse_prompt(html_content)
        result = model.generate_content(generated_prompt)
        fc = result.candidates[0].content.parts[0].function_call
        print("type", type(fc))
        output = json.dumps(type(fc).to_dict(fc), indent=4)
        parsed_json = json.loads(output)

        return parsed_json
    except Exception as e:
        print(f"Error in process_ai_response: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}
