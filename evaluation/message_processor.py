import re
import json
import ast
import uuid

def parse_arguments(json_value):
    try:
        parsed_value = json.loads(json_value)
        return parsed_value, isinstance(parsed_value, dict)
    except:
        return json_value, False

def get_argument_type(func_name: str, arg_key: str, defined_tools: list):
    name2tool = {tool["name"]: tool for tool in defined_tools}
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    if arg_key not in tool["parameters"]["properties"]:
        return None
    return tool["parameters"]["properties"][arg_key]["type"]

def parse_model_response(response: str, defined_tools: list):
    text = response.strip()
    reasoning_content = None
    content = None
    tool_calls = []

    if text.startswith('<think>'):
        if '</think>' in text:
            reasoning_content, text = text.rsplit('</think>', 1)
            reasoning_content = reasoning_content.removeprefix('<think>').strip()
            text = text.strip()
        else:
            reasoning_content = text.removeprefix('<think>').strip()
            text = ""

    if '<tool_call>' in text:
        index = text.find('<tool_call>')
        content = text[:index].strip()
        text = text[index:].strip()
    else:
        content = text.strip()
        text = ""

    tool_call_strs = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    for call in tool_call_strs:
        func_name_match = re.match(r'([^\n<]+)', call.strip())
        func_name = func_name_match.group(1).strip() if func_name_match else None
        if func_name:
            pairs = re.findall(r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>', call, re.DOTALL)
            arguments = {}
            for arg_key, arg_value in pairs:
                arg_key = arg_key.strip()
                arg_value = arg_value.strip()
                arg_type = get_argument_type(func_name, arg_key, defined_tools)
                if arg_type != 'string':
                    arg_value, is_good_json = parse_arguments(arg_value)
                arguments[arg_key] = arg_value

            tool_calls.append({
                'tool_call_id': "tool-call-" + str(uuid.uuid4()),
                'name': func_name,

                'arguments': arguments
            })

    message = {'role': 'assistant'}
    if reasoning_content:
        message['reasoning_content'] = reasoning_content
    if content:
        message['content'] = content
    if tool_calls:
        message['tool_calls'] = tool_calls

    return message
