# import os
# import openai
# import yaml
# import re
# import json
# import re
# with open("config.yaml") as f:
#  config_yaml = yaml.load(f, Loader=yaml.FullLoader)
#
# client = openai.OpenAI(api_key=config_yaml['token'])
#
# MODEL = "gpt-4o-mini"
#
# def llm(prompt, stop=["\n"]):
#
#     dialogs = [{"role": "user", "content": prompt}]
#
#     completion = client.chat.completions.create(
#         model=MODEL,
#         messages=dialogs
#     )
#     return completion.choices[0].message.content
#
#
# def generate_base_motion_prompt(action_description):
#     """Generate a prompt to extract the base (global) motion."""
#     prompt = f"""
# <TASK>
# You are tasked with analyzing the following action description and extracting the base (global) motion component.
#
# **Definition of Base Motion:**
# - The base motion refers to the primary, overall movement of the entire body.
# - It encompasses the general action without considering specific movements of individual body parts.
# - The base motion should include leg movements and global trajectories but exclude specific arm, hand, or head movements.
#
# <INPUT>
# {action_description}
# </INPUT>
#
# <REQUIREMENTS>
# - Focus solely on the primary, overall movement, including leg movements and general trajectories.
# - Exclude any specific movements of the arms, hands.
# - movements of the head should NOT be excluded.
# - The base motion description should be concise and clear, ideally in one sentence.
# - Use precise and unambiguous language.
# - Do not include any reasoning, explanations, or additional commentary.
#
#
# <FORMAT>
# Provide the base motion description enclosed in <BASE_MOTION> tags, like:
#
# <BASE_MOTION>
# [Your base motion description here]
# </BASE_MOTION>
#
# <END_SIGNAL>
# When you have finished, signal completion by saying '<BASEEND>'.
#
# Remember:
# - Do not include any local movements of the arms, hands, or head.
# - Do not include any explanations.
# - Output only the base motion description.
#
# </END_SIGNAL>
# </TASK>
# """
#     return prompt
#
# def generate_local_edits_prompt(action_description, base_description):
#     """Generate a prompt to extract local body part movements."""
#     prompt = f"""
# <TASK>
# You are tasked with analyzing the differences between the action description and the base motion to extract local body part movements that need to be applied as edits.
#
# **Definition of Local Edits:**
# - Local edits refer to specific movements of individual body parts (e.g., arms, hands, head) that occur simultaneously with the base motion.
# - These are detailed actions that modify the base motion.
#
# <BASE_MOTION>
# {base_description}
# </BASE_MOTION>
#
# <ACTION_DESCRIPTION>
# {action_description}
# </ACTION_DESCRIPTION>
#
# <REQUIREMENTS>
# - Identify all specific movements of individual body parts mentioned in the action description that are not included in the base motion.
# - For each local edit, specify:
#   - The body part involved (e.g., "left arm", "head").
#   - A concise and specific description of its movement.
# - Use clear and unambiguous language.
# - Do not include any reasoning, explanations, or additional commentary.
# - For body parts without specific movements (i.e., not mentioned in the action description), specify "none" as the description.
#
# <BODY_PARTS>
# The list of body parts to consider:
# - "head"
# - "left arm"
# - "right arm"
# - "left hand"
# - "right hand"
# </BODY_PARTS>
#
# <FORMAT>
# Provide the local edits in the following JSON format, enclosed in <LOCAL_EDITS> tags:
#
# <LOCAL_EDITS>
# [
#   {{
#     "body part": "[body part]",
#     "description": "[specific movement description or 'none']"
#   }},
#   ...
# ]
# </LOCAL_EDITS>
#
# <END_SIGNAL>
# When you have finished, signal completion by saying '<EDITEND>'.
#
# Remember:
# - Do not include any explanations.
# - Output only the local edits in the specified JSON format.
#
# </END_SIGNAL>
# </TASK>
# """
#     return prompt
#
# def generate_local_edits_json_prompt(edits_response):
#     """Generate a prompt to convert the edits response into JSON format."""
#     prompt = f"""
# <TASK>
# You are provided with the following description of local body part movements:
#
# <EDITS_RESPONSE>
# {edits_response}
# </EDITS_RESPONSE>
#
# <REQUIREMENTS>
# - Convert the description into a JSON-formatted list of dictionaries.
# - Each dictionary should have:
#   - "body part": The specific body part involved (e.g., "left arm", "head").
#   - "description": A concise description of its movement.
# - For body parts without specific movements, include them with "description": "none".
# - Only include the following body parts:
#   - "head"
#   - "left arm"
#   - "right arm"
#   - "left hand"
#   - "right hand"
#
# <FORMAT>
# Provide the local edits in the following JSON format, enclosed in <LOCAL_EDITS_JSON> tags:
#
# <LOCAL_EDITS_JSON>
# [
#   {{
#     "body part": "[body part]",
#     "description": "[specific movement description or 'none']"
#   }},
#   ...
# ]
# </LOCAL_EDITS_JSON>
#
# <END_SIGNAL>
# When you have finished, signal completion by saying '<JSONEND>'.
#
# Remember:
# - Include all specified body parts, even if their description is "none".
# - Do not include any explanations.
# - Output only the JSON-formatted local edits.
#
# </END_SIGNAL>
# </TASK>
# """
#     return prompt
#
# def test_motion_decomposition(action_description):
#     """Test motion decomposition with base motion and local edits."""
#     # Generate the base motion prompt
#     base_prompt = generate_base_motion_prompt(action_description)
#     base_response = llm(base_prompt, stop=["<BASEEND>"]).split("<BASEEND>")[0].strip()
#
#     # Extract the base motion description
#     base_description = base_response.replace("<BASE_MOTION>", "").replace("</BASE_MOTION>", "").strip()
#
#     # Generate the local edits prompt
#     edits_prompt = generate_local_edits_prompt(action_description, base_description)
#     edits_response = llm(edits_prompt, stop=["<EDITEND>"]).split("<EDITEND>")[0].strip()
#
#     # Generate the local edits JSON prompt
#     edits_json_prompt = generate_local_edits_json_prompt(edits_response)
#     edits_json_response = llm(edits_json_prompt, stop=["<JSONEND>"]).split("<JSONEND>")[0].strip()
#
#     return base_response, edits_json_response
#
# if __name__ == "__main__":
#     # Example actions to test
#     actions_to_test = [
#         "The man walks forward while raising his left hand and lowering his right hand at the same time.",
#         "A woman hops forward while holding a T-pose.",
#         "The person performs a rowing motion with their legs spread wide, moving their arms back and forth.",
#     ]
#
#     for action in actions_to_test:
#         base, edits = test_motion_decomposition(action)
#         print(f"Action: {action}")
#         print("Base Motion:")
#         print(base)
#         print("Local Edits JSON:")
#         print(edits)
#         print("=" * 50)
