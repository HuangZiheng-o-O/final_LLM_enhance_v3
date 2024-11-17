import os
import openai
import yaml
import json

# Load configuration to get the API key
with open("config.yaml") as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)

client = openai.OpenAI(api_key=config_yaml['token'])

MODEL = "gpt-4o-mini"

def llm(prompt, stop=["\n"]):
    dialogs = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=MODEL,
        messages=dialogs
    )
    return completion.choices[0].message.content

def generate_base_motion_prompt(action_description):
    """Generate a prompt to extract the base (global) motion."""
    prompt = f"""
<TASK>
You are tasked with analyzing the following action description and extracting the base (global) motion component.

**Definition of Base Motion:**
- The base motion refers to the primary, overall movement of the entire body.
- It encompasses the general action without considering specific movements of individual body parts.
- The base motion should include simple leg movements and global trajectories but exclude specific movements of the arms.
- If leg movements are highly complex, they can be excluded from the base motion and treated as local edits.
- Movements of the head should always be included in the base motion.

<INPUT>
{action_description}
</INPUT>

<REQUIREMENTS>
- Focus solely on the primary, overall movement, including simple leg movements, head movements, and general trajectories.
- Exclude any specific movements of the arms.
- Exclude complex leg movements if they are described in detail.
- The base motion description should be concise and clear, ideally in one sentence.
- Use precise and unambiguous language.
- Do not include any reasoning, explanations, or additional commentary.

<FORMAT>
Provide the base motion description enclosed in <BASE_MOTION> tags, like:

<BASE_MOTION>
[Your base motion description here]
</BASE_MOTION>

<END_SIGNAL>
When you have finished, signal completion by saying '<BASEEND>'.

Remember:
- Avoid including detailed limb movements (arms or complex legs) in the base motion.
- Do not include any explanations.
- Output only the base motion description.

</END_SIGNAL>
</TASK>
"""
    return prompt

def generate_local_edits_prompt(action_description, base_description):
    """Generate a prompt to extract local body part movements."""
    prompt = f"""
<TASK>
You are tasked with analyzing the differences between the action description and the base motion to extract local body part movements that need to be applied as edits.

**Definition of Local Edits:**
- Local edits refer to specific movements of individual body parts (arms and complex leg movements) that occur simultaneously with the base motion.
- These are detailed actions that modify the base motion.

<BASE_MOTION>
{base_description}
</BASE_MOTION>

<ACTION_DESCRIPTION>
{action_description}
</ACTION_DESCRIPTION>

<REQUIREMENTS>
- Identify all specific movements of the following body parts:
  - "left arm"
  - "right arm"
  - "left leg"
  - "right leg"
- For each body part, describe its movement concisely and specifically in the format:
  "A person's [body part] [action]".
- For body parts without specific movements, the description should be "none".
- Use clear and unambiguous language.
- Do not include any reasoning, explanations, or additional commentary.

<FORMAT>
Provide the local edits in the following JSON format, enclosed in <LOCAL_EDITS_JSON> tags:

<LOCAL_EDITS_JSON>
[
  {{
    "body part": "left arm",
    "description": "[specific movement or 'none']"
  }},
  {{
    "body part": "right arm",
    "description": "[specific movement or 'none']"
  }},
  {{
    "body part": "left leg",
    "description": "[specific movement or 'none']"
  }},
  {{
    "body part": "right leg",
    "description": "[specific movement or 'none']"
  }}
]
</LOCAL_EDITS_JSON>

<END_SIGNAL>
When you have finished, signal completion by saying '<EDITEND>'.

Remember:
- Include all specified body parts in the output.
- Avoid duplicating movements already covered in the base motion.
- Output only the JSON-formatted local edits.

</END_SIGNAL>
</TASK>
"""
    return prompt

def test_motion_decomposition(action_description):
    """Test motion decomposition with base motion and local edits."""
    # Generate the base motion prompt
    base_prompt = generate_base_motion_prompt(action_description)
    base_response = llm(base_prompt, stop=["<BASEEND>"]).split("<BASEEND>")[0].strip()

    # Extract the base motion description
    base_description = base_response.replace("<BASE_MOTION>", "").replace("</BASE_MOTION>", "").strip()

    # Generate the local edits prompt
    edits_prompt = generate_local_edits_prompt(action_description, base_description)
    edits_response = llm(edits_prompt, stop=["<EDITEND>"]).split("<EDITEND>")[0].strip()

    # Parse the JSON from the edits response
    edits_json_str = edits_response.replace("<LOCAL_EDITS_JSON>", "").replace("</LOCAL_EDITS_JSON>", "").strip()
    try:
        local_edits = json.loads(edits_json_str)
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        print("Edits Response:")
        print(edits_response)
        local_edits = []

    return base_response, local_edits

if __name__ == "__main__":
    # Example actions to test
    actions_to_test = [
        "The man walks forward while raising his left hand and lowering his right hand at the same time.",
        "A woman hops forward while holding a T-pose.",
        "The person performs a rowing motion with their legs spread wide, moving their arms back and forth.",
        # "A dancer spins gracefully with arms extended and legs crossed.",
        # "The runner sprints ahead without any arm movements."
    ]

    for action in actions_to_test:
        base, local_edits = test_motion_decomposition(action)
        print(f"Action: {action}")
        print("Base Motion:")
        print(base)
        print("Local Edits:")
        print(json.dumps(local_edits, indent=2))
        print("=" * 50)