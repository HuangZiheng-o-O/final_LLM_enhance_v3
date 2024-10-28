import os
import openai
import yaml
import re
import json
import re


def llm(prompt, stop=["\n"]):
    # Load API key from a YAML configuration file
    with open("config.yaml") as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize the OpenAI client with the API key
    client = openai.OpenAI(api_key=config_yaml['token'])

    # Set the model name
    MODEL = "gpt-4o-mini"

    # Prepare the dialog for the API request
    dialogs = [{"role": "user", "content": prompt}]

    completion = client.chat.completions.create(
        model=MODEL,
        messages=dialogs
    )
    return completion.choices[0].message.content


def generate_sequence_explanation_prompt(original_action):
    """
    Generate a prompt for the LLM to think about fine motion control for a given action.
    :param original_action: Perform a signature James Bond pose with a dramatic turn and gunpoint.
    :return:
    :example:
    The man pivots on his heel, shifting his weight to turn his body sharply while extending his opposite arm outward, creating a dramatic stance.
    The man's wrist flexes as he raises his hand to point the imaginary gun forward, aligning his arm with his shoulder for precision and balance.
    <DONE>
    """
    prompt = f"""
    The action 'original_action: {original_action}' may require detailed control over specific body parts. 
    Please evaluate the action and think carefully about how the movement breaks down into smaller steps. 
    You should independently decide the steps involved in completing this action.

    After thinking, provide a structured list of the steps involved in performing this action.

    <original_action>
    {original_action}
    </original_action>


    <REQUIREMENT>
    - Focus on describing the dynamic movement.
    - Highlight the necessary coordination between body parts.
    -Emphasize the importance of actions: Clearly require that each step must include key movement details, avoiding redundancy. 
    -Streamline the steps: Remind that the generated steps should be merged as much as possible, ensuring each step contains actual dynamic movements rather than empty descriptions.

        <FORMAT>

        The number of steps should be 1 or 2 or 3 or 4, depending on the TEMPORAL complexity of the action.Do not use too many steps if the action is simple. 2~3 steps are usually enough.  

         eg. 'run in a circle with one hand swinging',even if the action is complex by Spatial Composition, it is simple by Temporal. Apparently there is only one step of action and don't need to provide multiple steps. In this case, you can just provide one step. 

        For each step, use the words 'The man...'or 'The man's ...(body part)' to describe the action.
        Ensure the explanation is like:
        step1: The ...
        step2: The ...
        ...

        </FORMAT>
    </REQUIREMENT>



    <EXCLUSION>
    - Do not include any description of facial expressions or emotions.
    - Focus solely on the action and movement itself.
    </EXCLUSION>

    <END_SIGNAL>
    - When your explanation is finished, signal completion by saying '<SEQUENCEEND>'.
    </END_SIGNAL>

    Now think:
    """
    return prompt


def generate_sequence_explanation_prompt_json(original_action, sequence_explanation_prompt):
    prompt = f"""
<TASK>
Based on your breakdown of the action and the most important original_action, evaluate fine motion control for the following body parts:

    <original_action>
    {original_action}
    </original_action>

    <breakdown of the action>
    {sequence_explanation_prompt}
    </breakdown of the action>  


    <BODY_PARTS>
    - spine
    - Left Arm
    - Right Arm
    - Left Leg
    - Right Leg
    </BODY_PARTS>


</TASK>



<EXAMPLES>
EXAMPLE1:
{{"body part": "left arm", "description": "The man's left arm remains stationary at his side."}}
{{"body part": "right arm", "description": "The man's right arm moves in a waving motion."}}
{{"body part": "left leg", "description": "The man's left leg is stationary."}}
{{"body part": "right leg", "description": "The man's right leg is lifted slightly off the ground."}}
{{"body part": "spine", "description": "The man's spine moves in a wave-like motion."}}

EXAMPLE2:
{{"body part": "left arm", "description": "The man's left arm is bent at the elbow and held close to his body."}}
{{"body part": "right arm", "description": "The man's right arm moves in a circular motion."}}
{{"body part": "left leg", "description": "The man's left leg is bent at the knee."}}
{{"body part": "right leg", "description": "The man's right leg is bent at the knee."}}
{{"body part": "spine", "description": "The man's spine moves in a rhythmic motion."}}

EXAMPLE3:
{{"body part": "left arm", "description": "The man's left arm is raised in a bent position."}}
{{"body part": "right arm", "description": "The man's right arm is raised and bent at the elbow."}}
{{"body part": "left leg", "description": "The man's left leg is lifted and bent at the knee."}}
{{"body part": "right leg", "description": "The man's right leg is bent at the knee and lifted slightly off the ground."}}
{{"body part": "spine", "description": "The spine is arched slightly forward."}}
</EXAMPLES>

<FORMAT>
Ensure the explanation is in the following JSON-like format for each step and body part:



{{
    "step1": [
        {{
            "body part": "left arm",
            "description": "The man's left arm [specific movement description]."
        }},
        {{
            "body part": "right arm",
            "description": "The man's right arm [specific movement description]."
        }},
        {{
            "body part": "left leg",
            "description": "The man's left leg [specific movement description]."
        }},
        {{
            "body part": "right leg",
            "description": "The man's right leg [specific movement description]."
        }},
        {{
            "body part": "spine",
            "description": "The man's spine [specific movement description]."
        }}
    ],
    "step2": [
        {{
            "body part": "left arm",
            "description": "The man's left arm [specific movement description]."
        }},
        {{
            "body part": "right arm",
            "description": "The man's right arm [specific movement description]."
        }},
        {{
            "body part": "left leg",
            "description": "The man's left leg [specific movement description]."
        }},
        {{
            "body part": "right leg",
            "description": "The man's right leg [specific movement description]."
        }},
        {{
            "body part": "spine",
            "description": "The man's spine [specific movement description]."
        }}
    ],
    ...(continue for each step)
}}

Focus on the movement and positioning of each body part, similar to the provided examples. Be concise and avoid vague terms. Use clear and specific descriptions.
</FORMAT>

<REQUIREMENT>
Focus only on these body parts. DO NOT include any details about facial expressions, eyes, or emotions.
Be concise and AVOID providing any reasoning or explanation—focus only on the action of each body part.
</REQUIREMENT>

<END_SIGNAL>
When you finish the explanation for all steps, say '<SEQUENCEEND>'.
</END_SIGNAL>
    """


def sequence_analyze(action):
    sequence_explanation_prompt = generate_sequence_explanation_prompt(action)
    sequence_explanation = llm(sequence_explanation_prompt, stop=["<SEQUENCEEND>"]).split("<SEQUENCEEND>")[0].strip()
    print(sequence_explanation)

    # Use the updated format to generate JSON-like output for each body part
    sequence_explanation_prompt2 = generate_sequence_explanation_prompt_json(action, sequence_explanation)
    # print(sequence_explanation_prompt2)
    sequence_explanation2 = llm(sequence_explanation_prompt2, stop=["<SEQUENCEEND>"]).split("<SEQUENCEEND>")[0].strip()

    # Formatting the output into the desired JSON-like format for each body part
    output = (
            "Action: " + "\n" + action + "\n" +
            "Sequence Explanation: " + "\n" + sequence_explanation + "\n" +
            "Fine Motion Control Steps: " + "\n" +
            sequence_explanation2 + "\n" +
            "\n"
    )

    # Write to file
    with open("sequence_explanation.txt", "a") as file:
        file.write(output)

    # Print to console
    print(output)

    # Return results as well
    return sequence_explanation, sequence_explanation2


def generate_fine_motion_control_prompt(original_action, sequence_explanation):
    prompt = f"""
<TASK>
Generate fine motion control descriptions for each body part based on the action 'original_action' and the sequence_explanation.

<original_action>
{original_action}
</original_action>

<sequence_explanation>
{sequence_explanation}
</sequence_explanation>


</TASK>

<REQUIREMENT>
Focus on describing the specific actions for each body part without providing explanations or reasoning. Avoid vague terms such as "at an appropriate angle" or "adjust for balance." Instead, use precise descriptions like "Raise the right arm to shoulder height." 
Do not include any details about facial expressions, eyes, or emotions.
I don't like overly abstract language, such as "as if aiming a firearm." Understanding this action requires a high level of comprehension that general models may not possess, as this description does not clearly specify what kind of action it is. Please use more specific action descriptions.
</REQUIREMENT>

<EXAMPLES>
EXAMPLE1:
{{"body part": "left arm", "description": "The man's left arm remains stationary at his side."}}
{{"body part": "right arm", "description": "The man's right arm moves in a waving motion."}}
{{"body part": "left leg", "description": "The man's left leg is stationary."}}
{{"body part": "right leg", "description": "The man's right leg is lifted slightly off the ground."}}
{{"body part": "spine", "description": "The man's spine moves in a wave-like motion."}}

EXAMPLE2:
{{"body part": "left arm", "description": "The man's left arm is bent at the elbow and held close to his body."}}
{{"body part": "right arm", "description": "The man's right arm moves in a circular motion."}}
{{"body part": "left leg", "description": "The man's left leg is bent at the knee."}}
{{"body part": "right leg", "description": "The man's right leg is bent at the knee."}}
{{"body part": "spine", "description": "The man's spine moves in a rhythmic motion."}}

EXAMPLE3:
{{"body part": "left arm", "description": "The man's left arm is raised in a bent position."}}
{{"body part": "right arm", "description": "The man's right arm is raised and bent at the elbow."}}
{{"body part": "left leg", "description": "The man's left leg is lifted and bent at the knee."}}
{{"body part": "right leg", "description": "The man's right leg is bent at the knee and lifted slightly off the ground."}}
{{"body part": "spine", "description": "The spine is arched slightly forward."}}
</EXAMPLES>

<INPUT>
Using the action original_action and the sequence_explanation, generate fine motion control descriptions for the following body parts:

- spine
- left arm
- right arm
- left leg
- right leg

<original_action>
{original_action}
</original_action>

<sequence_explanation>
{sequence_explanation}
</sequence_explanation>

<FORMAT>
For each body part, provide a concise and specific action description in the following JSON format:

{{
    "body part": "left arm",
    "description": "The man's left arm [specific movement description]."
}},
{{
    "body part": "right arm",
    "description": "The man's right arm [specific movement description]."
}},
{{
    "body part": "left leg",
    "description": "The man's left leg [specific movement description]."
}},
{{
    "body part": "right leg",
    "description": "The man's right leg [specific movement description]."
}},
{{
    "body part": "spine",
    "description": "The man's spine [specific movement description]."
}}
</FORMAT>
</INPUT>

<END_SIGNAL>
When you finish, say '<CONTROLEND>'.
</END_SIGNAL>
"""
    return prompt


def analyze_fine_moton_control_txt(action):
    # Step 1: Get sequence explanation
    sequence_explanation_prompt = generate_sequence_explanation_prompt(action)
    sequence_explanation = llm(sequence_explanation_prompt, stop=["<SEQUENCEEND>"]).split("<SEQUENCEEND>")[0].strip()
    print(sequence_explanation)

    # Step 2: Evaluate fine motion control
    fine_moton_control_prompt = generate_fine_motion_control_prompt(action, sequence_explanation)
    control_evaluation = llm(fine_moton_control_prompt, stop=["<CONTROLEND>"]).split("<CONTROLEND>")[0].strip()

    # Parse the JSON objects from the control evaluation
    json_objects = re.findall(r'\{[^}]+\}', control_evaluation)
    control_results = [json.loads(obj) for obj in json_objects]

    # Output to file as well as print
    # output = {
    #     "Action": action,
    #     "Sequence Explanation": sequence_explanation,
    #     "Fine Motion Control Evaluation": control_results
    # }
    output = control_results

    # Write to file
    with open("analyze_fine_moton_control_complex.json", "a") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")

    output2 = {
        "Action": action,
        "Sequence Explanation": sequence_explanation,
        "Fine Motion Control Evaluation": control_results
    }
    with open("fine_control_complex.txt", "a") as file:
        file.write(json.dumps(output2, ensure_ascii=False, indent=2))
        file.write("\n")

    # Print to console
    print(json.dumps(output, ensure_ascii=False, indent=2))

    # Return results as well
    return sequence_explanation, control_results


# actions_to_test = [
#     "The person performs a rowing motion with their legs spread wide.",
#     "A woman hops forward while holding a T-pose.",
#     "The man executes a jump spin mid-air.",
#     "A person crawls on the ground in a baby-like motion.",
#     "A dancer spins gracefully in a ballet twirl.",
#     "The computer science student attempts one-armed push-ups.",
#     "The soccer player covers their ears during a goal celebration.",
#     "A dad crawls on the floor to retrieve his child’s toy.",
#     "The police officer sprints to chase someone on foot.",
#     "A bodybuilder performs a perfect pistol squat."
# ]

# Test each action and save the results in the fine_control.txt file
# for action in actions_to_test:
#     sequence_explanation, control_results = analyze_fine_moton_control_txt(action)
#
#     # Append the result to the fine_control.txt file
#     with open("fine_motion_control_log_complex.txt", "a") as file:
#         # Append both the sequence explanation and control results
#         file.write("========================")
#         file.write("Action: " + action + "\n")  # Write the action description
#         file.write("Sequence Explanation:\n" + sequence_explanation + "\n")  # Write the sequence explanation
#         file.write("Control Results:\n" + control_results.__str__() + "\n\n")  # Write the fine motion control results
#
