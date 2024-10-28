

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
    return prompt
