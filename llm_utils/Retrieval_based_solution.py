# %%
import os
import openai
import yaml

import re

import json
import re

# Load API key from a YAML configuration file
with open("/home/haoyum3/momask-codes-hzh/llm_utils/config.yaml") as f:
 config_yaml = yaml.load(f, Loader=yaml.FullLoader)

# Initialize the OpenAI client with the API key
client = openai.OpenAI(api_key=config_yaml['token'])

# Set the model name
MODEL = "gpt-4o"


def llm(prompt, stop=["\n"]):
    # Prepare the dialog for the API request
    dialogs = [{"role": "user", "content": prompt}]

    completion = client.chat.completions.create(
        model=MODEL,
        messages=dialogs
    )
    return completion.choices[0].message.content


def get_single_sentence(original_action,reasoning):
    prompt = f"""
     I want to refrase the action description "{original_action}" in to a simpler and clear and clean sentence. 
     you should see the import reasoning of the action
     
     <reasoning>
     original_action :"{original_action}"
     reasoning: "{reasoning}"
     </reasoning>
     
     The sentence should start with " A parson ... " and should be easy to understand and should only use words from the words list blow.
     A word can be used as long as it is mentioned in the **words_list**, regardless of its form. For example, if "walking" is in the **words_list**, then "walks" can also be used.

     <words_list>
     words = [
    'the', 'to', 'a', 'edge', 'rocking', 'fingers', 'marches', 
    'pretends', 'stirring', 'mixing', 'washes', 'pulling', 'also', 'style', 'shrugs', 'wipe', 'hed', 'forearms', 
    'limping'
]
     </words_list>

     The output can only be one sentence:" A parson ... " 
     The output can only be one sentence:" A parson ... " 
     do NOT add any other description!

     you should only use words from the words_list
 
    <END_SIGNAL>
    - When your sentence is finished, signal completion by saying '<DONE>'.
    </END_SIGNAL>

    """

    return prompt


def get_reasoning(original_action):
    prompt = f"""

    you are a reasoning and action model. your role is to, given a text description of a motion, first reason how the human body should look like, and then simplify the sentence into more universally understood body-part movement descriptions, so that anyone not familiar with the specifics of the phrase can understand the motion conatined in it. your reasoning step consists of understanding the motion, then synthesizing how the general motion pose looks like for the person, and then pinpoint how their arms and legs look like. It is essential that you keep the output sentence simple, even if the input motion is complicated, in which case you will do your best to encapsulate the key identifying points of the motion into, at most, two different motions within the output sentence. It is CRITICAL that you simplify the output to at most 2 separate motions, even if the originally described motion is a little less precise because of this.
    here is an example to get you started:
    motion: the worried contractor walks in a hurry
    reasoning 1: The prompt describes a person walking in a hurry. reasoning 2: How does a person look like when they are walking in a hurry? what are the main characteristics of the body when performing that action? reasoning 3: Walking in a hurry is walking with increased speed, so the body parts behave similar to a walk, that is, arms moving alternately, as well as legs.
    output: the man walks fast

    Now, I want to analyze the action description "{original_action}"  

    <END_SIGNAL>
    - When your sentence is finished, signal completion by saying '<DONE>'.
    </END_SIGNAL>

    """
    return prompt


def first_sequence_analyze(action):
    get_reasoning_prompt = get_reasoning(action)
    reasoning = llm(get_reasoning_prompt, stop=["<DONE>"]).split("<DONE>")[0].strip()
    print(reasoning)

    prompt = get_single_sentence(action,reasoning)
    result = llm(prompt, stop=["<DONE>"]).split("<DONE>")[0].strip()
    print(result)
    # print(result)
    return result


actions_to_test = [
    "The person performs a rowing motion with their legs spread wide.",
    "A woman hops forward while holding a T-pose.",
    "The man executes a jump spin mid-air.",

]

for action in actions_to_test:
    first_sequence_analyze(action)


