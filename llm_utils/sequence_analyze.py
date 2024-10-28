from llm_utils import generate_sequence_explanation_prompt
from llm_utils import generate_fine_motion_control_prompt
from llm_utils import generate_sequence_explanation_prompt_json
from llm_utils import llm

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
    with open("llm_result/sequence_explanation.txt", "a") as file:
        file.write(output)

    # Print to console
    print(output)

    # Return results as well
    return sequence_explanation, sequence_explanation2


