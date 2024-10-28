
from llm_utils import generate_sequence_explanation_prompt
from llm_utils import generate_fine_motion_control_prompt
from llm_utils import generate_sequence_explanation_prompt_json
from llm_utils import sequence_analyze
from llm_utils import analyze_fine_moton_control_txt


# txt = sequence_analyze("Perform a signature James Bond pose with a dramatic turn and gunpoint.")
# print("done")

# actions_to_test = [
#     "Perform a signature James Bond pose with a dramatic turn and gunpoint.",
#     # "Fight with the fluid precision and power of Bruce Lee.",
#     # "Perform a graceful spinning kick in the style of Jet Li.",
#     # "Execute a stealthy crouch and roll like a ninja.",
#     # "Leap over an obstacle with the agility of a parkour expert.",
#     # "Perform a slow-motion dive while firing two handguns like in an action movie.",
#     # "Execute a precise sword slash followed by a defensive stance like a samurai.",
#     # "Jump from a ledge and roll upon landing to minimize impact, as seen in martial arts films.",
#     # "Perform a quick disarm move to take an opponent’s weapon away.",
#     # "Perform a quick, tactical reload of a firearm while maintaining a defensive stance."
# ]
actions_to_test = [
    "The person performs a rowing motion with their legs spread wide.",
    # "A woman hops forward while holding a T-pose.",
    # "The man executes a jump spin mid-air.",
    # "A person crawls on the ground in a baby-like motion.",
    # "A dancer spins gracefully in a ballet twirl.",
    # "The computer science student attempts one-armed push-ups.",
    # "The soccer player covers their ears during a goal celebration.",
    # "A dad crawls on the floor to retrieve his child’s toy.",
    # "The police officer sprints to chase someone on foot.",
    # "A bodybuilder performs a perfect pistol squat."
]

for action in actions_to_test:
   sequence_explanation, sequence_explanation2 = sequence_analyze(action)

   ###########################################################################################


# Test each action and save the results in the fine_control.txt file
for action in actions_to_test:
    sequence_explanation, control_results = analyze_fine_moton_control_txt(action)

    # Append the result to the fine_control.txt file
    with open("llm_result/fine_motion_control_log_complex.txt", "a") as file:
        # Append both the sequence explanation and control results
        file.write("========================")
        file.write("Action: " + action + "\n")  # Write the action description
        file.write("Sequence Explanation:\n" + sequence_explanation + "\n")  # Write the sequence explanation
        file.write("Control Results:\n" + control_results.__str__() + "\n\n")  # Write the fine motion control results

