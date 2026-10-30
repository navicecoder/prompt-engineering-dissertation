'''This file implements Tree of Thought technique'''
import requests
import time
import re
import csv
from datasets import load_dataset
import os
from dotenv import load_dotenv
load_dotenv()

url = os.getenv("LLM_API_URL")
api_key = os.getenv("LLM_API_KEY")
if not url or not api_key:
    raise RuntimeError(
        "Set LLM_API_URL and LLM_API_KEY in your local .env file."
    )
headers = {
    "Authorization": f"Bearer {api_key}"
}

start = 7
end = 8

def ask_model(prompt):
    payload = {
        "model": "qwen3:32b",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }

    max_retries = 3
    for attempt in range(max_retries):
        response = requests.post(
            url,
            headers=headers,
            json=payload
        )
        if response.status_code == 200:
            return response.json()["message"]["content"]
        print(f"Attempt {attempt + 1} failed ({response.status_code})")
        time.sleep(2)

    raise RuntimeError("API request failed after 3 attempts")


def extract_answer(text):
    match = re.search(r"\**Final Answer\**\s*:?\s*\$?(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None

def extract_ground_truth(text):
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    
    return None

def tree_of_thought(question): #Generate different approaches
    prompt1 = f"""
            Solve this problem using 3 different approaches.Don't give the final answer yet.
            Question:
            {question}
            Give your responses as follows:
            Approach 1: ...
            Approach 2: ...
            Approach 3: ...
            """
    thoughts = ask_model(prompt1)
    print("GENERATED THOUGHTS:")
    print(thoughts)
    print("-" * 80)

    #Evaluate the approaches
    prompt2 = f""" Check these 3 approaches to the problem.
                Question:
                {question}
                Approaches:
                {thoughts}
                Check which approaches are correct and choose the best one.

                Return:
                Approach 1: ...
                Approach 2: ...
                Approach 3: ...
                Best approach: 1, 2, or 3
                """

    evaluation = ask_model(prompt2)
    print("EVALUATION:")
    print(evaluation)
    print("-" * 80)

    #Find the best approach
    match = re.search(
        r"Best approach.*?(1|2|3)",
        evaluation,
        re.IGNORECASE
    )
    if not match:
        raise RuntimeError("Could not find the best approach")

    best_approach = int(match.group(1))
    print("BEST APPROACH:", best_approach)

    #Get selected approach
    pattern = rf"Approach {best_approach}:\s*(.*?)(?=\n\s*Approach \d:|\Z)"
    match = re.search(pattern, thoughts, re.DOTALL)
    if not match:
        raise RuntimeError("Could not find selected approach")

    selected_thought = match.group(1).strip()

    print("SELECTED THOUGHT:")
    print(selected_thought)
    print("-" * 80)

    #Generate final answer
    prompt3 = f"""Solve the problem using the selected approach below.
                Question:
                {question}
                Selected approach:
                {selected_thought}
                At the end, write exactly: Final Answer: <number>
                Do not include any signs like commas, dollar signs, LaTeX or units in the final answer."""

    final_answer = ask_model(prompt3)
    return final_answer, thoughts, evaluation, selected_thought

dataset = load_dataset("openai/gsm8k", "main")
TECHNIQUE_NAME = "ToT"

numOfQues = end - start
correct = 0
total_time = 0

csv_file = open("results/results.csv", "a", newline="")
writer = csv.writer(csv_file)

if csv_file.tell() == 0:
    writer.writerow(["Dataset", "Question", "Technique", "GroundTruth", "Prediction", "Correct",
        "TimeSeconds"])
error_file = open("results/gsm8k_errors.txt", "a")

for i in range(start, end):
    question = dataset["test"][i]["question"]
    correct_answer = dataset["test"][i]["answer"]
    print("+" * 100)
    print(f"Question {i + 1}")
    print(question)
    print()

    start_time = time.perf_counter()
    final_answer, thoughts, evaluation, selected_thought = tree_of_thought(question)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    total_time += elapsed_time

    ground_truth = extract_ground_truth(correct_answer)
    prediction = extract_answer(final_answer)

    print("Expected:", ground_truth)
    print("Final predicted answer:", prediction)
    print(f"Time taken: {elapsed_time:.2f} seconds")

    if ground_truth is not None and prediction is not None and float(ground_truth) == float(prediction):
        print("Correct!")
        correct = correct+1
        is_correct = True
    else:
        print("Incorrect!")
        is_correct = False
        error_file.write("=" * 80 + "\n")
        error_file.write(f"Question {i + 1}\n")
        error_file.write(f"Technique: {TECHNIQUE_NAME}\n")
        error_file.write(f"Question:\n{question}\n")
        error_file.write(f"Expected: {ground_truth}\n")
        error_file.write(f"Predicted: {prediction}\n")
        error_file.write("Generated thoughts:\n")
        error_file.write(thoughts + "\n\n")
        error_file.write("Evaluation:\n")
        error_file.write(evaluation + "\n\n")
        error_file.write("Selected thought:\n")
        error_file.write(selected_thought + "\n\n")

        error_file.write("Final answer:\n")
        error_file.write(final_answer + "\n\n")
        error_file.flush()
    print()

    writer.writerow(["GSM8K", i + 1, TECHNIQUE_NAME, ground_truth, prediction, is_correct,
        round(elapsed_time, 2)])
    csv_file.flush()
print("=" * 80)
print("Final Results")
print(f"Correct: {correct}/{numOfQues}")
print(f"Accuracy: {(correct / numOfQues) * 100:.2f}%")
average_time = total_time / numOfQues
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per question: {average_time:.2f} seconds")
csv_file.close()
error_file.close()