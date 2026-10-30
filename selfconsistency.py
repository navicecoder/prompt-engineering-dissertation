'''this file is solely to implement self consistency technique'''
import requests
import re
import time
import csv
from datasets import load_dataset
from collections import Counter
import os
from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = "qwen3:32b"
TECHNIQUE_NAME = "Self Consistency"

start = 234
end = 235 #no. of questions
numOfQues = end - start
num_samples = 5
max_retry = 3

url = os.getenv("LLM_API_URL")
api_key = os.getenv("LLM_API_KEY")
if not url or not api_key:
    raise RuntimeError(
        "Set LLM_API_URL and LLM_API_KEY in your local .env file."
    )
headers = {
    "Authorization": f"Bearer {api_key}"
}

dataset = load_dataset("openai/gsm8k", "main") #loading gsm8k dataset

def self_consistency(question):
    new_prompt = f"""
    Solve this problem step by step.

    Question: {question}
    At the very end, write only:
    Final Answer: <number>
    Don't include commas, dollar signs, LaTeX, or unnecessary symbols in the final answer.
    """
    return new_prompt

def extract_answer(text):
    match = re.search(
        r"\**Final Answer\**\s*:?\s*\$?(-?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )

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


csv_file = open("results/results.csv", "a", newline="")
writer = csv.writer(csv_file)

if csv_file.tell() == 0:
    writer.writerow(["Dataset", "Question", "Technique", "GroundTruth", "Prediction", "Correct",
        "TimeSeconds"])

error_file = open("results/gsm8k_errors.txt", "a")

correct = 0
total_time = 0

for i in range(start, end):
    print(f"Processing question {i + 1}")
    question = dataset["test"][i]["question"]
    correct_answer = dataset["test"][i]["answer"]

    prompt = self_consistency(question)

    answers = []
    start_time = time.perf_counter()
    for j in range(num_samples):
        payload2 = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
            #"seed": 101,
                "temperature": 1.8
            }

        }

        for attempt in range(max_retry):

            response2 = requests.post(
                url,
                headers=headers,
                json=payload2
            )

            if response2.status_code == 200:
                break
            print(
                f"Attempt {attempt + 1} failed "
                f"({response2.status_code}). Retrying..."
            )
            time.sleep(2)

        else:
            print("Skipping this sample.")
            continue

        res2 = response2.json()

        answers.append(res2['message']['content'])
            #Get the final answer from each response

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    total_time += elapsed_time
    predicted_answers = []

    for answer in answers:
        #print(answer)
        print("FULL RESPONSE:")
        print(answer)
        print("-" * 50)

        predicted_answers.append(extract_answer(answer))

    #Choose the answer that appears most often
    counts = Counter(predicted_answers)
    prediction = counts.most_common(1)[0][0]

    ground_truth = extract_ground_truth(correct_answer)

    print("=" * 80)
    print(f"Question {i + 1}")
    print(question)
    print()
    print("Answers from samples:")

    for answer in predicted_answers:
        print(answer)

    print("Expected:", ground_truth)
    print("Predicted:", prediction)

    if ground_truth is not None and prediction is not None and float(ground_truth) == float(prediction):
        print("Correct!")
        correct += 1
        is_correct = True
    else:
        print("Incorrect!")
        is_correct = False

        error_file.write("=" * 80 + "\n")
        error_file.write(f"Question {i + 1}\n")
        error_file.write(f"Technique: {TECHNIQUE_NAME}\n")
        error_file.write(f"Question:\n{question}\n")
        error_file.write(f"Expected: {ground_truth}\n")
        error_file.write(f"Predicted: {prediction}\n\n")

        for j, answer in enumerate(answers):
            error_file.write(f"Sample {j + 1}:\n")
            error_file.write(answer + "\n")
            error_file.write("-" * 50 + "\n")

        error_file.flush()

    writer.writerow(["GSM8K", i + 1, TECHNIQUE_NAME, ground_truth, prediction, is_correct,
        round(elapsed_time, 2)])

    csv_file.flush()    

average_time = total_time / numOfQues

print("=" * 80)
print("Final Results")
print(f"Correct: {correct}/{numOfQues}")
print(f"Accuracy: {(correct / numOfQues) * 100:.2f}%")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per question: {average_time:.2f} seconds")

csv_file.close()
error_file.close()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              