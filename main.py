'''This file runs 4 prompting techniques - Zero shot, few shot, CoT, Role prompting on a configurable range of GSM8K mathematical reasoning questions.
Requires a compatible LLM API endpoint and credentials configured through
the LLM_API_URL and LLM_API_KEY environment variables.
For my dissertation submission I connected to university's VPN and logged in to the linux server to connect to uni's LLM and run this file.
'''
import requests
import re
import time
import csv
from prompts import zero_shot, few_shot, cot, role_prompt
from datasets import load_dataset
import os
from dotenv import load_dotenv
load_dotenv()

#select the prompting technique and model here
PROMPT_FUNC = role_prompt
MODEL_NAME = "qwen3:32b" #`gpt-oss:120b`
TECHNIQUE_NAME = "Role Prompting"

#range of questions you want to run at once
start = 5
end = 6

max_retry = 3 #max attempts if the API fails
error_file = open("results/gsm8k_errors.txt", "a") #created an error file in append mode

def extract_answer(text): #to extract numerical answer from the response
    match = re.search(
        r"\**Final Answer\**\s*:?\s*\$?(-?[\d,]+(?:\.\d+)?)", #looking for a number followed by Final Answer:
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).replace(",", "")
     #alternative if the model did not use Final Answer format
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        print("Final Answer format not found! Using last number.")
        return numbers[-1].replace(",", "")
    return None

def extract_ground_truth(text):
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    return None

url = os.getenv("LLM_API_URL")
api_key = os.getenv("LLM_API_KEY")

if not url or not api_key:
    raise RuntimeError(
        "Set LLM_API_URL and LLM_API_KEY in your local .env file."
    )

headers = {
    "Authorization": f"Bearer {api_key}"
}

dataset = load_dataset("openai/gsm8k", "main")

correct = 0
total_time = 0

csv_file = open("results/results.csv", "a", newline="") #creating results csv file
writer = csv.writer(csv_file) 
if csv_file.tell() == 0: #to check if the file is empty
    writer.writerow(["Dataset", "Question", "Technique", "GroundTruth", "Prediction",
        "Correct", "TimeSeconds"]) #writing columns

numOfQues = end - start
for i in range(start, end): #running selected prompting technique on each question
    print(f"Processing question {i+1}")
    question = dataset["test"][i]["question"]
    correct_answer = dataset["test"][i]["answer"]
    prompt = PROMPT_FUNC(question) #sending the question as a selected prompt to model

    #request sent to the llm gateway
    payload2 = {
        "model": MODEL_NAME, 
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }
    #Start timing immediately before sending the request
    start_time = time.perf_counter()
    
    for j in range(max_retry): 
        response2 = requests.post(
            url,
            headers=headers,
            json=payload2
        )
        if response2.status_code == 200: #stop retrying if the request was successful
            break
        print(f"Attempt {j+1} failed ({response2.status_code}). Retrying...")
        time.sleep(2)
    
    else:
        print("Skipping this question.")
        continue

    #stop the time after a response is received
    end_time = time.perf_counter() 
    elapsed_time = end_time - start_time
    total_time += elapsed_time

    res2 = response2.json()
    print("-"*80)
    ground_truth = extract_ground_truth(correct_answer) #correct answer in the dataset
    
    prediction = extract_answer(res2["message"]["content"]) #model predicted answer

    # Print the results for this question
    print("=" * 80)
    print(f"Question {i+1}")
    print()
    print("Question:")
    print(question)
    print()
    print("Expected:", ground_truth)
    print("Predicted:", prediction)
    print(f"Time taken: {elapsed_time:.2f} seconds")
    #print("-"*80)
    if ground_truth is not None and prediction is not None and float(ground_truth) == float(prediction): #compare both answers
        print("Correct!")
        correct += 1
        is_correct = True
    else:
        print("Incorrect")
        is_correct = False

        #Store incorrect cases in the error log 
        error_file.write("=" * 80 + "\n")
        error_file.write(f"Question {i + 1}\n")
        error_file.write(f"Technique: {TECHNIQUE_NAME}\n")
        error_file.write(f"Question:\n{question}\n")
        error_file.write(f"Expected: {ground_truth}\n")
        error_file.write(f"Predicted: {prediction}\n")
        error_file.write("Model response:\n")
        error_file.write(res2["message"]["content"] + "\n")
        error_file.flush()
    print("-"*80)
    writer.writerow(["GSM8K", i + 1, TECHNIQUE_NAME, ground_truth, prediction, is_correct,
        round(elapsed_time, 2) ])
    csv_file.flush()

average_time = total_time / numOfQues
print("="*80)
print("Final Results")
print(f"Correct: {correct}/{numOfQues}")
print(f"Accuracy: {(correct/numOfQues)*100:.2f}%")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per question: {average_time:.2f} seconds")
csv_file.close()
error_file.close()
