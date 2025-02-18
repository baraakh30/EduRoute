import time
from unsloth import FastLanguageModel
import torch

# Configuration
OUTPUT_DIR = "./career-qwen"
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=OUTPUT_DIR,  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# User profiles
user_profiles = [
    {
        "interests": ["Web Development"],
        "level": "Beginner",
        "learning_goal": "Career shift",
        "preferred_learning_method": "Videos",
        "previous_knowledge": ["HTML", "CSS"],
        "thinking_level": "intermediate"
    },
    {
        "interests": ["Data Science"],
        "level": "Beginner",
        "learning_goal": "Career shift",
        "preferred_learning_method": "Books",
        "previous_knowledge": ["Python"],
        "thinking_level": "beginner"
    },
    {
        "interests": ["Digital Marketing"],
        "level": "Intermediate",
        "learning_goal": "Enhance skills",
        "preferred_learning_method": "Online courses",
        "previous_knowledge": ["SEO", "Content Marketing"],
        "thinking_level": "intermediate"
    },
    {
        "interests": ["Graphic Design"],
        "level": "Advanced",
        "learning_goal": "Master design tools",
        "preferred_learning_method": "Hands-on projects",
        "previous_knowledge": ["Adobe Photoshop", "Illustrator"],
        "thinking_level": "advanced"
    }
]

# Alpaca prompt template
prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a career recommendation AI that MUST respond with VALID JSON ONLY. 
DO NOT INCLUDE ANY TEXT OUTSIDE THE JSON FORMAT. Your response will be parsed programmatically.

Follow these steps to complete the task:

1. Analyze the user's technical proficiency, existing skills, and interests.
2. Recommend the most suitable tech career based on the user's profile and aspirations.
3. Create a comprehensive, personalized learning path tailored to the user’s current level. The learning path should consist of 3 to 8 steps, starting from the user’s current skill level and progressing to a more advanced level.
4. Each learning step should be a brief but clear action or resource that directly helps the user progress in their recommended career.
5. Ensure that the suggested learning steps are actionable and practical for a user at their given level.
6. Return the response in this EXACT JSON format:

{{
    "career": "Career Name",
    "learning_path": [
        "Step 1: Description",
        "Step 2: Description",
        "Step 3: Description"
    ]
}}

### Example Input:
{{
    "interests": ["Web Development"]
    "level": "Beginner",
    "learning_goal" : "Career shift",
    "preferred_learning_method" : "Videos",
    "previous_knowledge": ["HTML", "CSS"],
    "thinking_level": "intermediate"
    
}}

### Example Output:
{{
    "career": "Frontend Developer",
    "learning_path": [
        "Step 1: Learn JavaScript fundamentals, focusing on basic syntax, variables, and control structures.",
        "Step 2: Understand and implement Responsive Web Design to make websites mobile-friendly.",
        "Step 3: Learn React.js and its core concepts to build interactive UIs.",
        "Step 4: Learn state management techniques with Redux.",
        "Step 5: Get comfortable with manipulating the DOM and utilizing browser APIs.",
        "Step 6: Learn performance optimization techniques for better web speed.",
        "Step 7: Study TypeScript to enhance your JavaScript code with strong typing.",
        "Step 8: Practice testing JavaScript code using Jest to ensure code quality."
    ]
}}

### Now process this user profile:

### Input:
{}
### Response:
{}
"""



prompt2 = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a career recommendation AI that MUST respond with VALID JSON ONLY. 
DO NOT INCLUDE ANY TEXT OUTSIDE THE JSON FORMAT. Your response will be parsed programmatically.

Follow these steps:
1. Analyze the user's technical level, existing skills, preferred learning method, thinking level and interests
2. Recommend the most suitable tech career
3. Create a comprehensive detailed learning path that will help the user achieve his recommended career with 3-8 steps and based on his level
4. Return the response in this EXACT format:
{{
    "career": "Career Name",
    "learning_path": ["Step 1", "Step 2", "Step 3"]
}}

Example Input:
{{
    "interests": ["Web Development"]
    "level": "Beginner",
    "learning_goal" : "Career shift",
    "preferred_learning_method" : "Videos",
    "previous_knowledge": ["HTML", "CSS"],
    "thinking_level": "intermediate"
}}
Example Output:
{{
    "career": "Frontend Developer",
    "learning_path": [
        "JavaScript Fundamentals",
        "Responsive Web Design",
        "React.js Core Concepts",
        "State Management with Redux",
        "Browser APIs and DOM Manipulation",
        "Web Performance Optimization",
        "TypeScript Basics",
        "Testing with Jest"
    ]
}}

Now process this user profile:

### Input:
{}
### Response:
{}"""

import re
import json

def extract_json_from_text(text):
    # Use regex to find the JSON-like structure, ignoring code block syntax (backticks)
    match = re.search(r'### Response:\s*(\{.*?\})', text.replace('```json', '').replace('```', ''), re.DOTALL)
    
    if match:
        # Extract the matched JSON string
        json_str = match.group(1)
        
        # Parse the JSON string
        try:
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError:
            return "Invalid JSON format"
    else:
        return "No response found"

def generate_repsonse(alpaca_prompt,profile):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                profile,  
                "", 
            )
        ],
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True,temperature = 0.6)
    response = tokenizer.batch_decode(outputs)
    return extract_json_from_text(response[0])


# Generate responses for all user profiles
for i, profile in enumerate(user_profiles):
    print(f"Processing profile {i + 1}/{len(user_profiles)}...")
    start_time = time.time()
    print(generate_repsonse(prompt,profile))
    print("Time : ",time.time() - start_time)

