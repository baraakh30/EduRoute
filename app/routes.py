from unsloth import FastLanguageModel
from flask import  render_template, request, jsonify, send_from_directory,Response
import json
import re
from app import app
from transformers import TextStreamer
import mimetypes
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')
# Configuration
MODEL_DIR = "./career-qwen"
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# make sure to uncomment these if this is your first time using the code

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/Qwen2.5-14B-Instruct-1M-bnb-4bit",
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
# )


# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_DIR,  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# Alpaca prompt template
prompt2 = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a career recommendation AI that MUST respond with VALID JSON ONLY. 
DO NOT INCLUDE ANY TEXT OUTSIDE THE JSON FORMAT. Your response will be parsed programmatically.

Follow these steps to complete the task:

1. Analyze the user's technical level, existing skills, preferred learning method, thinking level and interests
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



prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

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



prompt3 = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an AI chatbot that answers questions and your name is RouteGuide
### Input:
{}
### Response:
{}"""

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

def generate_response(user_profile, prompt_type=1):
    if prompt_type==1:
        alpaca_prompt = prompt
    elif prompt_type == 2:
        alpaca_prompt = prompt2

    inputs = tokenizer(
        [
            alpaca_prompt.format(json.dumps(user_profile),"")
        ],
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True,temperature = 0.6)
    response = tokenizer.batch_decode(outputs)
    return extract_json_from_text(response[0])


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chathome')
def chathome():
    return render_template('chat.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/form')
def form():
    return render_template('form.html')

# Flask route to generate response using prompt
@app.route("/generate_response", methods=["POST"])
def generate_response_api():
    data = request.get_json()
    profile = data.get("profile", None)
    prompt_type = int(data.get("prompt_type", 1))

    if not profile:
        return jsonify({"error": "Profile is required"}), 400

    response = generate_response(profile, prompt_type)
    return jsonify(response)


from threading import Thread
import queue

class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer,skip_prompt=True)
        self.queue = queue
    
    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)  # End-of-stream marker

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    message = data.get("prompt", None)
    if not message:
        return jsonify({"error": "Profile is required"}), 400

    response_queue = queue.Queue()
    
    # Format inputs
    inputs = tokenizer(
        [prompt3.format(json.dumps(message), "")],
        return_tensors="pt"
    ).to("cuda")

    # Create streamer with queue
    streamer = CustomStreamer(tokenizer, response_queue)

    # Generation thread
    def generate():
        model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=256,
            use_cache=True
        )

    # Start generation in background thread
    Thread(target=generate).start()

    # Streaming response generator
    def stream_generator():
        while True:
            chunk = response_queue.get()
            if chunk is None:
                break
            yield chunk.encode('utf-8')

    return Response(stream_generator(), mimetype='text/plain')