# README

## Prerequisites

To use this code, ensure you have the following:

1. **NVIDIA Graphics Card with CUDA Installed**: 
   - Make sure your system has an NVIDIA GPU and CUDA is properly installed.

2. **Install Requirements**:
   - Install the required Python packages by running:
     ```bash
     pip install -r requirements.txt
     ```

3. **Unsloth Installation**:
   - Visit the [Unsloth Installation Guide](https://docs.unsloth.ai/get-started/installing-+-updating) and follow the instructions for your specific system.

## First-Time Setup

- If this is your first time running the code, open `routes.py` and **uncomment the following lines** to download the necessary model files:

    ```python
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-14B-Instruct-1M-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    ```

- These lines will download the model and tokenizer from the specified repository. Ensure that your internet connection is stable during the download process.

## Running the Application

1. **Start the Flask Server**:
   - Run the Flask server using the following command:
     ```bash
     python run.py
     ```

2. **Access the Application**:
   - Open your web browser and navigate to:
     ```
     http://localhost:5000
     ```

## Notes

- Ensure all dependencies are correctly installed before running the application.
- If you encounter any issues, refer to the Unsloth documentation or check the console logs for more information.