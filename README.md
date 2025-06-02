# EduRoute 🎓

**Your ultimate guide to personalized learning and career growth**

EduRoute is an AI-powered career guidance platform that provides personalized learning paths and career recommendations for students and professionals in the tech industry. Built with Flask and powered by a fine-tuned Qwen 2.5 language model, EduRoute offers intelligent career counseling and interactive chat support.

## 🌟 Features

### 🎯 Personalized Career Recommendations
- **Smart Assessment**: Interactive questionnaire analyzing interests, experience level, learning goals, and preferred methods
- **AI-Powered Analysis**: Custom-trained Qwen 2.5 model provides tailored career suggestions
- **Comprehensive Learning Paths**: Step-by-step roadmaps from beginner to professional level

### 💬 Interactive AI Chat
- **RouteGuide Chatbot**: Real-time conversational AI for career guidance
- **Streaming Responses**: Live text generation for natural conversation flow
- **Context-Aware**: Understands career-related queries and provides relevant advice

### 🎨 Modern Web Interface
- **Responsive Design**: Bootstrap-powered UI that works on all devices
- **Animated Elements**: Engaging user experience with CSS animations
- **Professional Layout**: Clean, modern design focused on user experience

### 🧠 Advanced AI Technology
- **Fine-Tuned Model**: Custom-trained on career guidance and learning path datasets
- **4-bit Quantization**: Optimized for efficient GPU memory usage
- **Real-time Inference**: Fast response generation with streaming capabilities

## 🚀 Quick Start

### Prerequisites

1. **NVIDIA Graphics Card with CUDA**: Required for GPU acceleration
2. **Python 3.8+**: Ensure Python is installed on your system
3. **Git**: For cloning the repository

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd eduRoute
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Unsloth** (for model training/fine-tuning):
   - Visit the [Unsloth Installation Guide](https://docs.unsloth.ai/get-started/installing-+-updating)
   - Follow the instructions for your specific system

### First-Time Setup

**Important**: If this is your first time running the application, you need to download the base model:

1. Open `app/routes.py`
2. **Uncomment** the following lines (around line 16-21):
   ```python
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name = "unsloth/Qwen2.5-14B-Instruct-1M-bnb-4bit",
       max_seq_length = max_seq_length,
       dtype = dtype,
       load_in_4bit = load_in_4bit,
   )
   ```
3. **Comment out** the existing model loading lines (around line 24-29)
4. Run the application once to download the model
5. After successful download, reverse the changes (comment the download lines, uncomment the local model lines)

### Running the Application

1. **Start the Flask server**:
   ```bash
   python run.py
   ```

2. **Access the application**:
   Open your browser and navigate to: `http://localhost:5000`

## 📁 Project Structure

```
eduRoute/
├── app/
│   ├── __init__.py              # Flask app initialization
│   ├── routes.py                # Main application routes and AI logic
│   ├── templates/               # HTML templates
│   │   ├── index.html          # Homepage
│   │   ├── form.html           # Assessment questionnaire
│   │   ├── results.html        # Career recommendations
│   │   └── chat.html           # AI chatbot interface
│   └── static/                 # Static assets
│       ├── css/                # Stylesheets
│       ├── js/                 # JavaScript files
│       ├── images/             # Image assets
│       └── assets/             # Additional assets
├── career-qwen/                # Fine-tuned model directory
├── model/                      # Model artifacts
├── qwentraining.py            # Model training script
├── qwentesting.py             # Model testing utilities
├── run.py                     # Application entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🛠️ Technical Stack

### Backend
- **Flask**: Web framework for Python
- **Unsloth**: Efficient fine-tuning library
- **Transformers**: Hugging Face transformers library
- **PyTorch**: Deep learning framework
- **CUDA**: GPU acceleration

### Frontend
- **Bootstrap 5.3.3**: CSS framework
- **JavaScript**: Interactive functionality
- **HTML5/CSS3**: Modern web standards
- **Font Awesome**: Icon library
- **Google Fonts**: Typography

### AI/ML
- **Qwen 2.5-14B**: Base language model
- **LoRA**: Low-Rank Adaptation for efficient fine-tuning
- **4-bit Quantization**: Memory-efficient model loading
- **Custom Datasets**: Career guidance and learning path data

## 🎯 Usage Guide

### 1. Career Assessment
1. Navigate to the homepage
2. Click "Start your journey"
3. Complete the interactive questionnaire:
   - Enter your name
   - Select interests (Data Science, Web Development, AI, etc.)
   - Choose experience level
   - Define learning goals
   - Select preferred learning methods
   - Specify programming background

### 2. Get Recommendations
- Receive AI-generated career suggestions
- View detailed learning paths with actionable steps
- Download or save recommendations for future reference

### 3. Chat with RouteGuide
1. Click "chat with RouteGuide"
2. Ask career-related questions
3. Get real-time AI responses
4. Explore different career paths and learning strategies

## 🧪 Model Training

The project includes scripts for training and testing the AI model:

### Training a Custom Model
```bash
python qwentraining.py
```

This script:
- Loads the base Qwen 2.5-14B model
- Applies LoRA fine-tuning
- Trains on career guidance datasets
- Saves the fine-tuned model to `./career-qwen`

### Testing the Model
```bash
python qwentesting.py
```

## 📊 Datasets Used

1. **Career Guidance QA Dataset**: Question-answer pairs for career counseling
2. **Learning Path Dataset**: Structured learning paths for various tech skills
3. **Custom Prompts**: Alpaca-style instruction templates

## 🔧 Configuration

### Model Configuration
- **Max Sequence Length**: 2048 tokens
- **Quantization**: 4-bit for memory efficiency
- **LoRA Rank**: 16
- **Learning Rate**: 2e-4

### Server Configuration
- **Host**: 0.0.0.0 (accessible from network)
- **Port**: 5000
- **Debug Mode**: Enabled in development

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in training configuration
   - Ensure 4-bit quantization is enabled
   - Close other GPU-intensive applications

2. **Model Download Issues**:
   - Ensure stable internet connection
   - Check Hugging Face Hub access
   - Verify sufficient disk space

3. **Bootstrap Integrity Errors**:
   - The project uses Bootstrap 5.3.3 with verified integrity hashes
   - If issues persist, check network connectivity

### Performance Tips

- Use NVIDIA GPUs with at least 8GB VRAM for optimal performance
- Enable CUDA for faster inference
- Consider using smaller models for development/testing

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Integration with job boards
- [ ] Progress tracking and analytics
- [ ] Mobile application
- [ ] Advanced recommendation algorithms
- [ ] Community features and forums

## 📞 Support

For questions, issues, or contributions, please:
1. Check the troubleshooting section
2. Review existing issues in the repository
3. Contact the development team

---

**EduRoute** - Empowering your journey in tech education and career development! 🚀