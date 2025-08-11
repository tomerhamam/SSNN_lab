# SSL Lab: Open-Ended Assessment with Anthropic Claude

## 🚀 Quick Setup Guide

### 1. Get Your Anthropic API Key
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Create an account and generate an API key
3. You'll get $5 in free credits to start with

### 2. Set Up the API Key

Since you mentioned you already set it in `~/.bashrc`, you should be good to go! Just make sure it's exported:

```bash
# Add to ~/.bashrc (you've done this already)
export ANTHROPIC_API_KEY="your-api-key-here"

# Reload your shell
source ~/.bashrc

# Verify it's set
echo $ANTHROPIC_API_KEY
```

### 3. Install Required Packages

```bash
# Install the Anthropic SDK
pip install anthropic

# Or if using the project dependencies
pip install -e .
```

### 4. No Code Changes Required! 

The notebook `snn_lab_interactive_anthropic.ipynb` is **completely ready to use**. Just:

1. Open the notebook
2. Run all cells
3. Answer the questions in the provided cells
4. Use `submit_answer("q1", "your answer")` to get automated evaluation

## 🎯 How to Use

### Option 1: With API Key (Automatic Evaluation)
```python
# Example usage
my_answer = \"\"\"
Rotation prediction works because it forces the network to understand 
spatial relationships and geometric transformations in images...
\"\"\"

result = submit_answer("q1", my_answer)
# You'll get: score, strengths, improvements, and detailed feedback
```

### Option 2: Without API Key (Manual Evaluation)
```python
result = submit_answer("q1", my_answer)
# You'll get: rubric, sample answer, and self-evaluation guide
```

## 💰 Cost Information

- **Model**: Claude 3 Haiku (most cost-effective)
- **Cost per evaluation**: ~$0.0004 (less than half a cent!)
- **Perfect for learning**: Very affordable for educational use

## 📝 Three Assessment Questions

1. **Q1**: Why is rotation prediction effective for visual features?
2. **Q2**: Compare autoencoders vs contrastive learning methods
3. **Q3**: Design a novel pretext task for text data

Each question is worth 100 points based on a 4-item rubric.

## 🔧 Troubleshooting

### API Key Issues
```bash
# Check if your key is set
echo $ANTHROPIC_API_KEY

# If empty, add to ~/.bashrc:
echo 'export ANTHROPIC_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Installation Issues
```bash
# If anthropic package not found
pip install anthropic

# If other issues
pip install -r pyproject.toml
```

### Testing the Setup
Run the test cell in the notebook:
```python
test_assessment_system()
```

This will tell you if everything is working correctly.

## 📚 What's Different from Gemini Version?

✅ **Switched to Anthropic Claude 3 Haiku**
- More reliable API
- Better JSON parsing
- Consistent evaluation format
- 10x cheaper than Claude Sonnet

✅ **Simplified Setup**
- Just need ANTHROPIC_API_KEY environment variable
- No complex API configuration
- Better error handling

✅ **Enhanced Features**
- Automatic fallback to manual evaluation
- Better formatted feedback
- Cost tracking
- Test function included

## 🎓 Ready to Learn!

Open `snn_lab_interactive_anthropic.ipynb` and start learning! The assessment system will help you understand SSL concepts through thoughtful evaluation and feedback.

Remember: The goal is learning, not getting perfect scores. The AI evaluation is designed to be encouraging while helping you improve your understanding.