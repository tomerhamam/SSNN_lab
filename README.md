# Self-Supervised Neural Networks Interactive Lab

An interactive educational lab for learning self-supervised learning principles through hands-on implementation with AI-powered assessment.

## 🎯 Quick Start

### 1. **Get Your API Key**
- Visit [Anthropic Console](https://console.anthropic.com/)
- Create account and get API key (includes $5 free credits)
- Add to your environment:
  ```bash
  echo 'export MY_APP_ANTHROPIC_KEY="your-api-key-here"' >> ~/.bashrc
  source ~/.bashrc
  ```

### 2. **Install Dependencies**
```bash
pip install anthropic numpy scikit-learn matplotlib seaborn
```

### 3. **Test Your Setup**
```bash
python quick_test.py        # 30-second verification
python test_api_connection.py  # Comprehensive test (optional)
```

### 4. **Start Learning!**
```bash
jupyter notebook snn_lab_complete.ipynb
```

## 📚 What's Included

### **Main Notebook**: `snn_lab_complete.ipynb`
This is your complete 2-hour interactive learning experience with:

#### 🧠 **Interactive Learning Components**
- **10 Progressive Coding Exercises** (easy → advanced)
- **Fill-in-the-Blank Code Templates** with guided hints
- **Multiple Choice Questions** with instant feedback
- **3 Open-Ended Questions** with AI evaluation
- **Critical Thinking Prompts** throughout
- **Real-time Visualization** of learned features

#### 📖 **Educational Modules** (2-hour structured learning)
1. **Part 1: Introduction & Setup** (15 min) - Core concepts, data exploration
2. **Part 2: Computer Vision SSL** (45 min) - Rotation prediction, neural networks
3. **Part 3: Time Series SSL** (30 min) - Autoencoder implementation
4. **Part 4: Advanced Concepts** (15 min) - Creative extensions, novel pretext tasks  
5. **Part 5: Assessment & Reflection** (15 min) - AI-powered evaluation

#### 🤖 **AI-Powered Assessment**
- **Automatic Evaluation** using Claude 3 Haiku
- **Detailed Feedback** with scores, strengths, improvements
- **Cost-Effective** (~$0.0004 per evaluation)
- **Fallback Mode** for manual evaluation without API

## 🎓 Learning Path

### **Exercises Overview**
| Exercise | Difficulty | Topic | Skills |
|----------|------------|-------|--------|
| 1 | Easy | Data Visualization | NumPy, matplotlib, data exploration |
| 2 | Easy-Medium | Rotation Dataset Creation | Image transformations, labeling |
| 3 | Medium | Neural Network Forward Pass | Matrix operations, softmax |
| 4 | Medium-Hard | Training & Evaluation | Gradient descent, metrics |
| 5 | Hard | Transfer Learning Analysis | Feature extraction, performance |
| 6 | Medium | Time Series Generation | Synthetic data, signal processing |
| 7 | Hard | Autoencoder Implementation | Backpropagation, MSE loss |
| 8 | Advanced | Reconstruction Quality | Visualization, error analysis |
| 9 | Advanced | Embedding Classification | Dimensionality reduction, t-SNE |
| 10 | Creative | Novel Pretext Task Design | Innovation, domain knowledge |

### **Assessment Questions**
1. **Why is rotation prediction effective?** (Visual SSL concepts)
2. **Autoencoders vs Contrastive Learning** (Method comparison)
3. **Design a novel pretext task** (Creative application)

## 🔧 Technical Details

### **Implementation Features**
- ✅ Pure NumPy implementation (no deep learning frameworks)
- ✅ From-scratch neural networks with manual backpropagation
- ✅ Visualization of learned features and representations
- ✅ Transfer learning evaluation on downstream tasks
- ✅ Educational focus over performance optimization

### **Requirements**
- Python ≥ 3.8
- NumPy, scikit-learn, matplotlib, seaborn
- Anthropic SDK for AI evaluation
- Jupyter notebook environment

## 💰 Cost Information

**Very affordable for education:**
- **Model**: Claude 3 Haiku (most cost-effective)
- **Per assessment**: ~$0.0004 (less than half a cent)
- **All 3 questions**: ~$0.0012 total
- **$5 free credits**: Covers ~4,000 assessments!

## 🧪 Testing & Verification

### **Quick Test** (30 seconds)
```bash
python quick_test.py
```
Checks: API key, SDK installation, basic connection

### **Comprehensive Test** (2-3 minutes)  
```bash
python test_api_connection.py
```
Checks: Environment, dependencies, full API functionality, cost estimation

## 🚀 Key Concepts Covered

### **Self-Supervised Learning Paradigms**
- **Generative Methods**: Autoencoders, reconstruction tasks
- **Discriminative Methods**: Contrastive learning, representation learning

### **Pretext Tasks**
- Rotation prediction for visual understanding
- Sequence reconstruction for temporal patterns
- Feature learning without manual labels

### **Transfer Learning**
- Feature extraction from pretext tasks
- Downstream task performance evaluation
- Representation quality assessment

## 📁 Project Structure

```
snn_lab/
├── snn_lab_complete.ipynb              # Main comprehensive tutorial (NEW!)
├── quick_test.py                       # Fast API verification
├── test_api_connection.py              # Comprehensive testing
├── tests/snn_lab.py                    # Standalone implementation
├── CLAUDE.md                           # Development guidelines
├── snn_lab_interactive_anthropic.ipynb # Assessment-only notebook
└── TO_BE_DELETED/                      # Outdated files
```

## 🎯 Getting Started Checklist

- [ ] Get Anthropic API key from console.anthropic.com
- [ ] Add API key to environment: `export MY_APP_ANTHROPIC_KEY="key"`
- [ ] Install dependencies: `pip install anthropic`
- [ ] Test setup: `python quick_test.py`
- [ ] Open notebook: `jupyter notebook snn_lab_complete.ipynb`
- [ ] Complete the 2-hour interactive tutorial and get AI feedback!

## 🌟 Learning Outcomes

After completing this lab, you'll understand:
- Core principles of self-supervised learning
- How to design and implement pretext tasks
- Transfer learning with learned representations
- Trade-offs between different SSL approaches
- Practical implementation without frameworks

## 📚 Additional Resources

### **Research Papers**
- [Self-supervised Learning Survey](https://arxiv.org/abs/2301.05712)
- [SimCLR: Contrastive Learning](https://arxiv.org/abs/2002.05709)
- [Masked Autoencoders (MAE)](https://arxiv.org/abs/2111.06377)
- [Representation Learning Review](https://arxiv.org/abs/1206.5538)

### **Extensions & Challenges**
1. Implement SimCLR from scratch
2. Try masked patch prediction
3. Design novel pretext tasks
4. Apply to your own datasets
5. Combine multiple SSL objectives

## 🤝 Support

- **Setup Issues**: Run `python test_api_connection.py` for diagnostics
- **Learning Questions**: Use the AI evaluation system in the notebook
- **Technical Problems**: Check the built-in documentation and examples

## 📄 License

Educational use encouraged - adapt freely for teaching and learning!

---

**Ready to learn Self-Supervised Learning?** 🚀  
Start with `python quick_test.py` then dive into the interactive notebook!