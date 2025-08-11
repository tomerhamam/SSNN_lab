# API Key Migration Guide

## ⚠️ Important Change

To avoid conflicts with Claude Code's own API key, the lab now uses a different environment variable name.

## 🔄 Quick Migration

**OLD** (conflicts with Claude Code):
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

**NEW** (no conflicts):
```bash
export MY_APP_ANTHROPIC_KEY="your-key-here"
```

## 📝 Step-by-Step Update

### 1. Update your ~/.bashrc
```bash
# Remove or comment out the old line
# export ANTHROPIC_API_KEY="your-key-here"

# Add the new line
echo 'export MY_APP_ANTHROPIC_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 2. Verify the Change
```bash
echo $MY_APP_ANTHROPIC_KEY
```
You should see your API key displayed.

### 3. Test the Setup
```bash
python quick_test.py
```

## ✅ What's Been Updated

All files now use `MY_APP_ANTHROPIC_KEY`:
- ✅ `quick_test.py`
- ✅ `test_api_connection.py` 
- ✅ `README.md`
- ✅ `snn_lab_interactive_anthropic.ipynb`

## 🚀 Ready to Go!

Once you've updated your environment variable, everything should work exactly as before - just with no conflicts with Claude Code's API access.

Test with:
```bash
python quick_test.py
```

Then start learning:
```bash
jupyter notebook snn_lab_interactive_anthropic.ipynb
```