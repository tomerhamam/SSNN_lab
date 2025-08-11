#!/usr/bin/env python3
"""
Quick API test - minimal version for fast verification.
Run this for a super-quick check of your Anthropic API setup.
"""

import os

def quick_api_test():
    print("⚡ Quick API Test")
    print("=" * 30)
    
    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ No ANTHROPIC_API_KEY found")
        print("Set it with: export ANTHROPIC_API_KEY='your-key'")
        return False
    
    print(f"✅ API key found: {api_key[:8]}...")
    
    # Test import
    try:
        import anthropic
        print("✅ Anthropic SDK installed")
    except ImportError:
        print("❌ Run: pip install anthropic")
        return False
    
    # Quick API test
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=20,
            messages=[{"role": "user", "content": "Say 'working'"}]
        )
        
        result = response.content[0].text.strip()
        print(f"✅ API working: '{result}'")
        
        if "working" in result.lower():
            print("🎉 Perfect! You're ready to go!")
        else:
            print("✅ API responding (different response, but working)")
        
        return True
        
    except Exception as e:
        print(f"❌ API failed: {e}")
        return False

if __name__ == "__main__":
    success = quick_api_test()
    
    if success:
        print("\n🚀 Ready! Open snn_lab_interactive_anthropic.ipynb")
    else:
        print("\n🔧 Fix the issues above first")
        print("For detailed help, run: python test_api_connection.py")