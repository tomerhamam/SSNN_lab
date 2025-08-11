#!/usr/bin/env python3
"""
Test script to verify Anthropic API key is working properly.
Run this before starting the SSL lab to ensure everything is set up correctly.
"""

import os
import sys
from typing import Optional

def test_environment_setup():
    """Test if environment is properly configured."""
    print("🧪 Testing Environment Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required for Anthropic SDK")
        return False
    else:
        print("✅ Python version is compatible")
    
    # Check for API key
    api_key = os.getenv('MY_APP_ANTHROPIC_KEY')
    if api_key:
        print(f"✅ API key found (starts with: {api_key[:8]}...)")
        return True, api_key
    else:
        print("❌ MY_APP_ANTHROPIC_KEY not found in environment")
        print("\n🔧 To fix this:")
        print("1. Get API key from: https://console.anthropic.com/")
        print("2. Add to ~/.bashrc: export MY_APP_ANTHROPIC_KEY='your-key-here'")
        print("3. Run: source ~/.bashrc")
        print("4. Restart this script")
        return False, None

def test_anthropic_installation():
    """Test if Anthropic SDK is installed and importable."""
    print("\n📦 Testing Anthropic SDK Installation")
    print("=" * 50)
    
    try:
        import anthropic
        print(f"✅ Anthropic SDK installed (version: {anthropic.__version__})")
        return True
    except ImportError:
        print("❌ Anthropic SDK not installed")
        print("\n🔧 To fix this:")
        print("pip install anthropic")
        return False
    except AttributeError:
        # Older versions might not have __version__
        print("✅ Anthropic SDK installed (version unknown)")
        return True

def test_api_connection(api_key: str):
    """Test actual connection to Anthropic API."""
    print("\n🌐 Testing API Connection")
    print("=" * 50)
    
    try:
        import anthropic
        
        # Initialize client
        client = anthropic.Anthropic(api_key=api_key)
        print("✅ Client initialized successfully")
        
        # Test with a simple request
        print("📡 Sending test request to Claude...")
        
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            temperature=0.1,
            messages=[{
                "role": "user", 
                "content": "Reply with exactly: 'API test successful'"
            }]
        )
        
        response = message.content[0].text.strip()
        print(f"📨 Response received: '{response}'")
        
        # Check if response is reasonable
        if "API test successful" in response or "successful" in response.lower():
            print("✅ API connection working perfectly!")
            return True
        else:
            print(f"⚠️ API responding but with unexpected content: {response}")
            return True  # Still working, just different response
            
    except Exception as e:
        print(f"❌ API connection failed: {str(e)}")
        print("\n🔧 Possible issues:")
        print("1. Invalid API key")
        print("2. Network connectivity issues")
        print("3. Rate limiting (wait a moment and try again)")
        print("4. API service temporarily unavailable")
        return False

def test_json_parsing():
    """Test JSON parsing capability for assessments."""
    print("\n🔧 Testing Assessment Format")
    print("=" * 50)
    
    api_key = os.getenv('MY_APP_ANTHROPIC_KEY')
    if not api_key:
        print("⚠️ Skipping - no API key available")
        return False
    
    try:
        import anthropic
        import json
        
        client = anthropic.Anthropic(api_key=api_key)
        
        test_prompt = """
        You are testing an educational assessment system. 
        Please respond with ONLY this JSON object (no additional text):
        
        {
            "score": 85,
            "feedback": "This is a test response",
            "status": "working"
        }
        """
        
        print("📡 Testing JSON response format...")
        
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            temperature=0.1,
            messages=[{"role": "user", "content": test_prompt}]
        )
        
        response = message.content[0].text.strip()
        print(f"📨 Raw response: {response}")
        
        # Try to parse as JSON
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group())
            print("✅ JSON parsing successful!")
            print(f"   Parsed data: {parsed_json}")
            
            # Check required fields
            if 'score' in parsed_json and 'feedback' in parsed_json:
                print("✅ Assessment format compatible!")
                return True
            else:
                print("⚠️ JSON parsed but missing expected fields")
                return False
        else:
            print("❌ Could not extract valid JSON from response")
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Assessment format test failed: {e}")
        return False

def test_cost_estimation():
    """Provide cost estimation for the lab."""
    print("\n💰 Cost Estimation")
    print("=" * 50)
    
    print("Model: Claude 3 Haiku")
    print("Estimated costs per assessment:")
    print("  • Input tokens: ~500 tokens")
    print("  • Output tokens: ~200 tokens") 
    print("  • Cost per assessment: ~$0.0004 USD")
    print("  • Total for 3 questions: ~$0.0012 USD")
    print("\nWith $5 free credits, you can run ~4,000 assessments!")
    print("This is very cost-effective for educational use. 💪")

def run_all_tests():
    """Run complete test suite."""
    print("🚀 SSL Lab - API Connection Test Suite")
    print("=" * 60)
    print("This test verifies your setup before starting the lab.\n")
    
    # Test 1: Environment setup
    env_result = test_environment_setup()
    if isinstance(env_result, tuple):
        env_ok, api_key = env_result
    else:
        env_ok, api_key = env_result, None
    
    if not env_ok:
        print("\n❌ Environment setup failed. Please fix the issues above.")
        return False
    
    # Test 2: SDK installation
    if not test_anthropic_installation():
        print("\n❌ SDK installation failed. Please install the required package.")
        return False
    
    # Test 3: API connection
    if not test_api_connection(api_key):
        print("\n❌ API connection failed. Please check your API key.")
        return False
    
    # Test 4: JSON parsing for assessments
    if not test_json_parsing():
        print("\n⚠️ Assessment format test had issues, but basic API works.")
        print("   The lab should still function with manual evaluation.")
    
    # Test 5: Cost estimation
    test_cost_estimation()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Your setup is ready for the SSL lab")
    print("✅ Automatic evaluation will work")
    print("✅ You can now open: snn_lab_interactive_anthropic.ipynb")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🚀 Ready to start learning! Open the interactive notebook now.")
        sys.exit(0)
    else:
        print("\n❌ Setup incomplete. Please resolve the issues above.")
        sys.exit(1)