#!/usr/bin/env python3
"""
🌾 Crop Recommendation System - Deployment Monitoring Script
Tests all features after deployment to Render.com
"""

import sys
import time
import requests
import json

# Change this to your deployed Render URL
RENDER_URL = "https://crop-recommender.onrender.com"  # Update with your actual URL

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def test_endpoint(name, method, endpoint, **kwargs):
    """Test an API endpoint"""
    url = f"{RENDER_URL}{endpoint}"
    try:
        if method.upper() == "GET":
            resp = requests.get(url, timeout=10, **kwargs)
        else:
            resp = requests.post(url, timeout=10, **kwargs)
        
        status = f"✅ {resp.status_code}" if resp.status_code < 400 else f"❌ {resp.status_code}"
        print(f"{name:30} {status}")
        return resp
    except requests.exceptions.Timeout:
        print(f"{name:30} ❌ Timeout (>10s)")
        return None
    except Exception as e:
        print(f"{name:30} ❌ Error: {str(e)[:40]}")
        return None

def main():
    print_header("🚀 Crop Recommendation System - Deployment Check")
    
    print(f"Target: {RENDER_URL}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Basic connectivity
    print_header("1️⃣  Basic Connectivity")
    resp = test_endpoint("Homepage", "GET", "/")
    if not resp:
        print("\n❌ Cannot reach Render deployment. Is URL correct?")
        print(f"   Update RENDER_URL in this script to your actual URL")
        sys.exit(1)
    
    # 2. API endpoints (no auth needed for GET /api/schemes/options)
    print_header("2️⃣  API Endpoints (Public)")
    test_endpoint("Schemes Options", "GET", "/api/schemes/options")
    test_endpoint("Schemes List", "GET", "/api/schemes?page=1&per_page=5")
    
    # 3. Chatbot
    print_header("3️⃣  Chatbot (Gemini)")
    resp = test_endpoint("Chatbot Message", "POST", "/chatbot/message",
                        json={"message": "What is best crop for low nitrogen soil?", "lang": "en"})
    if resp and resp.status_code == 200:
        data = resp.json()
        if data.get("success"):
            print(f"   Reply: {data.get('reply', 'N/A')[:80]}...")
    
    # 4. Static assets
    print_header("4️⃣  Static Assets")
    test_endpoint("Home CSS", "GET", "/static/css/site.css")
    test_endpoint("Predict JS", "GET", "/static/js/predict.js")
    
    # 5. HTML Pages (redirects to login if not authenticated)
    print_header("5️⃣  Web Pages (HTML)")
    resp = test_endpoint("Login Page", "GET", "/login")
    resp = test_endpoint("Signup Page", "GET", "/signup")
    
    # 6. Summary
    print_header("✅ Deployment Verification Complete")
    print("""
    Next steps:
    1. Visit https://crop-recommender.onrender.com
    2. Sign up with a test account
    3. Verify OTP in Render logs
    4. Make a crop prediction
    5. Test chatbot and schemes
    
    Monitor logs at: Render Dashboard > Web Service > Logs
    """)

if __name__ == "__main__":
    # Ask for URL if not provided
    if len(sys.argv) > 1:
        RENDER_URL = sys.argv[1]
    else:
        print("Usage: python deployment_monitor.py <render-url>")
        print("Example: python deployment_monitor.py https://crop-recommender.onrender.com")
        print()
        url_input = input("Enter your Render URL (or press Enter to use default): ").strip()
        if url_input:
            RENDER_URL = url_input
        else:
            print(f"Using default: {RENDER_URL}")
    
    main()
