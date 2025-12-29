import src.llm_analyst as nexus

# 1. Mock Data (Simulating your CSV data)
mock_context = """
Channel: TV | Spend: $5M | ROI: 18.2 | Role: Efficiency Driver
Channel: Social | Spend: $2M | ROI: 10.1 | Role: Balanced
Channel: Billboard | Spend: $1M | ROI: 28.9 | Role: High Efficiency
"""

print("\n--- 🤖 NEXUS AI TERMINAL TEST ---")
print("Context Loaded: Mock TV, Social, and Billboard data.")
print("Type 'exit' to quit.\n")

# 2. Interactive Loop
while True:
    try:
        user_input = input("👤 YOU: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Exiting test.")
            break
        
        if not user_input:
            continue
            
        print("... Nexus is thinking ...")
        response = nexus.chat_with_data(user_input, mock_context)
        print(f"🤖 NEXUS: {response}\n")
        print("-" * 40)

    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        break