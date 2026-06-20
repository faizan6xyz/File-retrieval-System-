import json
from openai import OpenAI

# --- CONFIGURATION ---
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

def get_llm_step(goal, current_state, previous_steps):
    """
    Calls the LLM to get the next single step in JSON format.
    """
    system_prompt = """
You are a Browser Automation Planner. 
Output ONLY a single JSON object for the NEXT step.

ACTION TABLE:
| action   | target (ref_id)      | value          |
|----------|----------------------|----------------|
| navigate | null                 | URL            |
| click    | exact ref from state | null           |
| type     | exact ref from state | text           |
| search   | exact ref from state | query text     |
| scroll   | "up" or "down"       | null           |
| wait     | null                 | seconds (int)  |
| finish   | null                 | "true"/"false" |

RULES:
1. "target" MUST be a ref_id (e.g., "e22") found in the Current State.
2. Include a "description" field with simple English (e.g., "Click search").
3. If goal is done, use action "finish" with value "true".
"""

    # Format history for the LLM
    history_str = "\n".join([f"{i+1}. {s}" for i, s in enumerate(previous_steps)]) if previous_steps else "None"

    user_prompt = f"""
Goal: {goal}
Current State: {current_state}
Previous Steps: {history_str}

Output JSON only:
"""

    try:
        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

def print_human_readable_step(json_step):
    """
    Parses JSON and prints simple language like 'click search' or 'scroll down'.
    Returns True if finished, False otherwise.
    """
    try:
        # Clean markdown if present
        clean = json_step.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        
        action = data.get("action")
        desc = data.get("description", "")
        value = data.get("value")

        if action == "finish":
            print(f"✅ DONE: {desc}")
            return True
        
        # Map actions to simple verbs
        if action == "navigate":
            print(f"🌐 Navigate to {value}")
        elif action == "click":
            print(f"🖱️ {desc}")
        elif action == "type":
            print(f"⌨️ Type '{value}'")
        elif action == "search":
            print(f"🔍 Search for '{value}'")
        elif action == "scroll":
            print(f"📜 Scroll {data.get('target')}")
        elif action == "wait":
            print(f"⏳ Wait {value} seconds")
        else:
            print(f"⚙️ {desc}")
            
        return False

    except Exception as e:
        print(f"❌ Parse Error: {e}")
        return False

def run_automation_workflow(goal, max_steps=6):
    """
    Main loop that calls LLM and prints human-readable steps.
    """
    print(f"\n🚀 Goal: {goal}\n" + "-"*30)
    
    # Mock State (In real app, this comes from Playwright/Selenium)
    current_state = """
    - generic [ref=e1]:
      - combobox "Search" [ref=e22] [cursor=pointer]
      - link "Liked Videos" [ref=e50] [cursor=pointer]
      - button "Menu" [ref=e10] [cursor=pointer]
    """
    
    steps_list = []
    
    for i in range(max_steps):
        # 1. Call LLM
        raw_json = get_llm_step(goal, current_state, steps_list)
        if not raw_json: break
        
        # 2. Print Human Readable Step
        is_finished = print_human_readable_step(raw_json)
        
        # 3. Store step
        steps_list.append(raw_json)
        
        if is_finished: break
        
        # Simulate state change to prevent looping
        current_state = "State updated: Action performed."

# --- TEST IT ---
if __name__ == "__main__":
    # Example 1: YouTube Search
    usergoal = input ("what you want : ")
    run_automation_workflow(usergoal)
    
    print("\n" + "="*30 + "\n")