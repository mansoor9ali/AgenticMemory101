import json
import os
import time
from datetime import datetime

# ==========================================
# 1. THE MEMORY STRUCTURE (The Schema)
# ==========================================
INITIAL_MEMORY = {
    "project_name": "todo_cli_tool",
    "features": [
        {
            "id": "F001",
            "name": "Initialize Storage",
            "description": "Create tasks.json if it doesn't exist.",
            "status": "pending",
            "dependencies": []
        },
        {
            "id": "F002",
            "name": "Add Task Function",
            "description": "Create a function to add a task to the JSON file.",
            "status": "pending",
            "dependencies": ["F001"]
        },
        {
            "id": "F003",
            "name": "CLI Interface",
            "description": "Create main.py to handle user input.",
            "status": "pending",
            "dependencies": ["F001", "F002"]
        }
    ],
    "log": []
}

MEMORY_FILE = "project_spec.json"


# ==========================================
# 2. THE HARNESS (The Architect)
# ==========================================
class AgentHarness:
    def __init__(self):
        self.memory = {}

    def boot(self):
        """Step 1: Grounding - Wake up and read the state."""
        if not os.path.exists(MEMORY_FILE):
            print(f"[HARNESS] No memory found. Initializing new project...")
            self.memory = INITIAL_MEMORY
            self._save_memory()
        else:
            print(f"[HARNESS] Waking up. Reading domain memory from {MEMORY_FILE}...")
            with open(MEMORY_FILE, 'r') as f:
                self.memory = json.load(f)

    def select_next_task(self):
        """Step 2: Selection - Find the first unblocked, pending feature."""
        completed_ids = {f['id'] for f in self.memory['features'] if f['status'] == 'completed'}

        for feature in self.memory['features']:
            if feature['status'] == 'pending':
                # Check dependencies
                deps_met = all(dep in completed_ids for dep in feature['dependencies'])
                if deps_met:
                    print(f"[HARNESS] Selected Task: {feature['id']} - {feature['name']}")
                    return feature
                else:
                    print(f"[HARNESS] Skipping {feature['id']} (Dependencies not met)")

        print("[HARNESS] No pending tasks found. Project complete!")
        return None

    def execute_task(self, feature):
        """Step 3: Execution - The 'Two-Agent' Handoff."""
        print(f"[HARNESS] Bootstrapping Agent for {feature['id']}...")

        # 3a. Construct the Prompt (The Context)
        prompt = f"""
        YOU ARE: A Python Coding Agent.
        CONTEXT: We are building {self.memory['project_name']}.
        TASK: Implement feature {feature['id']}: {feature['description']}
        CONSTRAINT: Write valid Python code.
        """

        # 3b. Call the LLM (Simulated here for the example)
        # In a real app, this would be: response = openai.chat.completions.create(...)
        code_output, test_result = self._mock_llm_generation(feature['id'])

        print(f"[AGENT] Generated code for {feature['id']}.")
        print(f"[AGENT] Running Tests... {test_result}")

        # 3c. Update State (The 'Commit')
        if test_result == "PASS":
            feature['status'] = 'completed'
            self._log_work(f"Completed {feature['id']}. Tests passed.")

            # Actually write the code to a file to prove it's real
            filename = f"feature_{feature['id']}.py"
            with open(filename, "w") as f:
                f.write(code_output)
            print(f"[HARNESS] Code written to {filename}")
        else:
            feature['status'] = 'failed'
            self._log_work(f"Failed {feature['id']}. Tests failed.")

        # 3d. Save Memory
        self._save_memory()
        print(f"[HARNESS] Memory updated. Going to sleep.\n")

    def _save_memory(self):
        with open(MEMORY_FILE, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def _log_work(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.memory['log'].append(entry)

    # ==========================================
    # 3. THE LLM SIMULATION (The Brain)
    # ==========================================
    def _mock_llm_generation(self, feature_id):
        """
        Simulates the LLM receiving the prompt and generating code.
        Returns (code_string, test_status)
        """
        time.sleep(1.5)  # Thinking time...

        if feature_id == "F001":
            return ("# Storage Init\nimport json\nimport os\ndef init(): pass", "PASS")
        elif feature_id == "F002":
            return ("# Add Task\ndef add_task(task): pass", "PASS")
        elif feature_id == "F003":
            return ("# CLI Interface\ndef main(): pass", "PASS")
        else:
            return ("# Placeholder code", "PASS")


# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    # Clean up previous runs for this demo
    if os.path.exists(MEMORY_FILE): os.remove(MEMORY_FILE)

    harness = AgentHarness()

    # --- SESSION 1 ---
    print("--- STARTING SESSION 1 ---")
    harness.boot()  # 1. Wake up (creates file)
    task = harness.select_next_task()  # 2. Pick F001
    if task:
        harness.execute_task(task)  # 3. Do F001 -> Save -> Sleep

    time.sleep(1)  # Simulating time passing between sessions

    # --- SESSION 2 ---
    print("--- STARTING SESSION 2 ---")
    # Note: We create a NEW harness instance to prove no internal variables are saved.
    # It MUST read from the JSON file to know what happened.
    harness_new = AgentHarness()
    harness_new.boot()  # 1. Wake up (reads "F001 completed")
    task = harness_new.select_next_task()  # 2. Skips F001, Picks F002
    if task:
        harness_new.execute_task(task)  # 3. Do F002 -> Save -> Sleep

    # --- SESSION 3---
    print("--- STARTING SESSION 3 ---")
    harness_final = AgentHarness()
    harness_final.boot()  # 1. Wake up (reads "F001,F002 completed")
    task = harness_final.select_next_task()  # 2. Picks F003
    if task:
        harness_final.execute_task(task)  # 3. Do F003 -> Save -> Sleep