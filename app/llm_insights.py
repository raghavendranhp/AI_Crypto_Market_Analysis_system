import os
from groq import Groq
from dotenv import load_dotenv

#load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

def get_market_insights(metrics: dict, predicted_trend: str, anomaly_status: str) -> str:
    #read system prompt
    #ensure absolute path or path relative to execution dir is handled correctly.
    #assuming script runs from project_root or project_root/app
    prompt_path = "../prompts/system_prompt.txt"
    if not os.path.exists(prompt_path):
        prompt_path = "prompts/system_prompt.txt"
        
    with open(prompt_path, "r") as f:
        system_prompt = f.read()

    #initialize groq client
    client = Groq()
    
    #format metrics strictly
    metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
    state_str = f"trend prediction: {predicted_trend}, anomaly prediction: {anomaly_status}"
    
    #replace placeholders if we had them, or just append
    user_prompt = f"metrics:\\n{metrics_str}\\n\\nstate:\\n{state_str}"
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=150
        )
        insight = response.choices[0].message.content
        
        #enforce 500 character limit strictly
        if len(insight) > 500:
            insight = insight[:497] + "..."
            
        return insight
    except Exception as e:
        return f'{{"error": "{str(e)}"}}'
