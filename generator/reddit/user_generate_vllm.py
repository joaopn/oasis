# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import requests
import argparse

# vLLM server configuration
VLLM_SERVERS = [
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001", 
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8003"
]

def get_next_server():
    """Round-robin selection of vLLM servers"""
    if not hasattr(get_next_server, "current"):
        get_next_server.current = 0
    server = VLLM_SERVERS[get_next_server.current]
    get_next_server.current = (get_next_server.current + 1) % len(VLLM_SERVERS)
    return server

def vllm_completion(messages, max_retries=3):
    """Make a request to vLLM server in OpenAI-compatible format with retries"""
    # Add a strong system message about JSON-only output
    messages = [{
        "role": "system",
        "content": "You are a JSON generator. You MUST ONLY output valid JSON. No other text, no explanations, no markdown formatting. ONLY the JSON."
    }] + messages

    server_url = f"{get_next_server()}/v1/chat/completions"
    
    payload = {
        "model": "iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
        "stop": ["\n\n", "</s>", "```"]  # Stop on any markdown or extra newlines
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(server_url, json=payload, timeout=30)
            response.raise_for_status()
            response_json = response.json()
            
            if not response_json.get("choices") or not response_json["choices"][0].get("message"):
                raise ValueError("Invalid response structure from vLLM")
            
            content = response_json["choices"][0]["message"]["content"].strip()
            
            # Try to parse JSON directly first
            try:
                return {"choices": [{"message": {"content": json.dumps(json.loads(content))}}]}
            except json.JSONDecodeError:
                # If that fails, try to clean and extract
                content = content.replace("```json", "").replace("```", "").strip()
                if attempt == max_retries - 1:
                    print(f"Failed to parse JSON: {content}")
                    raise
                continue
                
        except (requests.exceptions.RequestException, ValueError, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                print(f"Error after {max_retries} attempts: {e}")
                print(f"Last response: {response.text if 'response' in locals() else 'No response'}")
                raise
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
            continue

# Gender ratio
gender_ratio = [0.351, 0.636]
genders = ['female', 'male']

# Age ratio
age_ratio = [0.44, 0.31, 0.11, 0.03, 0.11]
age_groups = ['18-29', '30-49', '50-64', '65-100', 'underage']

# MBTI ratio
p_mbti = [
    0.12625, 0.11625, 0.02125, 0.03125, 0.05125, 0.07125, 0.04625, 0.04125,
    0.04625, 0.06625, 0.07125, 0.03625, 0.10125, 0.11125, 0.03125, 0.03125
]
mbti_types = [
    "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP", "ESTP",
    "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"
]

# Country ratio
country_ratio = [0.4833, 0.0733, 0.0697, 0.0416, 0.0306, 0.3016]
countries = ["US", "UK", "Canada", "Australia", "Germany", "Other"]

# Profession ratio
p_professions = [1 / 16] * 16
professions = [
    "Agriculture, Food & Natural Resources", "Architecture & Construction",
    "Arts, Audio/Video Technology & Communications",
    "Business Management & Administration", "Education & Training", "Finance",
    "Government & Public Administration", "Health Science",
    "Hospitality & Tourism", "Human Services", "Information Technology",
    "Law, Public Safety, Corrections & Security", "Manufacturing", "Marketing",
    "Science, Technology, Engineering & Mathematics",
    "Transportation, Distribution & Logistics"
]

def get_random_gender():
    return random.choices(genders, gender_ratio)[0]

def get_random_age():
    group = random.choices(age_groups, age_ratio)[0]
    if group == 'underage':
        return random.randint(10, 17)
    elif group == '18-29':
        return random.randint(18, 29)
    elif group == '30-49':
        return random.randint(30, 49)
    elif group == '50-64':
        return random.randint(50, 64)
    else:
        return random.randint(65, 100)

def get_random_mbti():
    return random.choices(mbti_types, p_mbti)[0]

def get_random_country():
    country = random.choices(countries, country_ratio)[0]
    if country == "Other":
        response = vllm_completion([{
            "role": "system",
            "content": "Select a real country name randomly:"
        }])
        return response["choices"][0]["message"]["content"].strip()
    return country

def get_random_profession():
    return random.choices(professions, p_professions)[0]

def get_interested_topics(mbti, age, gender, country, profession):
    prompt = f"""Return ONLY a JSON array of 2-3 numbers representing chosen topics.
Input: {{"mbti": "{mbti}", "age": {age}, "gender": "{gender}", "country": "{country}", "profession": "{profession}"}}
Topics: [1: Economics, 2: IT, 3: Culture & Society, 4: General News, 5: Politics, 6: Business, 7: Fun]
Example output: [1,3,6]"""

    response = vllm_completion([{
        "role": "user",
        "content": prompt
    }])

    content = response["choices"][0]["message"]["content"].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract array from the content
        import re
        array_match = re.search(r'\[(.*?)\]', content)
        if array_match:
            array_str = array_match.group(0)
            return json.loads(array_str)
        raise

def generate_user_profile(age, gender, mbti, profession, topics):
    prompt = f"""Return ONLY a JSON object for a user profile.
Input: {{"age": {age}, "gender": "{gender}", "mbti": "{mbti}", "profession": "{profession}", "topics": {json.dumps(topics)}}}
Required format:
{{
    "realname": "string",
    "username": "string",
    "bio": "string",
    "persona": "string"
}}"""

    response = vllm_completion([{
        "role": "user",
        "content": prompt
    }])

    content = response["choices"][0]["message"]["content"].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract object from the content
        import re
        object_match = re.search(r'\{.*\}', content, re.DOTALL)
        if object_match:
            object_str = object_match.group(0)
            return json.loads(object_str)
        raise

def index_to_topics(index_lst):
    topic_dict = {
        '1': 'Economics',
        '2': 'Information Technology',
        '3': 'Culture & Society',
        '4': 'General News',
        '5': 'Politics',
        '6': 'Business',
        '7': 'Fun'
    }
    result = []
    for index in index_lst:
        topic = topic_dict[str(index)]
        result.append(topic)
    return result

def create_user_profile():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            gender = get_random_gender()
            age = get_random_age()
            mbti = get_random_mbti()
            country = get_random_country()
            profession = get_random_profession()
            topic_index_lst = get_interested_topics(mbti, age, gender, country, profession)
            topics = index_to_topics(topic_index_lst)
            profile = generate_user_profile(age, gender, mbti, profession, topics)
            profile['age'] = age
            profile['gender'] = gender
            profile['mbti'] = mbti
            profile['country'] = country
            profile['profession'] = profession
            profile['interested_topics'] = topics
            return profile
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Profile generation failed after {max_retries} attempts: {e}")
                raise
            print(f"Profile generation attempt {attempt + 1} failed: {e}. Retrying...")

def generate_user_data(n, max_workers=None):
    """Generate n user profiles using thread pool"""
    if max_workers is None:
        max_workers = min(32, n)  # Default to min of 32 or n workers
        
    user_data = []
    start_time = datetime.now()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(create_user_profile) for _ in range(n)]
        for i, future in enumerate(as_completed(futures)):
            try:
                profile = future.result()
                user_data.append(profile)
                elapsed_time = datetime.now() - start_time
                print(f"Generated {i+1}/{n} user profiles. Time elapsed: {elapsed_time}")
            except Exception as e:
                print(f"Failed to generate profile {i+1}: {e}")
    
    return user_data

def save_user_data(user_data, filename):
    with open(filename, 'w') as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate user profiles using vLLM')
    parser.add_argument('--num_users', type=int, default=36,
                      help='Number of user profiles to generate')
    parser.add_argument('--output', type=str, default='data/reddit/user_data_vllm.json',
                      help='Output file path')
    parser.add_argument('--max_workers', type=int, default=None,
                      help='Maximum number of worker threads')
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_users} user profiles...")
    user_data = generate_user_data(args.num_users, args.max_workers)
    save_user_data(user_data, args.output)
    print(f"Generated {len(user_data)} user profiles and saved to {args.output}") 