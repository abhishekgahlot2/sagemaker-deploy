import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

def test_endpoint():
    # Get endpoint from environment or endpoint_info.json
    endpoint_name = os.getenv('ENDPOINT_NAME')
    if not endpoint_name:
        try:
            with open('endpoint_info.json', 'r') as f:
                info = json.load(f)
                endpoint_name = info['endpoint_name']
        except FileNotFoundError:
            endpoint_name = input("Enter endpoint name: ")
    
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    # Initialize SageMaker runtime with extended timeouts for cold starts
    from botocore.config import Config
    config = Config(
        read_timeout=600,
        connect_timeout=60,
        retries={'max_attempts': 0}
    )
    runtime = boto3.client('sagemaker-runtime', region_name=region, config=config)
    
    print(f"Testing endpoint: {endpoint_name}")
    print("=" * 50)
    
    # Test with minimal token generation to reduce memory usage
    test_cases = [
        {
            "name": "Simple SQL Query",
            "payload": {
                "inputs": "Generate a SQL query to select all customers:",
                "parameters": {
                    "max_new_tokens": 50,  # Reduced for memory
                    "temperature": 0.7,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
        },
        {
            "name": "Complex SQL Query", 
            "payload": {
                "inputs": "Create a SQL query to find customers who made orders in the last 30 days:",
                "parameters": {
                    "max_new_tokens": 75,  # Reduced for memory
                    "temperature": 0.5,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"Input: {test_case['payload']['inputs']}")
        
        try:
            print("Calling endpoint (may take 2-5 minutes for first request)...")
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(test_case['payload'])
            )
            
            result = json.loads(response['Body'].read().decode())
            print(f"Success!")
            print(f"Response: {result}")
            
        except Exception as e:
            print(f"Failed: {str(e)}")
    
    print(f"\nRemember to run 'python cleanup.py' to delete endpoint when done")

if __name__ == "__main__":
    test_endpoint()