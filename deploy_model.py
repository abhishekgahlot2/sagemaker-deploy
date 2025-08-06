import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class SageMakerDeployer:
    def __init__(self, region=None):
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')
        self.session = sagemaker.Session(boto3.Session(region_name=self.region))
        
        try:
            self.role = sagemaker.get_execution_role()
            print(f"Found execution role: {self.role}")
        except ValueError:
            # Running locally - get role from env or IAM
            role_name = os.getenv('SAGEMAKER_ROLE_NAME')
            if role_name:
                try:
                    iam_client = boto3.client('iam', region_name=self.region)
                    role_response = iam_client.get_role(RoleName=role_name)
                    self.role = role_response['Role']['Arn']
                    print(f"Found role ARN: {self.role}")
                except Exception as e:
                    print(f"Error finding role: {e}")
                    self.role = input("Role ARN: ")
            else:
                print("SAGEMAKER_ROLE_NAME not found in .env file")
                self.role = input("Role ARN: ")
    
    def deploy_model(self, model_id=None, instance_type=None, initial_instance_count=1, endpoint_name=None):
        try:
            model_id = model_id or os.getenv('MODEL_ID')
            instance_type = instance_type or os.getenv('INSTANCE_TYPE')
            
            if not endpoint_name:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                endpoint_name = f"better-sql-agent-{timestamp}"
            
            print(f"Deploying model: {model_id}")
            print(f"Instance type: {instance_type}")
            print(f"Endpoint name: {endpoint_name}")
            
            # Create HuggingFace Model
            huggingface_model = HuggingFaceModel(
                model_data=None,
                role=self.role,
                transformers_version="4.49.0",
                pytorch_version="2.6.0",
                py_version="py312",
                model_server_workers=1,
                env={
                    'HF_MODEL_ID': model_id,
                    'HF_TASK': 'text-generation',
                    'SAGEMAKER_MODEL_SERVER_TIMEOUT': os.getenv('MODEL_SERVER_TIMEOUT', '600'),
                    'SAGEMAKER_MODEL_SERVER_WORKERS': '1',
                    'MAX_CONTEXT_LENGTH': os.getenv('MAX_CONTEXT_LENGTH', '4096'),
                    'MAX_NEW_TOKENS': os.getenv('MAX_NEW_TOKENS', '512'),
                    'SM_NUM_GPUS': '1',
                    'TRUST_REMOTE_CODE': 'true',
                    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                    'CUDA_LAUNCH_BLOCKING': '1',
                },
                sagemaker_session=self.session
            )
            
            print("Creating SageMaker endpoint...")
            print("This may take 10-15 minutes...")
            
            # Deploy to endpoint
            predictor = huggingface_model.deploy(
                initial_instance_count=initial_instance_count,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer(),
                wait=True,
                model_data_download_timeout=1200,
                container_startup_health_check_timeout=600
            )
            
            print(f"Model deployed successfully!")
            print(f"Endpoint name: {endpoint_name}")
            print(f"Region: {self.region}")
            
            # Test endpoint
            print("\nTesting the endpoint...")
            try:
                test_payload = {
                    "inputs": "Generate a SQL query to find all customers:",
                    "parameters": {
                        "max_new_tokens": 100,
                        "temperature": 0.7,
                        "do_sample": True,
                        "return_full_text": False
                    }
                }
                
                response = predictor.predict(test_payload)
                print(f"Test successful: {response}")
            except Exception as e:
                print(f"Test failed (endpoint is running): {str(e)}")
                print(f"You can still use the endpoint manually")
            
            return {
                "endpoint_name": endpoint_name,
                "predictor": predictor,
                "status": "success"
            }
            
        except Exception as e:
            print(f"Deployment failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def delete_endpoint(self, endpoint_name):
        try:
            predictor = sagemaker.predictor.Predictor(
                endpoint_name=endpoint_name,
                sagemaker_session=self.session
            )
            predictor.delete_endpoint(delete_endpoint_config=True)
            print(f"Endpoint {endpoint_name} deleted successfully!")
            return True
        except Exception as e:
            print(f"Failed to delete endpoint: {str(e)}")
            return False
    
    def list_endpoints(self):
        try:
            sagemaker_client = boto3.client('sagemaker', region_name=self.region)
            response = sagemaker_client.list_endpoints(
                StatusEquals='InService',
                MaxResults=50
            )
            
            endpoints = response.get('Endpoints', [])
            if endpoints:
                print("Active endpoints:")
                for endpoint in endpoints:
                    print(f"  - {endpoint['EndpointName']} ({endpoint['EndpointStatus']})")
                    print(f"    Created: {endpoint['CreationTime']}")
            else:
                print("No active endpoints found.")
            
            return endpoints
        except Exception as e:
            print(f"Failed to list endpoints: {str(e)}")
            return []

def main():
    print("SageMaker Model Deployment")
    print("=" * 50)
    
    deployer = SageMakerDeployer()
    
    # Deploy the model
    result = deployer.deploy_model()
    
    if result["status"] == "success":
        print(f"\nDeployment completed!")
        print(f"Endpoint: {result['endpoint_name']}")
        print(f"\nRemember to delete the endpoint when done to avoid charges")
        
        # Save endpoint info
        with open('endpoint_info.json', 'w') as f:
            json.dump({
                "endpoint_name": result['endpoint_name'],
                "instance_type": os.getenv('INSTANCE_TYPE'),
                "region": os.getenv('AWS_REGION', 'us-east-1'),
                "deployed_at": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"Endpoint info saved to endpoint_info.json")
    
    return result

if __name__ == "__main__":
    main()