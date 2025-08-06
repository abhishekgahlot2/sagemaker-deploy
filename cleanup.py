import boto3
import sagemaker
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class SageMakerCleanup:
    def __init__(self, region=None):
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')
        self.session = sagemaker.Session(boto3.Session(region_name=self.region))
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.region)
    
    def list_all_endpoints(self):
        try:
            response = self.sagemaker_client.list_endpoints(MaxResults=50)
            endpoints = response.get('Endpoints', [])
            
            if endpoints:
                print("All endpoints:")
                for endpoint in endpoints:
                    status = "Active" if endpoint['EndpointStatus'] == 'InService' else "Inactive"
                    print(f"  - {endpoint['EndpointName']} ({status})")
                    print(f"    Created: {endpoint['CreationTime']}")
            else:
                print("No endpoints found.")
            
            return endpoints
        except Exception as e:
            print(f"Failed to list endpoints: {str(e)}")
            return []
    
    def delete_endpoint(self, endpoint_name, delete_config=True):
        try:
            print(f"Deleting endpoint: {endpoint_name}")
            
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            print(f"Endpoint {endpoint_name} deletion initiated")
            
            if delete_config:
                try:
                    self.sagemaker_client.delete_endpoint_config(
                        EndpointConfigName=endpoint_name
                    )
                    print(f"Endpoint config {endpoint_name} deleted")
                except Exception as e:
                    print(f"Could not delete endpoint config: {str(e)}")
            
            return True
        except Exception as e:
            print(f"Failed to delete endpoint {endpoint_name}: {str(e)}")
            return False
    
    def delete_all_endpoints(self, confirm=True):
        endpoints = self.list_all_endpoints()
        
        if not endpoints:
            print("No endpoints to delete.")
            return True
        
        if confirm:
            print(f"\nThis will delete {len(endpoints)} endpoint(s):")
            for endpoint in endpoints:
                print(f"  - {endpoint['EndpointName']}")
            
            confirmation = input("\nAre you sure? (type 'yes' to confirm): ")
            if confirmation.lower() != 'yes':
                print("Deletion cancelled.")
                return False
        
        success_count = 0
        for endpoint in endpoints:
            if self.delete_endpoint(endpoint['EndpointName']):
                success_count += 1
        
        print(f"\nSuccessfully deleted {success_count}/{len(endpoints)} endpoints")
        return success_count == len(endpoints)
    
    def delete_endpoint_from_file(self, info_file='endpoint_info.json'):
        try:
            with open(info_file, 'r') as f:
                endpoint_info = json.load(f)
            
            endpoint_name = endpoint_info.get('endpoint_name')
            if endpoint_name:
                return self.delete_endpoint(endpoint_name)
            else:
                print(f"No endpoint name found in {info_file}")
                return False
                
        except FileNotFoundError:
            print(f"File {info_file} not found")
            return False
        except Exception as e:
            print(f"Error reading {info_file}: {str(e)}")
            return False
    
    def estimate_costs(self):
        try:
            endpoints = self.list_all_endpoints()
            active_endpoints = [ep for ep in endpoints if ep['EndpointStatus'] == 'InService']
            
            if not active_endpoints:
                print("No active endpoints - no ongoing costs")
                return 0
            
            print("Current cost estimation:")
            print("=" * 40)
            
            # Cost estimates per hour (US East 1 pricing)
            instance_costs = {
                'ml.g4dn.xlarge': 0.736,
                'ml.g4dn.2xlarge': 0.94,
                'ml.g4dn.4xlarge': 1.505,
                'ml.g5.xlarge': 1.408,
                'ml.g5.2xlarge': 1.515,
                'ml.g5.4xlarge': 2.03,
                'ml.g5.8xlarge': 3.06,
                'ml.p3.2xlarge': 3.825,
                'ml.p3.8xlarge': 14.688,
                'ml.p3.16xlarge': 28.152
            }
            
            total_hourly_cost = 0
            for endpoint in active_endpoints:
                try:
                    config_response = self.sagemaker_client.describe_endpoint_config(
                        EndpointConfigName=endpoint['EndpointName']
                    )
                    
                    for variant in config_response['ProductionVariants']:
                        instance_type = variant['InstanceType']
                        instance_count = variant['InitialInstanceCount']
                        hourly_cost = instance_costs.get(instance_type, 2.0) * instance_count
                        total_hourly_cost += hourly_cost
                        
                        print(f"  {endpoint['EndpointName']}:")
                        print(f"    Instance: {instance_type} (x{instance_count})")
                        print(f"    Cost: ${hourly_cost:.3f}/hour")
                        
                except Exception as e:
                    print(f"  {endpoint['EndpointName']}: Unable to get cost info")
            
            daily_cost = total_hourly_cost * 24
            monthly_cost = daily_cost * 30
            
            print(f"\nTotal estimated costs:")
            print(f"  Hourly: ${total_hourly_cost:.2f}")
            print(f"  Daily: ${daily_cost:.2f}")
            print(f"  Monthly: ${monthly_cost:.2f}")
            
            return total_hourly_cost
            
        except Exception as e:
            print(f"Error estimating costs: {str(e)}")
            return 0

def main():
    print("SageMaker Cleanup Utility")
    print("=" * 40)
    
    cleanup = SageMakerCleanup()
    
    while True:
        print("\nOptions:")
        print("1. List all endpoints")
        print("2. Delete specific endpoint")
        print("3. Delete endpoint from info file")
        print("4. Delete ALL endpoints")
        print("5. Estimate current costs")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ")
        
        if choice == '1':
            cleanup.list_all_endpoints()
            
        elif choice == '2':
            endpoint_name = input("Enter endpoint name to delete: ")
            cleanup.delete_endpoint(endpoint_name)
            
        elif choice == '3':
            info_file = input("Enter info file path (default: endpoint_info.json): ") or "endpoint_info.json"
            cleanup.delete_endpoint_from_file(info_file)
            
        elif choice == '4':
            cleanup.delete_all_endpoints()
            
        elif choice == '5':
            cleanup.estimate_costs()
            
        elif choice == '6':
            print("Goodbye!")
            break
            
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()