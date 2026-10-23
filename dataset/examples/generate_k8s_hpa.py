#!/usr/bin/env python3
"""
Example: Generating Kubernetes HPA configurations

This script demonstrates how to generate Kubernetes Horizontal Pod Autoscaler
configurations based on workload characteristics.
"""

import json
import yaml
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_workload(workload_id: str, dataset_path: str = '../workloads') -> dict:
    """Load a workload by ID"""
    filepath = Path(dataset_path) / f'{workload_id}.json'
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_hpa_config(workload: Dict[str, Any], 
                       deployment_name: str = None,
                       namespace: str = 'default',
                       min_replicas: int = None,
                       max_replicas: int = None) -> Dict[str, Any]:
    """Generate HPA configuration based on workload characteristics"""
    
    # Use workload ID as deployment name if not provided
    if deployment_name is None:
        deployment_name = f"app-{workload['workload_id']}"
    
    # Determine scaling parameters based on archetype
    archetype = workload['archetype']
    
    # Default configurations by archetype
    archetype_configs = {
        'spike': {
            'min_replicas': 2,  # Keep warm instances
            'max_replicas': 100,
            'target_cpu': 30,   # Low threshold for quick response
            'scale_down_stabilization': 300,  # 5 minutes
            'scale_up_stabilization': 0      # No delay for scale up
        },
        'periodic': {
            'min_replicas': 1,
            'max_replicas': 50,
            'target_cpu': 75,   # Higher threshold for efficiency
            'scale_down_stabilization': 120,  # 2 minutes
            'scale_up_stabilization': 30      # 30 seconds
        },
        'ramp': {
            'min_replicas': 1,
            'max_replicas': 75,
            'target_cpu': 60,   # Moderate threshold
            'scale_down_stabilization': 180,  # 3 minutes
            'scale_up_stabilization': 60      # 1 minute
        },
        'stationary_noisy': {
            'min_replicas': 1,
            'max_replicas': 30,
            'target_cpu': 50,   # Moderate threshold
            'scale_down_stabilization': 600,  # 10 minutes - avoid thrashing
            'scale_up_stabilization': 120     # 2 minutes
        }
    }
    
    config = archetype_configs.get(archetype, archetype_configs['stationary_noisy'])
    
    # Override with provided values
    if min_replicas is not None:
        config['min_replicas'] = min_replicas
    if max_replicas is not None:
        config['max_replicas'] = max_replicas
    
    # Calculate max replicas based on p99 resource requirements if not provided
    if max_replicas is None:
        cpu_p99 = workload['resource_requirements']['cpu_millicores']['p99']
        # Assume each pod can handle 1000 millicores
        config['max_replicas'] = max(config['min_replicas'] + 1, min(100, cpu_p99 // 1000 + 2))
    
    # Generate HPA v2 configuration
    hpa = {
        'apiVersion': 'autoscaling/v2',
        'kind': 'HorizontalPodAutoscaler',
        'metadata': {
            'name': f"{deployment_name}-hpa",
            'namespace': namespace,
            'labels': {
                'app': deployment_name,
                'archetype': archetype
            }
        },
        'spec': {
            'scaleTargetRef': {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'name': deployment_name
            },
            'minReplicas': config['min_replicas'],
            'maxReplicas': config['max_replicas'],
            'metrics': [
                {
                    'type': 'Resource',
                    'resource': {
                        'name': 'cpu',
                        'target': {
                            'type': 'Utilization',
                            'averageUtilization': config['target_cpu']
                        }
                    }
                },
                {
                    'type': 'Resource',
                    'resource': {
                        'name': 'memory',
                        'target': {
                            'type': 'Utilization',
                            'averageUtilization': 80
                        }
                    }
                }
            ],
            'behavior': {
                'scaleDown': {
                    'stabilizationWindowSeconds': config['scale_down_stabilization'],
                    'policies': [
                        {
                            'type': 'Percent',
                            'value': 10,
                            'periodSeconds': 60
                        },
                        {
                            'type': 'Pods',
                            'value': 2,
                            'periodSeconds': 60
                        }
                    ],
                    'selectPolicy': 'Min'
                },
                'scaleUp': {
                    'stabilizationWindowSeconds': config['scale_up_stabilization'],
                    'policies': [
                        {
                            'type': 'Percent',
                            'value': 100,
                            'periodSeconds': 15
                        },
                        {
                            'type': 'Pods',
                            'value': 4,
                            'periodSeconds': 15
                        }
                    ],
                    'selectPolicy': 'Max'
                }
            }
        }
    }
    
    return hpa


def generate_deployment_config(workload: Dict[str, Any],
                             deployment_name: str = None,
                             namespace: str = 'default',
                             image: str = 'nginx:latest') -> Dict[str, Any]:
    """Generate deployment configuration with resource requirements"""
    
    if deployment_name is None:
        deployment_name = f"app-{workload['workload_id']}"
    
    # Get resource requirements
    cpu_req = workload['resource_requirements']['cpu_millicores']
    mem_req = workload['resource_requirements']['memory_mb']
    
    deployment = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': deployment_name,
            'namespace': namespace,
            'labels': {
                'app': deployment_name,
                'archetype': workload['archetype']
            }
        },
        'spec': {
            'replicas': 2 if workload['archetype'] == 'spike' else 1,
            'selector': {
                'matchLabels': {
                    'app': deployment_name
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': deployment_name
                    }
                },
                'spec': {
                    'containers': [
                        {
                            'name': 'app',
                            'image': image,
                            'resources': {
                                'requests': {
                                    'cpu': f"{cpu_req['p50']}m",
                                    'memory': f"{mem_req['p50']}Mi"
                                },
                                'limits': {
                                    'cpu': f"{cpu_req['p99']}m",
                                    'memory': f"{mem_req['p99']}Mi"
                                }
                            }
                        }
                    ]
                }
            }
        }
    }
    
    return deployment


def save_yaml(config: Dict[str, Any], filename: str):
    """Save configuration to YAML file"""
    with open(filename, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved configuration to {filename}")


def main():
    """Example usage"""
    print("Generating Kubernetes configurations from workload data...\n")
    
    # Example 1: Generate HPA for spike workload
    print("1. Generating HPA for spike workload...")
    try:
        workload = load_workload('w_0001')
        hpa = generate_hpa_config(workload)
        
        print(f"   Archetype: {workload['archetype']}")
        print(f"   Min replicas: {hpa['spec']['minReplicas']}")
        print(f"   Max replicas: {hpa['spec']['maxReplicas']}")
        print(f"   Target CPU: {hpa['spec']['metrics'][0]['resource']['target']['averageUtilization']}%")
        
        save_yaml(hpa, 'spike-hpa.yaml')
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 2: Generate HPA for periodic workload
    print("\n2. Generating HPA for periodic workload...")
    try:
        workload = load_workload('w_0002')
        hpa = generate_hpa_config(workload, deployment_name='periodic-app')
        
        print(f"   Archetype: {workload['archetype']}")
        print(f"   Min replicas: {hpa['spec']['minReplicas']}")
        print(f"   Max replicas: {hpa['spec']['maxReplicas']}")
        print(f"   Target CPU: {hpa['spec']['metrics'][0]['resource']['target']['averageUtilization']}%")
        
        save_yaml(hpa, 'periodic-hpa.yaml')
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 3: Generate deployment + HPA
    print("\n3. Generating deployment and HPA configurations...")
    try:
        workload = load_workload('w_0001')
        
        deployment = generate_deployment_config(workload)
        hpa = generate_hpa_config(workload)
        
        # Save as combined manifest
        combined = f"---\n{yaml.dump(deployment, default_flow_style=False, sort_keys=False)}"
        combined += f"---\n{yaml.dump(hpa, default_flow_style=False, sort_keys=False)}"
        
        with open('app-manifest.yaml', 'w') as f:
            f.write(combined)
        
        print("   Saved combined deployment and HPA to app-manifest.yaml")
        print(f"   CPU requests: {deployment['spec']['template']['spec']['containers'][0]['resources']['requests']['cpu']}")
        print(f"   CPU limits: {deployment['spec']['template']['spec']['containers'][0]['resources']['limits']['cpu']}")
        print(f"   Memory requests: {deployment['spec']['template']['spec']['containers'][0]['resources']['requests']['memory']}")
        print(f"   Memory limits: {deployment['spec']['template']['spec']['containers'][0]['resources']['limits']['memory']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\nConfiguration generation complete!")
    print("\nTo apply these configurations:")
    print("  kubectl apply -f spike-hpa.yaml")
    print("  kubectl apply -f app-manifest.yaml")


if __name__ == '__main__':
    main()