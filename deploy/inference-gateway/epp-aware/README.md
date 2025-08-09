# Installing Inference Gateway with Dynamo (Experimental)

This guide provides instructions for setting up the Inference Gateway with Dynamo EPP for managing and routing inference requests.

## Prerequisites

- Kubernetes cluster with kubectl configured
- NVIDIA GPU drivers installed on worker nodes

## Installation Steps


1. **Deploy Inference Gateway**
Follow the [Inference Gateway README](../README.md) .
Finish the steps right before [Install dynamo model and dynamo gaie helm chart](../README.md#install-dynamo-model-and-dynamo-gaie-helm-chart)


3. **Apply Dynamo-specific manifests**

The Inference Gateway is configured.
Deploy DynamoGraph. We provide the example in the deployment.yaml which you can adapt.

```bash
cd deploy/inference-gateway/epp-aware
kubectl apply -f deployment.yaml
```

Deploy the Inference Gateway resources to your Kubernetes cluster:

```bash
cd deploy/inference-gateway/epp-aware
kubectl apply -f resources
```

Key configurations include:
- An InferenceModel resource for the Qwen model
- A service for the inference gateway
- Required RBAC roles and bindings
- RBAC permissions

5. **Verify Installation**

Check that all resources are properly deployed:

```bash
kubectl get inferencepool
kubectl get inferencemodel
kubectl get httproute
```

Sample output:

```bash
# kubectl get inferencepool
NAME              AGE
dynamo-qwen       6s

# kubectl get inferencemodel
NAME              MODEL NAME                                 INFERENCE POOL    CRITICALITY   AGE
qwen-model        Qwen/Qwen3-0.6B                            dynamo-qwen   Critical      6s

# kubectl get httproute
NAME        HOSTNAMES   AGE
llm-route               6s
```

## Usage

The Inference Gateway provides HTTP/2 endpoints for model inference. The default service is exposed on port 9002.

### 1: Populate gateway URL for your k8s cluster
```bash
export GATEWAY_URL=<Gateway-URL>
```

To test the gateway in minikube, use the following command:
```bash
minikube tunnel &

GATEWAY_URL=$(kubectl get svc inference-gateway -o yaml -o jsonpath='{.spec.clusterIP}')
echo $GATEWAY_URL
```

### 2: Check models deployed to inference gateway

Query models:
```bash
curl $GATEWAY_URL/v1/models | jq .
```

Send inference request to gateway:

```bash
MODEL_NAME="Qwen/Qwen3-0.6B"
curl $GATEWAY_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL_NAME}"'",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream":false,
    "max_tokens": 30
  }'
```