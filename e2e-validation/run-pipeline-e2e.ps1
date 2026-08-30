# 1. Apply all manifests:
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml

# 2. Once training completes, deploy the serving layer:
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# 3. Verify pods are running and healthy:
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

# 4. Test the prediction endpoint:
# Port-forward for local testing
kubectl port-forward svc/model-serving 8080:80 -n ml-training
# Send a prediction request
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"