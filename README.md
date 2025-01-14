# Diabetes Prediction Project

## Overview
This project demonstrates an end-to-end machine learning pipeline for diabetes prediction using Google Cloud Vertex AI. The pipeline is designed to:
1. Train a classifier for diabetes prediction locally.
2. Containerize the model and push it to Google Artifact Registry using Cloud Build.
3. Train and deploy the model using Vertex AI Pipelines.
4. Provide online predictions through a Vertex AI Endpoint.

## Project Architecture
<img width="874" alt="diabetes_pred" src="https://github.com/user-attachments/assets/8e612588-6a5b-47d6-872c-ffd664053ad8" />

The project follows these steps:

1. **Model Training Locally**
   - Develop and train a classifier locally using the diabetes dataset.
   - The dataset includes features such as age, gender, BMI, blood glucose level, HbA1c level, and medical history.

2. **Containerization and Deployment**
   - Containerize the model code along with the dataset and classifier using Docker.
   - Use Cloud Build to push the Docker image to Google Artifact Registry.

3. **Vertex AI Pipelines**
   - Use Vertex AI Pipelines to train the model on Vertex AI.
   - Evaluate the model's performance using a custom component.
   - Deploy the model to a Vertex AI Endpoint if it meets the performance threshold (e.g., ROC AUC > 90%).

4. **Online Predictions**
   - Use the deployed model to provide real-time predictions via the Vertex Endpoint.
   - Input: Patient data (e.g., age, BMI, medical history).
   - Output: Prediction indicating whether the patient is diabetic or non-diabetic.

## Instructions

### 1. Local Model Training
Train a diabetes classifier locally using any ML framework (e.g., scikit-learn, TensorFlow). Ensure the model saves as a `.pkl` or `.h5` file.

### 2. Containerize the Model
Create a Dockerfile to containerize the model code and dependencies. Push the container to Google Artifact Registry using:
```bash
gcloud builds submit --tag gcr.io/[PROJECT_ID]/diabetes-predictor
```

### 3. Compile and Deploy the Pipeline
- Compile the pipeline:
  ```python
  from kfp.v2 import compiler
  compiler.Compiler().compile(
      pipeline_func=pipeline, package_path="diabetes_pipeline.json"
  )
  ```

- Submit the pipeline:
  ```python
  from google.cloud import aiplatform

  ml_pipeline_job = aiplatform.PipelineJob(
      display_name="diabetes-prediction-pipeline",
      template_path="diabetes_pipeline.json",
      pipeline_root="gs://your-pipeline-root",
      parameter_values={"project": "your_project_id", "display_name": "diabetes-prediction-automl"},
      enable_caching=True,
  )
  ml_pipeline_job.submit()
  ```

### 4. Monitor Results
- Monitor the pipeline run in the Vertex AI console.
- Review evaluation metrics and deployment decisions.

### 5. Online Predictions
Query the deployed model using the Vertex AI Endpoint:
```python
from google.cloud import aiplatform

endpoint = aiplatform.Endpoint(endpoint_name="YOUR_ENDPOINT_NAME")
response = endpoint.predict([{
    "age": 45,
    "bmi": 28.5,
    "blood_glucose_level": 150,
    "HbA1c_level": 6.2,
    "gender": "male",
    "smoking_history": "non-smoker",
    "hypertension": 0,
    "heart_disease": 0
}])
print(response.predictions)
```
