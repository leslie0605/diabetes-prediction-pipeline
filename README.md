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
