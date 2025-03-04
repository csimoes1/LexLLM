import sagemaker
from sagemaker.jumpstart.estimator import JumpStartEstimator

session = sagemaker.Session()

# Define the model ID for LLaMA 3.1 8B base
model_id = "meta-textgeneration-llama-3-2-3b" #"meta-llama/Meta-Llama-3.1-8B"
role = "arn:aws:iam::203676937706:role/AmazonSageMakerFullAccessForSimoes"

# Initialize the estimator
estimator = JumpStartEstimator(
    model_id=model_id,
    role=role,
    instance_type="ml.g5.12xlarge",
    environment={"accept_eula": "true"}
)

# Set hyperparameters for SFT
estimator.set_hyperparameters(
    chat_dataset="True",
    # instruction_tuned="True",
    epoch="3",
    learning_rate="0.0002",
    max_input_length="1024",
)

# Start training with your S3 dataset
estimator.fit({"training": "s3://sagemaker-us-east-1-203676937706/trainingrun-6/training/",
               "validation": "s3://sagemaker-us-east-1-203676937706/trainingrun-6/validation/"})
