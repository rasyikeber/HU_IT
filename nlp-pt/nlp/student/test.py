import logging
from sentence_transformers import SentenceTransformer, util
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the path to the fine-tuned model
output_model_path = 'stud/fine_tuned_model'

# Load the fine-tuned model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(output_model_path, device=device)
print("model loaded successfully!")
# Log a message to confirm the model is loaded
logger.info("Model loaded successfully")



def check_similarity(sentence1, sentence2):
    # Compute embeddings
    embeddings1 = model.encode(sentence1, convert_to_tensor=True)
    embeddings2 = model.encode(sentence2, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    # Convert tensor to float
    similarity_score = cosine_scores.item()
    
    return similarity_score

 # Two different software project idea descriptions
project_description1 = "Build a mobile app that tracks fitness activities and provides health tips."
project_description2 = "Implement an e-commerce platform with user authentication and payment gateway integration."
    
# Check similarity between the two project descriptions
similarity_score = check_similarity(project_description1, project_description2)

# Print the results
print(f"Project Description 1: {project_description1}")
print(f"Project Description 2: {project_description2}")
print(f"Similarity Score: {similarity_score}")



   
