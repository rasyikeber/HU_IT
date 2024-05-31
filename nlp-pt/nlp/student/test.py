import logging
from sentence_transformers import SentenceTransformer, util
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the path to the fine-tuned model
output_model_path = 'stud\fine_tuned_model'

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
project_description1 = "Embark on a transformative journey in plant care with an innovative automated watering system that seamlessly integrates IoT technology. This groundbreaking project addresses the age-old challenge of maintaining optimal hydration levels for plants by harnessing the power of sensors, microcontrollers, and connectivity. At its core, the system features precision-engineered sensors strategically placed within plant pots to continuously monitor soil moisture levels with unparalleled accuracy. These sensors relay real-time data to a central microcontroller, which acts as the brain of the operation, orchestrating watering schedules and adjusting water delivery based on plant-specific needs and environmental conditions. What sets this system apart is its IoT connectivity, which enables users to remotely monitor and control their plants' hydration status from anywhere in the world. Imagine receiving notifications on your smartphone app when your plants need watering or accessing historical data and insights to optimize watering strategies over time. Moreover, the system could incorporate smart features like adaptive watering algorithms, automated fertilization, and integration with weather forecasts to further enhance plant health and vitality. By combining cutting-edge hardware design, software development, and IoT integration, this project offers a holistic solution for plant care automation, empowering users to nurture thriving green spaces effortlessly."
project_description2 = "Create a sophisticated recipe recommendation system that revolutionizes meal planning and culinary exploration. This system aims to cater to the diverse tastes, dietary preferences, and cooking abilities of users by leveraging advanced machine learning algorithms and natural language processing techniques. Users can interact with the system through a sleek and intuitive interface, where they input their dietary restrictions, preferred cuisines, cooking time constraints, and available ingredients. Behind the scenes, the system processes this input and sifts through a vast database of recipes, analyzing factors such as ingredient compatibility, flavor profiles, and nutritional content to generate personalized recipe suggestions. It doesn't stop there; the system could also take into account user feedback and past interactions to continuously refine its recommendations, ensuring a delightful and tailored experience for each user. Additionally, the system could offer features such as meal planning assistance, recipe variations, and cooking tips to enhance user engagement and satisfaction. With its ability to inspire creativity in the kitchen while accommodating individual preferences, this project promises to be a game-changer in the realm of culinary technology."
    
# Check similarity between the two project descriptions
similarity_score = check_similarity(project_description1, project_description2)

# Print the results
# print(f"Project Description 1: {project_description1}")
# print(f"Project Description 2: {project_description2}")
print(f"Similarity Score: {similarity_score}")



   
