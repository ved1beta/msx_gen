
# Start with basic image classification
from transformers import ViTImageProcessor, ViTForImageClassification

def simple_emotion_detector(image):
    # 1. Load a pre-trained model
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    
    # 2. Process the image
    inputs = processor(image, return_tensors="pt")
    
    # 3. Get predictions
    outputs = model(**inputs)
    
    return outputs
simple_emotion_detector("new.png")