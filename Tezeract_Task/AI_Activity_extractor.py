import cv2
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np
import os

# Load pre-trained models
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def generate_captions(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    captions = []
    frame_interval = fps * 5  # Analyze one frame every 5 seconds
    current_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or current_time > duration:
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_interval == 0:
            inputs = feature_extractor(images=frame, return_tensors="pt")
            pixel_values = inputs.pixel_values
            output_ids = model.generate(pixel_values)
            caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            captions.append((current_time, caption))
        
        current_time += 1 / fps

    cap.release()
    return captions

def extract_activity_clip(video_path, user_text, output_folder="clips"):
    # Generate captions for the video
    captions = generate_captions(video_path)
    
    # Compute embeddings
    user_embedding = sentence_model.encode(user_text, convert_to_tensor=True)
    caption_embeddings = sentence_model.encode([caption[1] for caption in captions], convert_to_tensor=True)
    
    # Find the most similar caption
    similarities = util.pytorch_cos_sim(user_embedding, caption_embeddings)[0]
    best_match_index = torch.argmax(similarities).item()
    best_time = captions[best_match_index][0]
    
    # Define the time window for the clip (e.g., 5 seconds before and after)
    start_time = max(0, best_time - 5)
    end_time = best_time + 5
    
    # Extract the video clip
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path = os.path.join(output_folder, f'clip_{int(best_time)}.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps)
    for _ in range(int((end_time - start_time) * fps)):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Clip saved to {output_path}")

# Example usage
video_path = "./1108862_1080p_high-five_standard_1920x1080.mp4"
user_text = "basket ball"
extract_activity_clip(video_path, user_text)