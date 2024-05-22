import cv2
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# Load pre-trained models for video feature extraction and caption generation
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

def add_text_to_frame(frame, text, font_path="./Lcd.ttf"):
    # Convert frame (numpy array) to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    
    # Use a commonly available font
    font = ImageFont.truetype(font_path, 24)
    
    # Define text position and background
    text_width, text_height = draw.textsize(text, font=font)
    text_position = (img.width // 2 - text_width // 2, img.height - text_height - 10)
    background_position = (text_position[0] - 10, text_position[1] - 5, text_position[0] + text_width + 10, text_position[1] + text_height + 5)
    
    # Draw background rectangle for text
    draw.rectangle(background_position, fill="black")
    draw.text(text_position, text, font=font, fill="white")
    
    # Convert PIL Image back to numpy array
    frame_with_text = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame_with_text

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

    return output_path

def add_captions_to_video(video_path, output_path, caption):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_with_text = add_text_to_frame(frame, caption)
        out.write(frame_with_text)

    cap.release()
    out.release()
    print(f"Captioned video saved to {output_path}")

# Provide the path to your video file
video_path = "./1108862_1080p_high-five_standard_1920x1080.mp4"
user_text = "basket ball"
output_folder = "./clips"

# Extract the activity clip based on user input
clip_path = extract_activity_clip(video_path, user_text, output_folder)

# Generate caption for the extracted clip
caption = generate_captions(clip_path)[0][1]  # Get the caption for the first frame

# Define the output path for the captioned video
output_captioned_path = os.path.join(output_folder, 'captioned_clip.mp4')

# Add captions to the extracted video clip
add_captions_to_video(clip_path, output_captioned_path, caption)
