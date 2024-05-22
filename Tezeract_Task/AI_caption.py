import cv2
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# Load pre-trained models for video feature extraction and caption generation
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def generate_caption(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Extract features from a frame (for simplicity, using the first frame)
    frame = frames[0]
    inputs = feature_extractor(images=frame, return_tensors="pt")

    # Generate captions
    pixel_values = inputs.pixel_values
    output_ids = model.generate(pixel_values)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    cap.release()
    return caption

def add_text_to_frame(frame, text):
    # Convert frame (numpy array) to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    
    # Use a commonly available font
    font_path = "D:/Lcd.ttf"  # Replace with the path to your font file
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

def add_captions_to_video(video_path, output_path):
    caption = generate_caption(video_path)
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

# Provide the path to your video file
video_path = "./856787-hd_1920_1080_30fps.mp4"

# Define the output path for the captioned video
output_path = './video.mp4'

# Call the function to add captions to the video
add_captions_to_video(video_path, output_path)

print(f"Captioned video saved to {output_path}")
