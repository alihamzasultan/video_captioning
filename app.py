import cv2
import whisper
import subprocess
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import textwrap
import streamlit as st
import tempfile
import shutil
import json
import os

# Function to load the visitor count from a JSON file
def load_visitor_count(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file).get("visitor_count", 0)
    return 0

# Function to save the visitor count to a JSON file
def save_visitor_count(file_path, count):
    with open(file_path, "w") as file:
        json.dump({"visitor_count": count}, file)

# File path for the visitor count
count_file_path = "visitor_count.json"

# Load the current visitor count
if 'visitor_count' not in st.session_state:
    st.session_state.visitor_count = load_visitor_count(count_file_path)

# Increment visitor count
st.session_state.visitor_count += 1

# Save the updated visitor count
save_visitor_count(count_file_path, st.session_state.visitor_count)

# Display visitor count
st.sidebar.markdown(f"**Total Visitor Count:** {st.session_state.visitor_count}")

# Your existing code...



# Function to map position string to vertical positioning
def get_text_y_position(position, text_height, height):
    if position == "top":
        return text_height + 50
    elif position == "center":
        return (height - text_height) // 2
    elif position == "bottom":
        return (height + text_height) // 2

# Initialize Whisper model
model = whisper.load_model("base")

# Streamlit app layout
st.title("Video Captioning Tool")



# User choices
choice = st.selectbox("Choose Number of words per frame:", [1, 0], format_func=lambda x: "One word" if x == 1 else "Sentences in each frame")
position = st.selectbox("Choose text position:", ["top", "center", "bottom"])
text_color = st.color_picker("Choose text color:", "#FFFFFF")
highlight_color = st.color_picker("Choose highlight color:", "#48FF00")  # Added highlight color picker


# Load fonts
fonts_dir = "fonts"
if not os.path.exists(fonts_dir):
    st.error(f"Fonts directory '{fonts_dir}' does not exist.")
else:
    font_files = [f for f in os.listdir(fonts_dir) if f.endswith('.ttf')]
    if not font_files:
        st.error("No .ttf font files found in the fonts directory.")
    else:
        font_style = st.selectbox("Choose font style:", [f[:-4] for f in font_files])

# Preview text box
if 'font_style' in locals():
    sample_text = "Selected Font"
    font_size = 45  # Adjust as necessary
    font_path = os.path.join(fonts_dir, f"{font_style}.ttf")
    font = ImageFont.truetype(font_path, font_size)

    # Create an image with sample text
    preview_image = Image.new("RGB", (800, 100), "black")
    draw = ImageDraw.Draw(preview_image)
    text_color_rgb = tuple(int(text_color[i:i + 2], 16) for i in (1, 3, 5))
    draw.text((10, 10), sample_text, font=font, fill=text_color_rgb)

    # Display the image
    st.image(preview_image, caption=f"Font: {font_style}")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4"])

if uploaded_video and 'font_style' in locals():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(uploaded_video.getbuffer())
        video_path = temp_video_file.name

    if st.button("Start Processing"):
        output_dir = tempfile.mkdtemp()
        output_path = os.path.join(output_dir, "captioned_video.mp4")
        audio_path = os.path.join(output_dir, "temp_audio.wav")

        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Unable to open video file.")
        else:
            # Get video properties
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Extract audio
            st.info("Extracting audio...")
            subprocess.run([
                'ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path
            ], check=True)

            # Transcribe audio with Whisper
            st.info("Transcribing audio...")
            result = model.transcribe(audio_path)

            # Check if the detected language is English
            if result['language'] != 'en':
                st.error("The detected language is not English. Terminating the process.")
                os.remove(audio_path)
                cap.release()
                out.release()
            else:
                custom_fonts_dir = "fonts"  # Your fonts folder

                # Initialize stop flag
                if 'stop_processing' not in st.session_state:
                    st.session_state.stop_processing = False

                # Function to add typing effect text to each frame using Pillow
                def add_typing_effect(frame, text, current_time, start_time, duration):
                    if choice == 1:
                        font_size = 30  # Larger font size for one word at a time
                        border_width = 3  # Shadow border width for one word at a time
                    else:
                        font_size = 15  # Smaller font size for whole sentences
                        border_width = 0  # Reduced shadow border width for sentences

                    font_path = os.path.join(custom_fonts_dir, f"{font_style}.ttf")  # Use font from the fonts folder
                    font = ImageFont.truetype(font_path, font_size)

                    if choice == 1:
                        chunks = text.split()
                    else:
                        words = text.split()
                        chunks = [' '.join(words[i:i + 4]) for i in range(0, len(words), 4)]  # Split into chunks of 3-4 words

                    text_color_rgb = tuple(int(text_color[i:i + 2], 16) for i in (1, 3, 5))

                    total_chunks = len(chunks)
                    typing_duration = duration / total_chunks if total_chunks > 0 else duration

                    elapsed_time = current_time - start_time
                    chunk_index = min(max(0, int(elapsed_time / typing_duration)), total_chunks - 1)
                    visible_text = chunks[chunk_index] if total_chunks > 0 else ""

                    wrapped_text = textwrap.fill(visible_text, width=40)

                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)

                    text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    text_x = (width - text_width) // 2
                    text_y = get_text_y_position(position, text_height, height)

                    for dx in range(-border_width, border_width + 1):
                        for dy in range(-border_width, border_width + 1):
                            if dx != 0 or dy != 0:
                                draw.text((text_x + dx, text_y + dy), wrapped_text, font=font, fill=(0, 0, 0))  # Shadow color

                    # Draw each word with possible highlight on the current word
                    words_in_chunk = visible_text.split()
                    word_elapsed_time = elapsed_time % typing_duration
                    word_duration = typing_duration / len(words_in_chunk) if words_in_chunk else typing_duration
                    word_index = min(max(0, int(word_elapsed_time / word_duration)), len(words_in_chunk) - 1)

                    current_x = text_x
                    for i, word in enumerate(words_in_chunk):
                        word_color = highlight_color if i == word_index else text_color_rgb
                        draw.text((current_x, text_y), word, font=font, fill=word_color)
                        word_bbox = draw.textbbox((current_x, text_y), word, font=font)
                        current_x += word_bbox[2] - word_bbox[0] + font_size // 2  # Adding space between words

                    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    return frame

                sentences = []
                for segment in result['segments']:
                    sentences.append((segment['start'], segment['end'], segment['text']))

                frame_number = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                progress_bar = st.progress(0)
                progress_text = st.empty()

                stop_button = st.button("Stop Processing")  # Added stop button
                if stop_button:
                    st.session_state.stop_processing = True

                for frame_number in tqdm(range(total_frames), desc="Processing frames"):
                    if st.session_state.stop_processing:
                        st.warning("Processing stopped by user.")
                        break
                    
                    ret, frame = cap.read()
                    if not ret:
                        break

                    current_time = frame_number / fps

                    for start_time, end_time, sentence in sentences:
                        if start_time <= current_time <= end_time:
                            frame = add_typing_effect(frame, sentence, current_time, start_time, end_time - start_time)
                            break

                    out.write(frame)
                    frame_number += 1
                    progress_bar.progress(frame_number / total_frames)
                    progress_text.text(f"Processing frame {frame_number}/{total_frames}")

                cap.release()
                out.release()

                if not st.session_state.stop_processing:
                    st.info("Finalizing the video, please wait...")

                    final_output_path = os.path.join(output_dir, "final_output.mp4")
                    subprocess.run([
                        'ffmpeg', '-i', output_path, '-i', video_path, '-c:v', 'libx264', '-crf', '23', '-preset', 'slow', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', final_output_path
                    ], check=True)

                    os.remove(audio_path)
                    os.remove(output_path)

                    st.success("Video processing complete!")

                    with open(final_output_path, "rb") as file:
                        btn = st.download_button(
                            label="Download Processed Video",
                            data=file,
                            file_name="final_output.mp4",
                            mime="video/mp4"
                        )

                shutil.rmtree(output_dir)

                # Reset stop flag after processing
                st.session_state.stop_processing = False

                

# Custom donation button styled using markdown
if st.markdown(
    """
    <a href="https://bymecoffee.vercel.app/" target="_blank" style="display: inline-block; 
    padding: 10px 20px; color: white; background-color: #FF7F50; border-radius: 5px; 
    text-align: center; text-decoration: none; font-size: 16px;">Buy Me a Coffee</a>
    """,
    unsafe_allow_html=True
):
    pass

# Display images demonstrating how the app works
st.header("100% Free Ai Tool")

# Path to your images directory
images_dir = "images"

# Ensure the directory exists
if os.path.exists(images_dir):
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if image_files:
        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            st.image(image_path, caption=image_file, use_column_width=True)
    else:
        st.warning("No image files found in the 'images' directory.")
else:
    st.error(f"Images directory '{images_dir}' does not exist.")
