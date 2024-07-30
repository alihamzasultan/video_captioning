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

# Function to map position string to vertical positioning
def get_text_y_position(position, text_height, height):
    if position == "top":
        return text_height + 20
    elif position == "below top":
        return text_height + 50
    elif position == "center":
        return (height - text_height) // 2
    elif position == "below center":
        return (height + text_height) // 2
    elif position == "bottom":
        return height - text_height - 20

# Initialize Whisper model
model = whisper.load_model("base")

# Streamlit app layout
st.title("Video Captioning Tool")

# User choices
choice = st.selectbox("Choose Number of words per frame:", [1, 0], format_func=lambda x: "One word" if x == 1 else "Sentences in each frame")
position = st.selectbox("Choose text position:", ["top", "below top", "center", "below center", "bottom"])
text_color = st.color_picker("Choose text color:", "#FFFFFF")

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

uploaded_video = st.file_uploader("Upload a video file", type=["mp4"])

if uploaded_video:
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

                # Function to add typing effect text to each frame using Pillow
                def add_typing_effect(frame, text, current_time, start_time, duration):
                    if choice == 1:
                        font_size = 60  # Larger font size for one word at a time
                    else:
                        font_size = 20  # Smaller font size for whole sentences

                    font_path = os.path.join(custom_fonts_dir, f"{font_style}.ttf")  # Use font from the fonts folder
                    font = ImageFont.truetype(font_path, font_size)

                    if choice == 1:
                        chunks = text.split()
                    else:
                        words = text.split()
                        chunks = [' '.join(words[i:i + 6]) for i in range(0, len(words), 6)]

                    text_color_rgb = tuple(int(text_color[i:i + 2], 16) for i in (1, 3, 5))
                    shadow_color = (0, 0, 0)  # Black for shadow

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

                    border_width = 3  # Increase this value for a thicker border
                    for dx in range(-border_width, border_width + 1):
                        for dy in range(-border_width, border_width + 1):
                            if dx != 0 or dy != 0:
                                draw.text((text_x + dx, text_y + dy), wrapped_text, font=font, fill=shadow_color)

                    draw.text((text_x, text_y), wrapped_text, font=font, fill=text_color_rgb)

                    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    return frame

                sentences = []
                for segment in result['segments']:
                    sentences.append((segment['start'], segment['end'], segment['text']))

                frame_number = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                progress_bar = st.progress(0)
                progress_text = st.empty()

                for frame_number in tqdm(range(total_frames), desc="Processing frames"):
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
