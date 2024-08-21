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
import os
import streamlit as st



# Function to map position string to vertical positioning
def get_text_y_position(position, text_height, height):
    if position == "top":
        return text_height + 50
    elif position == "center":
        return (height - text_height) // 2
    elif position == "bottom":
        return height - text_height - 50

# Initialize Whisper model
model = whisper.load_model("base")





# Streamlit app layout
st.title("Video Captioning Tool")

choice = st.selectbox("Choose Number of words per frame:", [1, 0], format_func=lambda x: "One word" if x == 1 else "Sentences in each frame")
position = st.selectbox("Choose text position:", ["top", "center", "bottom"])
text_color = st.color_picker("Choose text color:", "#FFFFFF")

highlight_color_rgb = "#ffFFff"  # Hardcoded highlight color



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
    # Font size selection
    font_size_option = st.selectbox("Choose font size:", ["Small", "Medium", "Large"])

    # Map selected font size to actual size
    font_size_map = {
    "Small": 40,
    "Medium": 45,
    "Large": 55
}
    font_size = font_size_map[font_size_option]
    sample_text = "Selected Font"
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

# Show "Start Processing" button only if the payment is successful

    if st.button("Start Processing"):
        st.write("Processing your video...")
        if choice == 1:
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
                            def add_typing_effect(frame, words, current_time):
                                font_size = 30  # Adjust font size as needed
                                border_width = 3  # Shadow border width

                                font_path = os.path.join(custom_fonts_dir, f"{font_style}.ttf")  # Use selected font
                                font = ImageFont.truetype(font_path, font_size)

                                text_color_rgb = tuple(int(text_color[i:i + 2], 16) for i in (1, 3, 5))

                                visible_text = ''
                                for word, start_time, end_time in words:
                                    if start_time <= current_time <= end_time:
                                        visible_text += f"{word} "
                                
                                wrapped_text = textwrap.fill(visible_text.strip(), width=40)
                                
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

                                draw.text((text_x, text_y), wrapped_text, font=font, fill=text_color_rgb)

                                frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                                return frame

                            # Modified transcription process to include word-level timestamps
                            st.info("Transcribing audio with word-level timestamps...")
                            result = model.transcribe(audio_path, word_timestamps=True)
                            progress_bar = st.progress(0)
                            progress_text = st.empty()
                            # Extract words with their corresponding timestamps
                            words_with_timestamps = []
                            for segment in result['segments']:
                                for word in segment['words']:
                                    words_with_timestamps.append((word['word'], word['start'], word['end']))

                            # Processing the video frames
                            frame_number = 0
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                            for frame_number in tqdm(range(total_frames), desc="Processing frames"):
                                if st.session_state.stop_processing:
                                    st.warning("Processing stopped by user.")
                                    break
                                
                                ret, frame = cap.read()
                                if not ret:
                                    break

                                current_time = frame_number / fps

                                # Render the words that are within the current timestamp
                                frame = add_typing_effect(frame, words_with_timestamps, current_time)

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


        if choice == 0:  # Sentences in each frame
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
                        result = model.transcribe(audio_path, word_timestamps=True)

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

                    st.info("Transcribing audio with sentence-level timestamps...")
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    # Extract sentences with their corresponding timestamps
                    sentences_with_timestamps = [(segment['text'], segment['start'], segment['end'], segment['words']) for segment in result['segments']]

                    # Create font object
                    font_path = os.path.join(custom_fonts_dir, f"{font_style}.ttf")
                    font = ImageFont.truetype(font_path, 30)  # Adjust font size if needed

                    # Processing the video frames
                    frame_number = 0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    for frame_number in tqdm(range(total_frames), desc="Processing frames"):
                        if st.session_state.stop_processing:
                            st.warning("Processing stopped by user.")
                            break

                        ret, frame = cap.read()
                        if not ret:
                            break

                        current_time = frame_number / fps

                        # Convert frame to PIL Image for text overlay
                        pil_img = Image.fromarray(frame)
                        draw = ImageDraw.Draw(pil_img)

                        # Add sentences to the frame based on the timestamp
                        for sentence, start_time, end_time, words in sentences_with_timestamps:
                            if start_time <= current_time <= end_time:
                                # Split sentence into chunks of 4 words and display all chunks within the timestamp
                                sentence_words = sentence.split()
                                num_chunks = (len(sentence_words) + 3) // 4  # Calculate number of chunks

                                # Determine which chunks to display based on the current time
                                chunk_index = int((current_time - start_time) / (end_time - start_time) * num_chunks)
                                chunk_index = min(chunk_index, num_chunks - 1)  # Ensure index is within bounds
                                current_chunk = sentence_words[chunk_index * 4:(chunk_index + 1) * 4]

                                text = ' '.join(current_chunk)
                                text_bbox = draw.textbbox((0, 0), text, font=font)
                                text_width = text_bbox[2] - text_bbox[0]
                                text_height = text_bbox[3] - text_bbox[1]
                                x_position = (width - text_width) // 2  # Center text horizontally
                                y_position = height - text_height - 20  # Position from bottom

                                # Draw each word in the chunk with proper highlighting
                                x_position = (width - text_width) // 2  # Recalculate x_position for center alignment
                                for word in current_chunk:
                                    word_bbox = draw.textbbox((0, 0), word, font=font)
                                    word_width = word_bbox[2] - word_bbox[0]
                                    word_height = word_bbox[3] - word_bbox[1]

                                    # Determine the highlight color for the current word
                                    word_start_time = next((w['start'] for w in words if w['word'] == word), None)
                                    word_end_time = next((w['end'] for w in words if w['word'] == word), None)

                                    color = highlight_color_rgb if word_start_time and word_end_time and word_start_time <= current_time <= word_end_time else text_color

                                    # Draw the text
                                    draw.text((x_position, y_position), word, font=font, fill=color)
                                    x_position += word_width + 5  # Adjust space between words if needed

                                break  # Only show the first sentence for the current frame

                        frame = np.array(pil_img)
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

st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """,
    unsafe_allow_html=True
)

# Example Output Section
st.markdown("## Product Example Output")

# Button to show the example output
if st.button("View Example Output"):
    # Example output can be an image, video, or any other media
    st.markdown("### Example Video")
    
    # You can embed a video from YouTube or display a locally stored video
    st.video("https://www.youtube.com/watch?v=1IGDQIdnhdM")  # Replace with your example video URL

st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """,
    unsafe_allow_html=True
)

# Footer with social links and icons, adapted for light and dark modes
footer = """
    <style>
        footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            padding: 10px;
            text-align: center;
            box-shadow: 0 -1px 5px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        
        /* Light mode styles */
        @media (prefers-color-scheme: light) {
            footer {
                background-color: #f1f1f1;
                color: #333;
            }
            footer a {
                color: #333;
            }
            footer a:hover {
                color: #555;
            }
        }

        /* Dark mode styles */
        @media (prefers-color-scheme: dark) {
            footer {
                background-color: #333;
                color: #f1f1f1;
            }
            footer a {
                color: #f1f1f1;
            }
            footer a:hover {
                color: #bbb;
            }
        }

        footer i {
            font-size: 24px;
            margin: 0 10px;
        }

        footer p {
            margin: 5px 0;
        }
    </style>
    <footer>
        <p>Contact us on:</p>
        <a href="https://github.com/alihamzasultan" target="_blank">
            <i class="fab fa-github"></i>
        </a>
        <a href="https://www.linkedin.com/in/ali-hamza-sultan-1ba7ba267/" target="_blank">
            <i class="fab fa-linkedin"></i>
        </a>
        <a href="https://www.youtube.com/channel/UCB3Cxjm-cCwV0Rn3WmKHF_A" target="_blank">
            <i class="fab fa-youtube"></i>
        </a>
        <p>&copy; 2024 Ali Hamza Sultan</p>
    </footer>
"""

st.markdown(footer, unsafe_allow_html=True)


from streamlit.components.v1 import html

button = """
<script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="alihamzasultan6" data-color="#FFDD00" data-emoji=""  data-font="Cookie" data-text="Buy me a coffee" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff" ></script>
"""

html(button, height=70, width=220)

st.markdown(
    """
    <style>
        iframe[width="220"] {
            position: fixed;
            bottom: 60px;
            right: 40px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
