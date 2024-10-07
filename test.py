import speech_recognition as sr
from openai import OpenAI
from moviepy.editor import VideoFileClip
import os
import dotenv

dotenv.load_dotenv()

print(os.environ.get('OPENAI_API_KEY'))
# Set up OpenAI API key
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def extract_audio_from_video(video_file_path, output_audio_path="audio_result.wav"):
    """
    Extracts the audio from a video file and saves it as a .wav file.
    """
    try:
        video = VideoFileClip(video_file_path)
        video.audio.write_audiofile(output_audio_path, codec="pcm_s16le")  # Save as WAV
        return output_audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def transcribe_audio_file(file_path):
    recognizer = sr.Recognizer()
    
    # Open the audio file and recognize its content
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    
    try:
        # Use Google's speech recognition to transcribe the audio to text
        text = recognizer.recognize_google(audio)
        print(f"Transcribed text: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError:
        print("Sorry, I am having trouble accessing the speech recognition service.")
        return None

def get_gpt_answer(prompt):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        print(f"Error getting response from GPT-4: {e}")
        return None

if __name__ == "__main__":
    # Specify the path to your audio or video file
    file_path = "query.wav"  # Change this to your file

    # Check if it's a video or audio file
    if file_path.endswith((".mp4", ".mov", ".avi", ".mkv")):
        # Extract audio from the video file
        audio_file_path = extract_audio_from_video(file_path)
    else:
        # If it's already an audio file
        audio_file_path = file_path

    if audio_file_path:
        # Step 1: Convert audio to text
        user_input_text = transcribe_audio_file(audio_file_path)
        
        if user_input_text:
            # Step 2: Use the transcribed text as a prompt for GPT-4
            print("Generating response from GPT-4...")
            gpt_response = get_gpt_answer(user_input_text)
            
            if gpt_response:
                print(f"GPT-4 response: {gpt_response}")
            else:
                print("No response from GPT-4.")
        else:
            print("Transcription failed.")
        
        # Clean up the extracted audio file if it was from a video
        if file_path.endswith((".mp4", ".mov", ".avi", ".mkv")) and os.path.exists(audio_file_path):
            os.remove(audio_file_path)
