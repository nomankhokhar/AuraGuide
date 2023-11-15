from PIL import Image
from pydub import AudioSegment
import numpy as np

def depth_to_audio(depth_map):
    # Map depth values to audio parameters
    # You may need to adjust these mapping functions based on your specific requirements
    pitch = np.interp(depth_map, (0, 255), (200, 1000))  # Map depth to pitch
    volume = np.interp(depth_map, (0, 255), (0, 100))     # Map depth to volume

    # Generate audio
    audio = AudioSegment.silent(duration=len(depth_map))
    for i, (p, v) in enumerate(zip(pitch, volume)):
        audio = audio.overlay(AudioSegment.sine(p, volume=v), position=i)

    return audio

def main():
    # Load depth map image (replace 'depth_map.png' with your file)
    depth_map_path = 'bunny.png'
    depth_map_image = Image.open(depth_map_path).convert('L')  # Convert to grayscale
    depth_map = np.array(depth_map_image)

    # Normalize depth values to the range [0, 255]
    depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255

    # Convert depth map to audio
    audio = depth_to_audio(depth_map)

    # Play the generated audio
    audio.export("depth_sound.wav", format="wav")
    audio.play()

if __name__ == "__main__":
    main()
