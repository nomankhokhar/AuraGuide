# import cv2

# # -1 -> Orgnal Image C0lor
# # 0 -> In GreyScal Color

# img = cv2.imread('assets/logo.jpg', -1)
# # Resize the image that we load
# img = cv2.resize(img , (0, 0) , fx=0.5 , fy=0.5)
# img = cv2.rotate(img , cv2.ROTATE_90_COUNTERCLOCKWISE)

# # this will create new Image of Calculation's
# cv2.imwrite('new_image.jpg', img)
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.io.wavfile as wav
# from PIL import Image

# # Load your black and white image and convert it to grayscale
# image_path = "new_image.jpg"
# image = Image.open(image_path).convert('L')

# # Normalize the pixel values to the desired frequency range
# pixels = np.array(image)
# min_pixel = np.min(pixels)
# max_pixel = np.max(pixels)
# normalized_pixels = (pixels - min_pixel) / (max_pixel - min_pixel)
# frequency_range = 100  # Adjust this based on your desired frequency range
# frequencies = normalized_pixels * frequency_range

# # Create a time array based on the number of pixels in the image
# sampling_rate = 44100  # Adjust this based on your preference
# duration = len(frequencies) / sampling_rate
# t = np.linspace(0, duration, len(frequencies), endpoint=False)

# # Generate the audio signal by using the frequencies as the signal amplitude
# audio_signal = np.sin(2 * np.pi * frequencies * t)

# # Save the audio signal as a WAV file
# output_file = "output_sound.wav"
# wav.write(output_file, sampling_rate, audio_signal.astype(np.float32))








# import numpy as np
# import matplotlib.pyplot as plt

# # Assuming you have your image data as a 2D array (e.g., grayscale values)
# image_data = np.random.rand(100, 100)  # Replace with your image data

# # Convert the image data to audio data (e.g., map pixel values to frequencies)
# audio_data = image_data * 1000  # Adjust scaling as needed

# # Generate a time series of audio data
# sample_rate = 44100  # Adjust as needed
# time = np.arange(0, len(audio_data)) / sample_rate

# # Perform FFT to create a spectrogram
# specgram, freqs, times = plt.specgram(audio_data, Fs=sample_rate)

# # Plot the spectrogram
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.title('Spectrogram of Image')
# plt.colorbar(label='Intensity (dB)')

# plt.show()







# import numpy as np
# # import matplotlib.pyplot as plt
# from scipy import signal
# import sounddevice as sd

# # Load an image (replace 'your_image.png' with the path to your image)
# image_path = 'your_image.png'
# image = plt.imread(image_path)

# # Convert the image to grayscale
# gray_image = np.mean(image, axis=-1)

# # Parameters for audio generation
# sample_rate = 44100  # Hz
# duration = 5  # seconds
# frequencies_per_pixel = 1000  # Adjust as needed

# # Create an audio signal by mapping grayscale values to frequencies
# audio_data = gray_image * frequencies_per_pixel

# # Create a spectrogram
# frequencies, times, spectrogram = signal.spectrogram(audio_data, fs=sample_rate)

# # Play the audio represented by the spectrogram
# sd.play(spectrogram, sample_rate)
# sd.wait()

# # # Display the spectrogram
# # plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram))
# # plt.colorbar(label='Intensity (dB)')
# # plt.xlabel('Time (s)')
# # plt.ylabel('Frequency (Hz)')
# # plt.title('Spectrogram of Image')
# # plt.show()










# import tkinter as tk
# from tkinter import filedialog
# import numpy as np
# from matplotlib import pyplot as plt
# from scipy.io import wavfile
# import sounddevice as sd
# from scipy import signal

# # Function to convert an image to audio
# def convert_image_to_audio():
#     file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
#     if file_path:
#         # Load the image and convert it to grayscale
#         image = plt.imread(file_path)
#         gray_image = np.mean(image, axis=-1)

#         # Parameters for audio generation
#         sample_rate = 44100  # Hz
#         frequencies_per_pixel = 1000

#         # Create an audio signal by mapping grayscale values to frequencies
#         audio_data = gray_image * frequencies_per_pixel

#         # Save the audio as a WAV file
#         wavfile.write("output_audio.wav", sample_rate, audio_data.astype(np.float32))

#         # Display a message
#         status_label.config(text="Image converted and saved as 'output_audio.wav'")

# # Function to play the generated audio
# def play_audio():
#     audio_file = "output_audio.wav"
#     if audio_file:
#         audio_data, sample_rate = sd.read(audio_file, dtype='float32')
#         sd.play(audio_data, sample_rate)
#         sd.wait()

# # Create the main application window
# app = tk.Tk()
# app.title("Image to Audio Converter")

# # Create buttons
# convert_button = tk.Button(app, text="Convert Image to Audio", command=convert_image_to_audio)
# play_button = tk.Button(app, text="Play Audio", command=play_audio)
# status_label = tk.Label(app, text="")

# # Place widgets in the window
# convert_button.pack()
# play_button.pack()
# status_label.pack()

# # Start the GUI event loop
# app.mainloop()
