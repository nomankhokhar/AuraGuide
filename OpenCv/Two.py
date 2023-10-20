# # # # # # import cv2
# # # # # # img = cv2.imread('assets/logo.jpg', -1)
 
# # # # # # print(img.shape)



# # # # # import numpy as np
# # # # # import matplotlib.pyplot as plt
# # # # # import scipy.io.wavfile as wavfile
# # # # # from pydub import AudioSegment
# # # # # from pydub.playback import play
# # # # # from scipy.io import wavfile

# # # # # # Load an image (replace 'your_image.png' with the path to your image)
# # # # # image_path = 'new_image.jpg'
# # # # # image = plt.imread(image_path)

# # # # # # Convert the image to grayscale
# # # # # gray_image = np.mean(image, axis=-1)

# # # # # # Parameters for audio generation
# # # # # sample_rate = 44100  # Hz
# # # # # duration = 5  # seconds
# # # # # frequency_scaling = 1000  # Adjust as needed

# # # # # # Create an audio signal by mapping grayscale values to frequencies
# # # # # audio_data = gray_image.flatten() * frequency_scaling

# # # # # # Normalize the audio data
# # # # # audio_data = audio_data / np.max(np.abs(audio_data))

# # # # # # Save the audio as a WAV file
# # # # # output_audio_file = 'image_to_audio.wav'
# # # # # wavfile.write(output_audio_file, sample_rate, audio_data.astype(np.float32))

# # # # # # Load the audio file
# # # # # sample_rate, audio_data = wavfile.read(output_audio_file)

# # # # # # Create a spectrogram from the audio
# # # # # plt.figure(figsize=(10, 6))
# # # # # plt.specgram(audio_data, Fs=sample_rate, cmap='viridis')
# # # # # plt.title('Spectrogram of Image-to-Audio Conversion')
# # # # # plt.xlabel('Time (s)')
# # # # # plt.ylabel('Frequency (Hz)')

# # # # # # Play the audio
# # # # # audio = AudioSegment.from_wav(output_audio_file)
# # # # # play(audio)

# # # # # plt.show()



# # # # import numpy as np
# # # # import matplotlib.pyplot as plt
# # # # import sounddevice as sd

# # # # # Parameters for the sine wave
# # # # frequency = 440  # Frequency in Hz (A4 note)
# # # # duration = 3  # Duration in seconds
# # # # sample_rate = 44100  # Samples per second (standard for audio)

# # # # # Create a time array
# # # # t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# # # # # Generate a sine wave
# # # # audio_signal = 0.5 * np.sin(2 * np.pi * frequency * t)

# # # # # Plot the sine wave
# # # # plt.figure(figsize=(10, 4))
# # # # plt.plot(t, audio_signal)
# # # # plt.title('Sine Wave (440 Hz)')
# # # # plt.xlabel('Time (s)')
# # # # plt.ylabel('Amplitude')
# # # # plt.grid(True)

# # # # # Play the audio
# # # # sd.play(audio_signal, sample_rate)
# # # # sd.wait()  # Wait for the audio to finish playing

# # # # # Show the plot
# # # # plt.show()






# # # import numpy as np
# # # import soundfile as sf
# # # import matplotlib.pyplot as plt

# # # # Load an image (replace 'your_image.png' with the path to your image)
# # # image_path = './assets/soccer_practice.jpg'
# # # image = plt.imread(image_path)

# # # # Convert the image to grayscale
# # # gray_image = np.mean(image, axis=-1)

# # # # Parameters for audio generation
# # # sample_rate = 44100  # Hz
# # # duration = 3  # seconds

# # # # Normalize the grayscale image values to the audio range (0-1)
# # # normalized_image = gray_image / np.max(gray_image)

# # # # Convert the image data to an audio signal
# # # audio_data = normalized_image.T.ravel()

# # # # Save the audio as a WAV file
# # # output_audio_file = 'image_to_sound.wav'
# # # sf.write(output_audio_file, audio_data, sample_rate)

# # # # Plot the grayscale image
# # # plt.figure(figsize=(6, 6))
# # # plt.imshow(gray_image, cmap='gray')
# # # plt.title('Grayscale Image')
# # # plt.axis('off')

# # # # Show the plot
# # # plt.show()

# # # # Play the audio
# # # import sounddevice as sd
# # # sd.play(audio_data, sample_rate)
# # # sd.wait()




# # import numpy as np
# # import scipy.io.wavfile as wavfile
# # from PIL import Image

# # # Load the grayscale image
# # image = Image.open("new_image.jpg").convert("L")
# # pixel_data = np.array(image)

# # # Define audio parameters
# # sample_rate = 44100  # CD-quality sample rate
# # duration = 3  # seconds

# # # Scale pixel values to fit within the range [-1.0, 1.0]
# # scaled_audio_data = (2 * (pixel_data / 255.0)) - 1.0

# # # Create the WAV file
# # wavfile.write("output_audio.wav", sample_rate, scaled_audio_data)







# # import numpy as np
# # import pyaudio
# # from PIL import Image

# # # Load the grayscale image
# # image = Image.open("new_image.jpg").convert("L")
# # pixel_data = np.array(image)

# # # Define audio parameters
# # sample_rate = 44100  # CD-quality sample rate
# # duration = 10  # seconds

# # # Scale pixel values to fit within the range [-1.0, 1.0]
# # scaled_audio_data = (2 * (pixel_data / 255.0)) - 1.0

# # # Initialize PyAudio
# # p = pyaudio.PyAudio()

# # # Open a new audio stream
# # stream = p.open(format=pyaudio.paFloat32,
# #                 channels=1,
# #                 rate=sample_rate,
# #                 output=True)

# # # Play the audio data
# # stream.start_stream()
# # stream.write(scaled_audio_data.tobytes())

# # # Stop and close the audio stream
# # stream.stop_stream()
# # stream.close()

# # # Terminate PyAudio
# # p.terminate()





















# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.io import wavfile
# # import cv2
# # import os

# # # Audio parameters
# # fs = 32000                          # Sampling frequency
# # frame_duration = 0.064              # Frame duration in seconds
# # M = int(frame_duration * fs)        # Window size
# # overlap = 0.875                     # Overlap between adjacent frames
# # H = int((1 - overlap) * M)          # Hop size

# # # Image parameters
# # height = M // 2 + 1                 # Image height
# # edge_detection = True               # Use edge detection?
# # scaling_factor = 10                 # Adjust loudness of output

# # # Read image using OpenCV
# # img_path = input('Filename: ')
# # while not os.path.exists(img_path):
# #     img_path = input('File does not exist. Try again: ')

# # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# # # Resize
# # img = cv2.resize(img, (img.shape[1], height))

# # # Edge detection using OpenCV's Sobel filter
# # if edge_detection:
# #     img = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)  # Applying Sobel filter for edge detection
# #     img = np.abs(img)                                # Convert to absolute values
# #     plt.imshow(img, cmap='gray')
# #     plt.show()

# # # Convert to double, flip upside down, and scale
# # img_double = scaling_factor * np.flipud(img.astype(float))

# # # Audio synthesis
# # nframes = img_double.shape[1]      # Number of frames in STFT
# # leny = M + (nframes - 1) * H + M   # Length of reconstructed signal (+M to avoid shape mismatch)
# # y = np.zeros(leny)                 # Initialize reconstructed signal
# # w = np.hamming(M)                  # Window
# # X = np.zeros((M, nframes), dtype=complex)  # Initialize STFT
# # offy = M // 2                      # Initialize offset in y

# # for i in range(nframes):
# #     samples = np.arange(offy - M // 2, offy + M // 2)  # Samples of y to be reconstructed
# #     phase = 2 * np.pi * (np.random.rand(M // 2) - 0.5)  # Randomize phase
# #     X[:, i] = np.concatenate((img_double[:, i] * np.exp(1j * phase), np.conj(np.flipud(img_double[:, i]) * np.exp(-1j * phase))), axis=0)
# #     yfw = np.fft.ifft(X[:, i]) * w    # Inverse FFT
# #     y[samples] += yfw.real            # Overlap add
# #     offy += H                        # Offset for the next iteration of the loop

# # # Plot spectrogram
# # plt.figure()
# # plt.specgram(y, NFFT=M, Fs=fs, window=w, noverlap=M - H)
# # plt.xlabel('Time (s)')
# # plt.ylabel('Frequency (Hz)')

# # # Write audio file
# # audio_path = os.path.splitext(img_path)[0] + '-img2sound.wav'
# # wavfile.write(audio_path, fs, y.astype(np.float32))













# # from PIL import Image
# # import math
# # import wave
# # import array
# # import sys
# # import getopt

# # def start(inputfile, outputfile, duration):
# #     im = Image.open(inputfile)
# #     width, height = im.size
# #     rgb_im = im.convert('RGB')

# #     duration_seconds = float(duration)
# #     tmp_data = []
# #     max_freq = 0
# #     data = array.array('h')
# #     sample_rate = 44100
# #     channels = 1
# #     data_size = 2

# #     num_samples = int(sample_rate * duration_seconds)
# #     samples_per_pixel = math.floor(num_samples / width)

# #     c = 20000 / height

# #     for x in range(num_samples):
# #         rez = 0

# #         pixel_x = int(x / samples_per_pixel)
# #         if pixel_x >= width:
# #             pixel_x = width - 1

# #         for y in range(height):
# #             r, g, b = rgb_im.getpixel((pixel_x, y))
# #             s = r + g + b

# #             volume = s * 100 / 765

# #             if volume == 0:
# #                 continue

# #             freq = int(c * (height - y + 1))

# #             rez += get_data(volume, freq, sample_rate, x)

# #         tmp_data.append(rez)
# #         if abs(rez) > max_freq:
# #             max_freq = abs(rez)

# #     for i in range(len(tmp_data)):
# #         data.append(int(32767 * tmp_data[i] / max_freq))

# #     with wave.open(outputfile, 'w') as f:
# #         f.setparams((channels, data_size, sample_rate, num_samples, "NONE", "Uncompressed"))
# #         f.writeframes(data.tobytes())

# # def get_data(volume, freq, sample_rate, index):
# #     return int(volume * math.sin(freq * math.pi * 2 * index / sample_rate))

# # if __name__ == '__main__':
# #     inputfile = './new_image.jpg'
# #     outputfile = './'
# #     duration = '4'

# #     try:
# #         opts, args = getopt.getopt(sys.argv[1:], "hi:o:t:")
# #     except getopt.GetoptError:
# #         print('imgencode.py -i <input_picture> -o <output.wav> -t <duration_seconds>')
# #         sys.exit(2)

# #     for opt, arg in opts:
# #         if opt == '-h':
# #             print('imgencode.py -i <input_picture> -o <output.wav> -t <duration_seconds>')
# #             sys.exit()
# #         elif opt == "-i":
# #             inputfile = arg
# #         elif opt == "-o":
# #             outputfile = arg
# #         elif opt == "-t":
# #             duration = arg

# #     start(inputfile, outputfile, duration)


# from PIL import Image  # Ensure you import Image from Pillow
# import numpy as np
# import wave
# import math
# import getopt
# import sys
# from scipy.io import wavfile

# def generate_audio_from_image(inputfile, outputfile, duration, sample_rate):
#     # Load the grayscale image
#     image = np.array(Image.open(inputfile).convert('L'))

#     # Normalize pixel intensities to the range [0, 1]
#     normalized_image = image / 255.0

#     # Parameters
#     num_samples = int(sample_rate * float(duration))
#     channels = 2  # Stereo audio
#     data_size = 2  # 16-bit audio
#     max_amplitude = 32767  # Maximum amplitude for 16-bit audio

#     # Create audio arrays for left and right channels
#     left_channel = np.zeros(num_samples)
#     right_channel = np.zeros(num_samples)

#     # Assign spatial positions (panning)
#     pan_factor = 0.5  # Adjust this for desired spatial effect
#     left_pan = 1.0 - pan_factor
#     right_pan = pan_factor

#     # Generate audio signals based on pixel intensities
#     for i in range(num_samples):
#         x = int(i * image.shape[1] / num_samples)
#         intensity = normalized_image[x, 0]  # Use the leftmost column intensity for audio

#         left_channel[i] = int(max_amplitude * intensity * left_pan)
#         right_channel[i] = int(max_amplitude * intensity * right_pan)

#     # Combine left and right channels
#     stereo_audio = np.column_stack((left_channel, right_channel))

#     # Write stereo audio to a WAV file
#     wavfile.write(outputfile, sample_rate, stereo_audio.astype(np.int16))

# if __name__ == '__main__':
#     inputfile = ''
#     outputfile = ''
#     duration = 5.0  # Duration of the audio in seconds
#     sample_rate = 44100  # Default sample rate

#     try:
#         opts, args = getopt.getopt(sys.argv[1:], "hi:o:d:s:")
#     except getopt.GetoptError:
#         print('audio_from_image.py -i <input_image> -o <output_audio.wav> -d <duration> -s <sample_rate>')
#         sys.exit(2)

#     for opt, arg in opts:
#         if opt == '-h':
#             print('audio_from_image.py -i <input_image> -o <output_audio.wav> -d <duration> -s <sample_rate>')
#             sys.exit()
#         elif opt == "-i":
#             inputfile = arg
#         elif opt == "-o":
#             outputfile = arg
#         elif opt == "-d":
#             duration = float(arg)
#         elif opt == "-s":
#             sample_rate = int(arg)

#     generate_audio_from_image(inputfile, outputfile, duration, sample_rate)









import numpy as np
import soundfile as sf

gray_image = np.array(
    [
    [100, 120, 140, 160, 180, 200, 220, 240, 250, 255],
    [90, 110, 130, 150, 170, 190, 210, 230, 245, 250],
    [80, 100, 120, 140, 160, 180, 200, 220, 240, 245],
    [70, 90, 110, 130, 150, 170, 190, 210, 230, 240],
    [60, 80, 100, 120, 140, 160, 180, 200, 220, 230],
    [50, 70, 90, 110, 130, 150, 170, 190, 210, 220],
    [40, 60, 80, 100, 120, 140, 160, 180, 200, 210],
    [30, 50, 70, 90, 110, 130, 150, 170, 190, 200],
    [20, 40, 60, 80, 100, 120, 140, 160, 180, 190],
    [10, 30, 50, 70, 90, 110, 130, 150, 170, 180],
], dtype=np.uint8)


sample_rate = 44100  
duration = 5.0       
num_channels = gray_image.shape[1]  

audio_signal = np.zeros((int(sample_rate * duration), num_channels), dtype=np.float32)

for channel_idx in range(num_channels):
    column = gray_image[:, channel_idx]
    audio_amplitudes = (column / 255.0) * 2.0 - 1.0
    num_samples_per_channel = len(audio_amplitudes)
    start_sample = int(channel_idx * (sample_rate * duration) / num_channels)
    end_sample = start_sample + num_samples_per_channel
    audio_signal[start_sample:end_sample, channel_idx] = audio_amplitudes


output_filename = "spatial_sound.wav"
sf.write(output_filename, audio_signal, sample_rate)

print(f"Spatial sound saved as {output_filename}")












# import numpy as np
# import soundfile as sf
# from PIL import Image
# from scipy.interpolate import interp1d  # Import interp1d from scipy.interpolate

# # Create a 1000x1000 grayscale image with random values (replace this with your own data)
# gray_image = np.random.randint(0, 256, size=(1000, 1000), dtype=np.uint8)

# # Define audio parameters
# sample_rate = 44100  # Sample rate in Hz
# duration = 10.0      # Duration in seconds
# num_channels = gray_image.shape[1]  # Number of audio channels (based on image columns)

# # Calculate the common length for all channels based on duration and sample rate
# common_length = int(sample_rate * duration)

# # Create an empty multi-channel audio signal
# audio_signal = np.zeros((common_length, num_channels), dtype=np.float32)

# # Map pixel intensities to audio amplitudes and distribute them across the audio signal
# for channel_idx in range(num_channels):
#     column = gray_image[:, channel_idx]
#     audio_amplitudes = (column / 255.0) * 2.0 - 1.0  # Map 0-255 to -1.0 to 1.0
    
#     # Use interpolation to resample audio amplitudes to the common length
#     resampled_amplitudes = interp1d(np.linspace(0, 1, len(audio_amplitudes)), audio_amplitudes)(np.linspace(0, 1, common_length))
    
#     audio_signal[:, channel_idx] = resampled_amplitudes

# # Save the multi-channel audio signal as a stereo WAV file
# output_filename = "spatial_sound.wav"
# sf.write(output_filename, audio_signal, sample_rate)

# print(f"Spatial sound saved as {output_filename}")
