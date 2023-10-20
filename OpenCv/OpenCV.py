# import numpy as np
# import soundfile as sf

# gray_image = np.array(
#     [
#     [100, 120, 140, 160, 180, 200, 220, 240, 250, 255],
#     [90, 110, 130, 150, 170, 190, 210, 230, 245, 250],
#     [80, 100, 120, 140, 160, 180, 200, 220, 240, 245],
#     [70, 90, 110, 130, 150, 170, 190, 210, 230, 240],
#     [60, 80, 100, 120, 140, 160, 180, 200, 220, 230],
#     [50, 70, 90, 110, 130, 150, 170, 190, 210, 220],
#     [40, 60, 80, 100, 120, 140, 160, 180, 200, 210],
#     [30, 50, 70, 90, 110, 130, 150, 170, 190, 200],
#     [20, 40, 60, 80, 100, 120, 140, 160, 180, 190],
#     [10, 30, 50, 70, 90, 110, 130, 150, 170, 180],
# ], dtype=np.uint8)

# sample_rate = 44100  
# duration = 5.0       
# num_channels = gray_image.shape[1]  

# audio_signal = np.zeros((int(sample_rate * duration), num_channels), dtype=np.float32)

# scaling_factor = 2.0  # Adjust the scaling factor as needed

# for channel_idx in range(num_channels):
#     column = gray_image[:, channel_idx]
#     audio_amplitudes = (column / 255.0) * 2.0 * scaling_factor - 1.0
#     num_samples_per_channel = len(audio_amplitudes)
#     start_sample = int(channel_idx * (sample_rate * duration) / num_channels)
#     end_sample = start_sample + num_samples_per_channel
#     audio_signal[start_sample:end_sample, channel_idx] = audio_amplitudes

# output_filename = "spatial_sound.wav"
# sf.write(output_filename, audio_signal, sample_rate)

# print(f"Spatial sound saved as {output_filename}")









# import numpy as np
# import soundfile as sf
# import cv2


# image_path = 'new_image.jpg'
# gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# if gray_image is None:
#     raise ValueError(f"Failed to load image from {image_path}")

# sample_rate = 44100  
# duration = 5.0       
# num_channels = gray_image.shape[1]  

# audio_signal = np.zeros((int(sample_rate * duration), num_channels), dtype=np.float32)

# scaling_factor = 2.0  # Adjust the scaling factor as needed

# for channel_idx in range(num_channels):
#     column = gray_image[:, channel_idx]
#     audio_amplitudes = (column / 255.0) * 2.0 * scaling_factor - 1.0
#     num_samples_per_channel = len(audio_amplitudes)
#     start_sample = int(channel_idx * (sample_rate * duration) / num_channels)
#     end_sample = start_sample + num_samples_per_channel
#     audio_signal[start_sample:end_sample, channel_idx] = audio_amplitudes

# output_filename = "spatial_sound.wav"
# sf.write(output_filename, audio_signal, sample_rate)

# print(f"Spatial sound saved as {output_filename}")
