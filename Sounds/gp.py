# [
#     0.8,
#     -0.6,
#     0.9,
#     -0.3,
#     0.5,
#     -0.2,
#     0.7,
#     -0.4,
#     0.2,
#     -0.8,
#     0.6,
#     -0.9,
#     0.3,
#     -0.5,
#     0.2,
#     -0.7,
#     0.4,
#     -0.2,
#     0.8,
#     -0.6,
# ]
# [
#     0.1,
#     -0.2,
#     0.3,
#     -0.4,
#     0.5,
#     -0.6,
#     0.7,
#     -0.8,
#     0.9,
#     -1.0,
#     0.1,
#     -0.2,
#     0.3,
#     -0.4,
#     0.5,
#     -0.6,
#     0.7,
#     -0.8,
#     0.9,
#     -1.0,
# ]

# [
#     0.1,
#     0.2,
#     0.3,
#     0.4,
#     0.5,
#     0.6,
#     0.7,
#     0.8,
#     0.9,
#     1.0,
#     1.1,
#     1.2,
#     1.3,
#     1.4,
#     1.5,
#     1.6,
#     1.7,
#     1.8,
#     1.9,
#     2.0,
# ]


import numpy as np
import scipy.io.wavfile as wav

sample_rate = 44100
duration = 3
num_samples = int(duration * sample_rate)
audio_data = np.array(
    [
        0.8,
        -0.6,
        0.9,
        -0.3,
        0.5,
        -0.2,
        0.7,
        -0.4,
        0.2,
        -0.8,
        0.6,
        -0.9,
        0.3,
        -0.5,
        0.2,
        -0.7,
        0.4,
        -0.2,
        0.8,
        -0.6,
    ]
)
audio_data = np.tile(audio_data, num_samples // len(audio_data) + 1)[:num_samples]

audio_data = np.int16(audio_data * 32767)
wav.write("output.wav", sample_rate, audio_data)
