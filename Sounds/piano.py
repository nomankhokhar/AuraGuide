import winsound

# Melody using winsound
notes = [1000, 587, 10000, 587, 660, 783, 698, 660, 587, 523, 587, 392]

for note in notes:
    winsound.Beep(note, note)
