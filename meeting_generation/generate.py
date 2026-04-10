import os

import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pathlib import Path

def main(simulate_folder, outfolder=None, scale=1, fs=16000, n_mics=4, mic_radius=0.1, n_speakers=4):
    if outfolder is None:
        outfolder = simulate_folder
    Path(outfolder).mkdir(parents=True, exist_ok=True)

    size = np.array([4,6,2]) * scale
    room = pra.ShoeBox(size, fs=fs)
    spk = 0

    # circular microphone array (n microphones) centered in the room at 1 m height
    mic_center = np.array([size[0]/2, size[1]/2, 1.0 * scale])
    angles = np.linspace(0, 2*np.pi, n_mics, endpoint=False)
    mics = np.stack([
        mic_center[0] + mic_radius * np.cos(angles),
        mic_center[1] + mic_radius * np.sin(angles),
        np.full(n_mics, mic_center[2])
    ])
    room.add_microphone_array(pra.MicrophoneArray(mics, fs))

    # speakers placed evenly around the microphone array
    speaker_radius = 0.2 * min(size[0], size[1])
    speaker_angles = np.linspace(0, 2*np.pi, n_speakers, endpoint=False)
    for file in Path(simulate_folder).iterdir():
        if not file.is_file() or not "Headset" in file.name or "Mix" in file.name:
            continue
        _, headset_sig = wavfile.read(file)  # f'/home/deegen/audio/audio/EN2001a.Headset-{i}.wav')
        src_pos = np.array([
            mic_center[0] + speaker_radius * np.cos(speaker_angles[spk]),
            mic_center[1] + speaker_radius * np.sin(speaker_angles[spk]),
            mic_center[2]
        ])
        room.add_source(src_pos, signal=headset_sig)
        spk += 1
        print(f"Added speaker {spk}")
        print(file.stem, )
        if spk == n_speakers:
            break

    # room.plot()
    # plt.title("2D shape of the room (the height of the room is 2 meters)")
    # plt.show()

    # Compute the RIR using the hybrid method
    # room.compute_rir() #mode='hybrid', nb_thetas=500, nb_phis=500, scatter_coef=0.)

    # Plot and apply the RIR on the audio file
    # room.plot_rir()
    # plt.show()
    # print("shown")
    room.simulate()
    print("simulated")
    room.mic_array.to_wav(f'{outfolder}/simulated_meeting_{file.name.split(".")[0]}.wav', norm=True, bitdepth=np.int16)


if __name__ == "__main__":
    main("/home/deegen/audio/audio", "/home/deegen/audio/audio/simulated_meeting_data_out")