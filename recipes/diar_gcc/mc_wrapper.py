import numpy as np
import soundfile as sf

def get_mc_file_wrapper(path, num_files, out_path):
    """Get multi-channel file wrapper

    Parameters
    ----------
    path : str : /net/db/ami/IS1002b/audio/IS1002b.Array1-01.wav
        Path to the single-channel file.
    num_files : int
        Number of files.
    out_path : str /net/vol/deegen/data/IS1002b/audio/IS1002b.Array1.wav
        Path to save the mc files.

    Returns
    """
    # read the first file to get the sample rate and number of samples
    data, sample_rate = sf.read(path)
    num_samples = data.shape[0]

    # create an empty array to hold the multi-channel data
    mc_data = np.zeros((num_samples, num_files), dtype=data.dtype)

    # read each file and fill the corresponding channel in the multi-channel array
    for i in range(num_files):
        print(i)
        file_path = path.replace("01", f"{i+1:02d}")
        data, _ = sf.read(file_path)
        mc_data[:, i] = data

    # save the multi-channel data to the output path
    sf.write(out_path, mc_data, sample_rate)
    return

if __name__ == "__main__":
    path = "/net/db/ami/IS1002b/audio/IS1002b.Array2-01.wav"
    num_files = 4
    out_path = "/net/vol/deegen/data/IS1002b/audio/IS1002b.Array2.wav"
    get_mc_file_wrapper(path, num_files, out_path)