import numpy as np
import os

def main(path, out_path=None):
    with open(path, "r") as f:
        lines = f.readlines()
    out_lines = []
    for line in lines:
        file = line.split("/")[-1]
        new_line = f"{line.split("/")[0]}/scratch/hpc-prf-nt2/db/AMI_AIS_ALI_NSF_CHiME7/wavs/test/{file}"
        out_lines.append(new_line)
    with open(out_path, "w") as f:
        f.writelines(out_lines)

if __name__ == "__main__":
    for d in ["AISHELL4",  "AliMeeting",  "AMI"]: #,  "NOTSOFAR1"]:
        main(f"/scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/data_no_chime/test_marc/{d}/wav.scp_{d}",
             out_path=f"/scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/data_no_chime/test_marc/{d}/wav.scp")