import collections
import numpy as np


def parse_rttm_line(line):
    """Parse one RTTM line -> (recording, start, end)."""
    parts = line.strip().split()
    if parts[0] != "SPEAKER":
        return None
    rec_id = parts[1]
    start = float(parts[3])
    dur = float(parts[4])
    end = start + dur
    return rec_id, start, end

def speaker_overlap_distribution(rttm_file):
    """
    Compute % of time with 0,1,2,... speakers active in a recording.
    """
    segments = []
    with open(rttm_file, "r") as f:
        for line in f:
            seg = parse_rttm_line(line)
            if seg:
                segments.append(seg)

    if not segments:
        return {}

    # pro Aufnahme getrennt
    by_rec = collections.defaultdict(list)
    for rec, start, end in segments:
        by_rec[rec].append((start, end))

    distributions = {}

    for rec, segs in by_rec.items():
        # Event-Liste bauen: +1 beim Start, -1 beim Ende
        events = []
        for s, e in segs:
            events.append((s, +1))
            events.append((e, -1))
        events.sort()

        counts = collections.Counter()
        active = 0
        last_t = events[0][0]

        for t, change in events:
            # Dauer seit letztem Event
            if t > last_t:
                counts[active] += (t - last_t)
            active += change
            last_t = t

        total_time = sum(counts.values())
        dist = {k: v / total_time for k, v in counts.items()}
        distributions[rec] = dist

    return distributions

# Beispiel
rttm_path = "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/data_no_chime/train/rttm"
dists = speaker_overlap_distribution(rttm_path)
total_counts = collections.Counter()
for rec, dist in dists.items():
    print(f"Recording: {rec}")
    for n_spk, frac in sorted(dist.items()):
        print(f"  {n_spk} Sprecher: {frac:.2%}")
for dist in dists.values():
    total_counts.update(dist)
avg_dist = {k: v / len(dists) for k, v in total_counts.items()}
print("\n=== Gemittelte Verteilung über alle Recordings ===")
for n_spk, frac in sorted(avg_dist.items()):
    print(f"  {n_spk} Sprecher: {frac:.2%}")