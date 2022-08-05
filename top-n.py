import numpy as np

NUM_EMB = 5

data = np.genfromtxt('test_set_vlad_embeddings_only_in_graph.txt', delimiter=' ')

room = data[:, 0]
scan = np.zeros(room.shape)
emb = data[:, 1:]

# Initialize numbering.
# Take advantage of the fact scan is all zeros.
prev_room = room[0]
prev_scan = 0
for i in range(NUM_EMB, emb.shape[0], NUM_EMB):
    if room[i] == prev_room:
        # If still same room we just increment scan number
        prev_scan += 1
        scan[i:i+NUM_EMB] = prev_scan
    else:
        # If new room reset scan counter
        # Again take advantage that the default value is zero
        prev_scan = 0
        prev_room = room[i]

top1_count = 0.0
top5_count = 0.0

# For each scan in the database
# 0, 5, 10, 15, 20
for i in range(0, emb.shape[0], NUM_EMB):
    # For each embedding in that scan
    # 0, 1, 2, 3, 4, 5
    for j in range(NUM_EMB):
        # Select the query embedding
        query = emb[i + j]

        # Calculate distance to all database embeddings
        dist = np.linalg.norm(emb - query, axis=1)
        # Make distance to embedding from same view infinity so that they
        # get automatically dropped out from the calculation
        dist[i:i + NUM_EMB] = np.inf

        # Sort from smallest to largest distance (get indices)
        sort_idx = np.argsort(dist)

        if room[sort_idx[0]] == room[i + j]:
            top1_count += 1.0

        if np.any(room[sort_idx[:5]] == room[i + j]):
            top5_count += 1.0

print('top1: ', top1_count / emb.shape[0])
print('top5: ', top5_count / emb.shape[0])