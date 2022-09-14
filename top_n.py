import numpy as np
# NUM_EMB = 5
def get_top_n_score(embeddings_path):

    data = np.genfromtxt(embeddings_path, delimiter=' ')

    room = data[:, 0]
    scan = np.zeros(room.shape)
    emb_size = data[:, 1]
    emb = data[:, 2:]

    # Initialize numbering.
    # Take advantage of the fact scan is all zeros.
    '''prev_room = room[0]
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
            prev_room = room[i]'''

    top1_count = 0.0
    top5_count = 0.0

    # For each scan in the database
    # 0, 5, 10, 15, 20
    i = 0
    while(i<emb.shape[0]):
        #print(emb_size[i], i)
        for j in range(int(emb_size[i])):
            # Select the query embedding
            query = emb[i + j]

            # Calculate distance to all database embeddings
            dist = np.linalg.norm(emb - query, axis=1)
            # Make distance to embedding from same view infinity so that they
            # get automatically dropped out from the calculation
            dist[i:i + int(emb_size[i])] = np.inf

            # Sort from smallest to largest distance (get indices)
            sort_idx = np.argsort(dist)

            if room[sort_idx[0]] == room[i + j]:
                top1_count += 1.0

            if np.any(room[sort_idx[:5]] == room[i + j]):
                top5_count += 1.0
        i+=emb_size[i]
    '''for i in range(0, emb.shape[0], int(emb_size[i])):
        # For each embedding in that scan
        # 0, 1, 2, 3, 4, 5
        print(emb_size[i], i)
        for j in range(int(emb_size[i])):
            # Select the query embedding
            query = emb[i + j]

            # Calculate distance to all database embeddings
            dist = np.linalg.norm(emb - query, axis=1)
            # Make distance to embedding from same view infinity so that they
            # get automatically dropped out from the calculation
            dist[i:i + int(emb_size[i])] = np.inf

            # Sort from smallest to largest distance (get indices)
            sort_idx = np.argsort(dist)

            if room[sort_idx[0]] == room[i + j]:
                top1_count += 1.0

            if np.any(room[sort_idx[:5]] == room[i + j]):
                top5_count += 1.0'''

    top1 = top1_count / emb.shape[0]
    top5 = top5_count / emb.shape[0]

    return top1, top5

def main():
    print(get_top_n_score("test_sets/test_set_vlad_embeddings_30_emb.txt"))

main()