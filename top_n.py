import numpy as np
# NUM_EMB = 5
def get_top_n_score(embeddings_path, is_vlad_emb=False):

    data = np.genfromtxt(embeddings_path, delimiter=' ')

    if is_vlad_emb:
        room = data[:, 0]
        emb_size = data[:, 1]
        emb = data[:, 2:]

        i = 0
        NUM_EMB = int(emb_size[int(i)])
    else:
        room = data[:, 0]
        emb = data[:, 1:]

        NUM_EMB = 5

    top1_count = 0.0
    top5_count = 0.0

    # For each scan in the database
    # 0, 5, 10, 15, 20
    i = 0
    while(i<emb.shape[0]):
        # print(emb_size[int(i)], int(i))
        for j in range(NUM_EMB):
            # Select the query embedding
            query = emb[int(i) + j]

            # Calculate distance to all database embeddings
            dist = np.linalg.norm(emb - query, axis=1)
            # Make distance to embedding from same view infinity so that they
            # get automatically dropped out from the calculation
            dist[int(i):int(i) + NUM_EMB] = np.inf

            # Sort from smallest to largest distance (get indices)
            sort_idx = np.argsort(dist)

            if room[sort_idx[0]] == room[int(i) + j]:
                top1_count += 1.0

            if np.any(room[sort_idx[:5]] == room[int(i) + j]):
                top5_count += 1.0
        if is_vlad_emb:    
            i+= int(emb_size[int(i)])
        else:
            i+=NUM_EMB
    
    top1 = top1_count / emb.shape[0]
    top5 = top5_count / emb.shape[0]

    return top1, top5

