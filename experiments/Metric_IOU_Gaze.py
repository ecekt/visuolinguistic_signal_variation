# Our implementation

import numpy as np

from torchvision import ops
import torch

class Metric():
    def __init__(self):
        # nothing to do here
        pass

    def similarity(self, v1, v2):

        # IOU similarity
        sim_score = ops.box_iou(torch.tensor([v1]), torch.tensor([v2])).item()

        return sim_score

    def find_closest(self, content_ref, content_gen):
        closest_data = []

        # finds the closest words to the words of content_gen in content_ref

        # this method is only called if there is at least one generated word

        for g in range(len(content_gen)):

            vector_g = content_gen[str(g)]

            closest_token = ''
            closest_sim = -1
            closest_pos = -1

            # loop over the other sentence
            for r in range(len(content_ref)):

                vector_r = content_ref[str(r)]

                # cosine similarity
                sim = self.similarity(vector_g, vector_r)

                # if this is equal to the closest token
                # in terms of similarity
                # then check if it is closer positionally

                if sim == closest_sim:

                    if abs(r - g) < abs(closest_pos - g):
                        # new one is closer in terms of position in the sentence
                        closest_token = vector_r
                        closest_sim = sim
                        closest_pos = r

                elif sim > closest_sim:
                    # if similarity is higher, set it directly as the closest token
                    closest_token = vector_r
                    closest_sim = sim
                    closest_pos = r

            closest_item = g, vector_g, closest_pos, closest_token, closest_sim  # similarity later to be converted to distance
            closest_data.append(closest_item)

        return closest_data

    def new_metric(self, ref, gen):

        if len(gen) == 0 or len(ref) == 0:
            return 2  # highest possible cost, in case we have an empty sentence

        closest_items = self.find_closest(ref, gen)

        score = 0

        # find the longest sentence and use its length in obtaining relative positional distances
        if len(ref) >= len(gen):
            longest_pos = len(ref)
        else:
            longest_pos = len(gen)

        for c in closest_items:
            g, token_g, closest_pos, closest_token, closest_sim = c

            # similarity to distance
            closest_dist = 1 - closest_sim
            pos_dist = abs(g - closest_pos)  # + 1

            rel_pos_dist = pos_dist / longest_pos  # relative positional cost

            contribution = closest_dist + rel_pos_dist  # summing two types of distances to obtain the final score

            score += contribution

        return score

    def get_corpus_score(self, all_gens, all_refs):

        # list of tokenized sentences
        # generated sentence and 1 reference sentence
        # image-participant pair

        corpus_size = len(all_gens)

        corpus_scores = []

        corpus_scores_1 = []
        corpus_scores_2 = []

        for i in range(corpus_size):
            gen_sent = all_gens[i]
            ref_sent = all_refs[i]

            score1 = self.new_metric(ref_sent, gen_sent)  # looping over generated words
            score2 = self.new_metric(gen_sent, ref_sent)  # looping over reference words

            corpus_scores_1.append(score1)
            corpus_scores_2.append(score2)

            score_avg = (score1 + score2) / 2  # average of two directions

            corpus_scores.append(score_avg)

            print('ref:',ref_sent)
            print('gen:',gen_sent)
            print('gen2ref:',score1)
            print('ref2gen:',score2)
            print()
            print('avgscore', score_avg)
            print('\n')

        corpus_score = np.mean(corpus_scores)  # arithmetic average of the full corpus score

        corpus_score_1 = np.mean(corpus_scores_1) #gen2ref
        corpus_score_2 = np.mean(corpus_scores_2) #ref2gen

        return corpus_score, corpus_score_1, corpus_score_2