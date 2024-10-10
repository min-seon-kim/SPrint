## the code is mostly taken from example_forgetting pytorch repo:
# https://github.com/mtoneva/example_forgetting

import numpy as np

def compute_forgetting_statistics(diag_stats, npresentations):
    presentations_needed_to_learn = {}
    unlearned_per_presentation = {}
    # margins_per_presentation = {}
    first_learned = {}

    for example_id, example_stats in diag_stats.items():

        # Skip 'train' and 'test' keys of diag_stats
        if not isinstance(example_id, str):

            # Forgetting event is a transition in accuracy from 1 to 0
            presentation_acc = np.array(example_stats[:npresentations])
            transitions = presentation_acc[1:] - presentation_acc[:-1]

            # Find all presentations when forgetting occurs
            if len(np.where(transitions == -1)[0]) > 0:
                unlearned_per_presentation[example_id] = np.where(
                    transitions == -1)[0] + 2
            else:
                unlearned_per_presentation[example_id] = []

            # Find number of presentations needed to learn example, 
            # e.g. last presentation when acc is 0
            if len(np.where(presentation_acc == 0)[0]) > 0:
                presentations_needed_to_learn[example_id] = np.where(
                    presentation_acc == 0)[0][-1] + 1
            else:
                presentations_needed_to_learn[example_id] = 0

            # Find the presentation at which the example was first learned, 
            # e.g. first presentation when acc is 1
            if len(np.where(presentation_acc == 1)[0]) > 0:
                first_learned[example_id] = np.where(
                    presentation_acc == 1)[0][0]
            else:
                first_learned[example_id] = np.nan

    return presentations_needed_to_learn, unlearned_per_presentation, first_learned


def sort_examples_by_forgetting(unlearned_per_presentation_all,
                                first_learned_all, npresentations):

    # Initialize lists
    example_original_order = []
    example_stats = []

    for example_id in unlearned_per_presentation_all[0].keys():

        # Add current example to lists
        example_original_order.append(example_id)
        example_stats.append(0)

        # Iterate over all training runs to calculate the total forgetting count for current example
        for i in range(len(unlearned_per_presentation_all)):

            # Get all presentations when current example was forgotten during current training run
            stats = unlearned_per_presentation_all[i][example_id]

            # If example was never learned during current training run, add max forgetting counts
            if np.isnan(first_learned_all[i][example_id]):
                example_stats[-1] += npresentations
            else:
                example_stats[-1] += len(stats)
    random_indices = np.random.permutation(len(example_original_order))

    print('Number of unforgettable examples: {}'.format(
        len(np.where(np.array(example_stats) == 0)[0])))
    # random order
    return np.array(example_original_order)[random_indices], np.array(example_stats)[random_indices]
    # reverse order
    # return np.array(example_original_order)[np.argsort(
    #     example_stats)[::-1]], np.sort(example_stats)[::-1]

    return np.array(example_original_order)[np.argsort(
        example_stats)], np.sort(example_stats)
    # origin order
    # return np.array(example_original_order), np.array(example_stats)