import matplotlib.pyplot as plt
import numpy as np


def visualize_images(img1, img2, similarity):
    fig, m_axs = plt.subplots(2, img1.shape[0], figsize=(12, 6))
    for c_a, c_b, c_d, (ax1, ax2) in zip(img1, img2, similarity, m_axs.T):
        ax1.imshow(c_a[:, :, 0])
        ax1.set_title('A')
        ax1.axis('off')
        ax2.imshow(c_b[:, :, 0])
        ax2.set_title('B\n %.2f%%' % (100 * c_d))
        ax2.axis('off')
    plt.show()


def visualize_accuracy(history, path):
    training_loss = history.history['accuracy']
    test_loss = history.history['val_accuracy']
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Acc', 'Validation Acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(path)
    plt.show()
    plt.close()


def gen_random_batch(in_groups, num_pairs, similarity):
    # Reference: https://www.kaggle.com/kmader/image-similarity-with-siamese-networks/notebook
    print('gen_random_batch')
    out_img_a, out_img_b, out_score = [], [], []
    all_groups = list(range(len(in_groups)))

    group_idx = np.random.choice(all_groups, size=num_pairs)
    out_img_a += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in group_idx]

    if similarity == 1:
        b_group_idx = group_idx
        out_score += [1] * num_pairs
    else:
        # anything but the same group
        non_group_idx = [np.random.choice([i for i in all_groups if i != c_idx]) for c_idx in group_idx]
        b_group_idx = non_group_idx
        out_score += [0] * num_pairs

    out_img_b += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in b_group_idx]

    return np.stack(out_img_a, 0), np.stack(out_img_b, 0), np.stack(out_score, 0)
