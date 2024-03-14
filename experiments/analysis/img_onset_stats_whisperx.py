import json
import matplotlib.pyplot as plt

splits = ['train', 'val', 'test']
stat_type = 'mean'  # ['mean', 'std', 'range']

for split in splits:
    print(split)
    count = 0
    stat_values = []

    with open(split + '_img_onset_stats_whisperx.json', 'r') as f:
        stats = json.load(f)

        for im in stats:
            print(im)
            print('mean', round(stats[im]['mean'], 3))
            print('std', round(stats[im]['std'], 3))
            print('range', round(stats[im]['range'], 3))
            count += 1
            print()

            stat_values.append(stats[im][stat_type])

    print()

    print('img_count', count, len(stat_values))

    plt.hist(stat_values)

    plt.title(split.upper())
    plt.xlabel(stat_type.upper())
    plt.ylabel('Frequency')
    plt.savefig(stat_type + '_onset_' + split + '_whisperx.png')
    plt.close()

