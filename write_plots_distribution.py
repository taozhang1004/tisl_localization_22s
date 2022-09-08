import matplotlib.pyplot as plt
from ast import literal_eval
import seaborn as sns

def get_distributions_lst(rooms_path):
    with open(rooms_path, 'r') as sample_rooms_file:
        sample_rooms_lst = sample_rooms_file.read().splitlines()
    sample_rooms_file.close()
    return sample_rooms_lst


def make_visualizatons(distributions_path):
    data = get_distributions_lst(distributions_path)
    for i in range(0, len(data), 2):
        headings = data[i]
        distr = data[i+1]
        distr_to_dict = literal_eval(distr)
        labels, values = zip(*distr_to_dict.items())
        c = '#7eb54e'
        sns.set()
        plt.figure(figsize=(9,6))
        plt.title(headings, fontsize=16, fontweight='bold')
        plt.bar(labels, values, width=0.3, color = c)
        plt.xticks(rotation=45, fontsize=7)
        image_name = headings + 'plot.png'
        image_path = 'test_distribution_images/' + image_name
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        #plt.show()

def main():
    distributions_path = 'distribution_objects_test_set.txt'
    make_visualizatons(distributions_path)

main()