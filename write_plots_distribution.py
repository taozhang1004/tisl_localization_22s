import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from ast import literal_eval

syn_dict = {'desk': 'table', 
          'coffee table' : 'table',
          'side table' : 'table', 
          'couch table' : 'table',
          'stand': 'table', 
          'nightstand' : 'table',
          'tv stand': 'table',
          'armchair': 'chair', 
          'dining chair' : 'chair',
          'desk chair' : 'chair',
          'stool' : 'chair', 
          'sofa': 'couch',
          'bath cabinet' : 'kitchen cabinet', 
          'showcase': 'kitchen cabinet', 
          'wardrobe': 'kitchen cabinet', 
          'cupboard': 'kitchen cabinet',
          'cushion':'pillow', 
          'shower curtain': 'curtain', 
          'monitor':'tv'
          }

#load entire json file for raw data
def load_json(name):
    with open(name) as f:
      data = json.load(f)
    return data

def get_sample_rooms_lst(sample_rooms_path):
    with open(sample_rooms_path, 'r') as sample_rooms_file:
        sample_rooms_lst = sample_rooms_file.read().splitlines()
    sample_rooms_file.close()
    return sample_rooms_lst


def extract_segGroup_objects(data):

    label_dict = {'sink': 0, 'microwave': 0, 'bottle': 0, 'oven': 0, 'regrigerator': 0, 'plant':0, 'toilet':0, 'clock':0,
            'bag':0, 'vase':0, 'telephone':0, 'backpack':0, 'laptop':0, 'bed':0, 'book':0, 'suitcase':0, 'bicycle':0,
            'clothes':0, 'shoes':0, 'shower':0, 'bathtub':0, 'plate':0, 'pillar':0, 'lamp':0, 'kettle':0, 'table':0,
            'chair':0, 'couch':0, 'kitchen cabinet':0, 'pillow':0, 'curtain':0, 'tv':0}
    #loop through all segGroup objects
    for object_dict in data["segGroups"]:
        #object_dict includes: objectId, id, partId, index, dominantNormal, obb, segments, label

        #get needed info
        label = object_dict["label"]

        if label in label_dict:
            label_dict[label] += 1
        else:
            if label in syn_dict:
              main_name = syn_dict.get(label)
              label_dict[main_name] += 1
            else:
              label_dict[label] = 1
    return label_dict

def get_distribution_of_objects(scan, distribution_file_path, room):
  rootdir = '3rscan'
  directory = os.path.join(rootdir, scan)
  scan_name = directory.split("/")[-1]
  for filename in os.listdir(directory):
    if 'semseg.v2.json' in filename:
      filename_extended = directory + '/' + filename
      data = load_json(filename_extended)
      label_dict = extract_segGroup_objects(data)
      with open(distribution_file_path, 'a') as distribution_objects_file:
        distribution_objects_file.write('room:')
        distribution_objects_file.write(str(room))
        distribution_objects_file.write(' ')
        distribution_objects_file.write('scan:')
        distribution_objects_file.write(' ')
        distribution_objects_file.write(scan_name)
        distribution_objects_file.write('\n')
        distribution_objects_file.write(str(label_dict))
        distribution_objects_file.write('\n')
      distribution_objects_file.close()


def get_distributions_lst(rooms_path):
    with open(rooms_path, 'r') as sample_rooms_file:
        sample_rooms_lst = sample_rooms_file.read().splitlines()
    sample_rooms_file.close()
    return sample_rooms_lst


def make_visualizatons(distributions_path, image_p):
    data = get_distributions_lst(distributions_path)
    for i in range(0, len(data), 2):
        headings = data[i]
        distr = data[i+1]
        distr_to_dict = literal_eval(distr)
        labels, values = zip(*distr_to_dict.items())
        c = '#7eb54e'
        sns.set()
        plt.figure(figsize=(20,10))
        plt.title(headings, fontsize=16, fontweight='bold')
        plt.bar(labels, values, width=0.3, color = c)
        plt.xticks(rotation=45, fontsize=7)
        image_name = headings + 'plot.png'
        image_path = '/' + image_name
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        #plt.show()

def get_distribution(sample_rooms_lst, distribution_objects_file_path):
    room_num = 0
    for scans in sample_rooms_lst:
        scans_lst = scans.split(" ")
        scans_lst = scans_lst[:-1]
        for scan in scans_lst:
            get_distribution_of_objects(scan, distribution_objects_file_path, room_num)
        room_num +=1 

def main():
    sample_rooms_path = 'rooms_cleaned_test.txt'
    distribution_objects_file_path = 'distribution_objects_test_set_original_synonyms.txt'
    image_path = 'test_distribution_images'
    sample_rooms_lst = get_sample_rooms_lst(sample_rooms_path)
    get_distribution(sample_rooms_lst, distribution_objects_file_path)
    make_visualizatons(distribution_objects_file_path, image_path)

main()
