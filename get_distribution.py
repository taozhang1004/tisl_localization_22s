import json
import os

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

    label_dict = {}
    #loop through all segGroup objects
    for object_dict in data["segGroups"]:
        #object_dict includes: objectId, id, partId, index, dominantNormal, obb, segments, label

        #get needed info
        label = object_dict["label"]

        if label in label_dict:
            label_dict[label] += 1
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

def main():
  sample_rooms_path = 'rooms_cleaned_test.txt'
  distribution_objects_file_path = 'distribution_objects_test_set.txt'
  sample_rooms_lst = get_sample_rooms_lst(sample_rooms_path)
  room = 0
  for scans in sample_rooms_lst:
    scans_lst = scans.split(" ")
    scans_lst = scans_lst[:-1]
    for scan in scans_lst:
      get_distribution_of_objects(scan, distribution_objects_file_path, room)
    room +=1 
  
main()
