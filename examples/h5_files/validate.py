import h5py
import json

# Open the HDF5 file
with h5py.File("./mmtest_memory_maps.h5", 'r') as f:
    # Print top-level groups
    for item in f.keys():
        print(item + ":", f[item])

    # Print the keys in the 'coord' group
    #for item in f.require_group('coord').keys():
    #    print('coord/' + item)

    # Print the keys in the 'scores' group
    #for item in f.require_group('scores').keys():
    #    print('scores/' + item)

    # Print the content of the 'coord' group
    #for item in f['coord'].keys():
    #    item_path = 'coord/' + item
    #    print(item_path + ':\t', f[item_path])

    # Print the content of the 'config' group
    #for item in f['config'].keys():
    #    item_path = 'config/' + item
    #    print(item_path + ':\t', f[item_path])
    #    
    #    # If the dataset contains JSON data, parse and print it
    #    if item == "config_data":
    #        config_json = f[item_path][()]  # Read the dataset
    #        config_dict = json.loads(config_json)  # Parse the JSON string
    #        print("Parsed Config:")
    #        for key, value in config_dict.items():
    #            print(f"  {key}: {value}")

    # Print the keys in the 'scores' group
    for item in f.require_group('scores').keys():
        item_path = 'scores/' + item

        read = f[item_path]
        print(read)