import h5py
f = h5py.File("./test_2_evolving_memory_maps.h5", 'r')
for item in f.keys():
    print(item + ":", f[item])

for item in f.require_group('coord').keys():
    print('scores/'+ item)

for item in f.require_group('scores').keys():
    print('scores/'+ item)

for item in f['scores/step_123'].keys():
    item_path = 'scores/step_123/'+item
    print(item_path+':\t',f[item_path])

for item in f['coord'].keys():
    item_path = 'coord/'+item
    print(item_path+':\t',f[item_path])