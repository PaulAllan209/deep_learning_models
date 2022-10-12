location_list1 = ['Calamba', 'Cabuyao', 'Los banos', 'San pablo']

def return_dict_of_locations(location_list):
    returned_dict = {}
    for loc in location_list:
        returned_dict[loc] = location_list.index(loc)
        
    return returned_dict
        
print(return_dict_of_locations(location_list1))