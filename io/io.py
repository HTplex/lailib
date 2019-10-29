def load_json(json_path):
    '''
    load one json file
    :param json_path: path to the json file, based on
    variable name the json file is assumed to be a
    dictionary.
    :return: load dictionary
    '''
    with open(json_path, 'r') as f:
        ret_dict = json.load(f)
    return ret_dict