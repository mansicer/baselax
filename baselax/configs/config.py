import collections
import os
import yaml


def init_config(alg: str):
    def _load_yaml_file(path):
        try:
            content = yaml.load(open(path), Loader=yaml.CLoader)
        except:
            content = yaml.load(open(path), Loader=yaml.Loader)
        return content

    def recursive_dict_update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = recursive_dict_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    default_config = _load_yaml_file(os.path.join(os.path.dirname(__file__)), "default.yaml")
    alg_config = _load_yaml_file(os.path.join(os.path.dirname(__file__), alg + ".yaml"))
    config = recursive_dict_update(default_config, alg_config)
    return config
