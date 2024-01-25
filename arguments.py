import os
import yaml


class DictProxy(object):

    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, key):
        return self._obj[key]

    def __setitem__(self, key, value):
        self._obj[key] = value

    def __getattr__(self, key):
        try:
            attr = getattr(self._obj, key, None)
            if attr is not None:
                return attr
            else:
                return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if key != '_obj':
            self._obj[key] = value
        else:
            super(DictProxy, self).__setattr__(key, value)

    def __iter__(self):
        return iter(self._obj)

    @property
    def instance(self):
        return self._obj

def wrap(name, cfg):
    var_name = f'{name}_config'
    globals()[var_name] = DictProxy(cfg)


CONFIG_PATH = 'config.yml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.CLoader)
    assert config


# load basic configs
for key, val in config.get('base', {}).items():
    globals()[key] = val
# load components configs
for category in config:
    if category == 'base': continue
    wrap(category, config[category])

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

api_config.url = f'{api_config.host}:{api_config.port}'