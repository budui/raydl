import importlib
import inspect
from copy import deepcopy
from typing import MutableMapping, Optional, Union

__all__ = ["initialize", "import_object"]


def import_object(object_path: str, reload_module=False):
    """
    Dynamically import a python object(e.g. class, function, ...) from a string reference if it's location.

    - `object_path`: The path to the Python object to import.
    Example: `"functools.lru_cache"`, `"torchvision.transforms.Resize"`
    """

    assert "." in object_path
    module, class_name = object_path.rsplit(".", 1)

    try:
        module_imp = importlib.import_module(module)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"failed to import `{object_path}`: {e}") from None

    if reload_module:
        importlib.reload(module_imp)
    try:
        cls = getattr(module_imp, class_name)
    except AttributeError as e:
        raise AssertionError(f"import module `{module}` over, but failed import class `{class_name}`: {e}") from None
    return cls


def _initialize(obj_type, arguments):
    if isinstance(obj_type, str):
        obj_cls = import_object(obj_type)
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"`obj_type` must be a class name of valid class, " f"but got {type(obj_type)}")

    try:
        obj = obj_cls(**arguments)
    except Exception as e:
        raise TypeError(f"Invalid arguments in {arguments} while " f"building class {obj_cls}") from e

    return obj


def initialize(
    config: Union[str, MutableMapping],
    default_arguments: Optional[MutableMapping] = None,
):
    """
    initialize a class instance with `config`.

    - `config` can be as class "location.name" str or a Mapping.
    `import_object` will be used to import desired class.

    - `default_arguments` will be used as the kwargs to initialize this class.

    Example:

    1. `config` as a str: `torchvision.models.resnet50`
    2. `config` as a dict: `dict(_type="torchvision.models.resnet50", pretrained=True)` or
      `{"torchvision.models.resnet50": dict(pretrained=True)}`

    `default_arguments` can be used to provide public arguments no matter what `config` you use:

    ```python
    raydl.initialize(config="torch.optim.SGD", default_arguments=dict(lr=0.0001))
    raydl.initialize(config="torch.optim.Adam", default_arguments=dict(lr=0.0001))
    ```

    """
    # get `obj_type``, `arguments`
    if isinstance(config, MutableMapping):
        if "_type" in config:
            arguments = deepcopy(config)
            obj_type = arguments.pop("_type")
            arguments = dict(arguments.items())
        elif len(config) == 1:
            obj_type, arguments = tuple(config.items())[0]
            arguments = dict(arguments.items())
        else:
            raise ValueError(
                "Invalid `config`, Mapping `config` must contain "
                "the type information by dict(_type=CLASS_NAME, ...arguments) "
                f"or dict(CLASS_NAME=dict(...arguments)), but got {config}"
            )
    elif isinstance(config, str):
        obj_type, arguments = config, {}
    else:
        raise TypeError(
            f"`config` must be a str or a `MutableMapping` like dict," f" but got {config} with type {type(config)}"
        )

    if not (isinstance(default_arguments, MutableMapping) or default_arguments is None):
        raise TypeError("default_args must be a dict or None, " f"but got {type(default_arguments)}")
    if default_arguments is not None:
        for k, v in default_arguments.items():
            arguments.setdefault(k, v)

    return _initialize(obj_type, arguments)
