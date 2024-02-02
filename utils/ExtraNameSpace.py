from typing import Callable


class NameSpace:
    __instance = None
    _args = None

    def __init__(self):
        if self.__instance is None:
            self.function_map = dict()
            NameSpace.__instance = self
        else:
            raise Exception("cannot instantiate a virtual Namespace again")

    @staticmethod
    def get_instance():
        if NameSpace.__instance is None:
            NameSpace()
        return NameSpace.__instance

    @classmethod
    def register(self, fn_name: str):
        def decorator(fn):
            func = Function(fn=fn, space_cls=self)
            name = func.register_key(fn_name=fn_name)  # 构造注册命名空间的name
            cls = DatasetsReaderNameSpace.get_instance()
            cls.function_map[name] = fn
            return func

        return decorator

    @classmethod
    def get(self, fn: Callable) -> Callable:
        cls = DatasetsReaderNameSpace.get_instance()

        func = Function(fn=fn, space_cls=self)
        fn_name = cls._args.dataset
        name = func.register_key(fn_name=fn_name)

        fn = cls.function_map.get(name)

        if not fn:
            fn_name = "Default"
            name = func.register_key(fn_name=fn_name)
            fn = cls.function_map.get(name)

        return fn

class Function(object):
    def __init__(self, fn: Callable, space_cls):
        self.fn = fn
        self.space_cls = space_cls

    def __call__(self, *args, **kwargs):
        fn = self.space_cls.get(self.fn)
        if not fn:
            raise Exception("no matching function found.")
        # invoking the wrapped function and returning the value.
        return fn(*args, **kwargs)

    def register_key(self, fn_name=None):
        return tuple([
            # self.fn.__module__,
            self.fn.__class__,
            self.fn.__name__, # 这个key目前无意义，但似乎不需要额外继承出DateReaderFunction之类的类
            self.space_cls.__name__,
            fn_name
        ])


class DatasetsReaderNameSpace(NameSpace):
    pass


class PredictionCleanNameSpace(NameSpace):
    pass

class ColdStartScoreNameSpace(NameSpace):
    pass

class RuleExtractionNameSpace(NameSpace):

    @classmethod
    def get(self, fn: Callable) -> Callable:
        cls = DatasetsReaderNameSpace.get_instance()

        func = Function(fn=fn, space_cls=self)
        fn_name = cls._args.cot_trigger_type
        name = func.register_key(fn_name=fn_name)

        fn = cls.function_map.get(name)

        if not fn:
            fn_name = "Default"
            name = func.register_key(fn_name=fn_name)
            fn = cls.function_map.get(name)

        return fn