import dataclasses
import json
import pickle
import sqlite3
import threading
import time
from abc import abstractstaticmethod
from collections import OrderedDict
from io import BytesIO
from typing import Any, Optional, Union, cast

import numpy as np
import PIL.Image


class TypeMapper:
    def __init__(self, sqlite3_type: str, py_type: Union[object, type]) -> None:
        self.sqlite3_type = sqlite3_type
        self.py_type = py_type

    @abstractstaticmethod
    def serialize(obj) -> Union[None, str, int, float, bytes]:
        ...

    @abstractstaticmethod
    def unserialize(val) -> Any:
        ...


class JSONTypeMapper(TypeMapper):
    def __init__(self) -> None:
        super().__init__("TEXT", py_type=Union[str, int, float, bool, None, dict[str, Any], list[Any]])

    @staticmethod
    def serialize(obj):
        return json.dumps(obj)

    @staticmethod
    def unserialize(val):
        return json.loads(val)


class NumpyArrayTypeMapper(TypeMapper):
    def __init__(self) -> None:
        super().__init__("BLOB", np.ndarray)

    @staticmethod
    def serialize(obj):
        buff = BytesIO()
        np.save(buff, obj)
        return buff.getvalue()

    @staticmethod
    def unserialize(val):
        return np.load(BytesIO(val))


class PickleTypeMapper(TypeMapper):
    def __init__(self) -> None:
        super().__init__("BLOB", Any)

    @staticmethod
    def serialize(obj):
        buff = BytesIO()
        pickle.dump(obj, buff)
        return buff.getvalue()

    @staticmethod
    def unserialize(val):
        return pickle.load(BytesIO(val))


class PILImageTypeMapper(TypeMapper):
    def __init__(self) -> None:
        super().__init__("BLOB", PIL.Image.Image)

    @staticmethod
    def serialize(obj: PIL.Image.Image):
        buff = BytesIO()
        obj.save(buff, format="webp")
        return buff.getvalue()

    @staticmethod
    def unserialize(val):
        return PIL.Image.open(BytesIO(val))


def find_type_mapper(_type) -> Optional[TypeMapper]:
    if _type in [dict, list]:
        return JSONTypeMapper()
    if _type == PIL.Image.Image:
        return PILImageTypeMapper()
    if _type == np.ndarray:
        return NumpyArrayTypeMapper()
    return None


def guess_type_mappers(data_class) -> dict:
    assert dataclasses.is_dataclass(data_class)

    type_mappers = dict()

    for f in dataclasses.fields(data_class):
        if f.type in DataclassSFDB.supported_type:
            continue

        guessed_type_mapper = find_type_mapper(f.type)
        if guess_type_mappers is not None:
            type_mappers[f.name] = guessed_type_mapper

    return type_mappers


class _Database:
    def __init__(self, filename, init_sql):
        self._filename = filename
        self._sqlite: sqlite3.Connection = sqlite3.connect(self._filename, check_same_thread=False)
        self._sqlite.execute(init_sql)
        self._sqlite.commit()
        self._lock = threading.Lock()
        self._iterating = False
        self._commit_timer = time.time()
        self._commit_counter = 0

        self._is_close = False

        self.__max_commit_waiting_time = 60
        self.__max_commit_waiting_updates = 1024 * 16

    @property
    def is_close(self):
        return self._is_close

    def _sanity_check(self):
        assert not self._is_close, f"SFDB[{self._filename}] Database already closed."
        assert not self._iterating, f"SFDB[{self._filename}] Database cannot be accessed inside an iterating loop."

    def _key_is_str(self, key):
        assert isinstance(
            key, str
        ), f'SFDB[{self._filename}] All keys must be str, get "{type(key).__name__}" instead.'

    def __len__(self):
        self._sanity_check()
        with self._lock:
            x = self._sqlite.execute("SELECT COUNT(ID) FROM DATA").fetchone()
            return x[0] if x is not None else 0

    def __contains__(self, key):
        self._sanity_check()
        self._key_is_str(key)
        with self._lock:
            return self._sqlite.execute("SELECT 1 FROM DATA WHERE ID = ?", (key,)).fetchone() is not None

    def __delitem__(self, key):
        self._sanity_check()
        self._key_is_str(key)
        with self._lock:
            self._sqlite.execute("DELETE FROM DATA WHERE ID = ?", (key,))
            self._commit_counter += 1
        self._auto_commit()
        return

    def commit(self):
        self._sanity_check()
        with self._lock:
            self._sqlite.commit()

            self._commit_timer = time.time()
            self._commit_counter = 0
        return

    def close(self):
        if self._sqlite is None:
            return
        self._sanity_check()
        self.commit()
        with self._lock:
            self._sqlite.close()
            self._is_close = True
        return

    def _auto_commit(self):
        if (
            time.time() > self._commit_timer + self.__max_commit_waiting_time
            or self._commit_counter > self.__max_commit_waiting_updates
        ):
            self.commit()
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return

    def __del__(self):
        self.close()
        return

    def keys(self):
        self._sanity_check()
        with self._lock:
            return [x[0] for x in self._sqlite.execute("SELECT ID FROM DATA").fetchall()]


class DataclassSFDB(_Database):
    supported_type = {int: "INTEGER", str: "TEXT", float: "REAL", bytes: "BLOB"}

    def __init__(
        self,
        filename: str,
        schema_dataclass,
        type_mappers: Optional[dict[str, TypeMapper]] = None,
        enable_mapper_guess=False,
    ):

        assert dataclasses.is_dataclass(schema_dataclass) and isinstance(schema_dataclass, type)

        self.schema = schema_dataclass

        guessed_type_mappers = guess_type_mappers(schema_dataclass) if enable_mapper_guess else dict()
        type_mappers = dict() if type_mappers is None else type_mappers
        type_mappers.update(guessed_type_mappers)
        self.type_mappers = type_mappers

        self._fields = self._schema_dataclass_to_fields(schema_dataclass)

        init_sql = (
            "CREATE TABLE IF NOT EXISTS DATA( ID TEXT NOT NULL UNIQUE, {}, "
            "last_update TIMESTAMP DEFAULT (datetime('now','localtime')) NOT NULL, PRIMARY KEY (ID))"
        )
        init_sql = init_sql.format(", ".join(f"{name} {_t} NOT NULL" for name, _t in self._fields.items()))

        super().__init__(filename, init_sql)

    def _schema_dataclass_to_fields(self, schema_dataclass):
        assert dataclasses.is_dataclass(
            schema_dataclass
        ), f"schema_dataclass must be dataclass, get {type(schema_dataclass).__name__} instead."

        fields = OrderedDict()

        for f in dataclasses.fields(schema_dataclass):
            if f.name in self.type_mappers:
                fields[f.name] = self.type_mappers[f.name].sqlite3_type
            else:
                assert (
                    f.type in self.supported_type
                ), f"{f.name} must be type in {self.supported_type}, but got {type(f.type)}"

                fields[f.name] = self.supported_type[f.type]
        return fields

    def fields(self):
        return ", ".join(self._fields)

    def _dataclass_from(self, *values):
        values = list(values)
        for i, (name, value) in enumerate(zip(self._fields, values)):
            if name in self.type_mappers:
                values[i] = self.type_mappers[name].unserialize(value)

        return self.schema(*values)

    def _dataclass_to(self, obj) -> list:
        assert isinstance(obj, self.schema), f"value must be a instance of {self.schema}, but got {type(obj)}"
        values = list(dataclasses.astuple(obj))
        for i, (name, value) in enumerate(zip(self._fields, values)):
            if name in self.type_mappers:
                values[i] = self.type_mappers[name].serialize(value)
        return values

    def __getitem__(self, key):
        self._sanity_check()
        self._key_is_str(key)
        with self._lock:
            item = self._sqlite.execute(f"SELECT {self.fields()} FROM DATA WHERE ID = ?", (key,)).fetchone()
        if item is None:
            raise KeyError(key)
        return self._dataclass_from(*item)

    def get(self, key, default=None):
        self._sanity_check()
        self._key_is_str(key)
        with self._lock:
            item = self._sqlite.execute(f"SELECT {self.fields()} FROM DATA WHERE ID = ?", (key,)).fetchone()
        return self._dataclass_from(*item) if item is not None else default

    def __setitem__(self, key, value):
        self._sanity_check()
        self._key_is_str(key)
        feed = (key, *self._dataclass_to(value))
        with self._lock:
            placeholders = ", ".join(["?"] * (1 + len(self._fields)))
            self._sqlite.execute(
                f"INSERT OR REPLACE INTO DATA(ID, {self.fields()}, last_update)"
                f" VALUES({placeholders}, datetime('now','localtime'))",
                feed,
            )
            self._commit_counter += 1
        self._auto_commit()
        return

    def __iter__(self):
        self._sanity_check()
        with self._lock:
            try:
                self._iterating = True
                for item in self._sqlite.execute(f"SELECT ID, {self.fields()} FROM DATA"):
                    yield item[0], self.schema(*item[1:])
            finally:
                self._iterating = False

    def todict(self):
        self._sanity_check()
        with self._lock:
            return {
                x[0]: self._dataclass_from(*x[1:])
                for x in self._sqlite.execute(f"SELECT ID, {self.fields()} FROM DATA").fetchall()
            }

    def tolist(self):
        self._sanity_check()
        with self._lock:
            return [
                (x[0], self._dataclass_from(*x[1:]))
                for x in self._sqlite.execute(f"SELECT ID, {self.fields()} FROM DATA").fetchall()
            ]


class KVSFDB(DataclassSFDB):
    def __init__(self, filename: str, value_type):
        assert value_type in self.supported_type or isinstance(value_type, TypeMapper)
        vtype = cast(type, value_type.py_type if isinstance(value_type, TypeMapper) else value_type)

        schema_dataclass = dataclasses.make_dataclass("KVData", fields=[("value", vtype)])
        type_mappers = None if not isinstance(value_type, TypeMapper) else dict(value=value_type)

        super().__init__(filename, schema_dataclass, type_mappers)

    def _dataclass_from(self, *values):
        obj = super()._dataclass_from(*values)
        return obj.value

    def _dataclass_to(self, obj) -> list:
        obj = self.schema(value=obj)
        return super()._dataclass_to(obj)


class JsonSFDB(KVSFDB):
    def __init__(self, filename: str):
        super().__init__(filename, JSONTypeMapper())
