from __future__ import annotations

import json
from hashlib import sha512
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from pydantic import BaseModel
from typing_extensions import Self


class _CyFunctionDetectorMeta(type):
    def __instancecheck__(self, instance):
        return instance.__class__.__name__ == "cython_function_or_method"


class CythonFunctionDetector(metaclass=_CyFunctionDetectorMeta):
    """Cython function detector for solution to pydantic/cython issue:
    https://github.com/pydantic/pydantic/issues/6670#issuecomment-1644799636"""


class HashableBaseModelIO(BaseModel, from_attributes=True, ignored_types=(CythonFunctionDetector,)):
    """Input/output utilities for the models with support for the following features:

    - Hashing of the model
    - Conversion to and from dictionaries, json, toml, and yaml files
    - Compatibility with pydantic v1 and v2
    """

    @property
    def exclude(self) -> set[int] | set[str] | dict[int, Any] | dict[str, Any] | None:
        """Fields to exclude from the model, typically used to exclude arbitrary types when it is allowed in the
        pydantic model to avoid hashing issues."""
        return None

    def __hash__(self) -> int:
        """Return the hash of the model."""
        string = f"{self.__class__.__qualname__}::{self.model_dump_json(exclude=self.exclude)}"
        return int.from_bytes(sha512(string.encode("utf-8", errors="ignore")).digest())

    def model_dump(self, **kwargs) -> dict[str, Any]:
        try:
            return super().model_dump(**kwargs)
        except AttributeError:
            return self.dict(**kwargs)

    def model_dump_json(self, **kwargs) -> str:
        try:
            return super().model_dump_json(**kwargs)
        except AttributeError:
            return self.json(**kwargs)

    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> Self:
        try:
            return super().model_validate(obj, **kwargs)
        except AttributeError:
            return super().parse_obj(obj, **kwargs)

    def update(self, data: dict = None, **kwargs) -> Self:
        """Update the options of the optimizer"""
        data = dict(data or {}, **kwargs)
        update = self.dict()
        update.update(data)
        for k, v in self.validate(update).dict(exclude_defaults=True).items():
            setattr(self, k, v)
        return self

    @classmethod
    def fromDict(cls, *, data: dict, **kwargs) -> Self:
        """Load the model from a dictionary."""
        return cls(**data, **kwargs)

    def toDict(self, **kwargs) -> dict:
        """Convert the model to a dictionary."""
        try:
            return self.model_dump(**kwargs)
        except AttributeError:
            return self.dict(**kwargs)  # noqa

    @classmethod
    def fromJson(cls, *, string: str | None = None, path: str | None = None, **kwargs):
        """Load the model from a json string or file if a path is provided."""
        assert string is not None or path is not None, "Either a string or a path must be provided."
        if string is None:
            with open(path, "r+", encoding="utf-8") as f:
                string = f.read()
        return cls.fromDict(data=json.loads(string), **kwargs)

    def toJson(self, path: Path | str | None = None, **kwargs) -> str | None:
        """Convert the model to a json string or save it to a file if a path is provided."""
        string = json.dumps(self.toDict(**kwargs), indent=4)
        if path is not None:
            with open(path, "w+", encoding="utf-8") as f:
                f.write(string)
        else:
            return string

    @classmethod
    def fromToml(cls, *, string: str | None = None, path: str | None = None, **kwargs):
        """Load the model from a toml string or file if a path is provided."""
        try:
            import tomllib as toml  # Python >= 3.11
        except ImportError:
            import tomli as toml  # noqa Python < 3.11

        assert string is not None or path is not None, "Either a string or a path must be provided."
        if string is None:
            with open(path, "r+", encoding="utf-8") as f:
                string = f.read()
        return cls.fromDict(data=toml.loads(string), **kwargs)

    def toToml(self, path: Path | str | None = None, **kwargs) -> str | None:
        """Convert the model to a toml string or save it to a file if a path is provided."""
        import tomli_w as toml

        string = toml.dumps(self.toDict(**kwargs))
        if path is not None:
            with open(path, "w+", encoding="utf-8") as f:
                f.write(string)
        else:
            return string

    @classmethod
    def fromYaml(cls, *, string: str | None = None, path: str | None = None, **kwargs):
        """Load the model from a yaml string or file if a path is provided."""
        import yaml

        assert string is not None or path is not None, "Either a string or a path must be provided."
        if string is None:
            with open(path, "r+", encoding="utf-8") as f:
                string = f.read()
        return cls.fromDict(data=yaml.safe_load(string), **kwargs)

    def toYaml(self, path: Path | str | None = None, **kwargs) -> str | None:
        """Convert the model to a yaml string or save it to a file if a path is provided."""
        import yaml

        string = yaml.safe_dump(self.toDict(**kwargs))
        if path is not None:
            with open(path, "w+", encoding="utf-8") as f:
                f.write(string)
        else:
            return string

    @classmethod
    def fromBytes(cls, *, binary: bytes | None = None, path: str | None = None, **kwargs) -> Self:
        """Load the model from a binary string or file if a path is provided."""
        import pickle

        assert binary is not None or path is not None, "Either a binary string or a path must be provided."
        if binary is None:
            with open(path, "rb+") as f:
                binary = f.read()
        return cls.fromDict(data=pickle.loads(binary), **kwargs)

    def toBytes(self, path: Path | str | None = None, **kwargs) -> bytes | None:
        """Convert the model to a binary string or save it to a file if a path is provided."""
        import pickle

        binary = pickle.dumps(self.toDict(**kwargs))
        if path is not None:
            with open(path, "wb+") as f:
                f.write(binary)
        else:
            return binary

    @classmethod
    def fromCryptography(cls, *, binary: bytes | None = None, path: str | None = None, **kwargs) -> Self:
        """Decrypt the model using the key."""
        assert binary is not None or path is not None, "Either a binary string or a path must be provided."
        if binary is None:
            with open(path, "rb") as f:
                binary = f.read()
        key = b"2I-1wc1I4gKoJ32bGBLudvxrBCdLsQjdeoRYQFekv7A="
        return cls.fromJson(string=Fernet(key).decrypt(binary).decode("utf-8"), **kwargs)

    def toCryptography(self, path: Path | str | None = None, **kwargs) -> bytes | None:
        """Encrypt the model using the key."""
        key = b"2I-1wc1I4gKoJ32bGBLudvxrBCdLsQjdeoRYQFekv7A="
        binary = Fernet(key).encrypt(self.toJson(**kwargs).encode("utf-8"))
        if path is not None:
            with open(path, "wb+") as f:
                f.write(binary)
        else:
            return binary
