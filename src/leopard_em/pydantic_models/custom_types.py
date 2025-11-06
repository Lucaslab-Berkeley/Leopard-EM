"""Additional type definitions and hints for Pydantic models."""

import json
import os
from typing import Annotated, Any, ClassVar, Optional

import torch
import yaml  # type: ignore
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

# Pydantic type-hint to exclude tensor from JSON schema/dump (still attribute)
ExcludedTensor = SkipJsonSchema[
    Annotated[Optional[torch.Tensor], Field(default=None, exclude=True)]
]


class BaseModel2DTM(BaseModel):
    """Implementation of a Pydantic BaseModel with additional, useful methods.

    Currently, only additional import/export methods are implemented and this
    class can effectively be treated as the `pydantic.BaseModel` class.

    Attributes
    ----------
    None

    Methods
    -------
    from_json(json_path: str | os.PathLike) -> BaseModel2DTM
        Load a BaseModel2DTM subclass from a serialized JSON file.
    from_yaml(yaml_path: str | os.PathLike) -> BaseModel2DTM
        Load a BaseModel2DTM subclass from a serialized YAML file.
    to_json(json_path: str | os.PathLike) -> None
        Serialize the BaseModel2DTM subclass to a JSON file.
    to_yaml(yaml_path: str | os.PathLike) -> None
        Serialize the BaseModel2DTM subclass to a YAML file.
    """

    model_config: ClassVar = ConfigDict(extra="forbid")

    #####################################
    ### Import/instantiation methods ###
    #####################################

    @classmethod
    def from_json(
        cls, json_path: str | os.PathLike, *args: tuple[Any], **kwargs: dict[Any, Any]
    ) -> "BaseModel2DTM":
        """Load a MatchTemplateManager from a serialized JSON file.

        Parameters
        ----------
        json_path : str | os.PathLike
            Path to the JSON file to load.
        *args : tuple[Any]
            Additional positional arguments to pass to the constructor.
        **kwargs : dict[Any, Any]
            Additional keyword arguments to pass to the constructor.

        Returns
        -------
        BaseModel2DTM
            Instance of the BaseModel2DTM subclass loaded from the JSON file.
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        return cls(*args, **data, **kwargs)

    @classmethod
    def from_yaml(
        cls, yaml_path: str | os.PathLike, *args: tuple[Any], **kwargs: dict[Any, Any]
    ) -> "BaseModel2DTM":
        """Load a MatchTemplateManager from a serialized YAML file.

        Parameters
        ----------
        yaml_path : str | os.PathLike
            Path to the YAML file to load.
        *args : tuple[Any]
            Additional positional arguments to pass to the constructor.
        **kwargs : dict[Any, Any]
            Additional keyword arguments to pass to the constructor.

        Returns
        -------
        BaseModel2DTM
            Instance of the BaseModel2DTM subclass loaded from the YAML file.
        """
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(*args, **data, **kwargs)

    ####################################
    ### Export/serialization methods ###
    ####################################

    def to_json(self, json_path: str | os.PathLike) -> None:
        """Serialize the MatchTemplateManager to a JSON file.

        Parameters
        ----------
        json_path : str | os.PathLike
            Path to the JSON file to save.

        Returns
        -------
        None
        """
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f)

    def to_yaml(self, yaml_path: str | os.PathLike) -> None:
        """Serialize the MatchTemplateManager to a YAML file.

        Parameters
        ----------
        yaml_path : str | os.PathLike
            Path to the YAML file to save.

        Returns
        -------
        None
        """
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f)
