from __future__ import annotations
from typing import Any
from pathlib import Path
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod

from scipy.io import loadmat
import spectral
import numpy as np
import skimage

from src.utilities.lib import ImageFormatEnum, ImageArray


@dataclass(frozen=True)
class Image:
    data: ImageListHandler
    dim_order: tuple[int, int, int, int] = (0, 1, 2, 3)

    @classmethod
    def create(
        cls,
        filelist: list[Path],
        image_format: ImageFormatEnum = ImageFormatEnum.ENVI,
        dim_order: tuple[int, int, int, int] = (0, 1, 2, 3),
        options: ImageHandlerFactoryOptions = None,
    ) -> "Image":
        if options is None:
            options = ImageHandlerFactoryOptions()
        data = image_handler_factory(
            fmt=image_format,
            filelist=filelist,
            options=options,
        )
        return cls(data=data, dim_order=dim_order)

    def open(self) -> Image:
        data = self.data.open()
        return replace(self, data=data)

    def close(self) -> None:
        self.data.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.data_list()[0].shape + (len(self.data_list()),)

    def __getitem__(
            self, item: tuple[slice, slice, slice, slice]
    ) -> np.ndarray[Any, np.float32]:
        """Returns the requested data as list of acquisitions"""
        shape = self.shape
        item = tuple(elem for _, elem in sorted(zip(self.dim_order, item)))
        dim = []
        for itm, shp in zip(item, shape):
            start = itm.start if itm.start is not None else 0
            stop = itm.stop if itm.stop is not None else shp
            step = itm.step if itm.step is not None else 1
            dim.append(len(range(start, stop, step)))
        array = np.empty(dim, dtype=np.float32)
        data_list = self.data_list()[item[3]]
        for ii, data in enumerate(data_list):
            array[:, :, :, ii] = data[item[:3]]
        array = np.moveaxis(
            array, source=self.dim_order, destination=(0, 1, 2, 3),
        )
        return array

    def data_list(self) -> list[ImageArray]:
        """Retrieves the image handler"""
        return self.data.data_list()


class ImageListHandler(ABC):
    """Abstract class to handle the image list."""

    @abstractmethod
    def open(self) -> ImageListHandler:
        """
        Returns the handler of the image, whose type depends on the particular
        image format.
        """

    @abstractmethod
    def close(self) -> None:
        """Closes all the opened image handlers."""

    @abstractmethod
    def data_list(self) -> list[ImageArray]:
        """Returns a list of 3d arrays (one for each acquisition)."""


@dataclass(frozen=True)
class NumpyHandler(ImageListHandler):
    """Manages image lists in the CSV file format"""
    filelist: list[Path]
    _handler: list[np.ndarray] = None

    @classmethod
    def from_filelist(
        cls, filelist: list[Path]
    ) -> "NumpyHandler":
        return cls(filelist=filelist)

    def open(self) -> "NumpyHandler":
        handler = [np.load(f"{file}", mmap_mode='r') for file in self.filelist]
        return replace(self, _handler=handler)

    def close(self) -> None:
        pass

    def handler(self) -> list[np.ndarray]:
        """Returns the handler for each file in the list."""
        if self._handler is not None:
            return self._handler
        raise IOError("File is not open")

    def data_list(self) -> list[ImageArray]:
        return [np.atleast_3d(arr) for arr in self.handler()]


@dataclass(frozen=True)
class SkimageHandler(ImageListHandler):
    """Manages image lists in the CSV file format"""
    filelist: list[Path]
    _handler: list[np.ndarray] = None

    @classmethod
    def from_filelist(
        cls, filelist: list[Path]
    ) -> "SkimageHandler":
        return cls(filelist=filelist)

    def open(self) -> "SkimageHandler":
        handler = [skimage.io.imread(f"{file}") for file in self.filelist]
        return replace(self, _handler=handler)

    def close(self) -> None:
        pass

    def handler(self) -> list[np.ndarray]:
        """Returns the handler for each file in the list."""
        if self._handler is not None:
            return self._handler
        raise IOError("File is not open")

    def data_list(self) -> list[ImageArray]:
        return [np.atleast_3d(arr) for arr in self.handler()]


@dataclass(frozen=True)
class EnviHandler(ImageListHandler):

    filelist: list[Path]
    _handler: list[spectral.SpyFile] = None

    @classmethod
    def from_filelist(cls, filelist: list[Path]) -> "EnviHandler":
        return cls(filelist=filelist)

    def open(self) -> "EnviHandler":
        handler = []
        for file in self.filelist:
            handler.append(spectral.open_image(f"{file}"))
        return replace(self, _handler=handler)

    def close(self) -> None:
        for handler in self.handler():
            handler.fid.close()

    def data_list(self) -> list[ImageArray]:
        data_list = []
        for handler in self.handler():
            image = handler.open_memmap()
            data_list.append(np.rot90(image, k=-1, axes=(0, 1)))
        return data_list

    def handler(self) -> list[spectral.SpyFile]:
        """Returns the handler for each file in the list."""
        if self._handler is not None:
            return self._handler
        raise IOError("File is not open")


@dataclass(frozen=True)
class MatFileHandler(ImageListHandler):
    """Manages image lists in the CSV file format"""
    filelist: list[Path]
    key: str
    _handler: list[np.ndarray] = None

    @classmethod
    def from_filelist(
        cls, filelist: list[Path], key: str,
    ) -> "MatFileHandler":
        return cls(filelist=filelist, key=key)

    def open(self) -> "MatFileHandler":
        handler = [loadmat(f"{file}")[self.key] for file in self.filelist]
        return replace(self, _handler=handler)

    def close(self) -> None:
        pass

    def handler(self) -> list[np.ndarray]:
        """Returns the handler for each file in the list."""
        if self._handler is not None:
            return self._handler
        raise IOError("File is not open")

    def data_list(self) -> list[ImageArray]:
        return [np.atleast_3d(arr) for arr in self.handler()]


@dataclass
class ImageHandlerFactoryOptions:
    key: str = None
    index_start: int = None
    index_stop: int = None

    def check_validity(self, fmt: ImageFormatEnum) -> None:
        if fmt == ImageFormatEnum.MAT_FILE and self.key is None:
            raise ValueError("key field is required for MAT file handler.")


def image_handler_factory(
        fmt: ImageFormatEnum,
        filelist: list[Path],
        options: ImageHandlerFactoryOptions,
) -> ImageListHandler:
    """Factory pattern method that returns an image list handler"""
    filelist.sort()
    options.check_validity(fmt=fmt)
    if fmt == ImageFormatEnum.ENVI:
        return EnviHandler.from_filelist(filelist=filelist)
    elif fmt == ImageFormatEnum.SKIMAGE:
        return SkimageHandler.from_filelist(filelist=filelist)
    elif fmt == ImageFormatEnum.MAT_FILE:
        return MatFileHandler.from_filelist(filelist=filelist, key=options.key)
    else:
        raise ValueError("Image format unknown")
