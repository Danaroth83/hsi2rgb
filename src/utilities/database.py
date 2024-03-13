from typing import Optional, Union, Sequence
from pathlib import Path
import csv
from dataclasses import dataclass
import json

import huggingface_hub
import numpy as np
from pydantic import BaseModel, RootModel, ConfigDict, model_validator, field_validator
import pint

from src.utilities.lib import ImageFormatEnum
from src.utilities.image import Image, ImageHandlerFactoryOptions
from src.utilities.validators import regex_file_list, is_id_unique, is_valid_unit, get_from_id


class DatasetSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str
    path: Path
    url: str
    wavelengths_file: Path
    wavelengths_unit: str = "nm"
    device_id: str = "unknown"
    bit_depth: Union[int, None] = None
    dim_order: tuple[int, int, int, int] = (0, 1, 2, 3)
    rgb: tuple[int, int, int] = (0, 1, 2)

    @field_validator("wavelengths_unit", mode="after")
    def check_linear_unit(cls, v: str):
        if not is_valid_unit(unit=v, dimensionality="[length]"):
            raise ValueError("Not a valid length unit")
        return v

    @field_validator("path", "wavelengths_file", mode="after")
    def make_path_absolute(cls, v: Path) -> Path:
        project_path = Path(__file__).resolve().parents[2]
        return project_path / v

    @field_validator("dim_order", mode="after")
    def verify_dim_order_fields(
            cls, v: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        valid_v = set(range(4))
        v_set = set(v)
        if not (len(v_set) == len(valid_v) and v_set.issubset(valid_v)):
            raise ValueError("dim_order must contain non-repeating integers from 0 to 3")
        return v

    @property
    def max_intensity(self) -> Union[int, None]:
        if self.bit_depth is None:
            return None
        return 2 ** self.bit_depth - 1

    def wavelengths(
            self,
            excluded_bands: list[int] = None,
    ) -> np.ndarray[tuple[int], np.float32]:
        """Returns a wavelength vector expressed in nanometers"""
        data = []
        with open(self.wavelengths_file, "r") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                data.append(float(row[0]))
        array = np.array(data, dtype=np.float32)
        if excluded_bands is not None:
            array = np.delete(array, excluded_bands)
        scale = pint.Quantity(1, self.wavelengths_unit).to("nm").magnitude
        return array * scale

    def download(
            self,
            verify_files: bool = False,
            force_download: bool = False,
    ) -> None:
        if not self.path.exists():
            verify_files = True
            self.path.mkdir(parents=False, exist_ok=True)
        _ = huggingface_hub.snapshot_download(
            repo_id=f"{self.url}",
            repo_type="dataset",
            local_dir=f"{self.path}",
            local_files_only=not verify_files,
            force_download=force_download,
        )


class DatasetListSchema(Sequence, RootModel):
    root: list[DatasetSchema]

    @field_validator("root", mode="after")
    def is_id_unique(cls, val):
        assert is_id_unique(obj=val, attrib="id")
        return val

    def __getitem__(self, item: str) -> DatasetSchema:
        return get_from_id(obj=self.root, index=item, attrib="id")

    def __len__(self) -> int:
        return len(self.root)


class AcquisitionSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: int
    dataset_id: str
    file_name: Optional[str] = None
    pattern: Optional[str] = None
    format: ImageFormatEnum
    excluded_bands: Optional[list[tuple[int, int]]] = None
    key: Optional[str] = None
    index_start: Optional[int] = None
    index_stop: Optional[int] = None

    @model_validator(mode="after")
    def check_format(self) -> "AcquisitionSchema":
        if self.format == ImageFormatEnum.MAT_FILE and self.key is None:
            raise ValueError("Field key is required for mat files")
        return self

    @model_validator(mode="after")
    def check_pattern_or_file_name(self) -> "AcquisitionSchema":
        if self.file_name is None and self.pattern is None:
            raise ValueError("Either field file_name or pattern required.")
        return self

    @field_validator("excluded_bands", mode="after")
    def fix_exclude_bands(
            cls, val: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        if val is None:
            val = []
        return val

    def filelist(self, path: Path) -> list[Path]:
        if self.file_name is not None:
            return [path / self.file_name]
        else:
            return regex_file_list(path, self.pattern)

    def excluded_bands_list(self) -> list[int]:
        if self.excluded_bands is None:
            return []
        excluded_bands = set()
        for band in self.excluded_bands:
            excluded_bands = excluded_bands.union(set(range(*band)))
        return list(excluded_bands)

    def wavelengths(
            self,
            dataset: DatasetSchema,
    ) -> np.ndarray[tuple[int], np.float32]:
        wavelengths = dataset.wavelengths(
            excluded_bands=self.excluded_bands_list()
        )
        return wavelengths

    def rgb(
            self,
            dataset: DatasetSchema,
    ) -> tuple[int, int, int]:
        device_wavelengths = dataset.wavelengths()
        wavelengths_rgb = [device_wavelengths[color] for color in dataset.rgb]
        wavelengths_cut = self.wavelengths(dataset=dataset)
        rgb = []
        for wavelength in wavelengths_rgb:
            rgb.append(int(np.argmin(np.abs(wavelengths_cut-wavelength))))
        return tuple(rgb)

    def image(self, path: Path, dim_order: tuple[int, int, int, int]) -> Image:
        options = ImageHandlerFactoryOptions(
            key=self.key,
            index_start=self.index_start,
            index_stop=self.index_stop,
        )
        return Image.create(
            image_format=self.format,
            filelist=self.filelist(path=path),
            dim_order=dim_order,
            options=options,
        )


class AcquisitionListSchema(Sequence, RootModel):
    root: list[AcquisitionSchema]

    @field_validator("root", mode="after")
    def is_id_unique(
            cls,
            val: list[AcquisitionSchema]
    ) -> list[AcquisitionSchema]:
        assert is_id_unique(obj=val, attrib="id")
        return val

    def id_from_dataset(self, dataset: str, item: int) -> int:
        count = -1
        for acquisition in self.root:
            if acquisition.dataset_id == dataset:
                count += 1
                if count == item:
                    return acquisition.id
        raise ValueError("Dataset id not available.")

    def get_from_dataset(self, dataset: str, item: int) -> AcquisitionSchema:
        item = self.id_from_dataset(dataset=dataset, item=item)
        return self[item]

    def __getitem__(self, item: int) -> AcquisitionSchema:
        return get_from_id(obj=self.root, index=item, attrib="id")

    def __len__(self):
        return len(self.root)


@dataclass(frozen=True)
class Database:
    acquisitions: AcquisitionListSchema
    datasets: DatasetListSchema

    @classmethod
    def create(cls) -> "Database":
        project_path = Path(__file__).resolve().parents[2]
        database_path = project_path / "data/database"
        with open(database_path / "acquisitions.json") as acq_opened:
            acquisition_dict = json.load(acq_opened)
            acquisitions = AcquisitionListSchema(root=acquisition_dict)
        with open(database_path / "datasets.json") as dataset_opened:
            datasets_dict = json.load(dataset_opened)
            datasets = DatasetListSchema(root=datasets_dict)
        return cls(
            datasets=datasets,
            acquisitions=acquisitions,
        )

    def wavelengths(self, image_id: int) -> np.ndarray[tuple[int], np.float32]:
        acquisition = self.acquisitions[image_id]
        dataset = self.datasets[acquisition.dataset_id]
        return acquisition.wavelengths(dataset=dataset)

    def image(self, image_id: int) -> Image:
        acquisition = self.acquisitions[image_id]
        dataset = self.datasets[acquisition.dataset_id]
        dataset.download()
        return acquisition.image(
            path=dataset.path,
            dim_order=dataset.dim_order,
        )

