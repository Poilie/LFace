from dataclasses import dataclass, field, fields, asdict
from datetime import date
from enum import Enum
from pydantic import BaseModel, ConfigDict
from collections import namedtuple


class PersonModel(BaseModel):
    id: str
    name: str
    position: str
    # image_urls contains a dict of image urls, with key being
    # the image type and value being the url
    image_urls: dict = field(
        default_factory=lambda: {'frontal': '', 'up': '',
                                 'left': '', 'right': '', 'down': ''}
    )
    _creation_time: date
    _update_time: date


class CompanyModel(BaseModel):
    id: str
    name: str
    _creation_time: date
    _update_time: date
