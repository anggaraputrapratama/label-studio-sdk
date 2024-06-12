# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ...core.datetime_utils import serialize_datetime
from ...core.pydantic_utilities import deep_union_pydantic_dicts, pydantic_v1
from .views_update_request_data_filters import ViewsUpdateRequestDataFilters
from .views_update_request_data_ordering_item import ViewsUpdateRequestDataOrderingItem


class ViewsUpdateRequestData(pydantic_v1.BaseModel):
    """
    Custom view data
    """

    filters: typing.Optional[ViewsUpdateRequestDataFilters] = pydantic_v1.Field(default=None)
    """
    Filters for the view
    """

    ordering: typing.Optional[typing.List[ViewsUpdateRequestDataOrderingItem]] = pydantic_v1.Field(default=None)
    """
    Ordering for the view
    """

    def json(self, **kwargs: typing.Any) -> str:
        kwargs_with_defaults: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        return super().json(**kwargs_with_defaults)

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        kwargs_with_defaults_exclude_unset: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        kwargs_with_defaults_exclude_none: typing.Any = {"by_alias": True, "exclude_none": True, **kwargs}

        return deep_union_pydantic_dicts(
            super().dict(**kwargs_with_defaults_exclude_unset), super().dict(**kwargs_with_defaults_exclude_none)
        )

    class Config:
        frozen = True
        smart_union = True
        extra = pydantic_v1.Extra.allow
        json_encoders = {dt.datetime: serialize_datetime}
