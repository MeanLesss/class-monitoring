"""A tiny custom Streamlit component for drawing/resizing MANY rectangles over an image.

Usage:
    result = seatdraw(image_url=<data-url>, rects=[], height=480, key="...")
    result is {"objects": [{"left":..., "top":..., "width":..., "height":...}, ...]}
"""

import os

import streamlit.components.v1 as components

_parent = os.path.dirname(os.path.abspath(__file__))
_frontend = os.path.join(_parent, "frontend")

_seatdraw = components.declare_component("seatdraw", path=_frontend)


def seatdraw(image_url, rects=None, height=480, key=None):
    return _seatdraw(
        image_url=image_url,
        rects=rects or [],
        height=height,
        key=key,
        default={"objects": []},
    )
