#
# Copyright (c) 2024, 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import Blueprint
from datetime import datetime

main = Blueprint('main', __name__)

from . import views, errors


def format_datetime(value: str, format='medium') -> str:
    if format == 'full':
        format= "%A, %d %B %Y, %H:%M"
    elif format == 'medium':
        format = "%d %b %Y"
    return datetime.fromisoformat(value).strftime(format)


@main.record
def register_template_filters(state):
    state.app.jinja_env.filters['format_datetime'] = format_datetime
