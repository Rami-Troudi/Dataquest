from __future__ import annotations

from core.constants import BUNDLE_NAME_BY_ID


def bundle_name(bundle_id: int) -> str:
    return BUNDLE_NAME_BY_ID.get(bundle_id, str(bundle_id))
