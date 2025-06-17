from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List


def _parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(v.strip().strip("'\"")) for v in value.split(",") if v]


def _default_group_ids() -> List[int]:
    val = os.getenv("GROUP_IDS") or os.getenv("GROUP_ID", "")
    return _parse_int_list(val)


@dataclass
class FreshServiceConfig:
    domain: str = os.getenv("FS_DOMAIN", "")
    masked_domain: str = os.getenv("FS_VANITY", "")
    api_key: str = os.getenv("FS_API_KEY", "")
    group_ids: List[int] = field(default_factory=_default_group_ids)

    status: List[int] = field(default_factory=lambda: [2, 3, 9, 12, 18, 21])
    per_page: int = int(os.getenv("FS_PER_PAGE", "100"))
    max_pages: int = int(os.getenv("FS_MAX_PAGES", "10"))

    ignore_subject_phrases: List[str] = field(
        default_factory=lambda: ["account migration", "meter reading request"]
    )

    def __post_init__(self) -> None:
        if not self.domain:
            raise ValueError("FS_DOMAIN required")
        if not self.group_ids:
            raise ValueError("GROUP_ID(S) required")
        if not self.api_key:
            key_file = os.getenv("FS_API_KEY_FILE")
            if key_file and Path(key_file).exists():
                self.api_key = Path(key_file).read_text().strip()
        if not self.api_key:
            raise ValueError("Freshservice API key missing")