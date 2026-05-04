from __future__ import annotations


class ExperimentNameSanitizer:
    """
    exp_name 全体の sanitize を担当する。
    旧 ExperimentNamer の挙動を基本的に維持する。
    """

    def sanitize(self, name: str) -> str:
        name = name.replace("/_", "/")
        name = name.replace("(", "").replace(")", "")
        name = name.replace(", ", "-")
        return name
