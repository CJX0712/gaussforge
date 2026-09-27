"""Error taxonomy. Codes: E100 env/backend, E200 data, E300 model/fit, E400 config, E500 runtime."""

from __future__ import annotations


class GaussForgeError(Exception):
    code = "E000"

    def __init__(self, message: str) -> None:
        super().__init__(f"[{self.code}] {message}")


class BackendUnavailableError(GaussForgeError):
    code = "E100"


class DataError(GaussForgeError):
    code = "E200"


class FitError(GaussForgeError):
    code = "E300"


class ConfigError(GaussForgeError):
    code = "E400"


class RuntimeFailure(GaussForgeError):
    code = "E500"
