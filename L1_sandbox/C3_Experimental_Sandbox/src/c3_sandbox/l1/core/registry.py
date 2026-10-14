"""Explicit factories plus opt-in installed entry-point plugins (trusted code)."""
from __future__ import annotations
from importlib.metadata import entry_points
from typing import Callable
from .contracts import ConfigurationError

class Registry:
    def __init__(self, group: str):
        self.group = group
        self._factories: dict[str, Callable] = {}
    def register(self, name: str, factory: Callable) -> None:
        if not name or name in self._factories or not callable(factory):
            raise ConfigurationError(f'duplicate/invalid plugin: {self.group}/{name}')
        self._factories[name] = factory
    def create(self, name: str, config: dict):
        if name not in self._factories:
            raise ConfigurationError(f'unknown {self.group} plugin {name}; available: {sorted(self._factories)}')
        # Plugin-specific settings live in options; adding B2/B3 does not require
        # widening core configuration keys. Factories receive a flattened view.
        settings=dict(config); options=settings.pop('options',{})
        if not isinstance(options,dict): raise ConfigurationError('plugin options must be an object')
        for key,value in options.items():
            if key in settings and settings[key]!=value:
                raise ConfigurationError(f'conflicting plugin option: {key}')
            settings[key]=value
        return self._factories[name](settings)
    def names(self) -> list[str]:
        return sorted(self._factories)
    def load_installed(self) -> None:
        for ep in entry_points(group=self.group):
            self.register(ep.name, ep.load())
