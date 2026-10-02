"""Regression coverage for Struphy's eager and lazy public domain exports."""

from abc import ABCMeta, abstractmethod
from types import ModuleType

import pytest
from struphy import domains
from struphy.geometry.base import Domain

import generate_domains


def exported_classes(module):
    return {
        "Cuboid": type("Cuboid", (domains.Cuboid,), {"__module__": module.__name__}),
        "HollowTorus": type(
            "HollowTorus", (domains.Cuboid,), {"__module__": f"{module.__name__}.hollow_torus"}
        ),
        "AbstractDomain": ABCMeta(
            "AbstractDomain", (Domain,),
            {"__module__": module.__name__, "mapping": abstractmethod(lambda self: None)},
        ),
        "Domain": Domain,
        "ForeignDomain": type("ForeignDomain", (domains.Cuboid,), {"__module__": "other.domains"}),
        "NotADomain": str,
    }


def test_lazy_exports_are_discovered(monkeypatch):
    module = ModuleType("struphy.geometry.domains")
    exports = exported_classes(module)
    module.__dir__ = lambda: sorted(exports)
    module.__getattr__ = exports.__getitem__
    assert "Cuboid" not in vars(module)
    monkeypatch.setattr(generate_domains, "domains", module)

    assert generate_domains.domain_classes() == [
        ("Cuboid", exports["Cuboid"]),
        ("HollowTorus", exports["HollowTorus"]),
    ]


def test_eager_exports_remain_supported(monkeypatch):
    module = ModuleType("struphy.geometry.domains")
    exports = exported_classes(module)
    vars(module).update(exports)
    monkeypatch.setattr(generate_domains, "domains", module)

    assert [name for name, _ in generate_domains.domain_classes()] == ["Cuboid", "HollowTorus"]


def test_installed_struphy_exports_are_discovered():
    names = [name for name, _ in generate_domains.domain_classes()]
    assert "Cuboid" in names
    assert "HollowTorus" in names
    assert "Domain" not in names


def test_empty_discovery_fails_before_writing_a_catalogue(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_domains, "domain_classes", lambda: [])

    with pytest.raises(RuntimeError, match="No concrete domains discovered"):
        generate_domains.export_domains(tmp_path)
    assert not (tmp_path / "catalogue.json").exists()
