from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Tuple
import pyvista as pv

Color = Tuple[float, float, float]


@dataclass(frozen=True)
class MeshComponent:
    name : str
    mesh : pv.PolyData
    color: Color

    @property
    def bounds(self):
        return self.mesh.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)

    @property
    def center(self):
        b = self.bounds
        return ((b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2)


class MeshComponentSet:
    def __init__(self, components: Dict[str, MeshComponent]):
        self._components = dict(components)

    def get(self, name: str) -> MeshComponent:
        return self._components[name]

    def names(self) -> Iterable[str]:
        return self._components.keys()

    def items(self):
        return self._components.items()

    def __iter__(self) -> Iterator[MeshComponent]:
        return iter(self._components.values())

    def to_legacy_dict(self) -> dict:
        """
        既存コード(pv_box_array_multi_type_obj.cast_mesh_to_box_array)が
        dict形式を要求する間だけ使う「変換アダプタ」。
        """
        out = {}
        for name, c in self._components.items():
            out[name] = {"mesh": c.mesh, "color": list(c.color)}
        return out
