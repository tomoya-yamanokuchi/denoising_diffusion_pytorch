from __future__ import annotations


DEFAULT_AXIS_ORDER = ("z", "x", "y")


def build_action_table(side_length: int, axis_order=DEFAULT_AXIS_ORDER) -> dict:
    """Build the cutting-action index table used by the visualizer.

    This is the standalone equivalent of ``dismantling_env.get_action_table``.
    The original implementation assigns action indices by iterating over axes
    in the order ``["z", "x", "y"]`` and then over slice locations from
    ``0`` to ``side_length - 1``.

    Args:
        side_length: Number of voxel slices along each axis.
        axis_order: Axis order used to assign consecutive action indices.

    Returns:
        Mapping from action index to ``{"axis": axis_name, "loc": slice_index}``.
    """
    side_length = int(side_length)
    if side_length <= 0:
        raise ValueError(f"side_length must be positive, got {side_length}")

    if len(axis_order) == 0:
        raise ValueError("axis_order must contain at least one axis")

    valid_axes = set(DEFAULT_AXIS_ORDER)
    invalid_axes = [axis for axis in axis_order if axis not in valid_axes]
    if invalid_axes:
        raise ValueError(f"axis_order contains unsupported axes: {invalid_axes}")

    action_table = {}
    action_idx = 0

    for axis in axis_order:
        for loc in range(side_length):
            action_table[action_idx] = {"axis": axis, "loc": loc}
            action_idx += 1

    return action_table


def get_action_table(grid_config: dict) -> dict:
    """Compatibility wrapper for code that passes a grid_config dict."""
    if "side_length" not in grid_config:
        raise KeyError("grid_config must contain 'side_length'")
    return build_action_table(side_length=int(grid_config["side_length"]))
