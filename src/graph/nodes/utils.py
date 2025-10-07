"""Utility functions for graph nodes."""


def get_state_attr(state, attr_name, default=None):
    """
    Safely get attribute from state object or dict.

    Args:
        state: State object or dict
        attr_name: Name of attribute to get
        default: Default value if not found

    Returns:
        Attribute value or default
    """
    if isinstance(state, dict):
        return state.get(attr_name, default)
    else:
        return getattr(state, attr_name, default)