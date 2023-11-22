from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import catboost
else:
    try:
        import catboost
    except ImportError:
        catboost = None

__all__ = ["catboost"]

