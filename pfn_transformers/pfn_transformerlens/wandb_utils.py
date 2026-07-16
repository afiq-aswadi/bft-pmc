"""Utilities for constructing wandb run names from config fields."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any


@dataclass
class RunNameScheme:
    """Defines which config fields to include in run names and use for filtering.

    Example:
        >>> scheme = RunNameScheme.from_templates(
        ...     model={'n_layers': None, 'd_model': None},
        ...     data={'num_tasks': None}
        ... )
        >>> scheme.model_fields
        ('n_layers', 'd_model')
    """

    model_fields: tuple[str, ...] = ()
    training_fields: tuple[str, ...] = ()
    data_fields: tuple[str, ...] = ()

    @classmethod
    def from_templates(
        cls,
        model: Any | None = None,
        training: Any | None = None,
        data: Any | None = None,
    ) -> "RunNameScheme":
        """Create scheme from template configs or dicts.

        Pass dicts or dataclass instances with only the fields you care about.
        Values don't matter, only keys/field names are extracted.

        Args:
            model: Model config template (dict or dataclass)
            training: Training config template (dict or dataclass)
            data: Data config template (dict or dataclass)

        Returns:
            RunNameScheme with extracted field names

        Example:
            >>> scheme = RunNameScheme.from_templates(
            ...     model={'n_layers': None, 'd_model': None, 'mask_type': None},
            ...     data={'num_tasks': None}
            ... )
        """
        return cls(
            model_fields=tuple(_extract_fields_from_template(model)),
            training_fields=tuple(_extract_fields_from_template(training)),
            data_fields=tuple(_extract_fields_from_template(data)),
        )


_DEFAULT_RUN_NAME_FIELDS: dict[str, tuple[str, ...]] = {
    "model": ("d_model", "n_layers", "n_heads"),
    "training": ("learning_rate", "num_steps", "batch_size"),
    "data": ("num_tasks", "task_diversity"),
}


def create_run_name(
    *,
    base: str,
    model_config: Any | None = None,
    training_config: Any | None = None,
    data_config: Any | None = None,
    scheme: RunNameScheme | None = None,
    include_fields: dict[str, Sequence[str]] | None = None,
    extra: Mapping[str, Any] | None = None,
    max_length: int = 128,
) -> str:
    """
    Construct a wandb run name from selected configuration fields.

    Args:
        base: Prefix for the run name (e.g., task identifier).
        model_config: Model configuration dataclass or mapping.
        training_config: Training configuration dataclass or mapping.
        data_config: Data/generator configuration dataclass or mapping.
        scheme: RunNameScheme defining which fields to include (overrides include_fields).
        include_fields: Optional overrides for which fields to include per section.
        extra: Additional key/value pairs to append to the name.
        max_length: Maximum length of the resulting name.

    Returns:
        Sanitised run name string containing the selected fields.

    Example:
        >>> scheme = RunNameScheme.from_templates(
        ...     model={'n_layers': None, 'd_model': None},
        ...     data={'num_tasks': None}
        ... )
        >>> create_run_name(base="pfn", model_config=cfg, scheme=scheme)
    """
    if not base:
        raise ValueError("base must be a non-empty string")

    if scheme is not None:
        field_map = {
            "model": scheme.model_fields,
            "training": scheme.training_fields,
            "data": scheme.data_fields,
        }
    else:
        field_map = include_fields or _DEFAULT_RUN_NAME_FIELDS
    sections: dict[str, tuple[Any | None, Sequence[str]]] = {
        "model": (model_config, field_map.get("model", ())),
        "training": (training_config, field_map.get("training", ())),
        "data": (data_config, field_map.get("data", ())),
    }

    components: list[str] = [_slugify(base)]
    for section, (src, fields) in sections.items():
        if src is None or not fields:
            continue
        values = _extract_fields(src, fields)
        if not values:
            continue
        tokens: list[str] = [section]
        for field in fields:
            if field not in values:
                continue
            value = values[field]
            token = f"{field}{_normalise_value(value)}"
            tokens.append(token)
        components.append(_slugify("-".join(tokens)))

    if extra:
        for key, value in extra.items():
            token = f"{key}{_normalise_value(value)}"
            components.append(_slugify(token))

    run_name = "-".join(filter(None, components))
    if len(run_name) > max_length:
        run_name = run_name[:max_length].rstrip("-")
    return run_name


def _extract_fields(source: Any, fields: Sequence[str]) -> dict[str, Any]:
    data = _to_mapping(source)
    return {field: data[field] for field in fields if field in data}


def _extract_fields_from_template(template: Any | None) -> list[str]:
    """Extract field names from a template dict or dataclass instance.

    Args:
        template: Dict or dataclass instance (values don't matter)

    Returns:
        List of field names
    """
    if template is None:
        return []

    if isinstance(template, Mapping):
        return [str(key) for key in template]

    if is_dataclass(template):
        from dataclasses import fields

        return [f.name for f in fields(template)]

    if hasattr(template, "__dict__"):
        return [k for k in vars(template).keys() if not k.startswith("_")]

    return []


def _to_mapping(source: Any | None) -> dict[str, Any]:
    if source is None:
        return {}
    if is_dataclass(source):
        return dict(asdict(source))
    if isinstance(source, Mapping):
        return dict(source)
    if hasattr(source, "__dict__"):
        return {
            key: value for key, value in vars(source).items() if not key.startswith("_")
        }
    return {}


def _normalise_value(value: Any) -> str:
    if isinstance(value, float):
        formatted = f"{value:.4g}"
        return formatted.rstrip("0").rstrip(".")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _slugify(token: str) -> str:
    lowered = token.lower()
    cleaned: list[str] = []
    last_was_dash = False
    for char in lowered:
        if char.isalnum() or char in {"_", "."}:
            cleaned.append(char)
            last_was_dash = False
        else:
            if not last_was_dash:
                cleaned.append("-")
                last_was_dash = True
    slug = "".join(cleaned).strip("-")
    return slug or "run"


__all__ = [
    "RunNameScheme",
    "create_run_name",
]
