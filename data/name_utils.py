"""Shared helpers for building unique clothing product names.

Used by both ``data/import_amazon_fashion.py`` (so fresh imports already produce
unique names) and ``data/migrate_clothing_names.py`` (to fix names already in
the graph). Centralised here so the two paths never drift.

Name format:  ``{brand} {base_name} #{id_last4}``
  - ``brand`` is omitted when empty
  - ``base_name`` is the current/title name (category_code with dots replaced)
  - ``id_last4`` is the last 4 chars of the product id (string form)

Idempotent: a name already ending in ``#{id_last4}`` is detected and left
untouched on re-runs, so the migration can be applied repeatedly.
"""
import re

# Matches a trailing " #dddd" disambiguating suffix added by this module
# (1-12 digits, so it won't accidentally strip a real "#" in a name).
_SUFFIX_RE = re.compile(r"\s*#\d{1,12}\s*$")


def build_unique_name(current_name, brand, product_id) -> str:
    """Compute the unique display name for a clothing item.

    Args:
        current_name: existing name (e.g. ``'Apparel Shoes Slipons'``). If the
            name was already migrated, the ``#dddd`` suffix is stripped first so
            re-running never stacks suffixes or duplicates the brand prefix.
        brand: brand string (may be empty/None).
        product_id: the item's id (int or str). Its last 4 chars form the
            disambiguating suffix.

    Returns:
        New unique name string like ``'Nike Apparel Shoes Slipons #0239'``.
    """
    pid_str = str(product_id).strip() if product_id is not None else ""
    last4 = pid_str[-4:] if pid_str else "0000"
    suffix = f" #{last4}"

    # Strip any prior suffix from a partial/previous migration, so we rebuild
    # from the human-readable base rather than stacking suffixes.
    base = _SUFFIX_RE.sub("", (current_name or "").strip()).strip()
    brand = (brand or "").strip()

    # Avoid doubling the brand prefix (e.g. base already starts with "Brand ").
    if brand and base.lower().startswith(brand.lower() + " "):
        base = base[len(brand):].strip()

    parts = [p for p in [brand, base] if p]
    full = " ".join(parts) + suffix
    return re.sub(r"\s+", " ", full).strip()


def is_name_migrated(name, product_id) -> bool:
    """True if ``name`` already carries an ``#{id_suffix}`` for this id.

    Accepts any suffix width from 4 up to the full id length, so names whose
    suffix was widened to resolve a collision are also recognised as migrated.
    """
    if not name:
        return False
    pid_str = str(product_id).strip() if product_id is not None else ""
    if not pid_str:
        return False
    name_stripped = name.rstrip()
    for width in range(4, len(pid_str) + 1):
        if name_stripped.endswith(f" #{pid_str[-width:]}"):
            return True
    return False


def extend_suffix(name, product_id, width) -> str:
    """Re-suffix ``name`` using the last ``width`` chars of the product id.

    Used to resolve collisions: when two items would share a name (same brand +
    base + last-4 id), widen the id suffix until they diverge.
    """
    pid_str = str(product_id).strip() if product_id is not None else ""
    base = _SUFFIX_RE.sub("", (name or "").strip()).strip()
    if not pid_str:
        return name
    width = min(width, len(pid_str))
    new_suffix = f" #{pid_str[-width:]}"
    return re.sub(r"\s+", " ", base + new_suffix).strip()
