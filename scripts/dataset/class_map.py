"""Raw RICO / Android view class -> 15 project classes.

Rules are applied to the *simple* name of the raw label (the part after the last
'.' or '$'), except the ad-web-view rule which looks at the whole label.

The order of `RULES` matters: specific classes come first so a generic rule never
steals them (EditText before Text, TabLayout before Layout, ImageButton before
Image, ...). First match wins; no match means "Other", which preprocessing drops.

Decisions taken with the user (see docs/memory.md):
  ImageButton, RadioButton, Switch/SwitchCompat -> Button
  Spinner, ActionMenuView, ActionMenuItemView   -> Other (dropped)
  gms.ads.internal.webview.*                    -> Ad
"""

from __future__ import annotations

import re

# Fixed order = YOLO class ids 0..14. Never reorder without re-exporting labels.
CLASSES: list[str] = [
    "Layout", "Text", "Image", "Button", "Toolbar", "ProgressBar", "List", "WebView",
    "Tab", "Drawer", "Ad", "Card", "ScrollView", "Checkbox", "ViewPager",
]
CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

# Structural containers: no visual element of their own.
STRUCTURAL: set[str] = {"View", "ViewStub", "ViewStubCompat", "DecorView"}

# Classes present in RICO but dropped: too few elements to learn a detector class.
LOW_COUNT: set[str] = {"Input", "SeekBar", "Dialog", "VideoView", "Spinner"}

# (regex, target class, which text to match). Runnable in order, first match wins.
#   scope "name"  = simple class name (after the last '.' or '$')
#   scope "label" = whole raw label (for obfuscated names like gms...webview.n)
# target "__x__" means dropped, x is the reason reported in stats.
RULES: list[tuple[str, str | None, str]] = [
    # --- structural, dropped -------------------------------------------------
    (r"^(View|ViewStub|ViewStubCompat|DecorView)$", "__structural__", "name"),
    # --- ad web views: whole label, ads win over WebView ---------------------
    (r"ads.*webview", "Ad", "label"),
    (r"ads.*adview", "Ad", "label"),
    # --- dropped low-count classes ------------------------------------------
    (r"EditText", "__Input__", "name"),
    (r"SeekBar", "__SeekBar__", "name"),
    (r"VideoView", "__VideoView__", "name"),
    (r"Dialog", "__Dialog__", "name"),
    (r"Spinner", "__Spinner__", "name"),
    # --- specific classes before the generic ones ---------------------------
    (r"CheckBox", "Checkbox", "name"),
    (r"TabLayout|TabView|TabWidget|TabHost", "Tab", "name"),
    (r"WebView", "WebView", "name"),
    (r"DrawerLayout", "Drawer", "name"),
    (r"CardView", "Card", "name"),
    (r"ViewPager", "ViewPager", "name"),
    (r"RecyclerView|ListView|GridView", "List", "name"),
    (r"ScrollView", "ScrollView", "name"),
    (r"Toolbar|ActionBar", "Toolbar", "name"),
    (r"ProgressBar", "ProgressBar", "name"),
    (r"AdView", "Ad", "name"),
    # --- clickable widgets ---------------------------------------------------
    (r"ImageButton", "Button", "name"),
    (r"Button", "Button", "name"),           # also catches RadioButton
    (r"Switch(?!er)", "Button", "name"),     # Switch/SwitchCompat, not ViewSwitcher
    # --- generic visual widgets ---------------------------------------------
    (r"ImageView", "Image", "name"),
    (r"TextView", "Text", "name"),
    (r"Layout$", "Layout", "name"),
]

_COMPILED: list[tuple[re.Pattern[str], str | None, str]] = [
    (re.compile(pat, re.IGNORECASE), target, scope) for pat, target, scope in RULES
]


def simple_name(raw_class: str) -> str:
    """`android.support.v7.widget.AppCompatTextView` -> `AppCompatTextView`."""
    return re.split(r"[.$]", raw_class.strip())[-1]


def map_label(raw_class: str) -> tuple[str | None, str]:
    """Return (class name or None, reason).

    The reason is the class name on success, or the drop reason
    (`structural`, `Input`, `Other`, ...) on failure.
    """
    raw_class = (raw_class or "").strip()
    if not raw_class:
        return None, "empty"

    name = simple_name(raw_class)
    for pattern, target, scope in _COMPILED:
        if pattern.search(raw_class if scope == "label" else name):
            if target is None:
                return None, "Other"
            if target.startswith("__"):
                return None, target.strip("_")
            return target, target
    return None, "Other"


def class_id(raw_class: str) -> int | None:
    """YOLO class id for a raw label, or None if the element is dropped."""
    name, _ = map_label(raw_class)
    return None if name is None else CLASS_TO_ID[name]
