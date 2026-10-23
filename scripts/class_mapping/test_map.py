"""Sanity checks for class_map.py. Run: python scripts/class_mapping/test_map.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from class_map import CLASS_TO_ID, CLASSES, class_id, map_label  # noqa: E402

# (raw label, expected class or None, expected reason when None)
CASES: list[tuple[str, str | None, str]] = [
    # generic containers
    ("android.widget.LinearLayout", "Layout", ""),
    ("android.widget.FrameLayout", "Layout", ""),
    ("android.widget.TableLayout", "Layout", ""),
    ("android.support.v7.widget.FitWindowsLinearLayout", "Layout", ""),
    # text, including the EditText trap
    ("android.widget.TextView", "Text", ""),
    ("android.support.v7.widget.AppCompatTextView", "Text", ""),
    ("com.yinzcam.common.android.ui.fonts.FontTextView", "Text", ""),
    ("android.support.v7.widget.AppCompatCheckedTextView", "Text", ""),
    ("android.widget.EditText", None, "Input"),
    ("android.support.v7.widget.AppCompatEditText", None, "Input"),
    ("android.widget.NumberPicker$CustomEditText", None, "Input"),
    ("android.support.design.widget.TextInputEditText", None, "Input"),
    # images vs image buttons
    ("android.widget.ImageView", "Image", ""),
    ("com.makeramen.roundedimageview.RoundedImageView", "Image", ""),
    ("android.widget.ImageButton", "Button", ""),
    ("android.support.v7.widget.AppCompatImageButton", "Button", ""),
    # buttons and toggles
    ("android.widget.Button", "Button", ""),
    ("android.widget.RadioButton", "Button", ""),
    ("android.support.v7.widget.SwitchCompat", "Button", ""),
    ("android.widget.ViewSwitcher", None, "Other"),
    ("android.widget.TextSwitcher", None, "Other"),
    ("android.support.v7.widget.AppCompatCheckBox", "Checkbox", ""),
    # bars
    ("android.support.v7.widget.Toolbar", "Toolbar", ""),
    ("com.android.internal.widget.ActionBarContainer", "Toolbar", ""),
    ("android.widget.ProgressBar", "ProgressBar", ""),
    ("android.support.v4.widget.ContentLoadingProgressBar", "ProgressBar", ""),
    # lists and scrolling
    ("android.widget.ListView", "List", ""),
    ("android.support.v7.widget.RecyclerView", "List", ""),
    ("android.widget.GridView", "List", ""),
    ("android.widget.ExpandableListView", "List", ""),
    ("android.widget.ScrollView", "ScrollView", ""),
    ("android.support.v4.widget.NestedScrollView", "ScrollView", ""),
    # tabs vs tables vs drawers
    ("android.support.design.widget.TabLayout", "Tab", ""),
    ("android.support.design.widget.TabLayout$TabView", "Tab", ""),
    ("com.android.browser.NavTabView", "Tab", ""),
    ("android.widget.TableRow", None, "Other"),
    ("android.support.v4.widget.DrawerLayout", "Drawer", ""),
    ("android.support.v7.widget.CardView", "Card", ""),
    ("android.support.v4.view.ViewPager", "ViewPager", ""),
    # web views and ads
    ("android.webkit.WebView", "WebView", ""),
    ("com.google.android.gms.ads.AdView", "Ad", ""),
    ("com.google.android.gms.ads.internal.webview.n", "Ad", ""),
    # dropped
    ("android.view.View", None, "structural"),
    ("android.view.ViewStub", None, "structural"),
    ("android.support.v7.widget.ViewStubCompat", None, "structural"),
    ("com.android.internal.policy.PhoneWindow$DecorView", None, "structural"),
    ("android.widget.SeekBar", None, "SeekBar"),
    ("android.widget.VideoView", None, "VideoView"),
    ("com.android.internal.widget.DialogTitle", None, "Dialog"),
    ("android.widget.Spinner", None, "Spinner"),
    # no match
    ("android.support.v7.widget.ab", None, "Other"),
    ("android.widget.Space", None, "Other"),
    ("android.support.v7.widget.ActionMenuView", None, "Other"),
    ("", None, "empty"),
]


def main() -> int:
    assert len(CLASSES) == 15, f"expected 15 classes, got {len(CLASSES)}"
    assert len(set(CLASSES)) == 15, "duplicate class names"
    assert sorted(CLASS_TO_ID.values()) == list(range(15)), "class ids must be 0..14"

    failures = 0
    for raw, want_name, want_reason in CASES:
        got_name, got_reason = map_label(raw)
        ok = got_name == want_name and (want_name is not None or got_reason == want_reason)
        if not ok:
            failures += 1
            print(f"FAIL {raw!r}: expected {want_name or want_reason}, got {got_name or got_reason}")
        cid = class_id(raw)
        if cid != (None if want_name is None else CLASS_TO_ID[want_name]):
            failures += 1
            print(f"FAIL {raw!r}: class_id mismatch")

    print(f"\n{len(CASES) - failures if failures == 0 else 'some'} / {len(CASES)} cases checked, failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
