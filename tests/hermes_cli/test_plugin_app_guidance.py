"""A pinned portable plugin's app requirement is a backend-host fact, even when the UI is remote."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import plugin_catalog_presence as presence
from hermes_cli import plugins_cmd_install as install
from hermes_cli.plugins_cmd import PluginOperationError
from hermes_platform.declaration import parse_declaration
from hermes_platform.resolver.availability import Availability

NEEDS_APP = {"extensions": {"com.nousresearch.hermes": {"servers": {"paint": {
    "app": {"linux": {"presence": "executable", "location": "/nonexistent/paint"}},
    "requires": {"app": True},
}}}}}


def _manifest_for(server):
    return {"extensions": {"com.nousresearch.hermes": {"servers": {
        server: NEEDS_APP["extensions"]["com.nousresearch.hermes"]["servers"]["paint"]}}}}


def _entry(name="nvidia-app"):
    return SimpleNamespace(name=name, title="NVIDIA App", repo="https://github.com/NousResearch/hermes-nvidia",
                           sha="0" * 40, subdir="nvidia-app")


def _decl():
    return parse_declaration("test", {"win32": {"presence": "executable", "location": "C:/NVIDIA App.exe", "version": {"kind": "pe_resource"}}},
                             {"app": True, "min_version": "11.0", "gpu": "nvidia"}, where="test")


@pytest.mark.parametrize("state,reason", [
    ("missing_app", "missing_app"), ("version_too_old", "version_too_old"),
    ("unsupported_os", "unsupported_os"), ("unsupported_gpu", "unsupported_gpu"),
])
def test_presence_distinguishes_backend_blockers(monkeypatch, state, reason):
    from hermes_platform.resolver import availability as availability_mod
    monkeypatch.setattr(availability_mod, "availability", lambda decl: Availability(
        state, min_version="11.0", version="10.9" if state == "version_too_old" else None))
    found = presence._server_presence(_decl(), None, "NVIDIA App")
    assert found.reason == reason
    assert found.state == "missing_app"  # old clients keep their card vocabulary
    assert found.required_version == "11.0"
    assert found.found_version == ("10.9" if state == "version_too_old" else None)
    assert found.target_host == "backend"


def test_unreadable_version_is_uncertain_not_a_claim_of_old_version(monkeypatch):
    from hermes_platform.resolver import availability as availability_mod
    monkeypatch.setattr(availability_mod, "availability", lambda decl: Availability(
        "version_too_old", min_version="11.0", version=None))
    assert presence._server_presence(_decl(), None, "NVIDIA App").reason == "uncertain"


def test_official_download_link_only_for_verified_catalog_identity(monkeypatch):
    from hermes_cli import plugin_catalog as catalog

    reviewed = next(e for e in catalog.load_catalog() if e.name == "nvidia-app")
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: [reviewed])
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: _manifest_for("nvidia-app"))
    assert presence.app_guidance(reviewed, "missing_app")["download_url"] == "https://www.nvidia.com/en-us/software/nvidia-app/"
    impostor = SimpleNamespace(**reviewed.__dict__)
    impostor.repo = "https://github.com/someone/other"
    assert presence.app_guidance(impostor, "missing_app")["download_url"] is None
    assert presence.app_guidance(reviewed, "unsupported_gpu")["download_url"] is None


def test_reviewed_catalog_metadata_requires_exact_pin_and_declared_app(monkeypatch):
    from hermes_cli import plugin_catalog as catalog

    reviewed = catalog.entry_from_mapping({
        "name": "paint-bridge", "repo": "https://github.com/vendor/paint-bridge", "sha": "a" * 40,
        "description": "Paint bridge.", "maintainer": "vendor", "title": "Paint Studio", "app_server": "paint",
        "app_download_url": "https://paint.example.com/download", "tier": "community",
    }, "paint-bridge")
    assert reviewed is not None
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: [reviewed])
    manifest = {"extensions": {"com.nousresearch.hermes": {"servers": {"paint": {
        "app": {"linux": {"presence": "executable", "location": "/nonexistent/paint"}},
        "requires": {"app": True},
    }}}}}
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: manifest)
    assert presence.app_guidance(reviewed, "missing_app")["download_url"] == "https://paint.example.com/download"
    assert presence.app_guidance(reviewed, "unsupported_os")["app_title"] == "Paint Studio"
    assert presence.app_guidance(reviewed, "unsupported_os")["download_url"] is None
    impostor = SimpleNamespace(**{**reviewed.__dict__, "sha": "b" * 40})
    assert presence.app_guidance(impostor, "missing_app")["download_url"] is None
    assert presence.app_guidance(impostor, "missing_app")["app_title"] == "paint-bridge"
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: {"name": "paint-bridge"})
    assert presence.app_guidance(reviewed, "missing_app")["download_url"] is None


def test_reviewed_link_does_not_name_a_different_declared_app(monkeypatch):
    from hermes_cli import plugin_catalog as catalog

    reviewed = next(e for e in catalog.load_catalog() if e.name == "nvidia-app")
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: [reviewed])
    # The reviewed pin is still identified, but the declaration no longer names the reviewed app.
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: {"extensions": {
        "com.nousresearch.hermes": {"servers": {"other-app": {
            "app": {"win32": {"presence": "executable", "location": "C:/Other.exe"}},
            "requires": {"app": True},
        }}}}})
    result = presence.app_guidance(reviewed, "missing_app")
    assert result["download_url"] is None
    assert result["app_title"] == "nvidia-app"
    # Isolate the declaration-name check from the real catalog's Windows-only gate.
    compatible = SimpleNamespace(**{**reviewed.__dict__, "platforms": []})
    assert presence.presence(compatible).sentence == "needs nvidia-app"


def test_missing_sibling_never_inherits_other_servers_vendor_link(monkeypatch):
    from hermes_cli import plugin_catalog as catalog

    reviewed = next(e for e in catalog.load_catalog() if e.name == "nvidia-app")
    # Exercise sibling aggregation separately from the real Windows-only catalog gate.
    compatible = SimpleNamespace(**{**reviewed.__dict__, "platforms": []})
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: [compatible])
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: {"extensions": {
        "com.nousresearch.hermes": {"servers": {
            "nvidia-app": {"app": {"win32": {"presence": "executable", "location": "C:/NVIDIA.exe"}},
                           "requires": {"app": True}},
            "other-app": {"app": {"win32": {"presence": "executable", "location": "C:/Other.exe"}},
                          "requires": {"app": True}},
        }}}})
    monkeypatch.setattr(presence, "_server_presence", lambda decl, live, title:
                        presence.Presence("missing_app", f"needs {title}", "missing_app")
                        if decl.name == "other-app" else presence.Presence("present", reason="available"))
    row, = presence.presence_for(["nvidia-app"])
    assert row["reason"] == "missing_app"
    assert row["download_url"] is None
    assert row["app_title"] == "nvidia-app"
    assert row["sentence"] == "needs nvidia-app"


def test_reviewed_catalog_metadata_rejects_forged_url_and_undeclared_app(monkeypatch):
    from hermes_cli import plugin_catalog as catalog

    reviewed = catalog.entry_from_mapping({
        "name": "paint-bridge", "repo": "https://github.com/vendor/paint-bridge", "sha": "a" * 40,
        "description": "Paint bridge.", "maintainer": "vendor", "title": "Paint Studio", "app_server": "paint",
        "app_download_url": "https://paint.example.com/download", "tier": "community",
    }, "paint-bridge")
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: [reviewed])
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: {
        "extensions": {"com.nousresearch.hermes": {"servers": {"paint": {"requires": {"app": False}}}}}})
    assert presence.app_guidance(reviewed, "missing_app")["download_url"] is None
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: {
        "extensions": {"com.nousresearch.hermes": {"servers": {"paint": {
            "app": {"linux": {"presence": "executable", "location": "/nonexistent/paint"}},
            "requires": {"app": True},
        }}}}})
    impostor = SimpleNamespace(**{**reviewed.__dict__, "app_download_url": "https://evil.example/download"})
    assert presence.app_guidance(impostor, "missing_app")["download_url"] is None


@pytest.mark.parametrize("name,repo,title,url", [
    ("nvidia-broadcast", "https://github.com/NousResearch/hermes-nvidia", "NVIDIA Broadcast",
     "https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-app/"),
    ("game-editor", "https://github.com/TakeshiTGAL/hermes-plugin-game-editor", "Godot 4",
     "https://godotengine.org/download/macos/"),
])
def test_other_hard_gated_plugins_have_curated_vendor_guidance(monkeypatch, name, repo, title, url):
    from hermes_cli import plugin_catalog as catalog

    entry = next(e for e in catalog.load_catalog() if e.name == name)
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: [entry])
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: _manifest_for(name))
    result = presence.app_guidance(entry, "version_too_old", required_version="4.7")
    assert result["app_title"] == title
    assert result["download_url"] == url


def test_malformed_liveness_is_uncertain_not_a_claim_the_app_is_ready(monkeypatch):
    from hermes_platform.resolver import availability as availability_mod
    monkeypatch.setattr(availability_mod, "availability", lambda decl: Availability("available"))
    found = presence._server_presence(_decl(), {"kind": "invalid"}, "NVIDIA App")
    assert found.state == "unknown" and found.reason == "uncertain"


def test_named_presence_reports_unsupported_backend_os_instead_of_hiding_row(monkeypatch):
    from hermes_cli import plugin_catalog as catalog
    from hermes_platform.host.facts import os_family

    other = "darwin" if os_family() != "darwin" else "win32"
    entry = catalog.entry_from_mapping({
        "name": "game-editor", "repo": "https://github.com/TakeshiTGAL/hermes-plugin-game-editor",
        "sha": "0" * 40, "description": "Edits scenes.", "maintainer": "fx", "tier": "community",
        "platforms": ["macos" if other == "darwin" else "windows"]}, "game-editor")
    manifest = {"extensions": {"com.nousresearch.hermes": {"servers": {"godot": {
        "app": {other: {"presence": "executable", "location":
                        "/Applications/Godot.app/Contents/MacOS/Godot" if other == "darwin" else "C:/Godot.exe"}},
        "requires": {"app": True}}}}}}
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: [entry])
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: manifest)
    row, = presence.presence_for(["game-editor"])
    assert row["state"] == "missing_app" and row["reason"] == "unsupported_os"
    assert row["download_url"] is None and row["target_host"] == "backend"


def test_one_unreadable_server_does_not_make_other_present_server_look_sufficient(monkeypatch):
    from hermes_cli import agent_plugins

    entry = _entry()
    manifest = {"extensions": {"com.nousresearch.hermes": {"servers": {"a": {}, "b": {}}}}}
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: manifest)
    monkeypatch.setattr(agent_plugins, "_server_declarations", lambda *_: {
        "a": SimpleNamespace(declaration="available", liveness=None),
        "b": SimpleNamespace(declaration="unreadable", liveness=None),
    })
    monkeypatch.setattr(presence, "_server_presence", lambda decl, live, title:
                        presence.Presence("present", reason="available") if decl == "available" else presence.UNKNOWN)
    assert presence.presence(entry).reason == "uncertain"

def test_no_requirements_sibling_does_not_mask_genuine_missing_app(monkeypatch):
    from hermes_cli import agent_plugins

    manifest = {"extensions": {"com.nousresearch.hermes": {"servers": {"a": {}, "b": {}}}}}
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: manifest)
    monkeypatch.setattr(agent_plugins, "_server_declarations", lambda *_: {
        "a": SimpleNamespace(declaration="unconstrained", liveness=None),
        "b": SimpleNamespace(declaration="missing", liveness=None),
    })
    monkeypatch.setattr(presence, "_server_presence", lambda decl, live, title:
                        presence.Presence("unknown", reason="no_requirements") if decl == "unconstrained"
                        else presence.Presence("missing_app", "needs NVIDIA App", "missing_app"))
    found = presence.presence(_entry())
    assert found.reason == "missing_app" and found.state == "missing_app"


def test_no_requirements_sibling_does_not_mask_app_not_running(monkeypatch):
    from hermes_cli import agent_plugins

    manifest = {"extensions": {"com.nousresearch.hermes": {"servers": {"a": {}, "b": {}}}}}
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: manifest)
    monkeypatch.setattr(agent_plugins, "_server_declarations", lambda *_: {
        "a": SimpleNamespace(declaration="unconstrained", liveness=None),
        "b": SimpleNamespace(declaration="stopped", liveness=None),
    })
    monkeypatch.setattr(presence, "_server_presence", lambda decl, live, title:
                        presence.Presence("unknown", reason="no_requirements") if decl == "unconstrained"
                        else presence.Presence("app_not_running", "start app", "app_not_running"))
    assert presence.presence(_entry()).reason == "app_not_running"

def test_no_requirements_sibling_does_not_mask_available_app(monkeypatch):
    from hermes_cli import agent_plugins
    manifest = {"extensions": {"com.nousresearch.hermes": {"servers": {"a": {}, "b": {}}}}}
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: manifest)
    monkeypatch.setattr(agent_plugins, "_server_declarations", lambda *_: {
        "a": SimpleNamespace(declaration="unconstrained", liveness=None),
        "b": SimpleNamespace(declaration="available", liveness=None),
    })
    monkeypatch.setattr(presence, "_server_presence", lambda decl, live, title:
                        presence.Presence("unknown", reason="no_requirements") if decl == "unconstrained"
                        else presence.Presence("present", reason="available"))
    found = presence.presence(_entry())
    assert found.reason == "available" and found.state == "present"

def test_no_requirements_alone_remains_unknown(monkeypatch):
    from hermes_platform.resolver import availability as availability_mod

    monkeypatch.setattr(availability_mod, "availability", lambda decl: Availability("no_requirements"))
    found = presence._server_presence(_decl(), None, "NVIDIA App")
    assert found.state == "unknown" and found.reason == "no_requirements"


def _github_entry(name):
    from hermes_cli import plugin_catalog as catalog

    return catalog.entry_from_mapping({
        "name": name, "repo": f"https://github.com/fx/{name}", "sha": "0" * 40,
        "description": f"{name} does things.", "maintainer": "fx", "tier": "community"}, name)


def test_named_presence_reports_a_plugin_without_app_declaration_as_no_requirements(monkeypatch):
    """The wire row, not just one server: the Desktop must not doubt an app nobody declared."""
    from hermes_cli import plugin_catalog as catalog

    entry = _github_entry("plain")
    no_app = {"extensions": {"com.nousresearch.hermes": {"servers": {"srv": {"command": "srv"}}}}}
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: [entry])
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: no_app)
    row, = presence.presence_for(["plain"])
    assert row["reason"] == "no_requirements" and row["state"] == "unknown" and row["sentence"] == ""


def test_pin_without_plugin_json_has_no_requirements_but_a_failed_read_stays_uncertain(monkeypatch):
    """A 404 is a native plugin (the install gate skips it too); a network failure proves nothing."""
    import httpx

    from hermes_cli import plugin_catalog as catalog

    entries = [_github_entry("native"), _github_entry("offline")]

    def get(url, **_kwargs):
        if "/fx/offline/" in url:
            raise httpx.ConnectError("unreachable")
        return SimpleNamespace(status_code=404, content=b"")

    monkeypatch.setattr(catalog, "load_catalog_live", lambda: entries)
    monkeypatch.setattr(presence, "_manifests", {})
    monkeypatch.setattr(httpx, "get", get)
    rows = {row["name"]: row for row in presence.presence_for(["native", "offline"])}
    assert rows["native"]["reason"] == "no_requirements"
    assert rows["offline"]["reason"] == "uncertain"


def _titled_entry(name, title):
    from hermes_cli import plugin_catalog as catalog

    return catalog.entry_from_mapping({
        "name": name, "repo": f"https://github.com/fx/{name}", "sha": "0" * 40, "title": title,
        "description": f"{name} does things.", "maintainer": "fx", "tier": "community"}, name)


def test_lone_app_requirement_keeps_the_catalog_title_without_annotation(monkeypatch):
    """pay-cli: hard-gated, titled "Pay CLI", no app_server; the name needs no reviewed vendor link."""
    from hermes_cli import plugin_catalog as catalog

    monkeypatch.setattr(catalog, "load_catalog_live", lambda: [_titled_entry("pay-cli", "Pay CLI")])
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: _manifest_for("pay-cli"))
    row, = presence.presence_for(["pay-cli"])
    assert row["sentence"] == "needs Pay CLI" and row["app_title"] == "Pay CLI"
    assert row["download_url"] is None


def test_several_unannotated_apps_do_not_borrow_the_catalog_title(monkeypatch):
    from hermes_cli import plugin_catalog as catalog

    servers = NEEDS_APP["extensions"]["com.nousresearch.hermes"]["servers"]["paint"]
    two_apps = {"extensions": {"com.nousresearch.hermes": {"servers": {"paint": servers, "print": servers}}}}
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: [_titled_entry("studio", "Paint Studio")])
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: two_apps)
    row, = presence.presence_for(["studio"])
    assert "Paint Studio" not in row["sentence"] and row["app_title"] == "studio"


def test_install_refusal_guidance_keeps_the_catalog_title_of_a_lone_app(monkeypatch, tmp_path):
    from hermes_platform.resolver import availability as availability_mod
    from hermes_cli import agent_plugins

    (tmp_path / "plugin.json").write_text("{}")
    monkeypatch.setattr(agent_plugins, "load_agent_plugin", lambda *_:
                        SimpleNamespace(server_declarations={"pay": SimpleNamespace(declaration=_decl())}))
    monkeypatch.setattr(availability_mod, "availability", lambda decl: Availability("missing_app", min_version="11.0"))
    with pytest.raises(PluginOperationError) as raised:
        install._refuse_unavailable_portable_plugin("pay-cli", tmp_path, "Pay CLI")
    assert raised.value.app_guidance["app_title"] == "Pay CLI"


def test_named_presence_reads_the_live_catalog_once_per_request(monkeypatch):
    """Each live-catalog load re-parses every in-tree entry (1.2 s on a Windows backend)."""
    from hermes_cli import plugin_catalog as catalog

    reviewed = [e for e in catalog.load_catalog() if e.name in ("nvidia-app", "nvidia-broadcast")]
    entries = [*reviewed, _github_entry("plain")]
    loads = []
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: loads.append(1) or entries)
    monkeypatch.setattr(presence, "_pinned_manifest", lambda repo, sha, subdir: _manifest_for(subdir or "srv"))
    rows = presence.presence_for(["nvidia-app", "nvidia-broadcast", "plain"])
    assert [row["name"] for row in rows] == ["nvidia-app", "nvidia-broadcast", "plain"]
    assert len(loads) == 1


def test_unreadable_installed_version_is_typed_non_looping_refusal(monkeypatch, tmp_path):
    from hermes_platform.resolver import availability as availability_mod
    from hermes_cli import agent_plugins

    (tmp_path / "plugin.json").write_text("{}")
    server = SimpleNamespace(declaration=_decl())
    monkeypatch.setattr(agent_plugins, "load_agent_plugin", lambda *_:
                        SimpleNamespace(server_declarations={"s": server}))
    monkeypatch.setattr(availability_mod, "availability", lambda decl: Availability(
        "version_too_old", min_version="11.0", version=None))
    with pytest.raises(PluginOperationError) as raised:
        install._refuse_unavailable_portable_plugin("nvidia-app", tmp_path, "NVIDIA App")
    assert raised.value.app_guidance["reason"] == "uncertain"
    assert raised.value.app_guidance["required_version"] == "11.0"
    assert raised.value.app_guidance["found_version"] is None
    assert "could not determine" in str(raised.value).lower()
    assert "update NVIDIA App" not in str(raised.value)


def test_unsupported_server_takes_priority_over_an_installable_missing_app(monkeypatch):
    from hermes_cli import agent_plugins

    manifest = {"extensions": {"com.nousresearch.hermes": {"servers": {"a": {}, "b": {}}}}}
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: manifest)
    monkeypatch.setattr(agent_plugins, "_server_declarations", lambda *_: {
        "a": SimpleNamespace(declaration="missing", liveness=None),
        "b": SimpleNamespace(declaration="unsupported", liveness=None),
    })
    monkeypatch.setattr(presence, "_server_presence", lambda decl, live, title:
                        presence.Presence("missing_app", reason="missing_app" if decl == "missing" else "unsupported_gpu"))
    found = presence.presence(_entry())
    assert found.reason == "unsupported_gpu"


def test_install_refusal_never_labels_a_sibling_app_as_reviewed_vendor(monkeypatch, tmp_path):
    from hermes_cli import agent_plugins
    from hermes_platform.resolver import availability as availability_mod

    (tmp_path / "plugin.json").write_text("{}")
    monkeypatch.setattr(agent_plugins, "load_agent_plugin", lambda *_: SimpleNamespace(server_declarations={
        "other-app": SimpleNamespace(declaration=SimpleNamespace(name="other-app", required_gpu=None)),
    }))
    monkeypatch.setattr(availability_mod, "availability", lambda *_: Availability("missing_app"))
    with pytest.raises(PluginOperationError) as raised:
        install._refuse_unavailable_portable_plugin("nvidia-app", tmp_path, "NVIDIA App", app_server="nvidia-app")
    assert "NVIDIA App" not in str(raised.value)
    assert raised.value.app_server == "other-app"

def test_install_refusal_prioritizes_unsupported_sibling_over_missing_app(monkeypatch, tmp_path):
    from hermes_cli import agent_plugins
    from hermes_platform.resolver import availability as availability_mod
    (tmp_path / "plugin.json").write_text("{}")
    monkeypatch.setattr(agent_plugins, "load_agent_plugin", lambda *_: SimpleNamespace(server_declarations={
        "nvidia-app": SimpleNamespace(declaration=SimpleNamespace(name="nvidia-app", required_gpu=None)),
        "other": SimpleNamespace(declaration=SimpleNamespace(name="other", required_gpu=None)),
    }))
    monkeypatch.setattr(availability_mod, "availability", lambda decl: Availability(
        "missing_app" if decl.name == "nvidia-app" else "unsupported_os"))
    with pytest.raises(PluginOperationError) as raised:
        install._refuse_unavailable_portable_plugin("nvidia-app", tmp_path, "NVIDIA App", app_server="nvidia-app")
    assert raised.value.app_guidance["reason"] == "unsupported_os"
    assert raised.value.app_server == "other"
    assert "NVIDIA App" not in str(raised.value)

def test_local_app_declaration_must_match_pin_before_vendor_guidance(monkeypatch):
    from hermes_cli import plugin_catalog as catalog
    reviewed = next(e for e in catalog.load_catalog() if e.name == "nvidia-app")
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: [reviewed])
    pinned = {"extensions": {"com.nousresearch.hermes": {"servers": {"nvidia-app": {
        "app": {"win32": {"presence": "executable", "location": "C:/NVIDIA.exe"}},
        "requires": {"app": True},
    }}}}}
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: pinned)
    from hermes_cli.agent_plugins import _server_declarations
    exact = _server_declarations(pinned, {"nvidia-app": {}})["nvidia-app"].declaration
    changed = parse_declaration("nvidia-app", {"win32": {"presence": "executable", "location": "C:/Other.exe"}},
                                {"app": True}, where="test")
    for actual, expect_url in ((exact, True), (changed, False)):
        refusal = PluginOperationError("missing", failure_class="incompatible")
        refusal.app_guidance = presence.app_guidance(SimpleNamespace(name="nvidia-app", repo=""), "missing_app")
        refusal.app_server = "nvidia-app"
        refusal.app_declaration = actual
        result = install._app_refusal_result(refusal, reviewed, at_reviewed_pin=True)["app_guidance"]
        assert bool(result["download_url"]) is expect_url
        assert result["app_title"] == ("NVIDIA App" if expect_url else "nvidia-app")


def test_install_refusal_carries_typed_reason_without_publishing(monkeypatch, tmp_path):
    from hermes_platform.resolver import availability as availability_mod
    from hermes_cli import agent_plugins
    from hermes_cli import plugins_cmd

    (tmp_path / "plugin.json").write_text("{}")
    server = SimpleNamespace(declaration=_decl())
    monkeypatch.setattr(agent_plugins, "load_agent_plugin", lambda *_: SimpleNamespace(server_declarations={"s": server}))
    monkeypatch.setattr(availability_mod, "availability", lambda decl: Availability(
        "missing_app", min_version="11.0"))
    with pytest.raises(PluginOperationError) as raised:
        install._refuse_unavailable_portable_plugin("nvidia-app", tmp_path, "NVIDIA App")
    # The same name the refusal sentence uses; only the vendor link needs the reviewed pin.
    assert raised.value.app_guidance == {
        "reason": "missing_app", "app_title": "NVIDIA App", "required_version": "11.0",
        "found_version": None, "download_url": None, "target_host": "backend"}


def test_post_preflight_install_refusal_survives_to_rpc_data(monkeypatch):
    from tui_gateway import server
    from hermes_cli import plugins_cmd

    def refused(_identifier, **_kwargs):
        exc = PluginOperationError("Can't install plugin 'nvidia-app': needs NVIDIA App", failure_class="incompatible")
        exc.app_guidance = {"reason": "missing_app", "app_title": "NVIDIA App",
                            "required_version": "11.0", "found_version": None,
                            "download_url": "https://www.nvidia.com/en-us/software/nvidia-app/", "target_host": "backend"}
        raise exc

    monkeypatch.setattr(install, "recorded_install", lambda fn, **kw: fn())
    monkeypatch.setattr(plugins_cmd, "_install_plugin_core", refused)
    monkeypatch.setattr(install, "_resolve_source", lambda ident, name: (_entry(), "source", [], None))
    monkeypatch.setattr(install, "_catalog_install_on_disk", lambda *_: None)
    from hermes_cli import plugins_cmd_catalog
    monkeypatch.setattr(plugins_cmd_catalog, "install_catalog_entry", lambda *_args, **_kw: refused("source"))
    response = server.handle_request({"id": 31, "method": "plugins.manage", "params": {
        "action": "install", "catalog_name": "nvidia-app", "enable": True}})
    assert response["error"]["code"] == 5026
    assert response["error"]["data"] == {
        "reason": "missing_app", "app_title": "nvidia-app", "required_version": "11.0",
        "found_version": None, "download_url": None,
        "target_host": "backend"}


def test_retry_from_disabled_installed_tree_rechecks_hard_app_gate(monkeypatch, tmp_path):
    """An earlier install on disk is not evidence the app is still installed on the backend."""
    from hermes_cli import plugins_cmd_catalog

    (tmp_path / "plugin.json").write_text("{}")
    monkeypatch.setattr(install, "_catalog_install_on_disk", lambda *_: (tmp_path, {}, "nvidia-app"))
    monkeypatch.setattr(plugins_cmd_catalog, "install_catalog_entry", lambda *_a, **_k:
                        pytest.fail("must not re-clone or bypass gate"))

    def blocked(name, tree, title, *, app_server=None):
        error = PluginOperationError("app disappeared", failure_class="incompatible")
        error.app_guidance = {"reason": "missing_app", "app_title": title, "required_version": "11.0",
                              "found_version": None, "download_url": None, "target_host": "backend"}
        raise error

    monkeypatch.setattr(install, "_refuse_unavailable_portable_plugin", blocked)
    result = install._place_tree(_entry(), "source", force=False, ref=None,
                                 assume_deps_consent=True, step=lambda *_: None)
    assert result["ok"] is False
    assert result["app_guidance"]["reason"] == "missing_app"
    assert result["app_guidance"]["download_url"] is None  # fixture has no reviewed pin


def test_off_pin_catalog_clone_does_not_label_a_different_app_with_reviewed_title(monkeypatch):
    from hermes_cli import plugin_catalog as catalog
    from hermes_cli import plugins_cmd_catalog
    from hermes_cli import plugins_cmd

    entry = next(e for e in catalog.load_catalog() if e.name == "nvidia-app")
    monkeypatch.setattr(plugins_cmd_catalog, "raise_if_removed", lambda *_: None)
    monkeypatch.setattr(plugins_cmd_catalog, "refuse_incompatible_catalog_entry", lambda *_: None)
    def refused(*args, **kwargs):
        assert kwargs["app_title"] is None
        raise PluginOperationError("blocked", failure_class="incompatible")
    monkeypatch.setattr(plugins_cmd, "_install_plugin_core", refused)
    with pytest.raises(PluginOperationError):
        plugins_cmd_catalog.install_catalog_entry(entry, force=False, ref="b" * 40)


def test_off_pin_refusal_never_uses_reviewed_vendor_metadata(monkeypatch, tmp_path):
    from hermes_cli import plugin_catalog as catalog

    entry = next(e for e in catalog.load_catalog() if e.name == "nvidia-app")
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: [entry])
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: NEEDS_APP)
    (tmp_path / "plugin.json").write_text("{}")
    monkeypatch.setattr(install, "_catalog_install_on_disk", lambda *_: (tmp_path, {}, "nvidia-app"))

    def refused(name, tree, title, *, app_server=None):
        error = PluginOperationError("app unavailable", failure_class="incompatible")
        error.app_guidance = {"reason": "missing_app", "app_title": title or name,
                              "required_version": "11.0", "found_version": None,
                              "download_url": None, "target_host": "backend"}
        raise error

    monkeypatch.setattr(install, "_refuse_unavailable_portable_plugin", refused)
    result = install._place_tree(entry, "source", force=False, ref="b" * 40,
                                 assume_deps_consent=True, step=lambda *_: None)
    assert result["app_guidance"]["download_url"] is None
    assert result["app_guidance"]["app_title"] == "nvidia-app"


def test_disabled_off_pin_tree_without_ref_does_not_gain_reviewed_link(monkeypatch, tmp_path):
    from hermes_cli import plugin_catalog as catalog
    from hermes_cli import plugins_cmd_catalog

    entry = next(e for e in catalog.load_catalog() if e.name == "nvidia-app")
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: [entry])
    monkeypatch.setattr(presence, "_pinned_manifest", lambda *_: NEEDS_APP)
    (tmp_path / "plugin.json").write_text("{}")
    monkeypatch.setattr(install, "_catalog_install_on_disk", lambda *_: (tmp_path, {}, "nvidia-app"))
    monkeypatch.setattr(plugins_cmd_catalog, "catalog_install_record", lambda *_: {"sha": "b" * 40})

    def refused(name, tree, title, *, app_server=None):
        error = PluginOperationError("app unavailable", failure_class="incompatible")
        error.app_guidance = {"reason": "missing_app", "app_title": title or name,
                              "required_version": None, "found_version": None,
                              "download_url": None, "target_host": "backend"}
        raise error

    monkeypatch.setattr(install, "_refuse_unavailable_portable_plugin", refused)
    result = install._place_tree(entry, "source", force=False, ref=None,
                                 assume_deps_consent=True, step=lambda *_: None)
    assert result["app_guidance"]["download_url"] is None
