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


def test_official_download_link_only_for_verified_catalog_identity():
    assert presence.app_guidance(_entry(), "missing_app")["download_url"] == "https://www.nvidia.com/en-us/software/nvidia-app/"
    impostor = _entry()
    impostor.repo = "https://github.com/someone/other"
    assert presence.app_guidance(impostor, "missing_app")["download_url"] is None
    assert presence.app_guidance(_entry(), "unsupported_gpu")["download_url"] is None


@pytest.mark.parametrize("name,repo,title,url", [
    ("nvidia-broadcast", "https://github.com/NousResearch/hermes-nvidia", "NVIDIA Broadcast",
     "https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-app/"),
    ("game-editor", "https://github.com/TakeshiTGAL/hermes-plugin-game-editor", "Godot 4",
     "https://godotengine.org/download/macos/"),
])
def test_other_hard_gated_plugins_have_curated_vendor_guidance(name, repo, title, url):
    entry = SimpleNamespace(name=name, repo=repo, title="")
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
        "reason": "missing_app", "app_title": "NVIDIA App", "required_version": "11.0",
        "found_version": None, "download_url": "https://www.nvidia.com/en-us/software/nvidia-app/",
        "target_host": "backend"}


def test_retry_from_disabled_installed_tree_rechecks_hard_app_gate(monkeypatch, tmp_path):
    """An earlier install on disk is not evidence the app is still installed on the backend."""
    from hermes_cli import plugins_cmd_catalog

    (tmp_path / "plugin.json").write_text("{}")
    monkeypatch.setattr(install, "_catalog_install_on_disk", lambda *_: (tmp_path, {}, "nvidia-app"))
    monkeypatch.setattr(plugins_cmd_catalog, "install_catalog_entry", lambda *_a, **_k:
                        pytest.fail("must not re-clone or bypass gate"))

    def blocked(name, tree, title):
        error = PluginOperationError("app disappeared", failure_class="incompatible")
        error.app_guidance = {"reason": "missing_app", "app_title": title, "required_version": "11.0",
                              "found_version": None, "download_url": None, "target_host": "backend"}
        raise error

    monkeypatch.setattr(install, "_refuse_unavailable_portable_plugin", blocked)
    result = install._place_tree(_entry(), "source", force=False, ref=None,
                                 assume_deps_consent=True, step=lambda *_: None)
    assert result["ok"] is False
    assert result["app_guidance"]["reason"] == "missing_app"
    assert result["app_guidance"]["download_url"] == "https://www.nvidia.com/en-us/software/nvidia-app/"
