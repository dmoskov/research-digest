"""Tests for the CLI: command dispatch and DIGEST_CONFIG resolution.

Config loading is the first thing a new deployment gets wrong, so every failure
mode here has to name what to fix rather than raise an ImportError traceback.
"""

from unittest.mock import patch

import pytest

from digest.cli import COMMANDS, CONFIG_ENV_VAR, load_config_from_env, main
from digest.settings import DigestConfig

VALID = DigestConfig(
    org_name="Acme",
    subtopics={
        "climate": {"name": "Climate", "description": "d", "team_context": "t"}
    },
    subtopic_topics={
        "climate": {"grid": {"name": "Grid", "description": "d", "keywords": ["grid"]}}
    },
)
NOT_A_CONFIG = "just a string"


class TestDispatch:
    def test_bare_invocation_lists_commands(self, capsys):
        assert main([]) == 0
        out = capsys.readouterr().out
        for name in COMMANDS:
            assert name in out
        assert CONFIG_ENV_VAR in out

    def test_help_flag_lists_commands(self, capsys):
        assert main(["--help"]) == 0
        assert "usage: research-digest" in capsys.readouterr().out

    def test_unknown_command_is_an_error_not_a_crash(self, capsys):
        assert main(["frobnicate"]) == 2
        assert "unknown command 'frobnicate'" in capsys.readouterr().err

    def test_every_command_has_help_text(self):
        assert all(help_text for _, help_text in COMMANDS.values())

    def test_crawl_delegates_to_the_pipeline(self):
        with patch("digest.cli.load_config_from_env"), \
             patch("digest.pipeline.main", return_value=0) as pipeline_main:
            assert main(["crawl", "--test-mode"]) == 0
        pipeline_main.assert_called_once_with(["--test-mode"])

    def test_classify_delegates_to_the_worker(self):
        with patch("digest.cli.load_config_from_env"), \
             patch("digest.classify_worker.main", return_value=0) as worker_main:
            assert main(["classify", "--workers", "8"]) == 0
        worker_main.assert_called_once_with(["--workers", "8"])


class TestConfigResolution:
    def test_missing_env_var_names_the_fix(self, monkeypatch):
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setattr("digest.settings._config", None)
        with pytest.raises(SystemExit, match=CONFIG_ENV_VAR):
            load_config_from_env()

    def test_malformed_spec_names_the_expected_format(self, monkeypatch):
        monkeypatch.setenv(CONFIG_ENV_VAR, "myorg.config")
        monkeypatch.setattr("digest.settings._config", None)
        with pytest.raises(SystemExit, match="expected 'module.path:ATTRIBUTE'"):
            load_config_from_env()

    def test_unimportable_module_reports_the_module(self, monkeypatch):
        monkeypatch.setenv(CONFIG_ENV_VAR, "no_such_module_xyz:CONFIG")
        monkeypatch.setattr("digest.settings._config", None)
        with pytest.raises(SystemExit, match="cannot import 'no_such_module_xyz'"):
            load_config_from_env()

    def test_missing_attribute_reports_the_attribute(self, monkeypatch):
        monkeypatch.setenv(CONFIG_ENV_VAR, f"{__name__}:NOPE")
        monkeypatch.setattr("digest.settings._config", None)
        with pytest.raises(SystemExit, match="has no attribute 'NOPE'"):
            load_config_from_env()

    def test_wrong_type_is_rejected(self, monkeypatch):
        """Pointing at the config *module* instead of the config object is a
        common slip; it must not be accepted as a taxonomy."""
        monkeypatch.setenv(CONFIG_ENV_VAR, f"{__name__}:NOT_A_CONFIG")
        monkeypatch.setattr("digest.settings._config", None)
        with pytest.raises(SystemExit, match="expected DigestConfig"):
            load_config_from_env()

    def test_valid_spec_installs_the_config(self, monkeypatch):
        monkeypatch.setenv(CONFIG_ENV_VAR, f"{__name__}:VALID")
        monkeypatch.setattr("digest.settings._config", None)
        load_config_from_env()
        from digest.settings import get_config

        assert get_config() is VALID

    def test_already_configured_process_ignores_the_env_var(self, monkeypatch):
        """A host app that configured itself in-band must not need the env var."""
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        from digest.settings import configure

        configure(VALID)
        load_config_from_env()  # must not raise
