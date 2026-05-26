import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config_store import ConfigLoader


class TestConfigLoader(unittest.TestCase):
    def test_initializes_first_class_config_and_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            loader = ConfigLoader(config_dir=Path(tmp) / "config")

            loader.initialize_for_launch(workspace)

            app_config = loader.load_app_config()
            self.assertEqual(app_config["schema_version"], 1)
            self.assertFalse(app_config["first_launch_completed"])
            self.assertEqual(app_config["workspace"]["root"], str(workspace))
            self.assertTrue(loader.model_profiles_file.exists())

    def test_migrates_legacy_model_settings_to_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "Qwen3.6-27B-Q8_0.gguf"
            model_file.write_text("fake model")
            legacy_file = Path(tmp) / "model_settings.json"
            legacy_file.write_text(
                json.dumps({
                    str(model_file): {
                        "ctx_size": 128000,
                        "gpu_layers": 128,
                        "inference_settings": {"temperature": 0.6},
                    }
                })
            )
            loader = ConfigLoader(
                config_dir=Path(tmp) / "config",
                legacy_model_settings_file=legacy_file,
            )

            loader.ensure_initialized(workspace_root=tmp)
            settings = loader.get_model_profile_settings(str(model_file))

            self.assertEqual(settings["ctx_size"], 128000)
            self.assertEqual(settings["gpu_layers"], 128)
            self.assertEqual(settings["inference_settings"]["temperature"], 0.6)

    def test_saves_profile_as_structured_record_and_legacy_registry_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "model.gguf"
            model_file.write_text("fake model")
            loader = ConfigLoader(config_dir=Path(tmp) / "config")

            profile = loader.save_model_profile(
                str(model_file),
                {
                    "ctx_size": 64000,
                    "gpu_layers": 80,
                    "inference_settings": {"top_p": 0.95},
                },
                model_name="Model",
            )

            self.assertEqual(profile["name"], "Model")
            self.assertEqual(profile["load"]["ctx_size"], 64000)
            self.assertEqual(profile["inference"]["top_p"], 0.95)
            registry = loader.get_model_settings_registry()
            self.assertEqual(registry[str(model_file.resolve())]["ctx_size"], 64000)
            self.assertEqual(
                registry[str(model_file.resolve())]["inference_settings"]["top_p"],
                0.95,
            )

    def test_launch_view_consumes_first_launch_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader = ConfigLoader(config_dir=Path(tmp) / "config")
            loader.ensure_initialized(workspace_root=tmp)

            first_view = loader.get_launch_view(
                model_loaded=False,
                models_available=True,
                consume_first_launch=True,
            )
            second_view = loader.get_launch_view(
                model_loaded=False,
                models_available=True,
                consume_first_launch=True,
            )

            self.assertEqual(first_view, "discover")
            self.assertEqual(second_view, "models")

    def test_launch_view_prefers_chat_when_model_loaded_after_first_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader = ConfigLoader(config_dir=Path(tmp) / "config")
            loader.ensure_initialized(workspace_root=tmp)
            config = loader.load_app_config()
            config["first_launch_completed"] = True
            loader.save_app_config(config)

            self.assertEqual(
                loader.get_launch_view(model_loaded=True, models_available=False),
                "chat",
            )
            self.assertEqual(
                loader.get_launch_view(model_loaded=False, models_available=True),
                "models",
            )
            self.assertEqual(
                loader.get_launch_view(model_loaded=False, models_available=False),
                "discover",
            )


if __name__ == "__main__":
    unittest.main()
