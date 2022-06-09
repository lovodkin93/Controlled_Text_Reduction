import json
import sys
from json_minify import json_minify
import os
from pathlib import Path
import re


def prepare_config_for_hf() -> dict:
    """
    Write minified config so huggingface can read it even if it has comments, and handle env variables
    """

    # Read non-minified config with comments
    config_file_path = sys.argv[1]
    with open(config_file_path, "r") as f:
        minified_config = json_minify(f.read())
        config = json.loads(minified_config)

    # Handle env variables templates: {env:ENV_VARIABLE}
    for key, value in config.items():
        if isinstance(value, str):
            results = re.findall(r'{env:(.*)}', value)
            for env_variable in results:
                value = value.replace(f"${{env:{env_variable}}}", os.environ[env_variable])
            
            if any(results):
                config[key] = value

    # Save minified config
    new_config_path = f"{os.environ['TMPDIR']}/controlled_reduction/minified_configs/{config_file_path}"
    Path(os.path.dirname(new_config_path)).mkdir(parents=True, exist_ok=True)
    with open(new_config_path, "w") as f:
        f.write(json.dumps(config))

    sys.argv[1] = new_config_path

    return config