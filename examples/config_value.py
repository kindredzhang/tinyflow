from app.config.helpers import get_config_value

if __name__ == '__main__':
    # test get value from param_value -> env_key -> settings_field -> default_value
    # make sure user can set env value in .env file
    result = get_config_value(None, "LLM_BASE_URL", "LLM_BASE_URL")
    print(result)