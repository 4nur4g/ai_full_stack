from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PBP AI"
    openai_api_key: str
    model_config = SettingsConfigDict(env_file=".env")
    psql_username: str
    psql_password: str
    psql_host: str
    psql_port: int
    psql_database: str
    psql_sslmode: str


settings = Settings()
