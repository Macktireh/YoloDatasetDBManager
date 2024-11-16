from pydantic import BaseModel, Field, field_validator


class ParamsConnection(BaseModel):
    """Encapsulates and validates PostgreSQL connection parameters."""

    dbname: str = Field(description="Database name.")
    user: str = Field(description="Login username.")
    password: str = Field(description="User password.")
    host: str = Field(description="PostgreSQL server address.")
    port: int = Field(default=5432, description="PostgreSQL server port.")

    @field_validator("dbname", "user", "password", "host")
    def validate_non_empty(cls, value, field) -> str:
        if not value:
            raise ValueError(f"`{field.name}` cannot be empty.")
        return value

    @field_validator("port")
    def validate_port(cls, value) -> str:
        if not isinstance(value, int):
            raise ValueError("`port` must be an integer.")
        return value
