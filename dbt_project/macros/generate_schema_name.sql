{#
    Standard dbt override (this exact macro is dbt's own documented
    example, not a custom invention): without it, a model's `+schema:
    staging` config produces "<profiles.yml target schema>_staging"
    (e.g. "dbt_staging"), not plain "staging". This project's whole point
    of naming schemas "staging" / "silver" is the medallion-architecture
    story (bronze via dlt, silver via this mart) — a prefixed schema name
    would quietly break that naming without this override.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
