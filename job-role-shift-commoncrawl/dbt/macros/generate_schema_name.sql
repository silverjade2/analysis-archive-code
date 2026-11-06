{# 기본 동작은 "<target schema>_<+schema>" (예: ANALYTICS_STAGING). 그대로 쓰되 의도를 남기려고 명시 #}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ target.schema }}_{{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
