{#
  Sphinx's stock module template, with one change: every listing is filtered to
  public names. Autosummary collects a module's functions without consulting the
  leading underscore, so the default template links `_parse` and `_unique` from
  the summary tables — private helpers that autodoc then declines to document,
  leaving a dangling reference in every module that has one.
#}
{%- set public_attributes = [] %}
{%- for item in attributes if not item.startswith("_") %}
{%- set _ = public_attributes.append(item) %}
{%- endfor %}
{%- set public_functions = [] %}
{%- for item in functions if not item.startswith("_") and not (fullname == "fastrag.normalization.models" and item in ["finish", "line_offsets"]) %}
{%- set _ = public_functions.append(item) %}
{%- endfor %}
{%- set public_classes = [] %}
{%- for item in classes if not item.startswith("_") %}
{%- set _ = public_classes.append(item) %}
{%- endfor %}
{%- set public_exceptions = [] %}
{%- for item in exceptions if not item.startswith("_") %}
{%- set _ = public_exceptions.append(item) %}
{%- endfor %}
{%- set public_modules = [] %}
{%- for item in modules if not item.split(".")[-1].startswith("_") %}
{%- set _ = public_modules.append(item) %}
{%- endfor %}

{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {%- if public_attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
   {% for item in public_attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block functions %}
   {%- if public_functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
   {% for item in public_functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block classes %}
   {%- if public_classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
   {% for item in public_classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block exceptions %}
   {%- if public_exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
   {% for item in public_exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

{%- block modules %}
{%- if public_modules %}
.. rubric:: {{ _('Modules') }}

.. autosummary::
   :toctree:
   :recursive:
{% for item in public_modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
