pybabel update -i i18n\messages.pot -d i18n
python translate.py
pybabel compile -d i18n
