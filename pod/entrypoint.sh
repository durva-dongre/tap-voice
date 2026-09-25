#!/bin/sh
python -m app.main; code=$?
python -c "from app.api import terminate_self; terminate_self()"
exit $code
