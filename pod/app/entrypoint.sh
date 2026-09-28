#!/bin/sh
if [ "${SELFTEST:-0}" = "1" ]; then
    python -m app.selftest
    exit $?
fi
python -m app.main
code=$?
python -c "from app.control import terminate_self; terminate_self()"
exit $code