#!/bin/sh

MODE_VALUE="${MODE:-production}"

terminate() {
    python -c "from app.control import terminate_self; terminate_self()"
}

if [ "${DEBUG_SLEEP:-0}" = "1" ] || [ "$MODE_VALUE" = "sleep" ]; then
    MAX="${SLEEP_MAX_SECONDS:-3600}"
    if [ "$MAX" = "0" ]; then
        exec sleep infinity
    fi
    sleep "$MAX"
    terminate
    exit 0
fi

if [ "${SELFTEST:-0}" = "1" ] || [ "$MODE_VALUE" = "selftest" ]; then
    SELFTEST=1 timeout -k 30 "${SELFTEST_TIMEOUT_SECONDS:-1200}" python -m app.selftest
    code=$?
    terminate
    exit $code
fi

if [ "$MODE_VALUE" = "test" ]; then
    LIMIT=$(( ${TEST_TIMEOUT_MINUTES:-90} * 60 + 900 ))
    timeout -k 30 "$LIMIT" python tests/run_suite.py
    code=$?
    if [ "$code" = "3" ] || [ "${STORAGE_BACKEND:-gcs}" = "local" ]; then
        sleep "${TEST_GRACE_SECONDS:-600}"
    fi
    terminate
    exit $code
fi

LIMIT=$(( ${POD_LIMIT_SECONDS:-7200} + 300 ))
timeout -k 30 "$LIMIT" python -m app.main
code=$?
terminate
exit $code