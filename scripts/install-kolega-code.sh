#!/bin/sh

set -eu

PACKAGE_NAME="kolega-code"
INSTALL_URL="https://astral.sh/uv/install.sh"

info() {
    printf '%s\n' "$*"
}

fail() {
    printf 'kolega-code install: %s\n' "$*" >&2
    exit 1
}

have() {
    command -v "$1" >/dev/null 2>&1
}

install_uv() {
    info "Installing uv..."

    if have curl; then
        curl -LsSf "$INSTALL_URL" | sh
    elif have wget; then
        wget -qO- "$INSTALL_URL" | sh
    else
        fail "curl or wget is required to install uv"
    fi

    PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    export PATH

    have uv || fail "uv was installed, but it is not available on PATH"
}

case "$(uname -s 2>/dev/null || printf unknown)" in
    Darwin|Linux)
        ;;
    *)
        fail "this installer supports macOS and Linux; install directly with: uv tool install $PACKAGE_NAME"
        ;;
esac

if ! have uv; then
    install_uv
else
    info "Using uv: $(uv --version)"
fi

PACKAGE_SPEC="$PACKAGE_NAME"
if [ "${KOLEGA_CODE_VERSION:-}" ]; then
    PACKAGE_SPEC="$PACKAGE_NAME==$KOLEGA_CODE_VERSION"
fi

info "Installing $PACKAGE_SPEC..."
uv tool install --force --upgrade "$PACKAGE_SPEC"

if ! have "$PACKAGE_NAME"; then
    uv tool update-shell >/dev/null 2>&1 || true
    PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    export PATH
fi

if ! have "$PACKAGE_NAME"; then
    cat >&2 <<EOF
kolega-code was installed, but the executable is not available in this shell.
Run:

  uv tool update-shell

Then restart your shell and run:

  kolega-code --version
EOF
    exit 1
fi

"$PACKAGE_NAME" --version
info "Kolega Code is installed. Run: kolega-code ."
