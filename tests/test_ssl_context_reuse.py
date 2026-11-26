"""The suite shares the default certifi client context, and only that one.

``ssl.create_default_context`` parses the ~140 certificates in the certifi
bundle (23 ms) and every LLM client the tests build asks for one. Reusing the
context for the certifi bundle only keeps that parse once per worker without
handing a mutable ``SSLContext`` to a library that tunes one it created.
"""

import shutil
import ssl

import certifi


def test_certifi_bundle_context_is_built_once():
    first = ssl.create_default_context(cafile=certifi.where())
    second = ssl.create_default_context(cafile=certifi.where())

    assert first is second
    assert first.verify_mode == ssl.CERT_REQUIRED
    assert first.check_hostname is True
    # A real certifi trust store, not an empty stand-in.
    assert len(first.get_ca_certs()) > 100


def test_a_test_supplied_bundle_gets_its_own_context(tmp_path):
    bundle = tmp_path / "bundle.pem"
    shutil.copyfile(certifi.where(), bundle)

    first = ssl.create_default_context(cafile=str(bundle))
    second = ssl.create_default_context(cafile=str(bundle))

    assert first is not second
    assert first is not ssl.create_default_context(cafile=certifi.where())


def test_contexts_the_caller_tunes_stay_private():
    # anyio, websockets and httpcore call create_default_context() with no
    # arguments and then change what they got; those calls must not hand back
    # the shared object.
    first = ssl.create_default_context()
    second = ssl.create_default_context()

    assert first is not second
    assert first is not ssl.create_default_context(cafile=certifi.where())
