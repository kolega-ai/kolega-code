"""Shared tiktoken helpers for local token *counting*.

Local token counting must never crash on content that merely contains a tiktoken
special-token literal — e.g. ``<|endoftext|>`` in a GPT-2 / tokenizer file, or
``<|im_start|>`` in a chat-format prompt. By default ``Encoding.encode`` uses
``disallowed_special="all"`` and raises a ``ValueError`` the moment it sees such
a string in ordinary text. That is the right default for *building a request*
(you don't want to smuggle real control tokens into a prompt), but here we only
want an approximate size for context management, so those strings should be
counted as normal text. Passing ``disallowed_special=()`` disables the check —
exactly what tiktoken's own error message recommends.
"""

from typing import List

import tiktoken


class CountingEncoding:
    """A tiktoken ``Encoding`` wrapper whose ``encode`` never raises on special
    tokens.

    It forwards to the wrapped encoding with ``disallowed_special=()`` so
    special-token literals in the text are counted as ordinary text instead of
    raising. Only ``encode`` is exposed because counting is the only thing this
    is used for.
    """

    __slots__ = ("_encoding",)

    def __init__(self, encoding: "tiktoken.Encoding") -> None:
        self._encoding = encoding

    def encode(self, text: str) -> List[int]:
        return self._encoding.encode(text, disallowed_special=())


def get_counting_encoding(name: str) -> CountingEncoding:
    """Return a :class:`CountingEncoding` for the named tiktoken encoding.

    ``tiktoken.get_encoding`` caches the underlying BPE table in its own registry,
    so repeated calls are cheap; the wrapper is a trivial object around it.
    """
    return CountingEncoding(tiktoken.get_encoding(name))
